"""Intake Agent — converses with user to produce a ProblemConfig.

Uses the Claude Agent SDK to run a conversation with data inspection tools.
The agent saves configs via a save_config MCP tool that validates against
the Pydantic schema and writes problem_config.json to the run directory.
"""

import asyncio
import json
import re
from collections.abc import Callable, Awaitable
from pathlib import Path

import anyio
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    AssistantMessage,
    ResultMessage,
    TextBlock,
)

from src.intake.prompts import INTAKE_SYSTEM_PROMPT
from src.intake.tools import describe_column, list_unique_values, peek_columns, sample_rows
from src.models.problem_config import ProblemConfig


try:
    from claude_agent_sdk import tool, create_sdk_mcp_server
except ImportError:
    tool = None
    create_sdk_mcp_server = None


def _build_mcp_server(run_dir: str):
    """Build an MCP server with data inspection and config-saving tools."""

    async def _peek(args: dict) -> dict:
        return {"content": [{"type": "text", "text": peek_columns(args["csv_path"])}]}

    async def _sample(args: dict) -> dict:
        return {"content": [{"type": "text", "text": sample_rows(args["csv_path"], args.get("n", 10))}]}

    async def _describe(args: dict) -> dict:
        return {"content": [{"type": "text", "text": describe_column(args["csv_path"], args["column"])}]}

    async def _unique(args: dict) -> dict:
        return {"content": [{"type": "text", "text": list_unique_values(args["csv_path"], args["column"])}]}

    async def _save_config(args: dict) -> dict:
        """Validate and save ProblemConfig JSON to the run directory."""
        config_json = args["config_json"]
        try:
            config = ProblemConfig.model_validate_json(config_json)
            config_path = Path(run_dir) / "problem_config.json"
            config_path.write_text(config.model_dump_json(indent=2))
            return {
                "content": [{
                    "type": "text",
                    "text": (
                        f"Config saved successfully to {config_path}.\n"
                        f"Validated fields: {len(config.model_fields_set)} set, "
                        f"{len(config.decision_variables)} decision variables, "
                        f"{len(config.uncertain_parameters)} uncertain parameters, "
                        f"{len(config.cost_structure)} cost items, "
                        f"{len(config.constraints)} constraints.\n\n"
                        f"Now present a readable summary to the user for approval."
                    ),
                }],
            }
        except Exception as exc:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Validation FAILED: {exc}\n\nFix the JSON and try again.",
                }],
                "isError": True,
            }

    peek_tool = tool(
        "peek_columns",
        "Return column names and dtypes from a CSV file",
        {"csv_path": str},
    )(_peek)

    sample_tool = tool(
        "sample_rows",
        "Return first N rows from a CSV as a formatted table",
        {"csv_path": str, "n": int},
    )(_sample)

    describe_tool = tool(
        "describe_column",
        "Return summary statistics for a single CSV column",
        {"csv_path": str, "column": str},
    )(_describe)

    unique_tool = tool(
        "list_unique_values",
        "Return unique values for a CSV column (capped at 50)",
        {"csv_path": str, "column": str},
    )(_unique)

    save_tool = tool(
        "save_config",
        "Validate a ProblemConfig JSON string against the Pydantic schema and save "
        "it to the run directory. Call this INSTEAD of outputting raw JSON in chat. "
        "If validation fails, you will get an error message — fix and retry.",
        {"config_json": str},
    )(_save_config)

    return create_sdk_mcp_server(
        "intake-tools",
        tools=[peek_tool, sample_tool, describe_tool, unique_tool, save_tool],
    )


_TOOL_CALL_PATTERN = re.compile(r'^mcp__\w+__\w+\(.*\)$', re.DOTALL)


def _is_tool_call_text(text: str) -> bool:
    """Check if text is a tool call string emitted by the Claude Code subprocess."""
    stripped = text.strip()
    return bool(_TOOL_CALL_PATTERN.match(stripped))


async def _default_callback(event: dict) -> None:
    """Default callback that prints to stdout (CLI behavior)."""
    if event["type"] == "message":
        print(f"\nAgent: {event['text']}")
    elif event["type"] == "tool_call":
        print(f"\n  [calling {event['tool']}]")
    elif event["type"] == "tool_result":
        preview = event["result"][:200] if len(event["result"]) > 200 else event["result"]
        print(f"  [result: {preview}]")


async def _process_response(
    client: ClaudeSDKClient,
    callback: Callable[[dict], Awaitable[None]],
    usage_log: list[dict],
    run_dir: str,
) -> ProblemConfig | None:
    """Process agent response messages. Check if config was saved to disk.

    Args:
        client: Active ClaudeSDKClient session.
        callback: Async callback for event routing.
        usage_log: List to append usage data to.
        run_dir: Run directory to check for saved config.

    Returns:
        ProblemConfig if one was saved to disk, None otherwise.
    """
    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            if message.usage:
                usage_log.append(message.usage)

            for block in message.content:
                if isinstance(block, TextBlock):
                    if _is_tool_call_text(block.text):
                        # Tool call text — suppress from chat
                        continue
                    await callback({
                        "type": "message",
                        "agent": "intake",
                        "text": block.text,
                    })
                elif hasattr(block, "name"):
                    # Tool use block — suppress from chat
                    pass

    # Check if the agent saved a valid config file
    config_path = Path(run_dir) / "problem_config.json"
    if config_path.exists():
        try:
            config = ProblemConfig.model_validate_json(config_path.read_text())
            await callback({
                "type": "config_ready",
                "config": json.loads(config.model_dump_json()),
            })
            return config
        except Exception as exc:
            print(f"Saved config failed re-validation: {exc}")

    return None


async def run_intake_agent(
    csv_path: str,
    run_dir: str,
    initial_description: str = "",
    callback: Callable[[dict], Awaitable[None]] | None = None,
    user_input_queue: asyncio.Queue | None = None,
) -> tuple[ProblemConfig | None, list[dict]]:
    """Run the intake agent conversation and return a ProblemConfig.

    Args:
        csv_path: Path to the CSV file for inspection.
        run_dir: Run directory for saving the config file.
        initial_description: Optional initial problem description from user.
        callback: Optional async callback for event routing.
        user_input_queue: Optional async queue for receiving user messages.
            If None, falls back to stdin input() for CLI mode.

    Returns:
        Tuple of (ProblemConfig or None if failed, list of usage dicts).
    """
    if callback is None:
        callback = _default_callback

    mcp_server = _build_mcp_server(run_dir)
    usage_log: list[dict] = []

    prompt = (
        f"## Data File\n\n"
        f"The CSV file you will be working with is at: {csv_path}\n"
        f"Pass this exact path as the `csv_path` argument to all your data inspection tools.\n\n"
        f"## Your First Step\n\n"
        f"Call `peek_columns` with csv_path=\"{csv_path}\" to see what columns are available, "
        f"then call `sample_rows` with the same path to see example data.\n\n"
    )
    if initial_description:
        prompt += (
            f"## Problem Description\n\n"
            f"{initial_description}\n\n"
            f"Use this description along with the data to formulate the ProblemConfig. "
            f"Ask follow-up questions for anything that is missing or unclear.\n\n"
        )
    else:
        prompt += (
            "## Task\n\n"
            "The user has not provided a problem description yet. Start by inspecting "
            "the data, then ask the user to describe their optimization problem. "
            "Use what you learn from the data to ask targeted follow-up questions.\n\n"
        )
    prompt += (
        "## Saving the Config\n\n"
        "When you have gathered enough information:\n"
        "1. Build the ProblemConfig JSON string.\n"
        "2. Call the `save_config` tool with that JSON. It will validate against the "
        "Pydantic schema and save it to disk. If validation fails, fix the errors and retry.\n"
        "3. After a successful save, present a SHORT, READABLE summary to the user — "
        "NOT the raw JSON. Summarize: objective, key costs, sizes, categories, constraints, "
        "and data requirements in plain English.\n"
        "4. Ask the user if they approve or want changes.\n\n"
        "Do NOT output raw JSON in the chat. Use the save_config tool instead.\n\n"
        "Key fields downstream modules depend on:\n"
        "- `data_requirements.required_columns`: the data cleaning agent uses this to "
        "know which columns to produce in the cleaned dataset\n"
        "- `uncertain_parameters[].data_column`: must match actual column names in the CSV "
        "so the prediction module can find them\n"
        "- `cost_structure`: the optimizer uses these exact keys to set up the objective function\n"
    )

    options = ClaudeAgentOptions(
        system_prompt=INTAKE_SYSTEM_PROMPT,
        mcp_servers={"intake": mcp_server},
        allowed_tools=[],
        permission_mode="bypassPermissions",
        max_turns=30,
    )

    config = None

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        config = await _process_response(client, callback, usage_log, run_dir)

        # Conversation loop — wait for user follow-ups until config is saved
        while config is None:
            if user_input_queue is not None:
                await callback({"type": "__waiting__"})
                user_input = await user_input_queue.get()
            else:
                user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            await client.query(user_input)
            config = await _process_response(client, callback, usage_log, run_dir)

    return config, usage_log

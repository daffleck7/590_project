"""Intake Agent — converses with user to produce a ProblemConfig.

Uses the Claude Agent SDK to run a conversation with data inspection tools.
"""

import json
import re

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


def _build_mcp_server():
    """Build an MCP server with data inspection tools."""
    peek_tool = tool(
        "peek_columns",
        "Return column names and dtypes from a CSV file",
        {"csv_path": str},
    )(lambda args: {"content": [{"type": "text", "text": peek_columns(args["csv_path"])}]})

    sample_tool = tool(
        "sample_rows",
        "Return first N rows from a CSV as a formatted table",
        {"csv_path": str, "n": int},
    )(lambda args: {"content": [{"type": "text", "text": sample_rows(args["csv_path"], args.get("n", 10))}]})

    describe_tool = tool(
        "describe_column",
        "Return summary statistics for a single CSV column",
        {"csv_path": str, "column": str},
    )(lambda args: {"content": [{"type": "text", "text": describe_column(args["csv_path"], args["column"])}]})

    unique_tool = tool(
        "list_unique_values",
        "Return unique values for a CSV column (capped at 50)",
        {"csv_path": str, "column": str},
    )(lambda args: {"content": [{"type": "text", "text": list_unique_values(args["csv_path"], args["column"])}]})

    return create_sdk_mcp_server(
        "intake-tools",
        tools=[peek_tool, sample_tool, describe_tool, unique_tool],
    )


def _extract_json_from_text(text: str) -> str | None:
    """Extract JSON from a text response, looking for ```json blocks."""
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return None


async def run_intake_agent(
    csv_path: str,
    initial_description: str = "",
) -> tuple[ProblemConfig | None, list[dict]]:
    """Run the intake agent conversation and return a ProblemConfig.

    Args:
        csv_path: Path to the CSV file for inspection.
        initial_description: Optional initial problem description from user.

    Returns:
        Tuple of (ProblemConfig or None if failed, list of usage dicts).
    """
    mcp_server = _build_mcp_server()
    usage_log = []

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
        "## Output Format\n\n"
        "When you have gathered enough information, produce a complete ProblemConfig "
        "as a JSON code block wrapped in ```json ... ``` markers. This JSON will be "
        "parsed by a Pydantic model and passed to the next pipeline stage (data cleaning), "
        "so every field must be present and correctly typed.\n\n"
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
        allowed_tools=["Read"],
        permission_mode="bypassPermissions",
        max_turns=30,
    )

    config = None

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                if message.usage:
                    usage_log.append(message.usage)

                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"\nAgent: {block.text}")

                        json_str = _extract_json_from_text(block.text)
                        if json_str:
                            try:
                                config = ProblemConfig.model_validate_json(json_str)
                            except Exception:
                                pass

            elif isinstance(message, ResultMessage):
                if message.result:
                    json_str = _extract_json_from_text(message.result)
                    if json_str:
                        try:
                            config = ProblemConfig.model_validate_json(json_str)
                        except Exception:
                            pass

        # Interactive loop: user reviews config
        while config is None:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            await client.query(user_input)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    if message.usage:
                        usage_log.append(message.usage)
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(f"\nAgent: {block.text}")
                            json_str = _extract_json_from_text(block.text)
                            if json_str:
                                try:
                                    config = ProblemConfig.model_validate_json(json_str)
                                except Exception:
                                    pass

    return config, usage_log

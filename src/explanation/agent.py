"""Explanation Agent — produces a final stakeholder report from pipeline outputs.

Uses the Claude Agent SDK to read all summaries and results, then generates
a well-formatted recommendation report.
"""

import json
import re
from collections.abc import Callable, Awaitable
from pathlib import Path

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    AssistantMessage,
    ResultMessage,
    TextBlock,
)

from src.explanation.prompts import EXPLANATION_SYSTEM_PROMPT


try:
    from claude_agent_sdk import tool, create_sdk_mcp_server
except ImportError:
    tool = None
    create_sdk_mcp_server = None


def _build_mcp_server(run_dir: str):
    """Build MCP server with file reading and report saving tools."""

    async def _read_file(args: dict) -> dict:
        """Read a file from the run directory."""
        filename = args["filename"]
        file_path = Path(run_dir) / filename
        if not file_path.exists():
            return {
                "content": [{"type": "text", "text": f"File not found: {filename}"}],
                "isError": True,
            }
        content = file_path.read_text(encoding="utf-8", errors="replace")
        # Truncate very large files
        if len(content) > 10000:
            content = content[:10000] + f"\n\n... (truncated, {len(content)} chars total)"
        return {"content": [{"type": "text", "text": content}]}

    async def _list_files(args: dict) -> dict:
        """List all files in the run directory."""
        files = sorted(Path(run_dir).iterdir())
        lines = [f"  {f.name} ({f.stat().st_size:,} bytes)" for f in files if f.is_file()]
        return {"content": [{"type": "text", "text": "Files in run directory:\n" + "\n".join(lines)}]}

    async def _save_report(args: dict) -> dict:
        """Save the final report."""
        report_path = Path(run_dir) / "final_report.md"
        report_path.write_text(args["report"], encoding="utf-8")
        return {"content": [{"type": "text", "text": f"Report saved to {report_path}"}]}

    read_tool = tool(
        "read_file",
        "Read a file from the run directory. Use this to read agent summaries, "
        "CSV results, and the ProblemConfig.",
        {"filename": str},
    )(_read_file)

    list_tool = tool(
        "list_files",
        "List all files in the run directory to see what's available.",
        {},
    )(_list_files)

    save_tool = tool(
        "save_report",
        "Save the final formatted report as Markdown. Call this as your FINAL action.",
        {"report": str},
    )(_save_report)

    return create_sdk_mcp_server(
        "explanation-tools",
        tools=[read_tool, list_tool, save_tool],
    )


async def _default_callback(event: dict) -> None:
    """Default callback that prints to stdout."""
    if event["type"] == "message":
        print(f"\nExplanation Agent: {event['text']}")
    elif event["type"] == "tool_call":
        print(f"\n  [calling {event['tool']}]")


async def run_explanation_agent(
    run_dir: str,
    callback: Callable[[dict], Awaitable[None]] | None = None,
) -> tuple[bool, list[dict]]:
    """Run the explanation agent to produce a final report.

    Args:
        run_dir: Run directory containing all pipeline outputs.
        callback: Optional async callback for event routing.

    Returns:
        Tuple of (success bool, usage log).
    """
    if callback is None:
        callback = _default_callback

    mcp_server = _build_mcp_server(run_dir)
    usage_log: list[dict] = []

    prompt = (
        f"## Task\n\n"
        f"The optimization pipeline has completed. All outputs are in: {run_dir}\n\n"
        f"## Steps\n\n"
        f"1. Call `list_files` to see what's available\n"
        f"2. Read `problem_config.json` to understand the business problem\n"
        f"3. Read `cleaning_summary.txt` for data cleaning details\n"
        f"4. Read `modeling_summary.txt` for prediction and optimization details\n"
        f"5. Read `best_optimizer_order_plan.csv` for the recommended order quantities\n"
        f"6. Read `optimizer_comparison.csv` for the optimizer comparison\n"
        f"7. Read `sensitivity_results.json` for cost sensitivity analysis\n"
        f"8. Read `baseline_results.json` for agent vs baseline comparison\n"
        f"9. Synthesize everything into a final report\n"
        f"10. Call `save_report` with the Markdown report\n"
    )

    options = ClaudeAgentOptions(
        system_prompt=EXPLANATION_SYSTEM_PROMPT,
        mcp_servers={"explanation": mcp_server},
        allowed_tools=[],
        permission_mode="bypassPermissions",
        max_turns=15,
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                if message.usage:
                    usage_log.append(message.usage)
                for block in message.content:
                    if isinstance(block, TextBlock):
                        if re.match(r'^mcp__\w+__\w+\(.*\)$', block.text.strip(), re.DOTALL):
                            continue
                        await callback({
                            "type": "message",
                            "agent": "explanation",
                            "text": block.text,
                        })
                    elif hasattr(block, "name"):
                        pass

    report_path = Path(run_dir) / "final_report.md"
    success = report_path.exists()
    return success, usage_log

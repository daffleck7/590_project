"""Data Cleaning Agent — cleans CSV data according to ProblemConfig.

Uses the Claude Agent SDK with sandboxed code execution.
"""

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

from src.data_ingestion.prompts import DATA_CLEANING_SYSTEM_PROMPT
from src.data_ingestion.sandbox import execute_code, verify_no_fabrication, verify_type_consistency
from src.data_ingestion.tools import run_cfa_cleaning
from src.intake.tools import describe_column, peek_columns, sample_rows
from src.models.problem_config import ProblemConfig


try:
    from claude_agent_sdk import tool, create_sdk_mcp_server
except ImportError:
    tool = None
    create_sdk_mcp_server = None


def _build_mcp_server(csv_path: str, output_dir: str):
    """Build an MCP server with data cleaning tools."""

    async def _peek(args: dict) -> dict:
        return {"content": [{"type": "text", "text": peek_columns(args["csv_path"])}]}

    async def _sample(args: dict) -> dict:
        return {"content": [{"type": "text", "text": sample_rows(args["csv_path"], args.get("n", 10))}]}

    async def _describe(args: dict) -> dict:
        return {"content": [{"type": "text", "text": describe_column(args["csv_path"], args["column"])}]}

    async def _cfa_clean(args: dict) -> dict:
        return {"content": [{"type": "text", "text": run_cfa_cleaning(args["csv_path"], args["output_dir"])}]}

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

    cfa_tool = tool(
        "run_cfa_cleaning",
        "Run the CFA-specific cleaning pipeline on Squarespace order data",
        {"csv_path": str, "output_dir": str},
    )(_cfa_clean)

    async def _execute_code_handler(args: dict) -> dict:
        work_dir = Path(output_dir)
        result = execute_code(args["code"], work_dir, timeout=args.get("timeout", 30))
        if result.success:
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            return {"content": [{"type": "text", "text": output}]}
        else:
            return {
                "content": [{"type": "text", "text": f"Error:\n{result.error}"}],
                "isError": True,
            }

    exec_tool = tool(
        "execute_code",
        "Execute Python code in a sandbox. Use for data cleaning with pandas.",
        {"code": str, "timeout": int},
    )(_execute_code_handler)

    async def _save_summary(args: dict) -> dict:
        summary_path = Path(output_dir) / "cleaning_summary.txt"
        summary_path.write_text(args["summary"], encoding="utf-8")
        return {"content": [{"type": "text", "text": "Summary saved."}]}

    summary_tool = tool(
        "save_summary",
        "Save a structured summary of your cleaning work for the user to review. "
        "Call this as your FINAL action after all cleaning is complete.",
        {"summary": str},
    )(_save_summary)

    return create_sdk_mcp_server(
        "data-cleaning-tools",
        tools=[peek_tool, sample_tool, describe_tool, cfa_tool, exec_tool, summary_tool],
    )


def _build_data_manifest(csv_path: Path) -> dict:
    """Build a manifest describing the cleaned dataset."""
    import pandas as pd

    df = pd.read_csv(csv_path, low_memory=False)
    return {
        "row_count": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": {col: int(df[col].isna().sum()) for col in df.columns},
    }


async def _default_callback(event: dict) -> None:
    """Default callback that prints to stdout (CLI behavior)."""
    if event["type"] == "message":
        print(f"\nCleaning Agent: {event['text']}")
    elif event["type"] == "tool_call":
        print(f"\n  [calling {event['tool']}]")
    elif event["type"] == "tool_result":
        preview = event["result"][:200] if len(event["result"]) > 200 else event["result"]
        print(f"  [result: {preview}]")


async def run_cleaning_agent(
    csv_path: str,
    config: ProblemConfig,
    output_dir: str,
    callback: Callable[[dict], Awaitable[None]] | None = None,
) -> tuple[str | None, dict | None, list[dict]]:
    """Run the data cleaning agent.

    Args:
        csv_path: Path to the raw CSV file.
        config: ProblemConfig specifying data requirements.
        output_dir: Directory to write cleaned data and manifest.

    Returns:
        Tuple of (cleaned CSV path or None, data manifest or None, usage log).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    usage_log = []

    if callback is None:
        callback = _default_callback

    mcp_server = _build_mcp_server(csv_path, output_dir)

    cleaned_csv_path = str(output_path / "cleaned_data.csv")
    prompt = (
        f"## Data File\n\n"
        f"Raw CSV path: {csv_path}\n"
        f"Pass this exact path as `csv_path` to your inspection tools and when reading "
        f"data in execute_code.\n\n"
        f"## Output Location\n\n"
        f"Save the final cleaned CSV to exactly this path: {cleaned_csv_path}\n"
        f"Working directory for intermediate files: {output_dir}\n\n"
        f"## Your First Step\n\n"
        f"Call `peek_columns` with csv_path=\"{csv_path}\" to understand the raw data "
        f"structure, then call `sample_rows` with the same path to see example rows.\n\n"
        f"## ProblemConfig\n\n"
        f"This config was produced by the intake agent and defines what the cleaned "
        f"data must contain. Pay close attention to `data_requirements` — the prediction "
        f"module downstream will expect exactly these columns.\n\n"
        f"```json\n{config.model_dump_json(indent=2)}\n```\n\n"
        f"## Output Requirements\n\n"
        f"The cleaned CSV at `{cleaned_csv_path}` must:\n"
        f"- Contain all columns listed in `data_requirements.required_columns`\n"
        f"- Have a column matching each `uncertain_parameters[].data_column` so the "
        f"prediction module can find them\n"
        f"- Use consistent dtypes (numeric columns as numbers, dates as dates)\n"
        f"- Print a final summary showing: row count, column names, null counts per column\n"
    )

    options = ClaudeAgentOptions(
        system_prompt=DATA_CLEANING_SYSTEM_PROMPT,
        mcp_servers={"cleaning": mcp_server},
        allowed_tools=["Read"],
        permission_mode="bypassPermissions",
        max_turns=20,
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
                        await callback({"type": "message", "agent": "data_cleaning", "text": block.text})
                    elif hasattr(block, "name"):
                        pass

    # Find the cleaned output
    cleaned_csv = output_path / "cleaned_data.csv"
    if not cleaned_csv.exists():
        cfa_cleaned = output_path / "cfa_cleaned.csv"
        if cfa_cleaned.exists():
            cleaned_csv = cfa_cleaned
        else:
            csvs = list(output_path.glob("*.csv"))
            if csvs:
                cleaned_csv = csvs[0]
            else:
                return None, None, usage_log

    # Run verification
    input_path = Path(csv_path)
    type_errors = verify_type_consistency(input_path, cleaned_csv)
    fab_errors = verify_no_fabrication(input_path, cleaned_csv)

    if type_errors:
        print(f"\nWarning — type consistency issues: {type_errors}")
    if fab_errors:
        print(f"\nWarning — possible data fabrication: {fab_errors}")

    manifest = _build_data_manifest(cleaned_csv)

    manifest_path = output_path / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return str(cleaned_csv), manifest, usage_log

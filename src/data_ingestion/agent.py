"""Data Cleaning Agent — cleans CSV data according to ProblemConfig.

Uses the Claude Agent SDK with sandboxed code execution.
"""

import json
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

    cfa_tool = tool(
        "run_cfa_cleaning",
        "Run the CFA-specific cleaning pipeline on Squarespace order data",
        {"csv_path": str, "output_dir": str},
    )(lambda args: {"content": [{"type": "text", "text": run_cfa_cleaning(args["csv_path"], args["output_dir"])}]})

    def _execute_code_handler(args):
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

    return create_sdk_mcp_server(
        "data-cleaning-tools",
        tools=[peek_tool, sample_tool, describe_tool, cfa_tool, exec_tool],
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


async def run_cleaning_agent(
    csv_path: str,
    config: ProblemConfig,
    output_dir: str,
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

    mcp_server = _build_mcp_server(csv_path, output_dir)

    prompt = (
        f"Clean the data at: {csv_path}\n\n"
        f"ProblemConfig:\n{config.model_dump_json(indent=2)}\n\n"
        f"Save cleaned output to: {output_path / 'cleaned_data.csv'}\n\n"
        f"Available output directory: {output_dir}"
    )

    options = ClaudeAgentOptions(
        system_prompt=DATA_CLEANING_SYSTEM_PROMPT,
        mcp_servers={"cleaning": mcp_server},
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
                        print(f"\nCleaning Agent: {block.text}")

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

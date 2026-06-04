"""Explanation Agent — LLM call #3 in the pipeline.

Takes optimizer results + cleaned data, runs pure-Python analysis,
then calls the LLM once to produce a plain-English report.
"""

import json
from pathlib import Path

import pandas as pd
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    AssistantMessage,
    TextBlock,
)

from src.explanation.baseline import baseline_comparison
from src.explanation.sensitivity import sensitivity_analysis
from src.explanation.prompts import EXPLANATION_SYSTEM_PROMPT
from src.models.optimization_result import OptimizationResult
from src.models.problem_config import ProblemConfig


async def run_explanation_agent(
    result: OptimizationResult,
    cleaned_csv_path: str,
    config: ProblemConfig,
    train_years: list[int] | None = None,
    test_years: list[int] | None = None,
) -> tuple[str, dict, dict]:
    """Run the explanation agent and return the report + analysis data.

    Args:
        result: OptimizationResult from Fabian's optimizer module.
        cleaned_csv_path: Path to the aggregated cleaned CSV.
        config: ProblemConfig used throughout the pipeline.
        train_years: Training years for baseline computation. Defaults to 2020-2024.
        test_years: Test years to evaluate on. Defaults to 2025-2026.

    Returns:
        Tuple of (report_text, baseline_data, sensitivity_data).
    """
    if train_years is None:
        train_years = [2020, 2021, 2022, 2023, 2024]
    if test_years is None:
        test_years = [2025, 2026]

    demand_df = pd.read_csv(cleaned_csv_path)

    baseline_data = baseline_comparison(
        agent_quantities=result.order_quantities,
        demand_df=demand_df,
        config=config,
        train_years=train_years,
        test_years=test_years,
    )

    test_df = demand_df[demand_df["year"].isin(test_years)].copy().reset_index(drop=True)
    sensitivity_data = sensitivity_analysis(
        order_quantities=result.order_quantities,
        demand_df=test_df,
        config=config,
    )

    payload = {
        "optimization_result": {
            "order_quantities": result.order_quantities,
            "objective_value": result.objective_value,
            "total_spend": result.total_spend,
            "shadow_prices": result.shadow_prices,
            "solver_status": result.solver_status,
            "selected_model": result.selected_model,
        },
        "baseline_comparison": baseline_data,
        "sensitivity_analysis": sensitivity_data,
    }

    prompt = (
        "Please write the four-section report based on the following analysis results.\n\n"
        f"```json\n{json.dumps(payload, indent=2)}\n```"
    )

    options = ClaudeAgentOptions(
        system_prompt=EXPLANATION_SYSTEM_PROMPT,
        permission_mode="bypassPermissions",
        max_turns=2,
    )

    report_text = ""

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        report_text += block.text

    return report_text, baseline_data, sensitivity_data


def save_explanation(
    run_dir: str,
    report_text: str,
    baseline_data: dict,
    sensitivity_data: dict,
) -> None:
    """Save all explanation outputs to the run directory.

    Args:
        run_dir: Path to the run directory.
        report_text: Plain-English report from the LLM.
        baseline_data: Output of baseline_comparison().
        sensitivity_data: Output of sensitivity_analysis().
    """
    out = Path(run_dir)
    (out / "report.md").write_text(report_text, encoding="utf-8")
    (out / "baseline_comparison.json").write_text(
        json.dumps(baseline_data, indent=2), encoding="utf-8"
    )
    (out / "sensitivity_analysis.json").write_text(
        json.dumps(sensitivity_data, indent=2), encoding="utf-8"
    )

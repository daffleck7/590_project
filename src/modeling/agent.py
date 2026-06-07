"""Modeling Agent — runs prediction and optimization to produce a defensible order plan.

Uses the Claude Agent SDK with tools that wrap the prediction module,
optimization module, and a code sandbox for custom analysis.
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

from src.modeling.prompts import MODELING_SYSTEM_PROMPT
from src.models.problem_config import ProblemConfig
from src.models.data_module import DataModule
from src.models.prediction_module import add_prediction, PredictionParams
from src.models.all_optimizers_combined_final import compare_optimizers
from src.explanation.sensitivity import sensitivity_analysis
from src.explanation.baseline import baseline_comparison
from src.data_ingestion.sandbox import execute_code


try:
    from claude_agent_sdk import tool, create_sdk_mcp_server
except ImportError:
    tool = None
    create_sdk_mcp_server = None


def _build_mcp_server(
    config: ProblemConfig,
    cleaned_csv_path: str,
    run_dir: str,
):
    """Build MCP server with prediction, optimization, and code execution tools."""

    # Shared state so tools can pass results between calls
    state: dict = {"bundle": None, "prediction_result": None}

    async def _load_data(args: dict) -> dict:
        """Load cleaned CSV through DataModule."""
        csv_path = args.get("cleaned_csv_path", cleaned_csv_path)
        dm = DataModule(config)
        bundle = dm.load(csv_path)
        state["bundle"] = bundle

        summary = (
            f"DataBundle loaded successfully.\n"
            f"  Train: {len(bundle.features_train)} rows\n"
            f"  Test: {len(bundle.features_test)} rows\n"
            f"  Demand groups: {len(bundle.demand_pivot)}\n"
            f"  Sizes: {bundle.metadata.get('sizes', [])}\n"
            f"  Categories: {bundle.metadata.get('categories', [])}\n"
            f"  Seasons: {bundle.metadata.get('seasons', [])}\n"
            f"  Train years: {bundle.metadata.get('train_years', [])}\n"
            f"  Test years: {bundle.metadata.get('test_years', [])}\n"
            f"\nDemand pivot sample (first 10 rows):\n"
            f"{bundle.demand_pivot.head(10).to_string()}\n"
            f"\nFeature columns: {list(bundle.features_train.columns)}"
        )
        return {"content": [{"type": "text", "text": summary}]}

    async def _run_prediction(args: dict) -> dict:
        """Run all prediction models with optional hyperparameters."""
        if state["bundle"] is None:
            return {
                "content": [{"type": "text", "text": "Error: call load_data first"}],
                "isError": True,
            }

        # Build params from any provided arguments
        param_kwargs = {}
        for key in [
            "xgb_n_estimators", "xgb_max_depth", "xgb_learning_rate",
            "xgb_subsample", "xgb_colsample_bytree",
            "lgbm_n_estimators", "lgbm_max_depth", "lgbm_learning_rate",
            "lgbm_subsample", "ridge_alpha", "pto_strength",
        ]:
            if key in args:
                param_kwargs[key] = args[key]

        params = PredictionParams(**param_kwargs) if param_kwargs else None
        run_label = f" (custom: {param_kwargs})" if param_kwargs else " (defaults)"

        result = add_prediction(state["bundle"], config, params)
        state["prediction_result"] = result

        lines = [
            f"Prediction complete{run_label}. Winner: {result.best_model_name}",
            f"Baseline cost: ${result.baseline_cost:,.0f}",
            "",
            "All model costs (sorted):",
        ]
        for name, cost in sorted(result.all_costs.items(), key=lambda x: x[1]):
            pct = (cost / result.baseline_cost - 1) * 100
            marker = " <-- WINNER" if name == result.best_model_name else ""
            lines.append(f"  {name}: ${cost:,.0f} ({pct:+.1f}%){marker}")

        lines.extend([
            "",
            f"Demand groups predicted: {len(result.predicted_demand)}",
            f"Feature importance (top 5):",
        ])
        for feat, imp in result.feature_importance.head(5).items():
            lines.append(f"  {feat}: {imp:.4f}")

        lines.extend([
            "",
            f"P10 range: [{result.P10.min():.1f}, {result.P10.max():.1f}]",
            f"P50 range: [{result.P50.min():.1f}, {result.P50.max():.1f}]",
            f"P90 range: [{result.P90.min():.1f}, {result.P90.max():.1f}]",
        ])

        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    async def _run_optimization(args: dict) -> dict:
        """Run optimizers and return comparison."""
        if state["prediction_result"] is None:
            return {
                "content": [{"type": "text", "text": "Error: call run_prediction first"}],
                "isError": True,
            }

        n_restarts = args.get("n_restarts", 15)
        comparison, results, best = compare_optimizers(
            state["prediction_result"], config, n_restarts=n_restarts
        )

        # Save outputs
        run_path = Path(run_dir)
        comparison.to_csv(run_path / "optimizer_comparison.csv", index=False)
        best.order_plan.to_csv(run_path / "best_optimizer_order_plan.csv", index=False)

        # Build all plans CSV
        all_plans = []
        for r in results:
            plan = r.order_plan.copy()
            plan["optimizer"] = r.method
            all_plans.append(plan)
        import pandas as pd
        pd.concat(all_plans).to_csv(
            run_path / "all_optimizer_order_plans.csv", index=False
        )

        lines = [
            f"Optimization complete. Best: {best.method}",
            f"Objective: ${best.objective_value:,.2f}",
            f"Total spend: ${best.total_spend:,.2f}",
            f"Feasible: {best.feasible}",
            "",
            "All optimizers (sorted by objective):",
        ]
        for _, row in comparison.iterrows():
            lines.append(
                f"  {row['optimizer']}: ${row['objective_value']:,.2f} "
                f"(spend=${row['total_spend']:,.2f}, "
                f"feasible={row['feasible']}, "
                f"time={row['runtime_seconds']:.1f}s)"
            )

        lines.extend([
            "",
            f"Order plan has {len(best.order_plan)} items.",
            f"Order plan sample (first 10):",
            best.order_plan.head(10).to_string(),
        ])

        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    async def _execute_code_handler(args: dict) -> dict:
        """Run Python code in sandbox for custom analysis."""
        work_dir = Path(run_dir)
        result = execute_code(args["code"], work_dir, timeout=args.get("timeout", 60))
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

    async def _validate_results(args: dict) -> dict:
        """Compare predicted order quantities against actual demand."""
        if state["prediction_result"] is None:
            return {
                "content": [{"type": "text", "text": "Error: call run_prediction first"}],
                "isError": True,
            }

        import pandas as pd
        result = state["prediction_result"]
        df = result.demand_df.copy().reset_index(drop=True)
        df["predicted_order"] = result.predicted_demand
        df["actual_demand"] = df["demand"]
        df["error"] = df["predicted_order"] - df["actual_demand"]
        df["abs_error"] = df["error"].abs()
        df["pct_error"] = (df["error"] / df["actual_demand"].clip(lower=1)) * 100

        lines = [
            "VALIDATION: Predicted Orders vs Actual Demand",
            f"  Total items: {len(df)}",
            f"  Mean absolute error: {df['abs_error'].mean():.1f} units",
            f"  Mean % error: {df['pct_error'].mean():.1f}%",
            f"  Median % error: {df['pct_error'].median():.1f}%",
            "",
            f"  Over-orders (predicted > actual): {(df['error'] > 0).sum()} items",
            f"  Under-orders (predicted < actual): {(df['error'] < 0).sum()} items",
            f"  Exact matches: {(df['error'] == 0).sum()} items",
            "",
            "Worst 10 over-orders:",
        ]
        worst_over = df.nlargest(10, "error")[["product_category", "size", "actual_demand", "predicted_order", "error"]]
        lines.append(worst_over.to_string(index=False))

        lines.extend(["", "Worst 10 under-orders:"])
        worst_under = df.nsmallest(10, "error")[["product_category", "size", "actual_demand", "predicted_order", "error"]]
        lines.append(worst_under.to_string(index=False))

        # By category
        lines.extend(["", "Summary by product category:"])
        cat_summary = df.groupby("product_category").agg(
            items=("error", "count"),
            mean_error=("error", "mean"),
            total_predicted=("predicted_order", "sum"),
            total_actual=("actual_demand", "sum"),
        ).reset_index()
        lines.append(cat_summary.to_string(index=False))

        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    async def _run_sensitivity(args: dict) -> dict:
        """Run sensitivity analysis on cost parameters."""
        if state["prediction_result"] is None:
            return {
                "content": [{"type": "text", "text": "Error: call run_optimization first"}],
                "isError": True,
            }

        # Build order quantities dict from the best optimizer's order plan
        order_plan_path = Path(run_dir) / "best_optimizer_order_plan.csv"
        if not order_plan_path.exists():
            return {
                "content": [{"type": "text", "text": "Error: call run_optimization first"}],
                "isError": True,
            }

        import pandas as pd
        plan_df = pd.read_csv(order_plan_path)
        order_quantities = {}
        for _, row in plan_df.iterrows():
            key = f"{row['product_category']}_{row['gender_age']}_{row['size']}".lower()
            order_quantities[key] = float(row["recommended_order_qty"])

        # Get test demand data
        demand_df = state["prediction_result"].demand_df.copy()

        result = sensitivity_analysis(order_quantities, demand_df, config)

        # Save to file for explanation agent
        import json as _json
        sensitivity_path = Path(run_dir) / "sensitivity_results.json"
        sensitivity_path.write_text(_json.dumps(result, indent=2), encoding="utf-8")

        lines = [
            f"Sensitivity Analysis (base cost: ${result['base_cost']:,.2f})",
            "",
            "Overage cost sensitivity (co shifted):",
        ]
        for row in result["co_sensitivity"]:
            lines.append(f"  {row['shift_pct']:+d}%: ${row['total_cost']:,.2f} ({row['pct_change']:+.1f}%)")
        lines.extend(["", "Underage cost sensitivity (cu shifted):"])
        for row in result["cu_sensitivity"]:
            lines.append(f"  {row['shift_pct']:+d}%: ${row['total_cost']:,.2f} ({row['pct_change']:+.1f}%)")

        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    async def _run_baseline(args: dict) -> dict:
        """Compare agent orders vs historical-average baseline."""
        if state["bundle"] is None:
            return {
                "content": [{"type": "text", "text": "Error: call load_data first"}],
                "isError": True,
            }

        order_plan_path = Path(run_dir) / "best_optimizer_order_plan.csv"
        if not order_plan_path.exists():
            return {
                "content": [{"type": "text", "text": "Error: call run_optimization first"}],
                "isError": True,
            }

        import pandas as pd
        plan_df = pd.read_csv(order_plan_path)
        agent_quantities = {}
        for _, row in plan_df.iterrows():
            key = f"{row['product_category']}_{row['gender_age']}_{row['size']}".lower()
            agent_quantities[key] = float(row["recommended_order_qty"])

        # Use the full demand data for baseline computation
        demand_pivot = state["bundle"].demand_pivot.copy()
        # Need unit_price — merge from cleaned data
        cleaned_df = pd.read_csv(cleaned_csv_path)
        if "unit_price" not in demand_pivot.columns:
            price_map = cleaned_df.groupby(
                ["product_category", "size"]
            )["unit_price"].mean().reset_index()
            demand_pivot = demand_pivot.merge(
                price_map, on=["product_category", "size"], how="left"
            )
            demand_pivot["unit_price"] = demand_pivot["unit_price"].fillna(25.0)

        # Need gender_age column
        if "gender_age" not in demand_pivot.columns:
            demand_pivot["gender_age"] = "unknown"

        train_years = config.data_requirements.train_years
        test_years = config.data_requirements.test_years

        result = baseline_comparison(
            agent_quantities, demand_pivot, config,
            train_years=train_years, test_years=test_years,
        )

        # Save to file for explanation agent
        import json as _json
        baseline_path = Path(run_dir) / "baseline_results.json"
        # Convert to serializable (by_sku can be large)
        save_result = {
            "agent_cost": result["agent_cost"],
            "baseline_cost": result["baseline_cost"],
            "cost_reduction_pct": result["cost_reduction_pct"],
            "savings_usd": result["savings_usd"],
            "by_category": result["by_category"],
        }
        baseline_path.write_text(_json.dumps(save_result, indent=2), encoding="utf-8")

        lines = [
            "Baseline Comparison: Agent vs Historical Average",
            f"  Agent total cost: ${result['agent_cost']:,.2f}",
            f"  Baseline total cost: ${result['baseline_cost']:,.2f}",
            f"  Savings: ${result['savings_usd']:,.2f} ({result['cost_reduction_pct']:.1f}%)",
            "",
            "By category:",
        ]
        for cat in result["by_category"]:
            lines.append(
                f"  {cat['product_category']}: agent=${cat['agent_cost']:,.2f}, "
                f"baseline=${cat['baseline_cost']:,.2f}, "
                f"savings=${cat['savings']:,.2f}"
            )

        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    async def _save_summary(args: dict) -> dict:
        """Save structured summary of modeling work."""
        summary_path = Path(run_dir) / "modeling_summary.txt"
        summary_path.write_text(args["summary"], encoding="utf-8")
        return {"content": [{"type": "text", "text": "Summary saved."}]}

    load_tool = tool(
        "load_data",
        "Load the cleaned CSV through DataModule. Returns train/test split "
        "stats, demand pivot sample, and metadata.",
        {"cleaned_csv_path": str},
    )(_load_data)

    predict_tool = tool(
        "run_prediction",
        "Train XGBoost, LightGBM, and Ridge models on demand data. Compares "
        "all models by newsvendor cost and selects the winner. Must call "
        "load_data first. You can tune hyperparameters by passing optional args.",
        {
            "xgb_n_estimators": int,
            "xgb_max_depth": int,
            "xgb_learning_rate": float,
            "xgb_subsample": float,
            "xgb_colsample_bytree": float,
            "lgbm_n_estimators": int,
            "lgbm_max_depth": int,
            "lgbm_learning_rate": float,
            "lgbm_subsample": float,
            "ridge_alpha": float,
            "pto_strength": float,
        },
    )(_run_prediction)

    optimize_tool = tool(
        "run_optimization",
        "Run 5 optimizers (SLSQP, L-BFGS-B, PGD, SGD, Adam) on the prediction "
        "results. Saves comparison CSV and best order plan. Must call "
        "run_prediction first.",
        {"n_restarts": int},
    )(_run_optimization)

    validate_tool = tool(
        "validate_results",
        "Compare predicted order quantities against actual demand from the test "
        "set. Shows error distribution, worst over/under-orders, and category "
        "breakdown. Use this to assess prediction quality before finalizing. "
        "Must call run_prediction first.",
        {},
    )(_validate_results)

    exec_tool = tool(
        "execute_code",
        "Execute Python code in a sandbox for custom analysis, parameter tuning, "
        "or validation. Import pandas/numpy/scipy as needed. The cleaned CSV is "
        f"at: {cleaned_csv_path}",
        {"code": str, "timeout": int},
    )(_execute_code_handler)

    sensitivity_tool = tool(
        "run_sensitivity",
        "Run sensitivity analysis — shows how total cost changes when overage "
        "and underage cost parameters shift by -30% to +30%. Must call "
        "run_optimization first. Saves results to sensitivity_results.json.",
        {},
    )(_run_sensitivity)

    baseline_tool = tool(
        "run_baseline",
        "Compare agent's optimized orders against a naive baseline (historical "
        "average demand). Shows cost savings by category. Must call "
        "run_optimization first. Saves results to baseline_results.json.",
        {},
    )(_run_baseline)

    summary_tool = tool(
        "save_summary",
        "Save your structured summary report. Call this as your FINAL action.",
        {"summary": str},
    )(_save_summary)

    return create_sdk_mcp_server(
        "modeling-tools",
        tools=[
            load_tool, predict_tool, optimize_tool,
            validate_tool, sensitivity_tool, baseline_tool,
            exec_tool, summary_tool,
        ],
    )


async def _default_callback(event: dict) -> None:
    """Default callback that prints to stdout."""
    if event["type"] == "message":
        print(f"\nModeling Agent: {event['text']}")
    elif event["type"] == "tool_call":
        print(f"\n  [calling {event['tool']}]")


async def run_modeling_agent(
    config: ProblemConfig,
    cleaned_csv_path: str,
    run_dir: str,
    callback: Callable[[dict], Awaitable[None]] | None = None,
) -> tuple[bool, list[dict]]:
    """Run the modeling agent to produce prediction + optimization results.

    Args:
        config: Validated ProblemConfig.
        cleaned_csv_path: Path to cleaned CSV from data cleaning agent.
        run_dir: Run directory for saving outputs.
        callback: Optional async callback for event routing.

    Returns:
        Tuple of (success bool, usage log).
    """
    if callback is None:
        callback = _default_callback

    mcp_server = _build_mcp_server(config, cleaned_csv_path, run_dir)
    usage_log: list[dict] = []

    prompt = (
        f"## Task\n\n"
        f"You have a cleaned dataset at: {cleaned_csv_path}\n"
        f"Your run directory for outputs is: {run_dir}\n\n"
        f"## ProblemConfig\n\n"
        f"```json\n{config.model_dump_json(indent=2)}\n```\n\n"
        f"## Steps\n\n"
        f"1. Call `load_data` with cleaned_csv_path=\"{cleaned_csv_path}\"\n"
        f"2. Call `run_prediction` to train models and find the best one\n"
        f"3. Call `run_optimization` to find the best solver\n"
        f"4. Analyze the results — use `execute_code` if you need to dig deeper\n"
        f"5. Call `save_summary` with your structured report\n"
    )

    options = ClaudeAgentOptions(
        system_prompt=MODELING_SYSTEM_PROMPT,
        mcp_servers={"modeling": mcp_server},
        allowed_tools=[],
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
                        await callback({
                            "type": "message",
                            "agent": "modeling",
                            "text": block.text,
                        })
                    elif hasattr(block, "name"):
                        pass

    summary_path = Path(run_dir) / "modeling_summary.txt"
    success = summary_path.exists()
    return success, usage_log

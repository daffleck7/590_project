"""
Re-run joint optimization (Model A vs Model B) with corrected budget ($40,000).

Fixes the original pipeline issue by including 'year' in groupby_keys so the
joint optimizer and baseline comparison can work properly.
"""
import sys
import io
import json
import copy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from src.models.problem_config import ProblemConfig
from src.models.data_module import DataModule
from src.models.prediction_module import PredictionModule, PredictionParams
from src.models.joint_optimizer import optimize_joint_slsqp, joint_saa_exact
from src.models.all_optimizers_combined_final import compare_optimizers


def load_config_with_fixes() -> ProblemConfig:
    """Load the problem config with year in groupby_keys and $40k budget."""
    with open("data/default_config.json") as fh:
        raw = json.load(fh)

    # Ensure year is in groupby_keys
    gk = raw.get("data_requirements", {}).get("groupby_keys", [])
    if "year" not in gk:
        gk.insert(0, "year")
        raw["data_requirements"]["groupby_keys"] = gk

    # Ensure test_years includes both 2025 and 2026
    raw["data_requirements"]["test_years"] = [2025, 2026]
    raw["data_requirements"]["train_years"] = [2020, 2021, 2022, 2023, 2024]

    # Ensure budget is 40000
    for constraint in raw.get("constraints", []):
        if "budget" in constraint.get("name", "").lower():
            for key in ["budget_usd", "budget_usd_per_season", "budget"]:
                if key in constraint.get("parameters", {}):
                    constraint["parameters"][key] = 40000

    return ProblemConfig(**raw)


def main():
    config = load_config_with_fixes()
    print(f"Budget: ${config.constraints[0].parameters}")
    print(f"Groupby keys: {config.data_requirements.groupby_keys}")
    print(f"Train years: {config.data_requirements.train_years}")
    print(f"Test years: {config.data_requirements.test_years}")

    # Load data with year in groupby
    dm = DataModule(config)
    bundle = dm.load("data/cleaned_orders.csv")
    print(f"\n{bundle.summary()}")
    print(f"Train years in data: {bundle.metadata.get('train_years')}")
    print(f"Test years in data: {bundle.metadata.get('test_years')}")

    # Run prediction with correct temporal split
    params = PredictionParams(
        train_years=[2020, 2021, 2022, 2023, 2024],
        test_years=[2025, 2026],
    )
    pred = PredictionModule(config, params).run(bundle)
    print(f"\n{pred.summary()}")

    # Check that demand_df has year and multiple test years
    print(f"\ndemand_df columns: {list(pred.demand_df.columns)}")
    if "year" in pred.demand_df.columns:
        print(f"Years in demand_df: {sorted(pred.demand_df['year'].unique())}")
    else:
        print("ERROR: year not in demand_df!")
        return

    # Run Model A (single-period, all optimizers)
    print("\n" + "=" * 60)
    print("MODEL A: Independent Single-Period Newsvendor")
    print("=" * 60)
    comparison_df, all_results, best_a = compare_optimizers(pred, config)
    print(f"\nBest Model A: {best_a.method}, obj=${best_a.objective_value:,.2f}, "
          f"spend=${best_a.total_spend:,.2f}")

    # Run Model B (joint two-period)
    print("\n" + "=" * 60)
    print("MODEL B: Joint Two-Period with Carryover")
    print("=" * 60)
    model_b = optimize_joint_slsqp(pred, config, n_restarts=10)
    print(f"\n{model_b.summary()}")

    # Compute Model A cost on same joint objective for fair comparison
    # We need Model A's quantities for Y1 and Y2 SKUs
    n = model_b.q1.shape[0]
    D1 = model_b.y1_df  # we need the scenarios
    D2 = model_b.y2_df

    # Get Model A's order plan for the test set
    plan_a = best_a.order_plan
    demand_df = pred.demand_df.reset_index(drop=True)

    # Attach year to plan_a
    if "year" not in plan_a.columns and "year" in demand_df.columns:
        plan_a = plan_a.copy()
        plan_a["year"] = demand_df["year"].values[:len(plan_a)]

    print(f"\nModel B objective: ${model_b.objective_value:,.2f}")
    print(f"Model B Y1 spend: ${model_b.y1_spend:,.2f}")
    print(f"Model B Y2 spend: ${model_b.y2_spend:,.2f}")
    print(f"Model B feasible: {model_b.feasible}")
    print(f"Model B Y1 total qty: {model_b.q1.sum():.0f}")
    print(f"Model B Y2 total qty: {model_b.q2.sum():.0f}")
    print(f"Model B expected carryover: {model_b.expected_carryover.sum():.1f}")
    print(f"Model B Y1 shadow price: ${model_b.y1_shadow_price:.4f}")

    # Also run Model A shadow price
    from src.models.joint_optimizer import _compute_shadow_price_model_a
    print("\nComputing Model A shadow price...")
    shadow_a = _compute_shadow_price_model_a(pred, config)
    print(f"Model A Y1 shadow price: ${shadow_a:.4f}")

    # Build comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Model A total cost: ${best_a.objective_value:,.2f}")
    print(f"Model B total cost: ${model_b.objective_value:,.2f}")
    cost_of_myopia = best_a.objective_value - model_b.objective_value
    if best_a.objective_value > 0:
        reduction_pct = cost_of_myopia / best_a.objective_value * 100
    else:
        reduction_pct = 0
    print(f"Cost of myopia: ${cost_of_myopia:,.2f} ({reduction_pct:.1f}%)")

    # Save results
    results = {
        "budget": 40000,
        "model_a_total_cost": round(best_a.objective_value, 2),
        "model_a_total_spend": round(best_a.total_spend, 2),
        "model_a_method": best_a.method,
        "model_b_total_cost": round(model_b.objective_value, 2),
        "model_b_y1_spend": round(model_b.y1_spend, 2),
        "model_b_y2_spend": round(model_b.y2_spend, 2),
        "cost_of_myopia_usd": round(cost_of_myopia, 2),
        "cost_reduction_pct": round(reduction_pct, 2),
        "total_expected_carryover_units": round(float(model_b.expected_carryover.sum()), 1),
        "y1_qty_model_a_total": int(plan_a[plan_a.get("year", pd.Series()) == sorted(pred.demand_df["year"].unique())[0]]["recommended_order_qty"].sum()) if "year" in plan_a.columns else "N/A",
        "y1_qty_model_b": int(model_b.q1.sum()),
        "y2_qty_model_b": int(model_b.q2.sum()),
        "model_a_y1_shadow_price": round(shadow_a, 4),
        "model_b_y1_shadow_price": round(model_b.y1_shadow_price, 4),
        "model_b_feasible": model_b.feasible,
    }

    with open("report/rerun_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved to report/rerun_results.json")

    # Save comparison table
    key_cols = [c for c in ["season", "product_category", "gender_age", "size"]
                if c in model_b.y1_df.columns]
    comp_df = model_b.y1_df[key_cols].copy().sort_values(key_cols).reset_index(drop=True)
    comp_df["B_y1_qty"] = model_b.q1.astype(int)
    comp_df["B_y2_qty"] = model_b.q2.astype(int)
    comp_df["B_exp_carryover"] = np.round(model_b.expected_carryover, 1)
    comp_df.to_csv("report/joint_comparison_40k.csv", index=False)
    print("Comparison table saved to report/joint_comparison_40k.csv")


if __name__ == "__main__":
    main()

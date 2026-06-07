"""Baseline comparison — agent quantities vs. president's fixed-order heuristic."""

import pandas as pd

from src.models.problem_config import ProblemConfig


def _sku_key(product_category: str, gender_age: str, size: str) -> str:
    return f"{product_category}_{gender_age}_{size}".lower()


def _nv_cost(q: float, demand: float, co: float, cu: float) -> float:
    return co * max(q - demand, 0.0) + cu * max(demand - q, 0.0)


def compute_baseline_quantities(
    demand_df: pd.DataFrame,
    train_years: list[int],
) -> dict[str, float]:
    """President's heuristic: per-SKU average demand across training seasons.

    Args:
        demand_df: Full aggregated demand data (all years).
        train_years: Years to compute averages from.

    Returns:
        Dict mapping SKU key -> average historical demand quantity.
    """
    train = demand_df[demand_df["year"].isin(train_years)].copy()
    avg = (
        train.groupby(["product_category", "gender_age", "size"])["quantity"]
        .mean()
        .reset_index()
    )
    return {
        _sku_key(row["product_category"], row["gender_age"], row["size"]): round(row["quantity"], 2)
        for _, row in avg.iterrows()
    }


def baseline_comparison(
    agent_quantities: dict[str, float],
    demand_df: pd.DataFrame,
    config: ProblemConfig,
    train_years: list[int] | None = None,
    test_years: list[int] | None = None,
) -> dict:
    """Compare agent order quantities vs. president's heuristic on the test set.

    Args:
        agent_quantities: SKU key -> quantity recommended by the optimizer.
        demand_df: Full aggregated demand data (all years).
        config: ProblemConfig with cost_structure multipliers.
        train_years: Years used to compute the baseline. Defaults to 2020-2024.
        test_years: Years to evaluate on. Defaults to 2025-2026.

    Returns:
        Dict with keys:
          baseline_quantities  dict[str, float]
          agent_cost           float
          baseline_cost        float
          cost_reduction_pct   float
          savings_usd          float
          by_category          list[dict]  — per product_category breakdown
          by_sku               list[dict]  — per SKU detail row
    """
    if train_years is None:
        train_years = [2020, 2021, 2022, 2023, 2024]
    if test_years is None:
        test_years = [2025, 2026]

    co_mult = float(config.cost_structure.get("overage_cost_multiplier", 1.0))
    cu_mult = float(config.cost_structure.get("underage_cost_multiplier", 1.3))

    baseline_qtys = compute_baseline_quantities(demand_df, train_years)
    test_df = demand_df[demand_df["year"].isin(test_years)].copy().reset_index(drop=True)

    agent_total = 0.0
    baseline_total = 0.0
    by_sku = []

    for _, row in test_df.iterrows():
        key = _sku_key(row["product_category"], row["gender_age"], row["size"])
        unit_price = float(row["unit_price"])
        demand = float(row["quantity"])
        co = unit_price * co_mult
        cu = unit_price * cu_mult

        agent_q = agent_quantities.get(key, 0.0)
        baseline_q = baseline_qtys.get(key, 0.0)

        agent_cost = _nv_cost(agent_q, demand, co, cu)
        baseline_cost = _nv_cost(baseline_q, demand, co, cu)

        agent_total += agent_cost
        baseline_total += baseline_cost

        by_sku.append({
            "sku": key,
            "product_category": row["product_category"],
            "gender_age": row["gender_age"],
            "size": row["size"].upper(),
            "actual_demand": demand,
            "agent_qty": agent_q,
            "baseline_qty": baseline_q,
            "agent_cost": round(agent_cost, 2),
            "baseline_cost": round(baseline_cost, 2),
            "savings": round(baseline_cost - agent_cost, 2),
        })

    by_category = []
    for cat in test_df["product_category"].unique():
        cat_rows = [r for r in by_sku if r["product_category"] == cat]
        by_category.append({
            "product_category": cat,
            "agent_cost": round(sum(r["agent_cost"] for r in cat_rows), 2),
            "baseline_cost": round(sum(r["baseline_cost"] for r in cat_rows), 2),
            "savings": round(sum(r["savings"] for r in cat_rows), 2),
        })

    savings = baseline_total - agent_total
    reduction_pct = (savings / baseline_total * 100) if baseline_total > 0 else 0.0

    return {
        "baseline_quantities": baseline_qtys,
        "agent_cost": round(agent_total, 2),
        "baseline_cost": round(baseline_total, 2),
        "cost_reduction_pct": round(reduction_pct, 2),
        "savings_usd": round(savings, 2),
        "by_category": by_category,
        "by_sku": by_sku,
    }

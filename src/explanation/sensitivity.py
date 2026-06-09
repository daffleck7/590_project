"""Sensitivity analysis — how total newsvendor cost changes as cost parameters shift."""

import pandas as pd

from src.models.problem_config import ProblemConfig


def _sku_key(product_category: str, gender_age: str, size: str) -> str:
    return f"{product_category}_{gender_age}_{size}".lower()


def _nv_cost(q: float, demand: float, co: float, cu: float) -> float:
    return co * max(q - demand, 0.0) + cu * max(demand - q, 0.0)


def _total_cost(
    order_quantities: dict[str, float],
    demand_df: pd.DataFrame,
    co_mult: float,
    cu_mult: float,
) -> float:
    total = 0.0
    for _, row in demand_df.iterrows():
        key = _sku_key(row["product_category"], row["gender_age"], row["size"])
        q = order_quantities.get(key, 0.0)
        unit_price = float(row["unit_price"])
        total += _nv_cost(q, float(row["quantity"]), unit_price * co_mult, unit_price * cu_mult)
    return total


def sensitivity_analysis(
    order_quantities: dict[str, float],
    demand_df: pd.DataFrame,
    config: ProblemConfig,
    shifts: list[float] | None = None,
) -> dict:
    """Compute how total NV cost changes as overage and underage cost multipliers shift.

    Args:
        order_quantities: SKU key -> ordered quantity from the optimizer.
        demand_df: Test-set rows with columns (product_category, gender_age, size,
                   quantity, unit_price). Each row is one season-year-SKU observation.
        config: ProblemConfig containing cost_structure multipliers.
        shifts: Fractional shifts to apply, e.g. [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3].

    Returns:
        Dict with keys:
          base_cost        float  — cost at zero shift
          co_sensitivity   list[dict]  — [{shift, co_mult, total_cost, pct_change}]
          cu_sensitivity   list[dict]  — [{shift, cu_mult, total_cost, pct_change}]
    """
    if shifts is None:
        shifts = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]

    cs = config.cost_structure
    if isinstance(cs, dict):
        co_mult_base = float(cs.get("overage_cost_multiplier", 1.0))
        cu_mult_base = float(cs.get("underage_cost_multiplier", 1.3))
    else:
        co_mult_base, cu_mult_base = 1.0, 1.3
        for item in cs:
            if "overage" in item.name.lower():
                co_mult_base = float(item.value)
            elif "underage" in item.name.lower():
                cu_mult_base = float(item.value)

    base_cost = _total_cost(order_quantities, demand_df, co_mult_base, cu_mult_base)

    co_sensitivity = []
    for shift in shifts:
        co_mult = co_mult_base * (1.0 + shift)
        cost = _total_cost(order_quantities, demand_df, co_mult, cu_mult_base)
        co_sensitivity.append({
            "shift_pct": round(shift * 100),
            "co_multiplier": round(co_mult, 4),
            "total_cost": round(cost, 2),
            "pct_change": round((cost / base_cost - 1) * 100, 2) if base_cost > 0 else 0.0,
        })

    cu_sensitivity = []
    for shift in shifts:
        cu_mult = cu_mult_base * (1.0 + shift)
        cost = _total_cost(order_quantities, demand_df, co_mult_base, cu_mult)
        cu_sensitivity.append({
            "shift_pct": round(shift * 100),
            "cu_multiplier": round(cu_mult, 4),
            "total_cost": round(cost, 2),
            "pct_change": round((cost / base_cost - 1) * 100, 2) if base_cost > 0 else 0.0,
        })

    return {
        "base_cost": round(base_cost, 2),
        "co_sensitivity": co_sensitivity,
        "cu_sensitivity": cu_sensitivity,
    }

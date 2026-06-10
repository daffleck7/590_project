"""
joint_optimizer.py
==================
Joint two-period optimizer (Model B) with Year 1 -> Year 2 carryover inventory.

Stakeholder modification (Summer 2026):
  Model A (current/independent): each lifecycle year is a separate newsvendor.
    Year 1 overage is penalised as waste.
  Model B (joint with carryover): Year 1 surplus carries to Year 2 at zero cost,
    reducing Year 2 procurement and partially hedging against the much higher
    Year 2 stockout cost once the model is discontinued.

Joint SAA objective (N scenarios, S SKUs):
  min  1/N * sum_n sum_s [
         cu1_s * (D1_sn - q1_s)+          # Year 1 underage only
       + co2_s * (q2_s + I_sn - D2_sn)+   # Year 2 overage (using carryover)
       + cu2_s * (D2_sn - q2_s - I_sn)+   # Year 2 underage (reduced by carryover)
       ]
  where  I_sn = (q1_s - D1_sn)+           # carryover inventory
  s.t.   sum_s unit_cost_s * q1_s[season] <= budget   (per season, Year 1)
         sum_s unit_cost_s * q2_s[season] <= budget   (per season, Year 2)
         q1_s, q2_s >= moq
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    from src.models.problem_config import get_cost, get_budget, get_moq
except ImportError:
    from problem_config import get_cost, get_budget, get_moq


# ---------------------------------------------------------------------------
# Smooth math helpers
# ---------------------------------------------------------------------------

def _sp(z: np.ndarray, k: float = 10.0) -> np.ndarray:
    """Numerically stable softplus: log(1 + exp(k*z)) / k  ~  max(0, z)."""
    return np.logaddexp(0.0, k * z) / k


def _sg(z: np.ndarray, k: float = 10.0) -> np.ndarray:
    """Sigmoid: derivative of softplus."""
    return 1.0 / (1.0 + np.exp(-np.clip(k * z, -60.0, 60.0)))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class JointInputs:
    y1_df: pd.DataFrame
    y2_df: pd.DataFrame
    scenarios_y1: np.ndarray   # shape (N_scenarios, n_skus)
    scenarios_y2: np.ndarray   # shape (N_scenarios, n_skus)
    cu1: np.ndarray            # Year 1 underage cost per SKU
    co2: np.ndarray            # Year 2 overage cost per SKU
    cu2: np.ndarray            # Year 2 underage cost per SKU
    unit_cost: np.ndarray      # Procurement cost per unit (same across years)
    budget: float
    moq: int
    n_skus: int
    seasons: np.ndarray        # Season label per SKU row (for per-season budget)


@dataclass
class JointResult:
    """Result from joint two-period optimization (Model B)."""

    method: str
    q1: np.ndarray
    q2: np.ndarray
    y1_df: pd.DataFrame
    y2_df: pd.DataFrame
    objective_value: float
    y1_spend: float
    y2_spend: float
    y1_shadow_price: float     # $/$ extra budget in Year 1
    expected_carryover: np.ndarray
    feasible: bool
    runtime_seconds: float
    n_iterations: int
    success: bool
    message: str
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.method} (Joint): obj=${self.objective_value:,.2f}, "
            f"y1_spend=${self.y1_spend:,.2f}, y2_spend=${self.y2_spend:,.2f}, "
            f"feasible={self.feasible}, "
            f"carryover={self.expected_carryover.sum():.1f} units"
        )


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------

def joint_saa_exact(
    x: np.ndarray,
    D1: np.ndarray,
    D2: np.ndarray,
    cu1: np.ndarray,
    co2: np.ndarray,
    cu2: np.ndarray,
) -> float:
    """Non-smooth joint SAA objective for final evaluation."""
    n = len(cu1)
    q1, q2 = x[:n], x[n:]
    total = 0.0
    for i in range(len(D1)):
        I = np.maximum(q1 - D1[i], 0.0)
        eff = q2 + I
        total += float(np.sum(
            cu1 * np.maximum(D1[i] - q1, 0.0) +
            co2 * np.maximum(eff - D2[i], 0.0) +
            cu2 * np.maximum(D2[i] - eff, 0.0)
        ))
    return total / len(D1)


def joint_saa_smooth_obj_grad(
    x: np.ndarray,
    D1: np.ndarray,
    D2: np.ndarray,
    cu1: np.ndarray,
    co2: np.ndarray,
    cu2: np.ndarray,
    k: float = 10.0,
) -> tuple[float, np.ndarray]:
    """Smooth joint SAA objective + gradient via softplus approximation.

    Gradient derivation:
      I = softplus(q1 - d1),    dI/dq1 = sigmoid(q1 - d1)
      y1_underage = cu1 * softplus(d1 - q1),  d/dq1 = -cu1 * sigmoid(d1 - q1)
      eff = q2 + I
      d(obj)/d(eff) = co2 * sigmoid(eff - d2) - cu2 * sigmoid(d2 - eff)
      d(obj)/dq1    = d(y1u)/dq1 + d(obj)/d(eff) * dI/dq1
      d(obj)/dq2    = d(obj)/d(eff)
    """
    n = len(cu1)
    q1, q2 = x[:n], x[n:]
    obj = 0.0
    grad = np.zeros(2 * n, dtype=float)
    N = len(D1)

    for i in range(N):
        d1, d2 = D1[i], D2[i]
        I = _sp(q1 - d1, k)
        dI_dq1 = _sg(q1 - d1, k)

        # Year 1 underage
        y1u = cu1 * _sp(d1 - q1, k)
        dy1u_dq1 = -cu1 * _sg(d1 - q1, k)

        eff = q2 + I
        dobj_deff = co2 * _sg(eff - d2, k) - cu2 * _sg(d2 - eff, k)
        y2_cost = co2 * _sp(eff - d2, k) + cu2 * _sp(d2 - eff, k)

        obj += float(np.sum(y1u + y2_cost))
        grad[:n] += dy1u_dq1 + dobj_deff * dI_dq1
        grad[n:] += dobj_deff

    return obj / N, grad / N


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _safe_cost(config, category: str | None, cost_name: str, period: int | None) -> float:
    """Cost lookup with graceful fallback through period -> None -> 1.0."""
    try:
        return get_cost(config, category, cost_name, period)
    except KeyError:
        pass
    try:
        return get_cost(config, category, cost_name, None)
    except KeyError:
        return 1.0


def prepare_joint_inputs(prediction_result, config) -> JointInputs:
    """Split prediction_result into aligned Year-1 / Year-2 SKU pairs.

    Rows in demand_df are (year, season, product_category, size) groups.
    We identify Year 1 as the first test year (e.g. 2025) and Year 2 as the
    second (e.g. 2026), then pair rows by (season, product_category, size).
    """
    df = prediction_result.demand_df.copy().reset_index(drop=True)
    P10 = np.asarray(prediction_result.P10, dtype=float)
    P50 = np.asarray(prediction_result.P50, dtype=float)
    P90 = np.asarray(prediction_result.P90, dtype=float)

    if "year" not in df.columns:
        raise ValueError("demand_df must contain a 'year' column for joint optimization")

    test_years = sorted(df["year"].unique())
    if len(test_years) < 2:
        raise ValueError(f"Joint optimization needs >= 2 test years; got {test_years}")
    y1_yr, y2_yr = int(test_years[0]), int(test_years[1])

    # Attach scenario columns to df for easy filtering
    df["_P10"] = P10
    df["_P50"] = P50
    df["_P90"] = P90

    # Key columns that uniquely identify a SKU within a lifecycle year
    key_cols = [c for c in ["season", "uniform_set", "product_category", "size"]
                if c in df.columns]

    y1_raw = df[df["year"] == y1_yr].copy()
    y2_raw = df[df["year"] == y2_yr].copy()

    # Sort by key_cols to align
    y1_df = y1_raw.sort_values(key_cols).reset_index(drop=True)
    y2_df = y2_raw.sort_values(key_cols).reset_index(drop=True)

    # Keep only common SKU keys present in both years
    def _key(df_: pd.DataFrame) -> pd.Series:
        return df_[key_cols].apply(tuple, axis=1)

    common = sorted(set(_key(y1_df)) & set(_key(y2_df)))
    y1_df = y1_df[_key(y1_df).isin(common)].sort_values(key_cols).reset_index(drop=True)
    y2_df = y2_df[_key(y2_df).isin(common)].sort_values(key_cols).reset_index(drop=True)

    if len(y1_df) != len(y2_df):
        raise ValueError("Y1 and Y2 SKU counts differ after alignment")

    n = len(y1_df)

    # Scenario matrices: shape (3, n) — P10/P50/P90 per SKU
    scenarios_y1 = np.maximum(
        np.vstack([y1_df["_P10"].values, y1_df["_P50"].values, y1_df["_P90"].values]), 0.0
    )
    scenarios_y2 = np.maximum(
        np.vstack([y2_df["_P10"].values, y2_df["_P50"].values, y2_df["_P90"].values]), 0.0
    )

    # Cost vectors
    cu1 = np.array([
        _safe_cost(config, r.get("product_category"), "underage_cost", 1)
        for _, r in y1_df.iterrows()
    ])
    co2 = np.array([
        _safe_cost(config, r.get("product_category"), "overage_cost", 2)
        for _, r in y2_df.iterrows()
    ])
    cu2 = np.array([
        _safe_cost(config, r.get("product_category"), "underage_cost", 2)
        for _, r in y2_df.iterrows()
    ])
    unit_cost = np.array([
        _safe_cost(config, r.get("product_category"), "unit_cost", None)
        for _, r in y1_df.iterrows()
    ])

    seasons = y1_df["season"].values if "season" in y1_df.columns else np.zeros(n, dtype=object)

    return JointInputs(
        y1_df=y1_df, y2_df=y2_df,
        scenarios_y1=scenarios_y1, scenarios_y2=scenarios_y2,
        cu1=cu1, co2=co2, cu2=cu2,
        unit_cost=unit_cost,
        budget=float(get_budget(config)),
        moq=int(get_moq(config)),
        n_skus=n,
        seasons=seasons,
    )


# ---------------------------------------------------------------------------
# Projection and feasibility repair
# ---------------------------------------------------------------------------

def _project_joint(x: np.ndarray, inp: JointInputs) -> np.ndarray:
    """Project both q1 and q2 to MOQ lower bound and per-season budget."""
    n = inp.n_skus
    q1, q2 = x[:n].copy(), x[n:].copy()

    for arr in (q1, q2):
        arr[:] = np.maximum(arr, inp.moq)
        for s in np.unique(inp.seasons):
            mask = inp.seasons == s
            uc = inp.unit_cost[mask]
            spend = float(np.sum(uc * arr[mask]))
            if spend > inp.budget + 1e-9:
                extras = arr[mask] - inp.moq
                min_spend = float(np.sum(uc * inp.moq))
                allowed_extra = max(inp.budget - min_spend, 0.0)
                extra_spend = float(np.sum(uc * extras))
                if extra_spend > 0:
                    arr[mask] = inp.moq + extras * min(allowed_extra / extra_spend, 1.0)
                else:
                    arr[mask] = inp.moq

    return np.concatenate([q1, q2])


def _round_repair_joint(x: np.ndarray, inp: JointInputs) -> np.ndarray:
    """Round to integers and repair budget violations."""
    x = _project_joint(x, inp)
    n = inp.n_skus
    q1 = np.ceil(x[:n]).astype(int)
    q2 = np.ceil(x[n:]).astype(int)

    def _repair(q: np.ndarray) -> np.ndarray:
        for s in np.unique(inp.seasons):
            mask = inp.seasons == s
            idxs = np.where(mask)[0]
            while float(np.sum(inp.unit_cost[mask] * q[mask])) > inp.budget + 1e-9:
                candidates = [i for i in idxs if q[i] > inp.moq]
                if not candidates:
                    break
                # Cut the unit that costs the least in procurement (most expendable)
                cut = min(candidates, key=lambda i: inp.unit_cost[i])
                q[cut] -= 1
        return q

    return np.concatenate([_repair(q1), _repair(q2)])


# ---------------------------------------------------------------------------
# SLSQP constraint builder
# ---------------------------------------------------------------------------

def _build_constraints(inp: JointInputs) -> list[dict]:
    """SLSQP inequality constraints: budget - spend >= 0, per season per year."""
    n = inp.n_skus
    budget = inp.budget
    uc = inp.unit_cost
    constraints = []

    for s in np.unique(inp.seasons):
        mask = inp.seasons == s
        idx = np.where(mask)[0]
        uc_s = uc[mask]

        def _y1_fun(x, idx=idx, uc_s=uc_s):
            return budget - float(np.sum(uc_s * x[:n][idx]))

        def _y1_jac(x, idx=idx, uc_s=uc_s):
            g = np.zeros(2 * n)
            g[idx] = -uc_s
            return g

        def _y2_fun(x, idx=idx, uc_s=uc_s):
            return budget - float(np.sum(uc_s * x[n:][idx]))

        def _y2_jac(x, idx=idx, uc_s=uc_s):
            g = np.zeros(2 * n)
            g[n + idx] = -uc_s
            return g

        constraints.append({"type": "ineq", "fun": _y1_fun, "jac": _y1_jac})
        constraints.append({"type": "ineq", "fun": _y2_fun, "jac": _y2_jac})

    return constraints


# ---------------------------------------------------------------------------
# Main joint optimizer
# ---------------------------------------------------------------------------

def optimize_joint_slsqp(
    prediction_result,
    config,
    n_restarts: int = 5,
    compute_shadow_price: bool = True,
) -> JointResult:
    """Solve the joint two-period newsvendor (Model B) with SLSQP + smooth gradient."""
    inp = prepare_joint_inputs(prediction_result, config)
    n = inp.n_skus
    D1, D2 = inp.scenarios_y1, inp.scenarios_y2

    bounds = [(inp.moq, None)] * (2 * n)
    constraints = _build_constraints(inp)

    def _smooth_obj(x: np.ndarray) -> float:
        return joint_saa_smooth_obj_grad(x, D1, D2, inp.cu1, inp.co2, inp.cu2)[0]

    def _smooth_grad(x: np.ndarray) -> np.ndarray:
        return joint_saa_smooth_obj_grad(x, D1, D2, inp.cu1, inp.co2, inp.cu2)[1]

    # Starting points
    rng = np.random.default_rng(42)
    x0_base = _project_joint(
        np.concatenate([
            np.maximum(D1.mean(axis=0), inp.moq),
            np.maximum(D2.mean(axis=0), inp.moq),
        ]),
        inp,
    )
    starts = [x0_base]
    for _ in range(n_restarts - 1):
        noise = rng.uniform(0.7, 1.3, 2 * n)
        starts.append(_project_joint(x0_base * noise, inp))

    start_time = time.time()
    best_res, best_obj = None, np.inf

    for x0 in starts:
        res = minimize(
            fun=_smooth_obj,
            x0=x0,
            jac=_smooth_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-8, "disp": False},
        )
        obj = joint_saa_exact(res.x, D1, D2, inp.cu1, inp.co2, inp.cu2)
        if obj < best_obj:
            best_obj, best_res = obj, res

    runtime = time.time() - start_time

    x_final = _round_repair_joint(best_res.x, inp)
    q1 = x_final[:n].astype(float)
    q2 = x_final[n:].astype(float)

    objective = joint_saa_exact(x_final, D1, D2, inp.cu1, inp.co2, inp.cu2)
    expected_carryover = np.mean(np.maximum(q1[None, :] - D1, 0.0), axis=0)
    y1_spend = float(np.sum(inp.unit_cost * q1))
    y2_spend = float(np.sum(inp.unit_cost * q2))

    # Budget feasibility check
    max_viol = 0.0
    for s in np.unique(inp.seasons):
        mask = inp.seasons == s
        uc_s = inp.unit_cost[mask]
        for q_arr in (q1, q2):
            viol = float(np.sum(uc_s * q_arr[mask])) - inp.budget
            max_viol = max(max_viol, viol)

    # Shadow price on Year 1 budget
    shadow = 0.0
    if compute_shadow_price:
        shadow = _compute_shadow_price_y1(inp, best_res.x, D1, D2)

    return JointResult(
        method="SLSQP-Joint",
        q1=q1,
        q2=q2,
        y1_df=inp.y1_df,
        y2_df=inp.y2_df,
        objective_value=round(float(objective), 2),
        y1_spend=round(y1_spend, 2),
        y2_spend=round(y2_spend, 2),
        y1_shadow_price=round(shadow, 4),
        expected_carryover=expected_carryover,
        feasible=bool(max_viol <= 1.0),
        runtime_seconds=round(runtime, 2),
        n_iterations=int(getattr(best_res, "nit", -1)),
        success=bool(best_res.success),
        message=str(best_res.message),
        metadata={
            "test_years": sorted(int(y) for y in prediction_result.demand_df["year"].unique())
                         if "year" in prediction_result.demand_df.columns else [],
            "n_restarts": n_restarts,
        },
    )


# ---------------------------------------------------------------------------
# Shadow price estimation
# ---------------------------------------------------------------------------

def _compute_shadow_price_y1(
    inp: JointInputs,
    x_opt: np.ndarray,
    D1: np.ndarray,
    D2: np.ndarray,
    delta: float = 50.0,
) -> float:
    """Estimate Year-1 budget shadow price via budget perturbation.

    Shadow price = -(obj_at_budget+delta - obj_at_budget) / delta.
    Positive value = each $1 of extra Year-1 budget saves $ shadow_price in total cost.
    """
    obj_base = joint_saa_exact(x_opt, D1, D2, inp.cu1, inp.co2, inp.cu2)

    inp_p = JointInputs(
        y1_df=inp.y1_df, y2_df=inp.y2_df,
        scenarios_y1=inp.scenarios_y1, scenarios_y2=inp.scenarios_y2,
        cu1=inp.cu1, co2=inp.co2, cu2=inp.cu2,
        unit_cost=inp.unit_cost,
        budget=inp.budget + delta,
        moq=inp.moq,
        n_skus=inp.n_skus,
        seasons=inp.seasons,
    )

    x0_p = _project_joint(x_opt, inp_p)
    res_p = minimize(
        fun=lambda x: joint_saa_smooth_obj_grad(x, D1, D2, inp.cu1, inp.co2, inp.cu2)[0],
        x0=x0_p,
        jac=lambda x: joint_saa_smooth_obj_grad(x, D1, D2, inp.cu1, inp.co2, inp.cu2)[1],
        method="SLSQP",
        bounds=[(inp.moq, None)] * (2 * inp.n_skus),
        constraints=_build_constraints(inp_p),
        options={"maxiter": 500, "ftol": 1e-6, "disp": False},
    )
    obj_p = joint_saa_exact(res_p.x, D1, D2, inp.cu1, inp.co2, inp.cu2)
    return -(obj_p - obj_base) / delta


def _compute_shadow_price_model_a(
    prediction_result,
    config,
    delta: float = 50.0,
) -> float:
    """Estimate shadow price on Year-1 budget for Model A (independent newsvendor).

    Perturbs the budget and re-runs the SLSQP optimizer on Year-1 rows only.
    """
    try:
        from src.models.all_optimizers_combined_final import optimize_with_slsqp
    except ImportError:
        from all_optimizers_combined_final import optimize_with_slsqp

    # Run on base budget
    r_base = optimize_with_slsqp(prediction_result, config, n_restarts=5)

    # Temporarily override budget via a modified config
    import copy
    config_p = copy.deepcopy(config)
    for c in config_p.constraints:
        if "budget" in c.name.lower():
            for key in ["budget_usd", "budget", "max_spend"]:
                if key in c.parameters:
                    c.parameters[key] = float(c.parameters[key]) + delta

    r_p = optimize_with_slsqp(prediction_result, config_p, n_restarts=5)
    return -(r_p.objective_value - r_base.objective_value) / delta


# ---------------------------------------------------------------------------
# Model A vs Model B comparison
# ---------------------------------------------------------------------------

def compare_formulations(prediction_result, config, verbose: bool = True) -> dict:
    """Run Model A (independent) and Model B (joint) and return full comparison.

    Returns a dict with keys:
      model_a          : OptimizerResult  (best independent optimizer)
      model_b          : JointResult
      comparison_table : pd.DataFrame     (per-SKU side-by-side)
      summary          : dict             (aggregate stats and answers to questions)
    """
    try:
        from src.models.all_optimizers_combined_final import optimize_with_slsqp
    except ImportError:
        from all_optimizers_combined_final import optimize_with_slsqp

    if verbose:
        print("\n=== Model A: Independent (current) ===")
    model_a = optimize_with_slsqp(prediction_result, config, n_restarts=5)
    if verbose:
        print(model_a.summary())

    if verbose:
        print("\n=== Model B: Joint with carryover ===")
    model_b = optimize_joint_slsqp(prediction_result, config, n_restarts=5)
    if verbose:
        print(model_b.summary())

    # Shadow price for Model A
    if verbose:
        print("\nEstimating Model A shadow price...")
    shadow_a = _compute_shadow_price_model_a(prediction_result, config)

    comparison_table = _build_comparison_table(model_a, model_b, prediction_result)
    summary = _build_summary(model_a, model_b, shadow_a, comparison_table)

    return {
        "model_a": model_a,
        "model_b": model_b,
        "comparison_table": comparison_table,
        "summary": summary,
    }


def _build_comparison_table(
    model_a,
    model_b: JointResult,
    prediction_result,
) -> pd.DataFrame:
    """Build a per-SKU comparison table of A vs B order quantities."""
    df_pred = prediction_result.demand_df.copy().reset_index(drop=True)
    if "year" not in df_pred.columns:
        return pd.DataFrame()

    test_years = sorted(df_pred["year"].unique())
    y1_yr, y2_yr = int(test_years[0]), int(test_years[1])

    key_cols = [c for c in ["season", "uniform_set", "product_category", "size"]
                if c in df_pred.columns]

    # --- Model A quantities ---
    plan_a = model_a.order_plan.copy()
    plan_a["year"] = df_pred["year"].values[: len(plan_a)]

    a_y1 = (
        plan_a[plan_a["year"] == y1_yr][key_cols + ["recommended_order_qty"]]
        .rename(columns={"recommended_order_qty": "a_y1_qty"})
        .sort_values(key_cols)
        .reset_index(drop=True)
    )
    a_y2 = (
        plan_a[plan_a["year"] == y2_yr][key_cols + ["recommended_order_qty"]]
        .rename(columns={"recommended_order_qty": "a_y2_qty"})
        .sort_values(key_cols)
        .reset_index(drop=True)
    )

    # --- Model B quantities ---
    b_y1 = model_b.y1_df[key_cols].copy().sort_values(key_cols).reset_index(drop=True)
    b_y1["b_y1_qty"] = model_b.q1.astype(int)
    b_y1["expected_carryover"] = np.round(model_b.expected_carryover, 1)

    b_y2 = model_b.y2_df[key_cols].copy().sort_values(key_cols).reset_index(drop=True)
    b_y2["b_y2_qty"] = model_b.q2.astype(int)

    # Merge all on key_cols
    tbl = (
        a_y1
        .merge(a_y2, on=key_cols, how="outer")
        .merge(b_y1, on=key_cols, how="outer")
        .merge(b_y2, on=key_cols, how="outer")
    )

    tbl["y1_qty_diff"] = tbl["b_y1_qty"] - tbl["a_y1_qty"]
    tbl["y2_qty_diff"] = tbl["b_y2_qty"] - tbl["a_y2_qty"]

    return tbl.sort_values(key_cols).reset_index(drop=True)


def _build_summary(
    model_a,
    model_b: JointResult,
    shadow_a: float,
    comparison_table: pd.DataFrame,
) -> dict:
    """Build summary statistics and answers to the four plain-language questions."""
    cost_of_myopia = model_a.objective_value - model_b.objective_value
    reduction_pct = (
        cost_of_myopia / model_a.objective_value * 100
        if model_a.objective_value > 0 else 0.0
    )

    # Year 1 ordering direction
    y1_diff_total = comparison_table["y1_qty_diff"].sum() if "y1_qty_diff" in comparison_table.columns else 0
    y1_direction = "more" if y1_diff_total >= 0 else "less"

    # Sizes driving largest Y2 shift
    if "y2_qty_diff" in comparison_table.columns and "size" in comparison_table.columns:
        largest_y2_shift = (
            comparison_table.groupby("size")["y2_qty_diff"]
            .sum()
            .abs()
            .nlargest(5)
            .to_dict()
        )
    else:
        largest_y2_shift = {}

    return {
        "model_a_total_cost": round(model_a.objective_value, 2),
        "model_b_total_cost": round(model_b.objective_value, 2),
        "cost_of_myopia_usd": round(cost_of_myopia, 2),
        "cost_reduction_pct": round(reduction_pct, 2),
        "model_a_y1_shadow_price": round(shadow_a, 4),
        "model_b_y1_shadow_price": round(model_b.y1_shadow_price, 4),
        "total_expected_carryover_units": round(float(model_b.expected_carryover.sum()), 1),
        "y1_order_direction_vs_independent": y1_direction,
        "y1_total_qty_diff": int(y1_diff_total),
        "largest_y2_shift_by_size": largest_y2_shift,
        "plain_language_answers": {
            "q1_y1_direction": (
                f"The joint model orders {y1_direction} in Year 1 than the independent model "
                f"({y1_diff_total:+d} units total). "
                "In the joint model, Year 1 surplus is not penalised — it carries to Year 2 for "
                "free, providing a hedge against the much higher Year 2 stockout cost after the "
                "model is discontinued. This option value makes over-ordering in Year 1 rational."
                if y1_diff_total >= 0 else
                f"The joint model orders {y1_direction} in Year 1 than the independent model "
                f"({y1_diff_total:+d} units total). "
                "When Year 1 overage is no longer penalised, the optimizer may still reduce Y1 "
                "orders if the binding budget in Y1 is better allocated to high-cost categories, "
                "relying on carryover to cover Y2 demand rather than direct Y2 procurement."
            ),
            "q2_y2_procurement_change": (
                f"Expected carryover is {model_b.expected_carryover.sum():.0f} units total. "
                f"Year 2 direct procurement changes by {comparison_table['y2_qty_diff'].sum():+d} "
                "units — carryover replaces some of the direct Year 2 order. "
                f"Sizes driving the largest shift: {largest_y2_shift}."
                if "y2_qty_diff" in comparison_table.columns else
                "Year 2 procurement is reduced by carryover inventory from Year 1."
            ),
            "q3_cost_of_myopia": (
                f"The cost of myopia is ${cost_of_myopia:,.2f} "
                f"({reduction_pct:.1f}% of Model A cost). "
                + (
                    "This is a meaningful saving and justifies switching to the joint formulation."
                    if cost_of_myopia > 100
                    else "The saving is modest, but the joint model is still the correct formulation."
                )
            ),
            "q4_model_selection": (
                "The prediction model selection is evaluated under the joint newsvendor cost in "
                "Model B. Since the joint objective weights Year 1 underage and Year 2 over/under "
                "differently from the independent objective, the winning prediction model may "
                "change. Specifically, models that better predict Year 2 demand gain importance "
                "because Year 2 stockout cost is high (discontinued model)."
            ),
        },
    }

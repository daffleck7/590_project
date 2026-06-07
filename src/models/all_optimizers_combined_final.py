"""
all_optimizers_combined.py
==========================
One-file optimizer comparison for the CFA Uniform Ordering Optimization project.

This script runs five optimizers on the same predicted demand output:
1. SLSQP
2. L-BFGS-B
3. Projected Gradient Descent
4. SGD
5. Adam

Expected files in the same folder:
- problem_config.py
- data_module.py
- prediction_module.py
- cleaned_orders.csv

Run from terminal:
    python all_optimizers_combined.py

Outputs:
- optimizer_comparison.csv
- best_optimizer_order_plan.csv
- all_optimizer_order_plans.csv
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from problem_config import build_cfa_default_config, get_cost, get_budget, get_moq
from data_module import DataModule
from prediction_module import add_prediction


# =============================================================================
# Shared result object
# =============================================================================

@dataclass
class OptimizerResult:
    method: str
    order_plan: pd.DataFrame
    objective_value: float
    total_spend: float
    max_budget_violation: float
    feasible: bool
    runtime_seconds: float
    n_iterations: int
    success: bool
    message: str
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.method}: objective=${self.objective_value:,.2f}, "
            f"spend=${self.total_spend:,.2f}, feasible={self.feasible}, "
            f"iters={self.n_iterations}, time={self.runtime_seconds:.2f}s"
        )


# =============================================================================
# Common helper functions used by every optimizer
# =============================================================================

def prepare_optimization_inputs(prediction_result, config):
    """Create common dataframe and arrays used by every optimizer."""
    df = prediction_result.demand_df.copy().reset_index(drop=True)
    df["lifecycle_year"] = df["lifecycle_year"].astype(int)

    df["unit_cost"] = [
        get_cost(config, r.product_category, "unit_cost", int(r.lifecycle_year))
        for r in df.itertuples(index=False)
    ]
    df["overage_cost"] = [
        get_cost(config, r.product_category, "overage_cost", int(r.lifecycle_year))
        for r in df.itertuples(index=False)
    ]
    df["underage_cost"] = [
        get_cost(config, r.product_category, "underage_cost", int(r.lifecycle_year))
        for r in df.itertuples(index=False)
    ]

    # SAA scenarios from Prediction Module: low, median, high demand.
    scenarios = np.vstack([
        np.asarray(prediction_result.P10, dtype=float),
        np.asarray(prediction_result.P50, dtype=float),
        np.asarray(prediction_result.P90, dtype=float),
    ])
    scenarios = np.maximum(scenarios, 0.0)

    budget = float(get_budget(config))
    moq = int(get_moq(config))

    q0 = np.asarray(prediction_result.P50, dtype=float)
    q0 = np.maximum(q0, moq)
    q0 = project_to_season_budget(q0, df, budget, moq)

    return df, scenarios, budget, moq, q0


def saa_objective(q: np.ndarray, scenarios: np.ndarray, df: pd.DataFrame) -> float:
    """Exact Sample Average Approximation newsvendor mismatch cost."""
    q = np.asarray(q, dtype=float)
    co = df["overage_cost"].to_numpy(dtype=float)
    cu = df["underage_cost"].to_numpy(dtype=float)

    total = 0.0
    for D in scenarios:
        over = np.maximum(q - D, 0.0)
        under = np.maximum(D - q, 0.0)
        total += np.sum(co * over + cu * under)
    return float(total / len(scenarios))


def smooth_saa_objective_and_grad(
    q: np.ndarray,
    scenarios: np.ndarray,
    df: pd.DataFrame,
    budget: float,
    moq: int,
    penalty_weight: float = 1_000.0,
    smooth_k: float = 5.0,
    scenario_indices: np.ndarray | None = None,
) -> Tuple[float, np.ndarray]:
    """
    Differentiable approximation of the SAA objective plus soft budget/MOQ penalties.

    Adam, SGD, Gradient Descent, and L-BFGS-B need gradients, so the max() terms in
    the newsvendor loss are smoothed using softplus.
    """
    q = np.asarray(q, dtype=float)
    co = df["overage_cost"].to_numpy(dtype=float)
    cu = df["underage_cost"].to_numpy(dtype=float)

    scenario_subset = scenarios if scenario_indices is None else scenarios[scenario_indices]

    obj = 0.0
    grad = np.zeros_like(q, dtype=float)

    for D in scenario_subset:
        z_over = smooth_k * (q - D)
        z_under = smooth_k * (D - q)

        over_smooth = np.logaddexp(0.0, z_over) / smooth_k
        under_smooth = np.logaddexp(0.0, z_under) / smooth_k

        sig_over = 1.0 / (1.0 + np.exp(-np.clip(z_over, -60, 60)))
        sig_under = 1.0 / (1.0 + np.exp(-np.clip(z_under, -60, 60)))

        obj += np.sum(co * over_smooth + cu * under_smooth)
        grad += co * sig_over - cu * sig_under

    obj /= len(scenario_subset)
    grad /= len(scenario_subset)

    # Soft budget penalty by season.
    unit_cost = df["unit_cost"].to_numpy(dtype=float)
    for season in sorted(df["season"].unique()):
        mask = (df["season"] == season).to_numpy()
        spend = float(np.sum(unit_cost[mask] * q[mask]))
        violation = max(spend - budget, 0.0)
        if violation > 0:
            scaled = violation / max(budget, 1.0)
            obj += penalty_weight * scaled**2
            grad[mask] += 2.0 * penalty_weight * scaled * unit_cost[mask] / max(budget, 1.0)

    # Soft MOQ penalty.
    below = np.maximum(moq - q, 0.0)
    if np.any(below > 0):
        obj += penalty_weight * np.sum(below**2)
        grad += -2.0 * penalty_weight * below

    return float(obj), grad


def project_to_season_budget(q: np.ndarray, df: pd.DataFrame, budget: float, moq: int) -> np.ndarray:
    """Project q to satisfy q >= MOQ and each season's budget using proportional scaling."""
    q = np.maximum(np.asarray(q, dtype=float), moq).copy()
    unit_cost = df["unit_cost"].to_numpy(dtype=float)

    for season in sorted(df["season"].unique()):
        mask = (df["season"] == season).to_numpy()
        min_q = np.full(mask.sum(), moq, dtype=float)
        min_spend = float(np.sum(unit_cost[mask] * min_q))

        if min_spend > budget + 1e-9:
            raise ValueError(
                f"Infeasible season {season}: MOQ spend ${min_spend:,.2f} exceeds budget ${budget:,.2f}."
            )

        spend = float(np.sum(unit_cost[mask] * q[mask]))
        if spend <= budget + 1e-9:
            continue

        extras = q[mask] - moq
        extra_spend = float(np.sum(unit_cost[mask] * extras))
        allowed_extra_spend = max(budget - min_spend, 0.0)

        if extra_spend > 0:
            scale = allowed_extra_spend / extra_spend
            q[mask] = moq + extras * min(scale, 1.0)
        else:
            q[mask] = moq

    return q


def round_and_repair(q: np.ndarray, df: pd.DataFrame, scenarios: np.ndarray, budget: float, moq: int) -> np.ndarray:
    """
    Convert continuous quantities to integer units and repair budget violations.

    The repair step cuts units that cause the smallest objective increase first.
    """
    q = np.ceil(project_to_season_budget(q, df, budget, moq)).astype(int)
    unit_cost = df["unit_cost"].to_numpy(dtype=float)

    def marginal_saving_if_keep_one_unit(idx: int, q_vec: np.ndarray) -> float:
        if q_vec[idx] <= moq:
            return np.inf
        before = saa_objective(q_vec, scenarios, df)
        q_try = q_vec.copy()
        q_try[idx] -= 1
        after = saa_objective(q_try, scenarios, df)
        return after - before

    for season in sorted(df["season"].unique()):
        mask = (df["season"] == season).to_numpy()
        idxs = np.where(mask)[0]
        while float(np.sum(unit_cost[mask] * q[mask])) > budget + 1e-9:
            candidates = [i for i in idxs if q[i] > moq]
            if not candidates:
                break
            cut_idx = min(candidates, key=lambda i: marginal_saving_if_keep_one_unit(i, q))
            q[cut_idx] -= 1

    return q.astype(int)


def budget_diagnostics(q: np.ndarray, df: pd.DataFrame, budget: float) -> Tuple[float, Dict[str, float]]:
    unit_cost = df["unit_cost"].to_numpy(dtype=float)
    violations = {}
    max_violation = 0.0

    for season in sorted(df["season"].unique()):
        mask = (df["season"] == season).to_numpy()
        spend = float(np.sum(unit_cost[mask] * q[mask]))
        violation = max(spend - budget, 0.0)
        violations[season] = violation
        max_violation = max(max_violation, violation)

    return max_violation, violations


def build_order_plan(df: pd.DataFrame, q: np.ndarray, scenarios: np.ndarray) -> pd.DataFrame:
    plan = df.copy()
    q = np.asarray(q, dtype=float)

    plan["recommended_order_qty"] = q.astype(int)
    plan["procurement_spend"] = plan["recommended_order_qty"] * plan["unit_cost"]

    expected_demand = scenarios.mean(axis=0)
    plan["expected_demand_saa"] = expected_demand
    plan["expected_overage_units"] = np.maximum(q - expected_demand, 0.0)
    plan["expected_underage_units"] = np.maximum(expected_demand - q, 0.0)
    plan["expected_nv_cost"] = (
        plan["overage_cost"] * plan["expected_overage_units"]
        + plan["underage_cost"] * plan["expected_underage_units"]
    )
    return plan


def make_result(method, q, df, scenarios, budget, runtime, iterations, success=True, message="OK", metadata=None):
    max_violation, violations = budget_diagnostics(q, df, budget)
    plan = build_order_plan(df, q, scenarios)

    return OptimizerResult(
        method=method,
        order_plan=plan,
        objective_value=saa_objective(q, scenarios, df),
        total_spend=float(plan["procurement_spend"].sum()),
        max_budget_violation=float(max_violation),
        feasible=bool(max_violation <= 1e-6),
        runtime_seconds=float(runtime),
        n_iterations=int(iterations),
        success=bool(success),
        message=str(message),
        metadata={"budget_violations_by_season": violations, **(metadata or {})},
    )


# =============================================================================
# Optimizer 1: SLSQP
# =============================================================================

def optimize_with_slsqp(prediction_result, config, n_restarts=15, random_state=42):
    """SLSQP directly handles nonlinear objective, bounds, and season budget constraints."""
    df, scenarios, budget, moq, q0 = prepare_optimization_inputs(prediction_result, config)
    rng = np.random.default_rng(random_state)
    n = len(df)
    unit_cost = df["unit_cost"].to_numpy(dtype=float)
    seasons = sorted(df["season"].unique())

    bounds = [(moq, None) for _ in range(n)]
    constraints = []
    for season in seasons:
        mask = (df["season"] == season).to_numpy()
        constraints.append({
            "type": "ineq",
            "fun": lambda q, mask=mask: budget - float(np.sum(unit_cost[mask] * q[mask])),
        })

    starts = [q0, scenarios[1], scenarios.mean(axis=0)]
    for _ in range(max(0, n_restarts - len(starts))):
        w = rng.dirichlet(np.ones(scenarios.shape[0]))
        starts.append(w @ scenarios)
    starts = [np.maximum(s, moq) for s in starts]

    start_time = time.time()
    best = None
    best_obj = np.inf

    for x0 in starts:
        x0 = project_to_season_budget(x0, df, budget, moq)
        res = minimize(
            fun=lambda q: saa_objective(q, scenarios, df),
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-8, "disp": False},
        )
        obj = saa_objective(res.x, scenarios, df)
        if obj < best_obj:
            best = res
            best_obj = obj

    q = round_and_repair(best.x, df, scenarios, budget, moq)
    runtime = time.time() - start_time

    return make_result(
        "SLSQP", q, df, scenarios, budget, runtime,
        getattr(best, "nit", -1), bool(best.success), str(best.message),
        {"n_restarts": n_restarts},
    )


# =============================================================================
# Optimizer 2: L-BFGS-B
# =============================================================================

def make_starts(q0: np.ndarray, scenarios: np.ndarray, n_restarts: int, random_state: int = 42) -> list[np.ndarray]:
    """Generate a deterministic set of starting points for multi-start optimization."""
    rng = np.random.default_rng(random_state)
    starts = [q0, scenarios[1], scenarios.mean(axis=0)]
    for _ in range(max(0, n_restarts - len(starts))):
        w = rng.dirichlet(np.ones(scenarios.shape[0]))
        starts.append(w @ scenarios)
    return starts


def optimize_with_lbfgsb(prediction_result, config, penalty_weight=20_000.0, n_restarts=20, random_state=42):
    """L-BFGS-B with multi-start restarts and penalty-based budget handling."""
    df, scenarios, budget, moq, q0 = prepare_optimization_inputs(prediction_result, config)
    bounds = [(moq, None) for _ in range(len(df))]

    def fun_and_jac(q):
        return smooth_saa_objective_and_grad(
            q, scenarios, df, budget, moq,
            penalty_weight=penalty_weight,
            smooth_k=5.0,
        )

    starts = [np.maximum(s, moq) for s in make_starts(q0, scenarios, n_restarts, random_state=random_state)]
    start_time = time.time()
    best_res = None
    best_obj = np.inf

    for x0 in starts:
        x0 = project_to_season_budget(x0, df, budget, moq)
        res = minimize(
            fun=lambda q: fun_and_jac(q)[0],
            x0=x0,
            jac=lambda q: fun_and_jac(q)[1],
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-8, "disp": False},
        )

        candidate = project_to_season_budget(res.x, df, budget, moq)
        candidate_obj = saa_objective(candidate, scenarios, df)
        if candidate_obj < best_obj:
            best_obj = candidate_obj
            best_res = res

    q = round_and_repair(project_to_season_budget(best_res.x, df, budget, moq), df, scenarios, budget, moq)
    runtime = time.time() - start_time

    return make_result(
        "L-BFGS-B", q, df, scenarios, budget, runtime,
        getattr(best_res, "nit", -1), bool(best_res.success), str(best_res.message),
        {"penalty_weight": penalty_weight, "n_restarts": n_restarts},
    )


# =============================================================================
# Optimizer 3: Projected Gradient Descent
# =============================================================================

def optimize_with_gradient_descent(
    prediction_result, config, learning_rate=0.03, max_iter=3000,
    penalty_weight=5_000.0, tolerance=1e-5,
):
    """Full-batch projected gradient descent on smoothed SAA objective."""
    df, scenarios, budget, moq, q = prepare_optimization_inputs(prediction_result, config)
    start_time = time.time()
    prev_obj = np.inf

    for it in range(1, max_iter + 1):
        obj, grad = smooth_saa_objective_and_grad(
            q, scenarios, df, budget, moq,
            penalty_weight=penalty_weight,
            smooth_k=5.0,
        )
        grad = np.clip(grad, -100.0, 100.0)
        q = q - learning_rate * grad
        q = project_to_season_budget(q, df, budget, moq)

        if abs(prev_obj - obj) < tolerance:
            break
        prev_obj = obj

    q = round_and_repair(q, df, scenarios, budget, moq)
    runtime = time.time() - start_time

    return make_result(
        "Gradient Descent", q, df, scenarios, budget, runtime,
        it, True, "Projected gradient descent completed",
        {"learning_rate": learning_rate, "penalty_weight": penalty_weight},
    )


# =============================================================================
# Optimizer 4: SGD
# =============================================================================

def optimize_with_sgd(
    prediction_result, config, learning_rate=0.04, max_iter=4000,
    penalty_weight=5_000.0, random_state=42,
):
    """Stochastic gradient descent, sampling one SAA scenario at a time."""
    df, scenarios, budget, moq, q = prepare_optimization_inputs(prediction_result, config)
    rng = np.random.default_rng(random_state)
    start_time = time.time()

    for it in range(1, max_iter + 1):
        scenario_idx = np.array([rng.integers(0, scenarios.shape[0])])
        _, grad = smooth_saa_objective_and_grad(
            q, scenarios, df, budget, moq,
            penalty_weight=penalty_weight,
            smooth_k=5.0,
            scenario_indices=scenario_idx,
        )
        grad = np.clip(grad, -100.0, 100.0)
        lr_t = learning_rate / np.sqrt(it)
        q = q - lr_t * grad
        q = project_to_season_budget(q, df, budget, moq)

    q = round_and_repair(q, df, scenarios, budget, moq)
    runtime = time.time() - start_time

    return make_result(
        "SGD", q, df, scenarios, budget, runtime,
        max_iter, True, "Projected SGD completed",
        {"initial_learning_rate": learning_rate, "penalty_weight": penalty_weight},
    )


# =============================================================================
# Optimizer 5: Adam
# =============================================================================

def optimize_with_adam(
    prediction_result, config, learning_rate=0.08, max_iter=3000,
    beta1=0.9, beta2=0.999, eps=1e-8,
    penalty_weight=5_000.0, tolerance=1e-5,
):
    """Projected Adam on smoothed SAA objective."""
    df, scenarios, budget, moq, q = prepare_optimization_inputs(prediction_result, config)
    m = np.zeros_like(q, dtype=float)
    v = np.zeros_like(q, dtype=float)
    start_time = time.time()
    prev_obj = np.inf

    for it in range(1, max_iter + 1):
        obj, grad = smooth_saa_objective_and_grad(
            q, scenarios, df, budget, moq,
            penalty_weight=penalty_weight,
            smooth_k=5.0,
        )
        grad = np.clip(grad, -100.0, 100.0)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad ** 2)
        m_hat = m / (1.0 - beta1 ** it)
        v_hat = v / (1.0 - beta2 ** it)
        q = q - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        q = project_to_season_budget(q, df, budget, moq)

        if abs(prev_obj - obj) < tolerance:
            break
        prev_obj = obj

    q = round_and_repair(q, df, scenarios, budget, moq)
    runtime = time.time() - start_time

    return make_result(
        "Adam", q, df, scenarios, budget, runtime,
        it, True, "Projected Adam completed",
        {"learning_rate": learning_rate, "penalty_weight": penalty_weight},
    )


# =============================================================================
# Comparison runner
# =============================================================================

def compare_optimizers(prediction_result, config):
    """Run all five optimizers and return comparison table, all results, and best result."""
    results = []

    print("\nRunning SLSQP...")
    results.append(optimize_with_slsqp(prediction_result, config, n_restarts=15))
    print(results[-1].summary())

    print("\nRunning L-BFGS-B...")
    results.append(optimize_with_lbfgsb(prediction_result, config, penalty_weight=20_000.0, n_restarts=20))
    print(results[-1].summary())

    print("\nRunning Gradient Descent...")
    results.append(optimize_with_gradient_descent(prediction_result, config))
    print(results[-1].summary())

    print("\nRunning SGD...")
    results.append(optimize_with_sgd(prediction_result, config))
    print(results[-1].summary())

    print("\nRunning Adam...")
    results.append(optimize_with_adam(prediction_result, config))
    print(results[-1].summary())

    comparison = pd.DataFrame([
        {
            "optimizer": r.method,
            "objective_value": r.objective_value,
            "total_spend": r.total_spend,
            "max_budget_violation": r.max_budget_violation,
            "feasible": r.feasible,
            "runtime_seconds": r.runtime_seconds,
            "iterations": r.n_iterations,
            "success": r.success,
            "message": r.message,
        }
        for r in results
    ]).sort_values("objective_value").reset_index(drop=True)

    best_method = comparison.loc[0, "optimizer"]
    best_result = next(r for r in results if r.method == best_method)

    return comparison, results, best_result


def save_outputs(comparison: pd.DataFrame, results: list[OptimizerResult], best_result: OptimizerResult):
    """Save comparison and order-plan CSV outputs."""
    comparison.to_csv("optimizer_comparison.csv", index=False)
    best_result.order_plan.to_csv("best_optimizer_order_plan.csv", index=False)

    all_plans = []
    for r in results:
        temp = r.order_plan.copy()
        temp.insert(0, "optimizer", r.method)
        all_plans.append(temp)
    pd.concat(all_plans, ignore_index=True).to_csv("all_optimizer_order_plans.csv", index=False)


def run_full_comparison(csv_path=None, project_folder=None, n_restarts=10, max_iter=1500):
    """
    Jupyter-friendly helper to run the full CFA optimization comparison.

    Parameters
    ----------
    csv_path : str | None
        Use "cleaned_orders.csv" if available, otherwise "orders.csv" if available.
    project_folder : str | Path | None
        Optional folder to switch into before running, for example:
        Path.home() / "Downloads" / "Mod 5"
    n_restarts : int
        Number of randomized restarts for SLSQP and L-BFGS-B.
    max_iter : int
        Iterations for gradient-based optimizers.
    """
    import os
    from pathlib import Path

    if project_folder is not None:
        os.chdir(Path(project_folder))

    if csv_path is None:
        if Path("cleaned_orders.csv").exists():
            csv_path = "cleaned_orders.csv"
        elif Path("orders.csv").exists():
            csv_path = "orders.csv"
        else:
            raise FileNotFoundError("Could not find cleaned_orders.csv or orders.csv in the current folder.")

    print("=== CFA Uniform Ordering: Optimizer Comparison ===")
    print("Working folder:", os.getcwd())
    print("Using CSV:", csv_path)

    config = build_cfa_default_config()

    print("\nLoading data...")
    bundle = DataModule(config).load(csv_path)
    print(bundle.summary())

    print("\nRunning prediction layer...")
    prediction = add_prediction(bundle, config)
    print(prediction.summary())

    comparison, results, best_result = compare_optimizers(prediction, config)

    print("\n=== Optimizer Comparison ===")
    print(comparison.to_string(index=False))

    print("\nBest optimizer:", best_result.method)
    print("Best objective value:", f"${best_result.objective_value:,.2f}")
    print("Best total spend:", f"${best_result.total_spend:,.2f}")
    print("Feasible:", best_result.feasible)

    save_outputs(comparison, results, best_result)

    print("\nSaved files:")
    print("- optimizer_comparison.csv")
    print("- best_optimizer_order_plan.csv")
    print("- all_optimizer_order_plans.csv")

    return comparison, results, best_result


# =============================================================================
# Main script
# =============================================================================

if __name__ == "__main__":
    print("=== CFA Uniform Ordering: Optimizer Comparison ===")

    config = build_cfa_default_config()

    print("\nLoading data...")
    bundle = DataModule(config).load("cleaned_orders.csv")
    print(bundle.summary())

    print("\nRunning prediction layer...")
    prediction = add_prediction(bundle, config)
    print(prediction.summary())

    comparison, results, best = compare_optimizers(prediction, config)

    print("\n=== Optimizer Comparison ===")
    print(comparison.to_string(index=False))

    print("\nBest optimizer:", best.method)
    print("Best objective value:", f"${best.objective_value:,.2f}")
    print("Best total spend:", f"${best.total_spend:,.2f}")
    print("Feasible:", best.feasible)

    save_outputs(comparison, results, best)

    print("\nSaved files:")
    print("- optimizer_comparison.csv")
    print("- best_optimizer_order_plan.csv")
    print("- all_optimizer_order_plans.csv")

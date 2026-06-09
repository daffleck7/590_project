"""Tests for the explanation module: sensitivity_analysis and baseline_comparison."""

import pytest
import pandas as pd

from src.models.problem_config import (
    CostItem,
    Constraint,
    DecisionVariable,
    Objective,
    ProblemConfig,
    UncertainParameter,
)
from src.explanation.sensitivity import sensitivity_analysis
from src.explanation.baseline import baseline_comparison


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_config() -> ProblemConfig:
    """Config with list[CostItem] cost_structure (current schema)."""
    return ProblemConfig(
        problem_description="Test",
        decision_variables=[DecisionVariable(name="qty", type="integer")],
        objective=Objective(direction="minimize", description="Minimize cost"),
        constraints=[],
        uncertain_parameters=[
            UncertainParameter(name="demand", description="", data_column="quantity")
        ],
        cost_structure=[
            CostItem(name="overage_cost_multiplier", value=1.0),
            CostItem(name="underage_cost_multiplier", value=1.3),
        ],
        data_requirements={},
    )


def _make_config_dict_cost() -> ProblemConfig:
    """Config with dict cost_structure (legacy format, bypasses Pydantic validation)."""
    return ProblemConfig.model_construct(
        problem_description="Test",
        cost_structure={"overage_cost_multiplier": 1.0, "underage_cost_multiplier": 1.3},
        decision_variables=[],
        objective=Objective(direction="minimize", description="Minimize cost"),
        constraints=[],
        uncertain_parameters=[],
        data_requirements={},
        solver_hint=None,
        problem_title="Test",
        item_categories=[],
        item_sizes=[],
        time_periods=[],
        custom_fields={},
    )


def _test_demand() -> pd.DataFrame:
    """Single test-year demand row, useful for math verification."""
    return pd.DataFrame([
        {"product_category": "top", "gender_age": "mens_youth", "size": "ym",
         "quantity": 12.0, "unit_price": 25.0, "year": 2025},
    ])


def _demand_with_train() -> pd.DataFrame:
    """Two SKUs across train (2024) and test (2025) years."""
    return pd.DataFrame([
        {"product_category": "top", "gender_age": "mens_youth", "size": "ym",
         "quantity": 10.0, "unit_price": 25.0, "year": 2024},
        {"product_category": "top", "gender_age": "mens_youth", "size": "yl",
         "quantity": 8.0,  "unit_price": 25.0, "year": 2024},
        {"product_category": "top", "gender_age": "mens_youth", "size": "ym",
         "quantity": 12.0, "unit_price": 25.0, "year": 2025},
        {"product_category": "top", "gender_age": "mens_youth", "size": "yl",
         "quantity": 6.0,  "unit_price": 25.0, "year": 2025},
    ])


# ---------------------------------------------------------------------------
# sensitivity_analysis
# ---------------------------------------------------------------------------

class TestSensitivityAnalysis:

    def test_returns_required_keys(self):
        result = sensitivity_analysis(
            {"top_mens_youth_ym": 15.0}, _test_demand(), _make_config()
        )
        assert "base_cost" in result
        assert "co_sensitivity" in result
        assert "cu_sensitivity" in result

    def test_default_produces_seven_shift_points(self):
        result = sensitivity_analysis(
            {"top_mens_youth_ym": 15.0}, _test_demand(), _make_config()
        )
        assert len(result["co_sensitivity"]) == 7
        assert len(result["cu_sensitivity"]) == 7

    def test_zero_shift_gives_zero_pct_change(self):
        result = sensitivity_analysis(
            {"top_mens_youth_ym": 15.0}, _test_demand(), _make_config()
        )
        zero_co = next(r for r in result["co_sensitivity"] if r["shift_pct"] == 0)
        zero_cu = next(r for r in result["cu_sensitivity"] if r["shift_pct"] == 0)
        assert zero_co["pct_change"] == 0.0
        assert zero_cu["pct_change"] == 0.0

    def test_base_cost_overage_math(self):
        # q=15, demand=12 → 3 units over × unit_price=25 × co_mult=1.0 = 75
        result = sensitivity_analysis(
            {"top_mens_youth_ym": 15.0}, _test_demand(), _make_config()
        )
        assert result["base_cost"] == pytest.approx(75.0)

    def test_base_cost_underage_math(self):
        # q=9, demand=12 → 3 units under × unit_price=25 × cu_mult=1.3 = 97.5
        result = sensitivity_analysis(
            {"top_mens_youth_ym": 9.0}, _test_demand(), _make_config()
        )
        assert result["base_cost"] == pytest.approx(97.5)

    def test_perfect_order_zero_base_cost(self):
        result = sensitivity_analysis(
            {"top_mens_youth_ym": 12.0}, _test_demand(), _make_config()
        )
        assert result["base_cost"] == pytest.approx(0.0)

    def test_overage_position_sensitive_to_co_shift(self):
        # Over-ordered: cost must rise when co increases
        result = sensitivity_analysis(
            {"top_mens_youth_ym": 20.0}, _test_demand(), _make_config()
        )
        co_plus  = next(r for r in result["co_sensitivity"] if r["shift_pct"] == 10)
        co_minus = next(r for r in result["co_sensitivity"] if r["shift_pct"] == -10)
        assert co_plus["total_cost"]  > result["base_cost"]
        assert co_minus["total_cost"] < result["base_cost"]

    def test_underage_position_sensitive_to_cu_shift(self):
        # Under-ordered: cost must rise when cu increases
        result = sensitivity_analysis(
            {"top_mens_youth_ym": 5.0}, _test_demand(), _make_config()
        )
        cu_plus  = next(r for r in result["cu_sensitivity"] if r["shift_pct"] == 10)
        cu_minus = next(r for r in result["cu_sensitivity"] if r["shift_pct"] == -10)
        assert cu_plus["total_cost"]  > result["base_cost"]
        assert cu_minus["total_cost"] < result["base_cost"]

    def test_overage_position_insensitive_to_cu_shift(self):
        # Pure overage: underage cost shift should have no effect
        result = sensitivity_analysis(
            {"top_mens_youth_ym": 20.0}, _test_demand(), _make_config()
        )
        for row in result["cu_sensitivity"]:
            assert row["pct_change"] == pytest.approx(0.0)

    def test_dict_cost_structure_matches_list(self):
        order_quantities = {"top_mens_youth_ym": 15.0}
        result_list = sensitivity_analysis(order_quantities, _test_demand(), _make_config())
        result_dict = sensitivity_analysis(order_quantities, _test_demand(), _make_config_dict_cost())
        assert result_list["base_cost"] == pytest.approx(result_dict["base_cost"])
        assert [r["pct_change"] for r in result_list["co_sensitivity"]] == \
               [r["pct_change"] for r in result_dict["co_sensitivity"]]

    def test_custom_shifts(self):
        result = sensitivity_analysis(
            {"top_mens_youth_ym": 15.0}, _test_demand(), _make_config(),
            shifts=[-0.1, 0.0, 0.1]
        )
        assert len(result["co_sensitivity"]) == 3
        assert len(result["cu_sensitivity"]) == 3

    def test_missing_sku_treated_as_zero_order(self):
        # SKU not in order_quantities → q=0, all demand is underage
        result = sensitivity_analysis(
            {}, _test_demand(), _make_config()
        )
        # 12 units under × 25 × 1.3
        assert result["base_cost"] == pytest.approx(12 * 25 * 1.3)


# ---------------------------------------------------------------------------
# baseline_comparison
# ---------------------------------------------------------------------------

class TestBaselineComparison:

    def test_returns_required_keys(self):
        result = baseline_comparison(
            {"top_mens_youth_ym": 12.0, "top_mens_youth_yl": 6.0},
            _demand_with_train(), _make_config(),
            train_years=[2024], test_years=[2025],
        )
        for key in ("agent_cost", "baseline_cost", "cost_reduction_pct",
                    "savings_usd", "by_category", "baseline_quantities"):
            assert key in result

    def test_perfect_agent_zero_cost(self):
        # Agent orders exactly test demand → zero NV cost
        result = baseline_comparison(
            {"top_mens_youth_ym": 12.0, "top_mens_youth_yl": 6.0},
            _demand_with_train(), _make_config(),
            train_years=[2024], test_years=[2025],
        )
        assert result["agent_cost"] == pytest.approx(0.0)

    def test_perfect_agent_beats_imperfect_baseline(self):
        # Baseline is train avg (ym=10, yl=8). Test demand (ym=12, yl=6) differs.
        # Agent at exact demand → savings > 0
        result = baseline_comparison(
            {"top_mens_youth_ym": 12.0, "top_mens_youth_yl": 6.0},
            _demand_with_train(), _make_config(),
            train_years=[2024], test_years=[2025],
        )
        assert result["savings_usd"] > 0.0
        assert result["cost_reduction_pct"] == pytest.approx(100.0)

    def test_bad_agent_worse_than_baseline(self):
        # Agent orders 0 → all underage, worse than baseline ordering avg demand
        result = baseline_comparison(
            {"top_mens_youth_ym": 0.0, "top_mens_youth_yl": 0.0},
            _demand_with_train(), _make_config(),
            train_years=[2024], test_years=[2025],
        )
        assert result["agent_cost"] > result["baseline_cost"]
        assert result["savings_usd"] < 0.0

    def test_baseline_quantities_are_train_averages(self):
        result = baseline_comparison(
            {"top_mens_youth_ym": 12.0, "top_mens_youth_yl": 6.0},
            _demand_with_train(), _make_config(),
            train_years=[2024], test_years=[2025],
        )
        # Only one train year, so baseline qty = train demand exactly
        assert result["baseline_quantities"]["top_mens_youth_ym"] == pytest.approx(10.0)
        assert result["baseline_quantities"]["top_mens_youth_yl"] == pytest.approx(8.0)

    def test_by_category_sums_match_totals(self):
        result = baseline_comparison(
            {"top_mens_youth_ym": 11.0, "top_mens_youth_yl": 7.0},
            _demand_with_train(), _make_config(),
            train_years=[2024], test_years=[2025],
        )
        assert sum(c["agent_cost"] for c in result["by_category"]) == \
               pytest.approx(result["agent_cost"])
        assert sum(c["baseline_cost"] for c in result["by_category"]) == \
               pytest.approx(result["baseline_cost"])

    def test_savings_equals_baseline_minus_agent(self):
        result = baseline_comparison(
            {"top_mens_youth_ym": 11.0, "top_mens_youth_yl": 7.0},
            _demand_with_train(), _make_config(),
            train_years=[2024], test_years=[2025],
        )
        expected_savings = result["baseline_cost"] - result["agent_cost"]
        assert result["savings_usd"] == pytest.approx(expected_savings)

    def test_dict_cost_structure_matches_list(self):
        agent_quantities = {"top_mens_youth_ym": 11.0, "top_mens_youth_yl": 7.0}
        kwargs = dict(
            demand_df=_demand_with_train(), train_years=[2024], test_years=[2025]
        )
        r_list = baseline_comparison(agent_quantities, config=_make_config(), **kwargs)
        r_dict = baseline_comparison(agent_quantities, config=_make_config_dict_cost(), **kwargs)
        assert r_list["agent_cost"]    == pytest.approx(r_dict["agent_cost"])
        assert r_list["baseline_cost"] == pytest.approx(r_dict["baseline_cost"])

    def test_baseline_cost_math(self):
        # Baseline ym=10 vs test demand=12 → 2 underage × 25 × 1.3 = 65
        # Baseline yl=8  vs test demand=6  → 2 overage  × 25 × 1.0 = 50
        # Total baseline = 115
        result = baseline_comparison(
            {"top_mens_youth_ym": 12.0, "top_mens_youth_yl": 6.0},
            _demand_with_train(), _make_config(),
            train_years=[2024], test_years=[2025],
        )
        assert result["baseline_cost"] == pytest.approx(115.0)

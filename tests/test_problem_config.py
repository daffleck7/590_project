"""Tests for ProblemConfig Pydantic model."""

import pytest
from pydantic import ValidationError

from src.models.problem_config import (
    CostItem,
    Constraint,
    DecisionVariable,
    Objective,
    ProblemConfig,
    UncertainParameter,
)


def _make_valid_config() -> dict:
    """Return a minimal valid ProblemConfig as a dict."""
    return {
        "problem_description": "Minimize uniform ordering cost for a youth soccer league",
        "decision_variables": [
            {"name": "order_qty_ym", "type": "integer", "lower_bound": 0, "upper_bound": 500},
        ],
        "objective": {
            "direction": "minimize",
            "description": "Total procurement cost including rush and salvage",
        },
        "constraints": [
            {
                "name": "budget",
                "description": "Total spend must not exceed seasonal budget",
                "type": "hard",
                "parameters": {"max_budget": 50000},
            },
        ],
        "uncertain_parameters": [
            {
                "name": "demand_ym",
                "description": "Demand for Youth Medium jerseys",
                "data_column": "quantity",
            },
        ],
        "cost_structure": [
            {"name": "overage_cost", "value": 10.0},
            {"name": "underage_cost", "value": 25.0},
        ],
        "data_requirements": {
            "required_columns": ["size", "quantity", "product_category"],
        },
    }


class TestProblemConfigValidation:
    """Test ProblemConfig validates correctly."""

    def test_valid_config_parses(self) -> None:
        config = ProblemConfig(**_make_valid_config())
        assert config.problem_description.startswith("Minimize")
        assert len(config.decision_variables) == 1
        assert config.objective.direction == "minimize"

    def test_roundtrip_json(self) -> None:
        config = ProblemConfig(**_make_valid_config())
        json_str = config.model_dump_json(indent=2)
        parsed = ProblemConfig.model_validate_json(json_str)
        assert parsed.problem_description == config.problem_description
        assert len(parsed.constraints) == len(config.constraints)

    def test_missing_required_field_raises(self) -> None:
        data = _make_valid_config()
        del data["problem_description"]
        with pytest.raises(ValidationError):
            ProblemConfig(**data)

    def test_solver_hint_optional(self) -> None:
        data = _make_valid_config()
        config = ProblemConfig(**data)
        assert config.solver_hint is None

        data["solver_hint"] = "newsvendor"
        config = ProblemConfig(**data)
        assert config.solver_hint == "newsvendor"

    def test_empty_decision_variables_allowed(self) -> None:
        data = _make_valid_config()
        data["decision_variables"] = []
        config = ProblemConfig(**data)
        assert config.decision_variables == []

    def test_cost_structure_list_of_items(self) -> None:
        config = ProblemConfig(**_make_valid_config())
        assert len(config.cost_structure) == 2
        assert config.cost_structure[0].name == "overage_cost"
        assert config.cost_structure[1].value == 25.0


class TestDecisionVariable:
    """Test DecisionVariable model."""

    def test_upper_bound_optional(self) -> None:
        dv = DecisionVariable(name="qty", type="integer")
        assert dv.upper_bound is None

    def test_lower_bound_defaults_to_zero(self) -> None:
        dv = DecisionVariable(name="qty", type="integer")
        assert dv.lower_bound == 0.0

    def test_with_explicit_bounds(self) -> None:
        dv = DecisionVariable(name="qty", type="continuous", lower_bound=0.0, upper_bound=100.0)
        assert dv.lower_bound == 0.0
        assert dv.upper_bound == 100.0


class TestConstraint:
    """Test Constraint model."""

    def test_valid_constraint(self) -> None:
        c = Constraint(
            name="budget",
            description="Stay under budget",
            type="hard",
            parameters={"max_budget": 50000},
        )
        assert c.type == "hard"
        assert c.parameters["max_budget"] == 50000

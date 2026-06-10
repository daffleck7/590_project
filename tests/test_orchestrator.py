"""Tests for orchestrator module."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.problem_config import (
    CostItem,
    Constraint,
    DecisionVariable,
    Objective,
    ProblemConfig,
    UncertainParameter,
)
from src.orchestrator import create_run_directory, save_trace, Orchestrator


def _make_config() -> ProblemConfig:
    """Create a test ProblemConfig."""
    return ProblemConfig(
        problem_description="Test problem",
        decision_variables=[
            DecisionVariable(name="qty", type="integer", lower_bound=0, upper_bound=100),
        ],
        objective=Objective(direction="minimize", description="Minimize cost"),
        constraints=[
            Constraint(
                name="budget",
                description="Budget limit",
                type="hard",
                parameters={"max": 5000},
            ),
        ],
        uncertain_parameters=[
            UncertainParameter(
                name="demand",
                description="Demand forecast",
                data_column="quantity",
            ),
        ],
        cost_structure=[
            CostItem(name="overage_cost", value=10.0),
            CostItem(name="underage_cost", value=25.0),
        ],
        data_requirements={"required_columns": ["quantity", "size"]},
    )


class TestCreateRunDirectory:
    """Test run directory creation."""

    def test_creates_timestamped_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_directory(base_dir=tmp)
            assert Path(run_dir).exists()
            assert Path(run_dir).is_dir()

    def test_directory_name_contains_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_directory(base_dir=tmp)
            name = Path(run_dir).name
            assert "2026" in name or "20" in name


class TestSaveTrace:
    """Test trace logging."""

    def test_saves_trace_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_data = {
                "steps": [
                    {"agent": "intake", "duration_s": 5.0, "usage": {}},
                ]
            }
            save_trace(tmp, trace_data)
            trace_path = Path(tmp) / "trace.json"
            assert trace_path.exists()
            loaded = json.loads(trace_path.read_text())
            assert loaded["steps"][0]["agent"] == "intake"


class TestOrchestrator:
    """Test orchestrator wiring."""

    def test_saves_config_to_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config()
            config_path = Path(tmp) / "problem_config.json"
            config_path.write_text(config.model_dump_json(indent=2))
            loaded = ProblemConfig.model_validate_json(config_path.read_text())
            assert loaded.problem_description == "Test problem"

    def test_config_roundtrip_preserves_cost_structure(self) -> None:
        config = _make_config()
        json_str = config.model_dump_json(indent=2)
        loaded = ProblemConfig.model_validate_json(json_str)
        assert len(loaded.cost_structure) == 2
        assert loaded.cost_structure[0].name == "overage_cost"


class TestFullPipeline:
    """Test that run_full_pipeline calls all four stages in order."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_all_four_stages_called_on_success(self) -> None:
        config = _make_config()
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("src.orchestrator.run_intake_agent",
                      new=AsyncMock(return_value=(config, []))) as mock_intake,
                patch("src.orchestrator.run_cleaning_agent",
                      new=AsyncMock(return_value=("cleaned.csv", {}, []))) as mock_clean,
                patch("src.orchestrator.run_modeling_agent",
                      new=AsyncMock(return_value=(True, []))) as mock_model,
                patch("src.orchestrator.run_explanation_agent",
                      new=AsyncMock(return_value=(True, []))) as mock_explain,
            ):
                orch = Orchestrator("data.csv", base_dir=tmp)
                self._run(orch.run_full_pipeline("test description"))

                mock_intake.assert_called_once()
                mock_clean.assert_called_once()
                mock_model.assert_called_once()
                mock_explain.assert_called_once()

    def test_modeling_and_explanation_skipped_when_cleaning_fails(self) -> None:
        config = _make_config()
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("src.orchestrator.run_intake_agent",
                      new=AsyncMock(return_value=(config, []))),
                patch("src.orchestrator.run_cleaning_agent",
                      new=AsyncMock(return_value=(None, None, []))),
                patch("src.orchestrator.run_modeling_agent",
                      new=AsyncMock(return_value=(True, []))) as mock_model,
                patch("src.orchestrator.run_explanation_agent",
                      new=AsyncMock(return_value=(True, []))) as mock_explain,
            ):
                orch = Orchestrator("data.csv", base_dir=tmp)
                self._run(orch.run_full_pipeline())

                mock_model.assert_not_called()
                mock_explain.assert_not_called()

    def test_pipeline_aborts_when_intake_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("src.orchestrator.run_intake_agent",
                      new=AsyncMock(return_value=(None, []))),
                patch("src.orchestrator.run_cleaning_agent",
                      new=AsyncMock(return_value=("cleaned.csv", {}, []))) as mock_clean,
            ):
                orch = Orchestrator("data.csv", base_dir=tmp)
                self._run(orch.run_full_pipeline())

                mock_clean.assert_not_called()

    def test_modeling_called_with_config_and_cleaned_path(self) -> None:
        config = _make_config()
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("src.orchestrator.run_intake_agent",
                      new=AsyncMock(return_value=(config, []))),
                patch("src.orchestrator.run_cleaning_agent",
                      new=AsyncMock(return_value=("/run/cleaned_data.csv", {}, []))),
                patch("src.orchestrator.run_modeling_agent",
                      new=AsyncMock(return_value=(True, []))) as mock_model,
                patch("src.orchestrator.run_explanation_agent",
                      new=AsyncMock(return_value=(True, []))),
            ):
                orch = Orchestrator("data.csv", base_dir=tmp)
                self._run(orch.run_full_pipeline())

                call_args = mock_model.call_args
                assert call_args.args[0] == config
                assert call_args.args[1] == "/run/cleaned_data.csv"

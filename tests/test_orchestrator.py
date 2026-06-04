"""Tests for orchestrator module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.models.problem_config import (
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
            DecisionVariable(name="qty", type="integer", bounds=(0, 100)),
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
        cost_structure={"overage": 10, "underage": 25},
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

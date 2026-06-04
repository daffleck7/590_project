# Intake + Data Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-agent system (Intake + Data Cleaning) that converses with users to formulate optimization problems and cleans CSV data for downstream prediction/optimization.

**Architecture:** Two Claude Agent SDK agents orchestrated by a Python module. The Intake Agent converses with the user and produces a ProblemConfig JSON. The Data Cleaning Agent receives that config and transforms raw CSV into clean data via sandboxed code execution. A thin CLI entry point wires everything together.

**Tech Stack:** Python 3.11+, uv, Claude Agent SDK (`claude-agent-sdk`), Pydantic, pandas, pytest, pytest-asyncio

---

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Project metadata, dependencies, scripts |
| `.env.example` | Example environment config |
| `src/__init__.py` | Package root |
| `src/models/__init__.py` | Models package |
| `src/models/problem_config.py` | ProblemConfig Pydantic model |
| `src/intake/__init__.py` | Intake package |
| `src/intake/tools.py` | Read-only CSV inspection tools (MCP tools for Agent SDK) |
| `src/intake/prompts.py` | System prompt for intake agent |
| `src/intake/agent.py` | Intake agent — launches Agent SDK query, returns ProblemConfig |
| `src/data_ingestion/__init__.py` | Data ingestion package |
| `src/data_ingestion/tools.py` | CFA cleaner wrapper + execute_code tool |
| `src/data_ingestion/sandbox.py` | Sandboxed subprocess execution with verification |
| `src/data_ingestion/prompts.py` | System prompt for data cleaning agent |
| `src/data_ingestion/agent.py` | Data cleaning agent — launches Agent SDK query, returns cleaned data path |
| `src/orchestrator.py` | Wires intake → data ingestion, trace logging, run directory management |
| `src/cli.py` | argparse entry point |
| `tests/test_problem_config.py` | ProblemConfig validation tests |
| `tests/test_intake_tools.py` | CSV inspection tool tests |
| `tests/test_sandbox.py` | Sandbox execution + verification tests |
| `tests/test_data_ingestion_tools.py` | CFA cleaner wrapper tests |
| `tests/test_orchestrator.py` | Orchestrator flow tests |
| `tests/fixtures/sample_orders.csv` | Small test CSV (10 rows of CFA-like data) |

---

### Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `src/models/__init__.py`
- Create: `src/intake/__init__.py`
- Create: `src/data_ingestion/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "cfa-optimization-agent"
version = "0.1.0"
description = "AI-powered optimization agent for uniform ordering"
requires-python = ">=3.11"
dependencies = [
    "claude-agent-sdk",
    "pydantic>=2.0",
    "pandas>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]

[project.scripts]
cfa-agent = "src.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Create .env.example**

```
# No API key needed — Agent SDK uses Claude Code subscription
# Optional: override model
# CLAUDE_MODEL=claude-sonnet-4-20250514
```

- [ ] **Step 3: Create package init files**

Create these empty files:
- `src/__init__.py`
- `src/models/__init__.py`
- `src/intake/__init__.py`
- `src/data_ingestion/__init__.py`

Each should contain only:
```python
"""Package docstring."""
```

With docstrings:
- `src/__init__.py`: `"""CFA Optimization Agent."""`
- `src/models/__init__.py`: `"""Data models for the optimization pipeline."""`
- `src/intake/__init__.py`: `"""Intake agent for problem formulation."""`
- `src/data_ingestion/__init__.py`: `"""Data ingestion agent for CSV cleaning."""`

- [ ] **Step 4: Create test fixture**

Create `tests/fixtures/sample_orders.csv` with 10 rows of realistic CFA data:

```csv
Order ID,Created at,Financial Status,Lineitem name,Lineitem variant,Lineitem quantity,Lineitem price
5001,2024-08-15T10:00:00-07:00,PAID,Fall 2024 Game Jersey - Mens/Youth,Navy/YM,1,55.00
5001,2024-08-15T10:00:00-07:00,PAID,Fall 2024 Game Short - Mens/Youth,Navy/YM,1,30.00
5001,2024-08-15T10:00:00-07:00,PAID,Fall 2024 Game Sock - Mens/Youth,Navy/YM,1,15.00
5002,2024-08-16T14:30:00-07:00,PAID,Fall 2024 Game Jersey - Womens,White/WS,2,55.00
5002,2024-08-16T14:30:00-07:00,PAID,Fall 2024 Game Short - Womens,White/WS,2,30.00
5003,2024-09-01T09:00:00-07:00,PAID,Fall 2024 Game Jersey - Mens/Youth,Navy/AL,1,55.00
5004,2024-09-05T11:00:00-07:00,PAID,Fall 2024 Game Jersey - Mens/Youth,Navy/YL,3,55.00
5005,2024-11-10T08:00:00-07:00,PAID,Winter 2024 Game Jersey - Mens/Youth,Black/AM,1,55.00
5006,2024-03-20T12:00:00-07:00,PAID,Spring 2024 Game Jersey - Mens/Youth,Royal/YS,1,55.00
5007,2024-08-20T10:00:00-07:00,REFUNDED,Fall 2024 Practice Jersey - Mens/Youth,Navy/YM,1,35.00
```

Row 10 is REFUNDED + practice gear — should be filtered out by both status and category.

- [ ] **Step 5: Install dependencies**

Run: `uv sync --all-extras`
Expected: Dependencies install successfully, `.venv` created

- [ ] **Step 6: Verify setup**

Run: `uv run python -c "import pydantic; import pandas; print('OK')"`
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .env.example src/__init__.py src/models/__init__.py src/intake/__init__.py src/data_ingestion/__init__.py tests/fixtures/sample_orders.csv
git commit -m "Add project structure with pyproject.toml and package scaffolding"
```

---

### Task 2: ProblemConfig Pydantic Model

**Files:**
- Create: `src/models/problem_config.py`
- Create: `tests/test_problem_config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_problem_config.py`:

```python
"""Tests for ProblemConfig Pydantic model."""

import json

import pytest
from pydantic import ValidationError

from src.models.problem_config import (
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
            {"name": "order_qty_ym", "type": "integer", "bounds": (0, 500)},
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
        "cost_structure": {
            "overage_cost_per_unit": 10.0,
            "underage_cost_per_unit": 25.0,
        },
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
        del data["objective"]
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


class TestDecisionVariable:
    """Test DecisionVariable model."""

    def test_bounds_optional(self) -> None:
        dv = DecisionVariable(name="qty", type="integer")
        assert dv.bounds is None

    def test_with_bounds(self) -> None:
        dv = DecisionVariable(name="qty", type="continuous", bounds=(0.0, 100.0))
        assert dv.bounds == (0.0, 100.0)


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_problem_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.models.problem_config'`

- [ ] **Step 3: Implement ProblemConfig model**

Create `src/models/problem_config.py`:

```python
"""ProblemConfig Pydantic model — central data contract for the optimization pipeline."""

from pydantic import BaseModel


class DecisionVariable(BaseModel):
    """A variable the optimizer controls."""

    name: str
    type: str  # "continuous" | "integer" | "binary"
    bounds: tuple[float, float] | None = None


class Objective(BaseModel):
    """What the optimizer is trying to achieve."""

    direction: str  # "minimize" | "maximize"
    description: str


class Constraint(BaseModel):
    """A constraint on the optimization problem."""

    name: str
    description: str
    type: str  # "hard" | "soft"
    parameters: dict


class UncertainParameter(BaseModel):
    """A parameter that must be predicted from data."""

    name: str
    description: str
    data_column: str


class ProblemConfig(BaseModel):
    """Complete specification of an optimization problem.

    Generated by the Intake Agent from user conversation.
    Consumed by Data Cleaning, Prediction, Optimizer, and Explanation modules.
    """

    problem_description: str
    decision_variables: list[DecisionVariable]
    objective: Objective
    constraints: list[Constraint]
    uncertain_parameters: list[UncertainParameter]
    cost_structure: dict
    data_requirements: dict
    solver_hint: str | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_problem_config.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/problem_config.py tests/test_problem_config.py
git commit -m "Add ProblemConfig Pydantic model with validation tests"
```

---

### Task 3: CSV Inspection Tools

**Files:**
- Create: `src/intake/tools.py`
- Create: `tests/test_intake_tools.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_intake_tools.py`:

```python
"""Tests for CSV inspection tools."""

from pathlib import Path

import pytest

from src.intake.tools import describe_column, list_unique_values, peek_columns, sample_rows

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_orders.csv"


class TestPeekColumns:
    """Test peek_columns tool."""

    def test_returns_column_names_and_dtypes(self) -> None:
        result = peek_columns(str(FIXTURE_PATH))
        assert "Order ID" in result
        assert "Lineitem name" in result
        assert "dtype" in result.lower() or "object" in result or "int" in result

    def test_nonexistent_file_returns_error(self) -> None:
        result = peek_columns("/nonexistent/file.csv")
        assert "error" in result.lower()


class TestSampleRows:
    """Test sample_rows tool."""

    def test_returns_rows(self) -> None:
        result = sample_rows(str(FIXTURE_PATH), n=3)
        assert "5001" in result  # first order ID
        assert "Game Jersey" in result

    def test_default_n_is_10(self) -> None:
        result = sample_rows(str(FIXTURE_PATH))
        assert "5007" in result  # 10th row order ID

    def test_nonexistent_file_returns_error(self) -> None:
        result = sample_rows("/nonexistent/file.csv")
        assert "error" in result.lower()


class TestDescribeColumn:
    """Test describe_column tool."""

    def test_numeric_column(self) -> None:
        result = describe_column(str(FIXTURE_PATH), "Lineitem price")
        assert "55" in result  # most common price
        assert "count" in result.lower()

    def test_string_column(self) -> None:
        result = describe_column(str(FIXTURE_PATH), "Financial Status")
        assert "PAID" in result

    def test_nonexistent_column_returns_error(self) -> None:
        result = describe_column(str(FIXTURE_PATH), "nonexistent_col")
        assert "error" in result.lower()


class TestListUniqueValues:
    """Test list_unique_values tool."""

    def test_returns_unique_values(self) -> None:
        result = list_unique_values(str(FIXTURE_PATH), "Financial Status")
        assert "PAID" in result
        assert "REFUNDED" in result

    def test_caps_at_50(self) -> None:
        result = list_unique_values(str(FIXTURE_PATH), "Order ID")
        # Only 7 unique order IDs in fixture, so all should show
        assert "5001" in result

    def test_nonexistent_column_returns_error(self) -> None:
        result = list_unique_values(str(FIXTURE_PATH), "nonexistent")
        assert "error" in result.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_intake_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.intake.tools'`

- [ ] **Step 3: Implement inspection tools**

Create `src/intake/tools.py`:

```python
"""Read-only CSV inspection tools for the Intake Agent.

These tools let the agent understand CSV structure and content without
loading the full dataset into its context window.
"""

import pandas as pd


def peek_columns(csv_path: str) -> str:
    """Return column names and dtypes from a CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Formatted string of column names and their dtypes.
    """
    try:
        df = pd.read_csv(csv_path, nrows=0, low_memory=False)
        dtypes = df.dtypes
        lines = [f"  {col}: {dtype}" for col, dtype in dtypes.items()]
        return f"Columns ({len(lines)}):\n" + "\n".join(lines)
    except Exception as exc:
        return f"Error reading {csv_path}: {exc}"


def sample_rows(csv_path: str, n: int = 10) -> str:
    """Return first N rows from a CSV as a formatted table.

    Args:
        csv_path: Path to the CSV file.
        n: Number of rows to return. Defaults to 10.

    Returns:
        Formatted table string of the first N rows.
    """
    try:
        df = pd.read_csv(csv_path, nrows=n, low_memory=False)
        return df.to_string(index=False)
    except Exception as exc:
        return f"Error reading {csv_path}: {exc}"


def describe_column(csv_path: str, column: str) -> str:
    """Return summary statistics for a single column.

    Args:
        csv_path: Path to the CSV file.
        column: Name of the column to describe.

    Returns:
        Formatted string with count, nulls, unique values, min/max.
    """
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        if column not in df.columns:
            return f"Error: column '{column}' not found. Available: {list(df.columns)}"

        series = df[column]
        parts = [
            f"Column: {column}",
            f"  dtype: {series.dtype}",
            f"  count: {series.count()}",
            f"  nulls: {series.isna().sum()}",
            f"  unique: {series.nunique()}",
        ]

        if pd.api.types.is_numeric_dtype(series):
            parts.append(f"  min: {series.min()}")
            parts.append(f"  max: {series.max()}")
            parts.append(f"  mean: {series.mean():.2f}")
        else:
            top_values = series.value_counts().head(5)
            parts.append("  top values:")
            for val, count in top_values.items():
                parts.append(f"    {val}: {count}")

        return "\n".join(parts)
    except Exception as exc:
        return f"Error: {exc}"


def list_unique_values(csv_path: str, column: str, max_values: int = 50) -> str:
    """Return unique values for a column, capped at max_values.

    Args:
        csv_path: Path to the CSV file.
        column: Name of the column.
        max_values: Maximum number of unique values to return. Defaults to 50.

    Returns:
        Formatted list of unique values.
    """
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        if column not in df.columns:
            return f"Error: column '{column}' not found. Available: {list(df.columns)}"

        unique_vals = df[column].dropna().unique()
        total = len(unique_vals)
        display = unique_vals[:max_values]

        lines = [f"Unique values in '{column}' ({total} total):"]
        for val in sorted(str(v) for v in display):
            lines.append(f"  {val}")

        if total > max_values:
            lines.append(f"  ... and {total - max_values} more")

        return "\n".join(lines)
    except Exception as exc:
        return f"Error: {exc}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_intake_tools.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/intake/tools.py tests/test_intake_tools.py
git commit -m "Add CSV inspection tools for intake agent"
```

---

### Task 4: Sandbox Execution with Verification

**Files:**
- Create: `src/data_ingestion/sandbox.py`
- Create: `tests/test_sandbox.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sandbox.py`:

```python
"""Tests for sandboxed code execution and verification."""

import csv
import tempfile
from pathlib import Path

import pytest

from src.data_ingestion.sandbox import SandboxResult, execute_code, verify_no_fabrication, verify_type_consistency


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a small CSV for testing."""
    csv_path = tmp_path / "input.csv"
    csv_path.write_text(
        "id,name,value,date\n"
        "1,Alice,10.5,2024-01-01\n"
        "2,Bob,20.0,2024-02-01\n"
        "3,Charlie,30.5,2024-03-01\n"
    )
    return csv_path


class TestExecuteCode:
    """Test sandboxed code execution."""

    def test_simple_print(self, tmp_path: Path) -> None:
        result = execute_code("print('hello')", tmp_path)
        assert result.success is True
        assert "hello" in result.stdout

    def test_pandas_code(self, sample_csv: Path, tmp_path: Path) -> None:
        code = f"""
import pandas as pd
df = pd.read_csv(r'{sample_csv}')
print(f"rows: {{len(df)}}")
"""
        result = execute_code(code, tmp_path)
        assert result.success is True
        assert "rows: 3" in result.stdout

    def test_timeout(self, tmp_path: Path) -> None:
        code = "import time; time.sleep(60)"
        result = execute_code(code, tmp_path, timeout=2)
        assert result.success is False
        assert "timeout" in result.error.lower()

    def test_syntax_error(self, tmp_path: Path) -> None:
        result = execute_code("def bad(:", tmp_path)
        assert result.success is False
        assert result.error != ""

    def test_can_write_output_file(self, sample_csv: Path, tmp_path: Path) -> None:
        output_path = tmp_path / "output.csv"
        code = f"""
import pandas as pd
df = pd.read_csv(r'{sample_csv}')
df.to_csv(r'{output_path}', index=False)
print('saved')
"""
        result = execute_code(code, tmp_path)
        assert result.success is True
        assert output_path.exists()


class TestVerifyTypeConsistency:
    """Test type consistency verification."""

    def test_consistent_types_pass(self, sample_csv: Path, tmp_path: Path) -> None:
        output_path = tmp_path / "output.csv"
        # Write output with same types
        output_path.write_text(
            "id,name,value,date\n"
            "1,Alice,10.5,2024-01-01\n"
            "2,Bob,20.0,2024-02-01\n"
        )
        errors = verify_type_consistency(sample_csv, output_path)
        assert errors == []

    def test_numeric_to_string_fails(self, sample_csv: Path, tmp_path: Path) -> None:
        output_path = tmp_path / "output.csv"
        output_path.write_text(
            "id,name,value,date\n"
            "1,Alice,not_a_number,2024-01-01\n"
        )
        errors = verify_type_consistency(sample_csv, output_path)
        assert len(errors) > 0
        assert any("value" in e for e in errors)


class TestVerifyNoFabrication:
    """Test no-fabrication verification."""

    def test_subset_passes(self, sample_csv: Path, tmp_path: Path) -> None:
        output_path = tmp_path / "output.csv"
        output_path.write_text(
            "id,name,value,date\n"
            "1,Alice,10.5,2024-01-01\n"
            "2,Bob,20.0,2024-02-01\n"
        )
        errors = verify_no_fabrication(sample_csv, output_path)
        assert errors == []

    def test_fabricated_value_detected(self, sample_csv: Path, tmp_path: Path) -> None:
        output_path = tmp_path / "output.csv"
        output_path.write_text(
            "id,name,value,date\n"
            "1,Alice,10.5,2024-01-01\n"
            "99,FakePerson,999.0,2024-12-31\n"
        )
        errors = verify_no_fabrication(sample_csv, output_path)
        assert len(errors) > 0

    def test_derived_columns_allowed(self, sample_csv: Path, tmp_path: Path) -> None:
        output_path = tmp_path / "output.csv"
        # New column 'value_doubled' is derived — should be allowed
        output_path.write_text(
            "id,name,value,date,value_doubled\n"
            "1,Alice,10.5,2024-01-01,21.0\n"
            "2,Bob,20.0,2024-02-01,40.0\n"
        )
        errors = verify_no_fabrication(sample_csv, output_path)
        assert errors == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sandbox.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data_ingestion.sandbox'`

- [ ] **Step 3: Implement sandbox module**

Create `src/data_ingestion/sandbox.py`:

```python
"""Sandboxed code execution with verification checks.

Executes agent-generated Python code in an isolated subprocess.
Verifies output data for type consistency and no fabrication.
"""

import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class SandboxResult:
    """Result of a sandboxed code execution."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    error: str = ""


def execute_code(code: str, work_dir: Path, timeout: int = 30) -> SandboxResult:
    """Execute Python code in a subprocess with timeout.

    Args:
        code: Python code string to execute.
        work_dir: Working directory for the subprocess.
        timeout: Maximum execution time in seconds.

    Returns:
        SandboxResult with success status, stdout, stderr, and error info.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    script_path = work_dir / "_sandbox_script.py"
    script_path.write_text(code, encoding="utf-8")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(work_dir),
        )
        return SandboxResult(
            success=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
            error=result.stderr if result.returncode != 0 else "",
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            success=False,
            error=f"Timeout: code exceeded {timeout} second limit",
        )
    except Exception as exc:
        return SandboxResult(success=False, error=str(exc))
    finally:
        script_path.unlink(missing_ok=True)


def verify_type_consistency(
    input_csv: Path, output_csv: Path
) -> list[str]:
    """Check that shared columns maintain their dtypes after cleaning.

    Args:
        input_csv: Path to the original CSV.
        output_csv: Path to the cleaned CSV.

    Returns:
        List of error messages. Empty list means all checks passed.
    """
    errors = []
    try:
        df_in = pd.read_csv(input_csv, low_memory=False)
        df_out = pd.read_csv(output_csv, low_memory=False)
    except Exception as exc:
        return [f"Failed to read CSVs: {exc}"]

    shared_cols = set(df_in.columns) & set(df_out.columns)

    for col in shared_cols:
        in_numeric = pd.api.types.is_numeric_dtype(df_in[col])
        out_numeric = pd.api.types.is_numeric_dtype(df_out[col])

        if in_numeric and not out_numeric:
            errors.append(
                f"Column '{col}' was numeric in input but is not numeric in output"
            )

    return errors


def verify_no_fabrication(
    input_csv: Path, output_csv: Path
) -> list[str]:
    """Check that raw columns contain only values from the original data.

    Columns that exist in both input and output are checked. New columns
    (derived/computed) are allowed and not checked.

    Args:
        input_csv: Path to the original CSV.
        output_csv: Path to the cleaned CSV.

    Returns:
        List of error messages. Empty list means no fabrication detected.
    """
    errors = []
    try:
        df_in = pd.read_csv(input_csv, low_memory=False)
        df_out = pd.read_csv(output_csv, low_memory=False)
    except Exception as exc:
        return [f"Failed to read CSVs: {exc}"]

    shared_cols = set(df_in.columns) & set(df_out.columns)

    for col in shared_cols:
        in_values = set(df_in[col].dropna().astype(str))
        out_values = set(df_out[col].dropna().astype(str))

        fabricated = out_values - in_values
        if fabricated:
            sample = list(fabricated)[:5]
            errors.append(
                f"Column '{col}' contains values not in original data: {sample}"
            )

    return errors
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sandbox.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_ingestion/sandbox.py tests/test_sandbox.py
git commit -m "Add sandboxed code execution with type and fabrication verification"
```

---

### Task 5: Data Ingestion Tools (CFA Wrapper)

**Files:**
- Create: `src/data_ingestion/tools.py`
- Create: `tests/test_data_ingestion_tools.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data_ingestion_tools.py`:

```python
"""Tests for data ingestion tools."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.data_ingestion.tools import run_cfa_cleaning


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_orders.csv"


class TestRunCfaCleaning:
    """Test CFA cleaning wrapper tool."""

    def test_returns_cleaned_data_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            result = run_cfa_cleaning(str(FIXTURE_PATH), str(output_dir))
            assert "cleaned" in result.lower() or "saved" in result.lower() or "output" in result.lower()

    def test_output_file_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_cfa_cleaning(str(FIXTURE_PATH), str(output_dir))
            output_files = list(output_dir.glob("*.csv"))
            assert len(output_files) >= 1

    def test_invalid_csv_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cfa_cleaning("/nonexistent/file.csv", tmp)
            assert "error" in result.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data_ingestion_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data_ingestion.tools'`

- [ ] **Step 3: Implement data ingestion tools**

Create `src/data_ingestion/tools.py`:

```python
"""Tools for the Data Cleaning Agent.

Includes a wrapper around the existing CFA cleaning pipeline and
data inspection tools shared with the intake agent.
"""

from pathlib import Path

import pandas as pd

from src.intake.tools import describe_column, list_unique_values, peek_columns, sample_rows


def run_cfa_cleaning(csv_path: str, output_dir: str) -> str:
    """Run the CFA-specific cleaning pipeline on a Squarespace export.

    Wraps scripts/clean_data.py functions. Use this for known CFA uniform
    order data. For other datasets, use execute_code instead.

    Args:
        csv_path: Path to the raw Squarespace CSV export.
        output_dir: Directory to save the cleaned CSV.

    Returns:
        Status message with output file path or error description.
    """
    try:
        import sys
        project_root = Path(__file__).resolve().parent.parent.parent
        scripts_dir = str(project_root / "scripts")

        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from clean_data import clean_data, RAW_PATH

        # Temporarily override the path used by clean_data
        import clean_data as cd_module
        original_raw = cd_module.RAW_PATH
        cd_module.RAW_PATH = Path(csv_path)

        try:
            cleaned_df = cd_module.clean_data()
        finally:
            cd_module.RAW_PATH = original_raw

        output_path = Path(output_dir) / "cfa_cleaned.csv"
        cleaned_df.to_csv(output_path, index=False)

        return (
            f"CFA cleaning complete. Output: {output_path}\n"
            f"Rows: {len(cleaned_df)}, Columns: {list(cleaned_df.columns)}"
        )
    except Exception as exc:
        return f"Error running CFA cleaning: {exc}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data_ingestion_tools.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_ingestion/tools.py tests/test_data_ingestion_tools.py
git commit -m "Add CFA cleaning wrapper tool for data ingestion agent"
```

---

### Task 6: Intake Agent Prompts and Agent

**Files:**
- Create: `src/intake/prompts.py`
- Create: `src/intake/agent.py`

- [ ] **Step 1: Create intake system prompt**

Create `src/intake/prompts.py`:

```python
"""System prompt for the Intake Agent."""

INTAKE_SYSTEM_PROMPT = """\
You are an optimization problem formulator. Your job is to understand the user's \
business problem and produce a structured ProblemConfig JSON that fully specifies \
an optimization problem.

## Your Process

1. Read the user's initial problem description.
2. Use the data inspection tools to examine the uploaded CSV — look at column names, \
   sample rows, and value distributions to ground your questions in the actual data.
3. Identify which ProblemConfig fields you can fill from the description vs. which \
   are missing or ambiguous.
4. Ask ONE follow-up question at a time for missing fields. Prefer multiple-choice \
   when possible. Reference actual data columns/values in your questions.
5. Once you have enough information, produce the complete ProblemConfig JSON.
6. Present it to the user and ask if they want to modify anything.

## ProblemConfig Schema

You must produce a JSON object with these fields:

- problem_description (str): The original problem in plain English
- decision_variables (list): What the optimizer controls
  - Each: {name, type ("continuous"|"integer"|"binary"), bounds (optional [min, max])}
- objective: {direction ("minimize"|"maximize"), description}
- constraints (list): Limitations on the solution
  - Each: {name, description, type ("hard"|"soft"), parameters (dict)}
- uncertain_parameters (list): What needs prediction from data
  - Each: {name, description, data_column}
- cost_structure (dict): Overage costs, underage costs, unit costs, etc.
- data_requirements (dict): What columns/features the data must have
- solver_hint (str, optional): Problem class if known (e.g., "newsvendor", "LP")

## Tools Available

- peek_columns: See column names and types in the CSV
- sample_rows: See first N rows of the CSV
- describe_column: Get stats for a specific column
- list_unique_values: See unique values in a column

## Rules

- Always inspect the data before asking questions about it.
- Never guess column names — use peek_columns first.
- Ask at most 6-8 follow-up questions total.
- When you produce the final ProblemConfig, output it as a JSON code block \
  wrapped in ```json ... ``` markers.
- If the user says the config looks good, output it one final time as clean JSON.
"""
```

- [ ] **Step 2: Create intake agent module**

Create `src/intake/agent.py`:

```python
"""Intake Agent — converses with user to produce a ProblemConfig.

Uses the Claude Agent SDK to run a conversation with data inspection tools.
"""

import json
import re

import anyio
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    AssistantMessage,
    ResultMessage,
    TextBlock,
)

from src.intake.prompts import INTAKE_SYSTEM_PROMPT
from src.intake.tools import describe_column, list_unique_values, peek_columns, sample_rows
from src.models.problem_config import ProblemConfig


try:
    from claude_agent_sdk import tool, create_sdk_mcp_server
except ImportError:
    tool = None
    create_sdk_mcp_server = None


def _build_mcp_server():
    """Build an MCP server with data inspection tools."""
    peek_tool = tool(
        "peek_columns",
        "Return column names and dtypes from a CSV file",
        {"csv_path": str},
    )(lambda args: {"content": [{"type": "text", "text": peek_columns(args["csv_path"])}]})

    sample_tool = tool(
        "sample_rows",
        "Return first N rows from a CSV as a formatted table",
        {"csv_path": str, "n": int},
    )(lambda args: {"content": [{"type": "text", "text": sample_rows(args["csv_path"], args.get("n", 10))}]})

    describe_tool = tool(
        "describe_column",
        "Return summary statistics for a single CSV column",
        {"csv_path": str, "column": str},
    )(lambda args: {"content": [{"type": "text", "text": describe_column(args["csv_path"], args["column"])}]})

    unique_tool = tool(
        "list_unique_values",
        "Return unique values for a CSV column (capped at 50)",
        {"csv_path": str, "column": str},
    )(lambda args: {"content": [{"type": "text", "text": list_unique_values(args["csv_path"], args["column"])}]})

    return create_sdk_mcp_server(
        "intake-tools",
        tools=[peek_tool, sample_tool, describe_tool, unique_tool],
    )


def _extract_json_from_text(text: str) -> str | None:
    """Extract JSON from a text response, looking for ```json blocks."""
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return None


async def run_intake_agent(
    csv_path: str,
    initial_description: str = "",
) -> tuple[ProblemConfig | None, list[dict]]:
    """Run the intake agent conversation and return a ProblemConfig.

    Args:
        csv_path: Path to the CSV file for inspection.
        initial_description: Optional initial problem description from user.

    Returns:
        Tuple of (ProblemConfig or None if failed, list of usage dicts).
    """
    mcp_server = _build_mcp_server()
    usage_log = []

    prompt = f"CSV file is at: {csv_path}\n\n"
    if initial_description:
        prompt += f"Problem description: {initial_description}"
    else:
        prompt += (
            "Please help me formulate an optimization problem. "
            "Start by inspecting the data file, then ask me questions."
        )

    options = ClaudeAgentOptions(
        system_prompt=INTAKE_SYSTEM_PROMPT,
        mcp_servers={"intake": mcp_server},
        max_turns=30,
    )

    config = None

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                if message.usage:
                    usage_log.append(message.usage)

                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"\nAgent: {block.text}")

                        json_str = _extract_json_from_text(block.text)
                        if json_str:
                            try:
                                config = ProblemConfig.model_validate_json(json_str)
                            except Exception:
                                pass

            elif isinstance(message, ResultMessage):
                if message.result:
                    json_str = _extract_json_from_text(message.result)
                    if json_str:
                        try:
                            config = ProblemConfig.model_validate_json(json_str)
                        except Exception:
                            pass

        # Interactive loop: user reviews config
        while config is None:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            await client.query(user_input)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    if message.usage:
                        usage_log.append(message.usage)
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(f"\nAgent: {block.text}")
                            json_str = _extract_json_from_text(block.text)
                            if json_str:
                                try:
                                    config = ProblemConfig.model_validate_json(json_str)
                                except Exception:
                                    pass

    return config, usage_log
```

- [ ] **Step 3: Commit**

```bash
git add src/intake/prompts.py src/intake/agent.py
git commit -m "Add intake agent with system prompt and conversation loop"
```

---

### Task 7: Data Cleaning Agent Prompts and Agent

**Files:**
- Create: `src/data_ingestion/prompts.py`
- Create: `src/data_ingestion/agent.py`

- [ ] **Step 1: Create data cleaning system prompt**

Create `src/data_ingestion/prompts.py`:

```python
"""System prompt for the Data Cleaning Agent."""

DATA_CLEANING_SYSTEM_PROMPT = """\
You are a data cleaning specialist. Your job is to take a raw CSV file and \
clean it according to a ProblemConfig specification, producing a dataset ready \
for machine learning prediction.

## Your Process

1. Inspect the raw data using peek_columns, sample_rows, and describe_column.
2. Review the ProblemConfig's data_requirements to understand what the cleaned \
   data must contain.
3. Plan your cleaning steps.
4. Execute cleaning code step by step using execute_code.
5. After each step, inspect the results to verify correctness.
6. Save the final cleaned CSV to the specified output path.

## Tools Available

- peek_columns: See column names and types
- sample_rows: See first N rows
- describe_column: Get stats for a specific column
- run_cfa_cleaning: One-shot CFA uniform data cleaner (use for CFA Squarespace data)
- execute_code: Run Python/pandas code in sandbox

## Rules for execute_code

- Always import pandas at the top of each code block.
- Read input from the provided csv_path.
- Write output to the provided output_path.
- Print summary information (row counts, column lists) after each step.
- Never fabricate data — only transform, filter, aggregate, or derive from existing values.
- If a cleaning step drops more than 50% of rows, print a warning and explain why.

## When to Use run_cfa_cleaning

If the data looks like a Squarespace order export with columns like "Lineitem name", \
"Lineitem variant", "Financial Status" — this is CFA uniform data. Call run_cfa_cleaning \
first for the base cleaning, then do additional transformations on top.

## Output Requirements

Your final output must be a CSV with:
- All columns specified in data_requirements.required_columns
- No duplicate rows (unless duplicates are meaningful)
- Consistent dtypes (numbers as numbers, dates as dates)
- A printed summary of the final dataset (row count, columns, null counts)
"""
```

- [ ] **Step 2: Create data cleaning agent module**

Create `src/data_ingestion/agent.py`:

```python
"""Data Cleaning Agent — cleans CSV data according to ProblemConfig.

Uses the Claude Agent SDK with sandboxed code execution.
"""

import json
from pathlib import Path

import anyio
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    AssistantMessage,
    ResultMessage,
    TextBlock,
)

from src.data_ingestion.prompts import DATA_CLEANING_SYSTEM_PROMPT
from src.data_ingestion.sandbox import execute_code, verify_no_fabrication, verify_type_consistency
from src.data_ingestion.tools import run_cfa_cleaning
from src.intake.tools import describe_column, peek_columns, sample_rows
from src.models.problem_config import ProblemConfig


try:
    from claude_agent_sdk import tool, create_sdk_mcp_server
except ImportError:
    tool = None
    create_sdk_mcp_server = None


def _build_mcp_server(csv_path: str, output_dir: str):
    """Build an MCP server with data cleaning tools."""
    peek_tool = tool(
        "peek_columns",
        "Return column names and dtypes from a CSV file",
        {"csv_path": str},
    )(lambda args: {"content": [{"type": "text", "text": peek_columns(args["csv_path"])}]})

    sample_tool = tool(
        "sample_rows",
        "Return first N rows from a CSV as a formatted table",
        {"csv_path": str, "n": int},
    )(lambda args: {"content": [{"type": "text", "text": sample_rows(args["csv_path"], args.get("n", 10))}]})

    describe_tool = tool(
        "describe_column",
        "Return summary statistics for a single CSV column",
        {"csv_path": str, "column": str},
    )(lambda args: {"content": [{"type": "text", "text": describe_column(args["csv_path"], args["column"])}]})

    cfa_tool = tool(
        "run_cfa_cleaning",
        "Run the CFA-specific cleaning pipeline on Squarespace order data",
        {"csv_path": str, "output_dir": str},
    )(lambda args: {"content": [{"type": "text", "text": run_cfa_cleaning(args["csv_path"], args["output_dir"])}]})

    def _execute_code_handler(args):
        work_dir = Path(output_dir)
        result = execute_code(args["code"], work_dir, timeout=args.get("timeout", 30))
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

    exec_tool = tool(
        "execute_code",
        "Execute Python code in a sandbox. Use for data cleaning with pandas.",
        {"code": str, "timeout": int},
    )(_execute_code_handler)

    return create_sdk_mcp_server(
        "data-cleaning-tools",
        tools=[peek_tool, sample_tool, describe_tool, cfa_tool, exec_tool],
    )


def _build_data_manifest(csv_path: Path) -> dict:
    """Build a manifest describing the cleaned dataset."""
    import pandas as pd

    df = pd.read_csv(csv_path, low_memory=False)
    return {
        "row_count": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": {col: int(df[col].isna().sum()) for col in df.columns},
    }


async def run_cleaning_agent(
    csv_path: str,
    config: ProblemConfig,
    output_dir: str,
) -> tuple[str | None, dict | None, list[dict]]:
    """Run the data cleaning agent.

    Args:
        csv_path: Path to the raw CSV file.
        config: ProblemConfig specifying data requirements.
        output_dir: Directory to write cleaned data and manifest.

    Returns:
        Tuple of (cleaned CSV path or None, data manifest or None, usage log).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    usage_log = []

    mcp_server = _build_mcp_server(csv_path, output_dir)

    prompt = (
        f"Clean the data at: {csv_path}\n\n"
        f"ProblemConfig:\n{config.model_dump_json(indent=2)}\n\n"
        f"Save cleaned output to: {output_path / 'cleaned_data.csv'}\n\n"
        f"Available output directory: {output_dir}"
    )

    options = ClaudeAgentOptions(
        system_prompt=DATA_CLEANING_SYSTEM_PROMPT,
        mcp_servers={"cleaning": mcp_server},
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
                        print(f"\nCleaning Agent: {block.text}")

    # Find the cleaned output
    cleaned_csv = output_path / "cleaned_data.csv"
    if not cleaned_csv.exists():
        # Check for CFA cleaner output
        cfa_cleaned = output_path / "cfa_cleaned.csv"
        if cfa_cleaned.exists():
            cleaned_csv = cfa_cleaned
        else:
            # Look for any CSV in output dir
            csvs = list(output_path.glob("*.csv"))
            if csvs:
                cleaned_csv = csvs[0]
            else:
                return None, None, usage_log

    # Run verification
    input_path = Path(csv_path)
    type_errors = verify_type_consistency(input_path, cleaned_csv)
    fab_errors = verify_no_fabrication(input_path, cleaned_csv)

    if type_errors:
        print(f"\nWarning — type consistency issues: {type_errors}")
    if fab_errors:
        print(f"\nWarning — possible data fabrication: {fab_errors}")

    manifest = _build_data_manifest(cleaned_csv)

    manifest_path = output_path / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return str(cleaned_csv), manifest, usage_log
```

- [ ] **Step 3: Commit**

```bash
git add src/data_ingestion/prompts.py src/data_ingestion/agent.py
git commit -m "Add data cleaning agent with system prompt and sandboxed execution"
```

---

### Task 8: Orchestrator

**Files:**
- Create: `src/orchestrator.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_orchestrator.py`:

```python
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
            assert "2026" in name or "20" in name  # contains year


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.orchestrator'`

- [ ] **Step 3: Implement orchestrator**

Create `src/orchestrator.py`:

```python
"""Orchestrator — wires Intake Agent and Data Cleaning Agent together.

Manages run directories, trace logging, and the handoff between agents.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import anyio

from src.intake.agent import run_intake_agent
from src.data_ingestion.agent import run_cleaning_agent
from src.models.problem_config import ProblemConfig


def create_run_directory(base_dir: str = "runs") -> str:
    """Create a timestamped run directory.

    Args:
        base_dir: Parent directory for runs. Defaults to "runs".

    Returns:
        Path to the created run directory.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def save_trace(run_dir: str, trace_data: dict) -> None:
    """Save trace data to the run directory.

    Args:
        run_dir: Path to the run directory.
        trace_data: Dictionary of trace information to save.
    """
    trace_path = Path(run_dir) / "trace.json"
    trace_path.write_text(json.dumps(trace_data, indent=2, default=str))


class Orchestrator:
    """Orchestrates the intake and data cleaning pipeline.

    Manages the flow: user input -> intake agent -> ProblemConfig ->
    data cleaning agent -> cleaned data + manifest.
    """

    def __init__(self, csv_path: str, base_dir: str = "runs") -> None:
        """Initialize the orchestrator.

        Args:
            csv_path: Path to the raw CSV file.
            base_dir: Parent directory for run outputs.
        """
        self.csv_path = csv_path
        self.run_dir = create_run_directory(base_dir)
        self.trace: dict = {"steps": [], "run_dir": self.run_dir}

    async def run_intake(
        self, description: str = ""
    ) -> ProblemConfig | None:
        """Run the intake agent to produce a ProblemConfig.

        Args:
            description: Optional initial problem description.

        Returns:
            ProblemConfig if successful, None otherwise.
        """
        print(f"\n{'='*60}")
        print("INTAKE AGENT")
        print(f"{'='*60}")

        start = time.time()
        config, usage_log = await run_intake_agent(self.csv_path, description)
        duration = time.time() - start

        self.trace["steps"].append({
            "agent": "intake",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(duration, 2),
            "usage": usage_log,
            "success": config is not None,
        })

        if config:
            config_path = Path(self.run_dir) / "problem_config.json"
            config_path.write_text(config.model_dump_json(indent=2))
            print(f"\nProblemConfig saved to {config_path}")

        return config

    async def run_cleaning(
        self, config: ProblemConfig
    ) -> tuple[str | None, dict | None]:
        """Run the data cleaning agent.

        Args:
            config: ProblemConfig specifying data requirements.

        Returns:
            Tuple of (cleaned CSV path, data manifest) or (None, None).
        """
        print(f"\n{'='*60}")
        print("DATA CLEANING AGENT")
        print(f"{'='*60}")

        start = time.time()
        cleaned_path, manifest, usage_log = await run_cleaning_agent(
            self.csv_path, config, self.run_dir
        )
        duration = time.time() - start

        self.trace["steps"].append({
            "agent": "data_cleaning",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(duration, 2),
            "usage": usage_log,
            "success": cleaned_path is not None,
        })

        return cleaned_path, manifest

    async def run_full_pipeline(self, description: str = "") -> None:
        """Run the complete intake + data cleaning pipeline.

        Args:
            description: Optional initial problem description.
        """
        config = await self.run_intake(description)
        if config is None:
            print("\nFailed to produce a valid ProblemConfig. Aborting.")
            save_trace(self.run_dir, self.trace)
            return

        cleaned_path, manifest = await self.run_cleaning(config)

        save_trace(self.run_dir, self.trace)

        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"Run directory: {self.run_dir}")

        if cleaned_path:
            print(f"Cleaned data:  {cleaned_path}")
            print(f"Manifest:      {Path(self.run_dir) / 'data_manifest.json'}")
        else:
            print("Warning: Data cleaning did not produce output.")

        print(f"Trace log:     {Path(self.run_dir) / 'trace.json'}")

    async def run_cleaning_only(self, config_path: str) -> None:
        """Re-run just the data cleaning step with an existing config.

        Args:
            config_path: Path to a ProblemConfig JSON file.
        """
        config = ProblemConfig.model_validate_json(
            Path(config_path).read_text()
        )
        print(f"Loaded config from {config_path}")

        cleaned_path, manifest = await self.run_cleaning(config)

        save_trace(self.run_dir, self.trace)

        print(f"\n{'='*60}")
        print("DATA CLEANING COMPLETE")
        print(f"{'='*60}")
        print(f"Run directory: {self.run_dir}")

        if cleaned_path:
            print(f"Cleaned data:  {cleaned_path}")
        else:
            print("Warning: Data cleaning did not produce output.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator.py tests/test_orchestrator.py
git commit -m "Add orchestrator with trace logging and re-run support"
```

---

### Task 9: CLI Entry Point

**Files:**
- Create: `src/cli.py`
- Create: `src/__main__.py`

- [ ] **Step 1: Create CLI module**

Create `src/cli.py`:

```python
"""CLI entry point for the CFA Optimization Agent."""

import argparse
import sys
from pathlib import Path

import anyio

from src.orchestrator import Orchestrator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. Defaults to sys.argv[1:].

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="AI-powered optimization agent for business decision-making",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Interactive intake — agent asks questions
  python -m src data/orders.csv

  # Front-load a problem description
  python -m src data/orders.csv --description "Optimize uniform ordering..."

  # Re-run data cleaning with a modified config
  python -m src data/orders.csv --rerun-from data --config runs/20260603/problem_config.json
""",
    )

    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the raw CSV data file",
    )
    parser.add_argument(
        "--description", "-d",
        type=str,
        default="",
        help="Initial problem description (optional — agent will ask questions)",
    )
    parser.add_argument(
        "--rerun-from",
        type=str,
        choices=["data"],
        default=None,
        help="Skip intake and re-run from a specific stage",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to existing ProblemConfig JSON (required with --rerun-from)",
    )

    args = parser.parse_args(argv)

    # Validate CSV path
    if not Path(args.csv_path).exists():
        parser.error(f"CSV file not found: {args.csv_path}")

    # Validate rerun args
    if args.rerun_from and not args.config:
        parser.error("--config is required when using --rerun-from")
    if args.config and not Path(args.config).exists():
        parser.error(f"Config file not found: {args.config}")

    return args


async def async_main(args: argparse.Namespace) -> None:
    """Run the pipeline asynchronously.

    Args:
        args: Parsed command-line arguments.
    """
    orchestrator = Orchestrator(args.csv_path)

    if args.rerun_from == "data":
        await orchestrator.run_cleaning_only(args.config)
    else:
        await orchestrator.run_full_pipeline(args.description)


def main() -> None:
    """Entry point for the CLI."""
    args = parse_args()
    anyio.run(async_main, args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create __main__.py**

Create `src/__main__.py`:

```python
"""Allow running as python -m src."""

from src.cli import main

main()
```

- [ ] **Step 3: Verify CLI help works**

Run: `uv run python -m src --help`
Expected: Help text displays with usage, examples, and argument descriptions

- [ ] **Step 4: Commit**

```bash
git add src/cli.py src/__main__.py
git commit -m "Add CLI entry point with argparse and re-run support"
```

---

### Task 10: Integration Smoke Test

**Files:**
- No new files — verify everything wires together

- [ ] **Step 1: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All tests pass (approximately 21 tests across 5 test files)

- [ ] **Step 2: Verify CLI starts without errors**

Run: `uv run python -m src data/orders.csv --help`
Expected: Help text displays correctly

- [ ] **Step 3: Verify imports**

Run:
```bash
uv run python -c "
from src.models.problem_config import ProblemConfig
from src.intake.tools import peek_columns
from src.data_ingestion.sandbox import execute_code
from src.orchestrator import Orchestrator
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "Complete intake and data ingestion module implementation"
```

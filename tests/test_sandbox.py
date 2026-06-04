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
        output_path.write_text(
            "id,name,value,date,value_doubled\n"
            "1,Alice,10.5,2024-01-01,21.0\n"
            "2,Bob,20.0,2024-02-01,40.0\n"
        )
        errors = verify_no_fabrication(sample_csv, output_path)
        assert errors == []

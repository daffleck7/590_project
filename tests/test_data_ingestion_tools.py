"""Tests for data ingestion tools."""

import tempfile
from pathlib import Path

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

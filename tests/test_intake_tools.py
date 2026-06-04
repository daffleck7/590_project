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
        assert "5001" in result
        assert "Game Jersey" in result

    def test_default_n_is_10(self) -> None:
        result = sample_rows(str(FIXTURE_PATH))
        assert "5007" in result

    def test_nonexistent_file_returns_error(self) -> None:
        result = sample_rows("/nonexistent/file.csv")
        assert "error" in result.lower()


class TestDescribeColumn:
    """Test describe_column tool."""

    def test_numeric_column(self) -> None:
        result = describe_column(str(FIXTURE_PATH), "Lineitem price")
        assert "55" in result
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
        assert "5001" in result

    def test_nonexistent_column_returns_error(self) -> None:
        result = list_unique_values(str(FIXTURE_PATH), "nonexistent")
        assert "error" in result.lower()

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

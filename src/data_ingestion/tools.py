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

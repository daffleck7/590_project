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

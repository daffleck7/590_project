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
from src.explanation.agent import run_explanation_agent, save_explanation
from src.models.optimization_result import OptimizationResult
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

    async def run_explanation(
        self,
        result: OptimizationResult,
        cleaned_csv_path: str,
        config: ProblemConfig,
        train_years: list[int] | None = None,
        test_years: list[int] | None = None,
    ) -> str | None:
        """Run the explanation agent to produce a plain-English report.

        Args:
            result: OptimizationResult from the optimizer module.
            cleaned_csv_path: Path to the aggregated cleaned CSV.
            config: ProblemConfig used throughout the pipeline.
            train_years: Training years for baseline. Defaults to 2020-2024.
            test_years: Test years to evaluate on. Defaults to 2025-2026.

        Returns:
            Path to the saved report.md, or None on failure.
        """
        print(f"\n{'='*60}")
        print("EXPLANATION AGENT")
        print(f"{'='*60}")

        start = time.time()
        report_text, baseline_data, sensitivity_data = await run_explanation_agent(
            result=result,
            cleaned_csv_path=cleaned_csv_path,
            config=config,
            train_years=train_years,
            test_years=test_years,
        )
        duration = time.time() - start

        self.trace["steps"].append({
            "agent": "explanation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(duration, 2),
            "success": bool(report_text),
        })

        if report_text:
            save_explanation(self.run_dir, report_text, baseline_data, sensitivity_data)
            report_path = str(Path(self.run_dir) / "report.md")
            print(f"\nReport saved to {report_path}")
            return report_path

        print("\nWarning: Explanation agent did not produce a report.")
        return None

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

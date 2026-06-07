"""Orchestrator — wires Intake Agent and Data Cleaning Agent together.

Manages run directories, trace logging, and the handoff between agents.
"""

import asyncio
import json
import time
from collections.abc import Callable, Awaitable
from datetime import datetime, timezone
from pathlib import Path

import anyio

from src.intake.agent import run_intake_agent
from src.data_ingestion.agent import run_cleaning_agent
from src.modeling.agent import run_modeling_agent
from src.explanation.agent import run_explanation_agent
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

    def __init__(
        self,
        csv_path: str,
        base_dir: str = "runs",
        callback: Callable[[dict], Awaitable[None]] | None = None,
        interactive: bool = True,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            csv_path: Path to the raw CSV file.
            base_dir: Parent directory for run outputs.
            callback: Optional async callback for event routing.
            interactive: If True, allow stdin prompts (CLI mode).
        """
        self.csv_path = csv_path
        self.run_dir = create_run_directory(base_dir)
        self.trace: dict = {"steps": [], "run_dir": self.run_dir}
        self.callback = callback
        self.interactive = interactive
        self.user_input_queue: asyncio.Queue[str] = asyncio.Queue()
        self.activity_log: list[dict] = []

    async def _log_event(self, event: dict) -> None:
        """Log an event and forward to callback."""
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.activity_log.append(event)
        if self.callback:
            await self.callback(event)

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
        input_queue = None if self.interactive else self.user_input_queue
        config, usage_log = await run_intake_agent(
            self.csv_path, self.run_dir, description, callback=self._log_event,
            user_input_queue=input_queue,
        )
        duration = time.time() - start

        self.trace["steps"].append({
            "agent": "intake",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(duration, 2),
            "usage": usage_log,
            "success": config is not None,
        })

        if config:
            print(f"\nProblemConfig saved to {Path(self.run_dir) / 'problem_config.json'}")

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

        await self._log_event({"type": "cleaning_start"})

        start = time.time()
        cleaned_path, manifest, usage_log = await run_cleaning_agent(
            self.csv_path, config, self.run_dir, callback=self._log_event
        )
        duration = time.time() - start

        self.trace["steps"].append({
            "agent": "data_cleaning",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(duration, 2),
            "usage": usage_log,
            "success": cleaned_path is not None,
        })

        await self._log_event({
            "type": "cleaning_done",
            "cleaned_path": cleaned_path,
            "manifest": manifest,
        })

        return cleaned_path, manifest

    async def run_modeling(
        self, config: ProblemConfig, cleaned_csv_path: str
    ) -> bool:
        """Run the modeling agent (prediction + optimization).

        Args:
            config: ProblemConfig for cost structure and constraints.
            cleaned_csv_path: Path to the cleaned CSV.

        Returns:
            True if successful, False otherwise.
        """
        print(f"\n{'='*60}")
        print("MODELING AGENT")
        print(f"{'='*60}")

        await self._log_event({"type": "modeling_start"})

        start = time.time()
        success, usage_log = await run_modeling_agent(
            config, cleaned_csv_path, self.run_dir, callback=self._log_event
        )
        duration = time.time() - start

        self.trace["steps"].append({
            "agent": "modeling",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(duration, 2),
            "usage": usage_log,
            "success": success,
        })

        await self._log_event({"type": "modeling_done"})

        # Read and emit the summary
        summary_path = Path(self.run_dir) / "modeling_summary.txt"
        if summary_path.exists():
            await self._log_event({
                "type": "stage_summary",
                "stage": "modeling",
                "summary": summary_path.read_text(encoding="utf-8", errors="replace"),
            })

        return success

    async def run_explanation(self) -> bool:
        """Run the explanation agent to produce the final report.

        Returns:
            True if report was generated, False otherwise.
        """
        print(f"\n{'='*60}")
        print("EXPLANATION AGENT")
        print(f"{'='*60}")

        await self._log_event({"type": "explanation_start"})

        start = time.time()
        success, usage_log = await run_explanation_agent(
            self.run_dir, callback=self._log_event
        )
        duration = time.time() - start

        self.trace["steps"].append({
            "agent": "explanation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(duration, 2),
            "usage": usage_log,
            "success": success,
        })

        await self._log_event({"type": "explanation_done"})

        # Read and emit the report
        report_path = Path(self.run_dir) / "final_report.md"
        if report_path.exists():
            await self._log_event({
                "type": "stage_summary",
                "stage": "explanation",
                "summary": report_path.read_text(encoding="utf-8", errors="replace"),
            })

        return success

    async def run_full_pipeline(self, description: str = "") -> None:
        """Run the complete intake + data cleaning pipeline.

        Args:
            description: Optional initial problem description.
        """
        config = await self.run_intake(description)
        if config is None:
            await self._log_event({"type": "error", "message": "Failed to produce ProblemConfig"})
            print("\nFailed to produce a valid ProblemConfig. Aborting.")
            self.trace["activity_log"] = self.activity_log
            save_trace(self.run_dir, self.trace)
            return

        cleaned_path, manifest = await self.run_cleaning(config)

        self.trace["activity_log"] = self.activity_log
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

        self.trace["activity_log"] = self.activity_log
        save_trace(self.run_dir, self.trace)

        print(f"\n{'='*60}")
        print("DATA CLEANING COMPLETE")
        print(f"{'='*60}")
        print(f"Run directory: {self.run_dir}")

        if cleaned_path:
            print(f"Cleaned data:  {cleaned_path}")
        else:
            print("Warning: Data cleaning did not produce output.")

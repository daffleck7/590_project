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

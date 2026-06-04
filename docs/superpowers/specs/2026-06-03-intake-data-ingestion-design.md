# Intake + Data Ingestion Module Design

## Overview

Two-agent system for the first stage of the optimization pipeline. The **Intake Agent**
converses with the user in natural language to understand the optimization problem and
produces a structured `ProblemConfig`. The **Data Cleaning Agent** receives that config
and transforms raw CSV data into a clean dataset ready for the prediction module.

Both agents are powered by the Claude Agent SDK, which runs on the user's Claude
subscription (no separate API spend).

## Architecture

```
User (CLI)
    |
    v
Orchestrator (src/orchestrator.py)
    |
    +---> Intake Agent (src/intake/agent.py)
    |       - Converses with user (hybrid: NL description + follow-up questions)
    |       - Inspects uploaded CSV via read-only tools
    |       - Outputs: ProblemConfig JSON
    |
    +---> Data Cleaning Agent (src/data_ingestion/agent.py)
            - Receives ProblemConfig + raw CSV path
            - Generates + executes pandas code in sandbox
            - Can call CFA-specific cleaner (scripts/clean_data.py) as a tool
            - Outputs: cleaned CSV + data manifest JSON
```

## Project Structure

```
src/
    intake/
        __init__.py
        agent.py          # Intake agent conversation loop
        prompts.py         # System prompt for intake agent
        tools.py           # Data inspection tools (peek, describe, sample)
    data_ingestion/
        __init__.py
        agent.py           # Data cleaning agent
        prompts.py         # System prompt for cleaning agent
        tools.py           # CFA cleaner wrapper + sandbox execution
        sandbox.py         # Sandboxed code execution with verification
    models/
        __init__.py
        problem_config.py  # ProblemConfig Pydantic model
    orchestrator.py         # Wires intake -> data ingestion, manages runs
    cli.py                  # Entry point: python -m src.cli
```

The existing `scripts/clean_data.py` is preserved and wrapped as a callable tool.

## ProblemConfig Schema

Central data contract consumed by all downstream modules.

```python
from pydantic import BaseModel

class DecisionVariable(BaseModel):
    name: str
    type: str  # "continuous" | "integer" | "binary"
    bounds: tuple[float, float] | None = None

class Objective(BaseModel):
    direction: str  # "minimize" | "maximize"
    description: str

class Constraint(BaseModel):
    name: str
    description: str
    type: str  # "hard" | "soft"
    parameters: dict

class UncertainParameter(BaseModel):
    name: str
    description: str
    data_column: str

class ProblemConfig(BaseModel):
    problem_description: str
    decision_variables: list[DecisionVariable]
    objective: Objective
    constraints: list[Constraint]
    uncertain_parameters: list[UncertainParameter]
    cost_structure: dict
    data_requirements: dict
    solver_hint: str | None = None
```

Serialized to `runs/<timestamp>/problem_config.json` after validation.

## Intake Agent

### Interaction Style

Hybrid: the user provides an initial problem description (can be detailed or brief), and
the agent asks targeted follow-up questions for missing or ambiguous ProblemConfig fields.
During the conversation, the agent inspects the uploaded CSV to ask grounded questions
(e.g., "I see a `variant` column -- does that contain size information?").

### System Prompt Responsibilities

1. Read the user's initial problem description
2. Identify which ProblemConfig fields are clear vs. missing/ambiguous
3. Inspect the CSV via data inspection tools to ground questions in actual data
4. Ask follow-up questions one at a time
5. Once all fields are understood, produce a complete ProblemConfig JSON
6. Present it to the user for review/edits

### Tools

| Tool | Description | Output Size |
|---|---|---|
| `peek_columns` | Column names + dtypes from CSV | Small |
| `sample_rows` | First N rows as formatted table (default 10) | Small |
| `describe_column` | Summary stats for one column (count, nulls, uniques, min/max) | Small |
| `list_unique_values` | Unique values for a column (capped at 50) | Small |

All tools are read-only. The agent cannot modify data. Output sizes are kept small
to avoid context overload -- the full CSV never enters the conversation context.

### Output

Validated ProblemConfig JSON. The orchestrator parses with Pydantic and retries up to
3 times if validation fails.

## Data Cleaning Agent

### Responsibilities

Receives ProblemConfig + raw CSV path. Produces a cleaned dataset matching the config's
`data_requirements`.

### System Prompt Responsibilities

1. Inspect the raw data using peek/describe tools
2. Determine cleaning steps based on ProblemConfig.data_requirements
3. Generate and execute pandas code to clean the data
4. Verify the output meets requirements
5. Save cleaned CSV to the run directory

### Tools

| Tool | Description |
|---|---|
| `peek_columns` | Same as intake -- column names and dtypes |
| `sample_rows` | Same as intake -- first N rows |
| `describe_column` | Same as intake -- summary stats |
| `run_cfa_cleaning` | Wraps `scripts/clean_data.py` -- one-shot CFA-specific cleaning |
| `execute_code` | Runs arbitrary pandas code in sandbox, returns stdout/stderr |

### Sandbox (`execute_code`)

Generated Python code runs in an isolated subprocess with:
- 30-second timeout
- Access only to the raw CSV and a designated output directory
- No network access

### Verification Checks

Two checks run after each code execution:

1. **Type consistency** -- numeric columns remain numeric, dates remain dates after
   cleaning. Prevents accidental type coercion.
2. **No fabrication** -- the input CSV is hashed before cleaning. Raw data columns
   in the output must contain only values present in the original source (no
   hallucinated rows or invented raw values). Computed/derived columns (aggregations,
   flags, encodings) are allowed since they are transformations of existing data.
   The verification compares raw column values against the original CSV's value sets.

### Flow

- **CFA data:** Agent calls `run_cfa_cleaning` for the known pipeline, then runs
  additional transformations on top (lifecycle labeling, feature engineering, etc.)
- **Unknown datasets:** Agent uses `execute_code` to build cleaning from scratch,
  iterating through inspect-execute-verify cycles
- Multiple `execute_code` calls allowed -- agent inspects results, fixes issues, refines

### Output

- Cleaned CSV at `runs/<timestamp>/cleaned_data.csv`
- Data manifest JSON at `runs/<timestamp>/data_manifest.json` containing:
  row count, column list, dtypes, null counts

## Orchestrator

Plain Python module wiring the two agents together.

### Flow

```
1. User runs: python -m src.cli <csv_path> [--description "..."]
2. CLI parses arguments
3. Orchestrator creates run directory: runs/<timestamp>/
4. Launch Intake Agent
   - Input: initial description + CSV path
   - Output: ProblemConfig JSON
   - Validate with Pydantic (retry up to 3x)
   - Save: runs/<timestamp>/problem_config.json
5. Launch Data Cleaning Agent
   - Input: ProblemConfig + raw CSV path
   - Output: cleaned CSV path + data manifest
   - Run verification checks (type consistency, no fabrication)
   - Save: runs/<timestamp>/cleaned_data.csv
   - Save: runs/<timestamp>/data_manifest.json
6. Print summary and exit
```

### Trace Logging

Each agent call is logged to `runs/<timestamp>/trace.json` with:
- Timestamp
- Agent name
- Inputs (description, config, CSV path)
- Outputs (config JSON, cleaned data path, manifest)
- Duration
- Token usage (from Agent SDK's AssistantMessage.usage)

### Re-run Support

The orchestrator supports skipping intake and re-running just the data cleaning agent
with a modified config:

```
python -m src.cli data/orders.csv --rerun-from data --config runs/20260603/problem_config.json
```

This enables stakeholder modification: edit the ProblemConfig JSON, re-run cleaning +
downstream pipeline without repeating the intake conversation.

## CLI

```
Usage:
  python -m src.cli <csv_path> [--description "..."] [--rerun-from data --config <path>]

Examples:
  # Interactive intake
  python -m src.cli data/orders.csv

  # Front-load a description
  python -m src.cli data/orders.csv --description "Optimize uniform ordering..."

  # Re-run data cleaning with modified config
  python -m src.cli data/orders.csv --rerun-from data --config runs/20260603/problem_config.json
```

## Tech Stack

- Python 3.11+
- Package manager: uv
- Agent framework: Claude Agent SDK (`claude-agent-sdk`)
- Validation: Pydantic
- Data: pandas
- Testing: pytest, pytest-asyncio
- Model: claude-sonnet-4-20250514 (via Agent SDK, runs on subscription)

## What This Does NOT Cover

- Web UI (FastAPI + frontend) -- deferred to a later module
- Prediction module -- separate team member's responsibility
- Optimization engine -- separate team member's responsibility
- Explanation module -- separate team member's responsibility

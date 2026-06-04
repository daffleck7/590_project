# Web UI Design — Chat + Results

## Overview

A localhost web interface for the intake + data ingestion pipeline. Two pages:
a **chat page** where the user uploads a CSV and converses with the intake agent,
and a **results page** showing the ProblemConfig, cleaned data preview, and full
agent activity log. Dark terminal aesthetic. FastAPI backend with Jinja templates
and vanilla JS — no build step, no npm.

## Architecture

```
Browser (localhost:8000)
    |
    v
FastAPI Backend (src/web/app.py)
    |
    +-- GET /              -> Chat page (Jinja template)
    +-- POST /upload       -> CSV file upload, saves to runs/<id>/
    +-- POST /chat/start   -> Starts intake agent, returns run_id
    +-- GET /chat/stream   -> SSE endpoint, streams agent messages + tool calls
    +-- POST /chat/respond -> User sends a message back to the agent
    +-- POST /run/clean    -> Triggers data cleaning agent
    +-- GET /run/<id>      -> Results page (Jinja template)
    +-- GET /run/<id>/data -> JSON API for results (config, manifest, trace, data preview)
    |
    v
Orchestrator (existing src/orchestrator.py)
    |
    +-- Intake Agent (streams messages back via SSE)
    +-- Data Cleaning Agent (streams messages back via SSE)
```

The orchestrator and agents stay the same. The web layer adds streaming message
capture via a callback pattern — agents accept a callback function instead of
printing to stdout.

## Chat Page

Two-panel layout, dark theme.

### Left Panel (Main) — Chat Conversation

- **Top bar:** Project title + file upload area (drag-and-drop or click to browse)
- **Chat messages:** Scrolling container. Agent messages on the left (dark card),
  user messages on the right (slightly lighter card).
- **Tool calls inline:** Agent tool calls rendered as collapsible blocks showing
  tool name, arguments, and result (e.g., "Called peek_columns -> 7 columns found").
  Lets the user see what the agent is doing in real time.
- **Text input:** Bottom of panel, with send button.
- **ProblemConfig card:** When the agent produces the final JSON, it renders as a
  formatted card with a "Proceed to Data Cleaning" button.

### Right Panel (Sidebar) — Status

- Upload status: filename, size, upload time
- Pipeline progress indicator: Intake -> Cleaning -> Done
- "View Results" button appears once cleaning completes

### Streaming

Agent messages stream to the browser via Server-Sent Events (SSE). The user sees
the agent's text appearing in real-time and can watch tool calls execute. Each SSE
event has a type:

- `message` — agent text block (streamed incrementally)
- `tool_call` — agent called a tool (name + args)
- `tool_result` — tool returned a result
- `config_ready` — agent produced a valid ProblemConfig
- `cleaning_start` — data cleaning agent started
- `cleaning_done` — pipeline complete, results ready
- `error` — something went wrong

## Results Page

Single scrollable page, same dark theme. Three collapsible sections:

### Section 1: ProblemConfig

- Formatted JSON in a syntax-highlighted code block
- Starts expanded

### Section 2: Cleaned Data Preview

- Table showing first 20 rows of cleaned CSV, styled like a terminal table
- Summary stats below: row count, column count, null counts per column
- Starts expanded

### Section 3: Agent Activity Log

- Timeline of what each agent did, in chronological order
- Each entry shows: agent name, action type (message / tool call / tool result),
  content, timestamp
- Tool calls show: tool name, arguments passed, result returned
- Agent reasoning (text messages) shown as chat-style blocks
- Long outputs (e.g., full sample_rows results) are expandable/collapsible
- Token usage and duration summary at the bottom

This section demonstrates that the agent made grounded, traceable decisions —
critical for the class demo's explainability requirement.

## Visual Style

- Dark background (#1a1a2e or similar dark navy/charcoal)
- Monospace fonts for data, JSON, code, and tool calls
- Sans-serif for chat text and UI labels
- Green/cyan accent colors for agent actions and highlights
- Subtle borders and card shadows for depth
- Terminal/dev-tool aesthetic throughout

## Backend Changes

### Message Capture (callback pattern)

The existing agents (`src/intake/agent.py`, `src/data_ingestion/agent.py`) currently
print messages to stdout. They need to accept a callback function so messages can be
routed differently depending on the caller:

- **CLI path:** callback prints to terminal (current behavior preserved)
- **Web path:** callback pushes messages to an async queue that feeds the SSE stream

The callback receives structured events (message text, tool calls, tool results)
so the web layer can render them appropriately.

### Modified Files

- `src/intake/agent.py` — add callback parameter to `run_intake_agent()`
- `src/data_ingestion/agent.py` — add callback parameter to `run_cleaning_agent()`
- `src/orchestrator.py` — pass callback through to agents
- `src/cli.py` — pass a print-based callback (preserves current CLI behavior)

### No Changes To

- ProblemConfig model (`src/models/problem_config.py`)
- Sandbox execution (`src/data_ingestion/sandbox.py`)
- CSV inspection tools (`src/intake/tools.py`)
- Agent system prompts (`src/intake/prompts.py`, `src/data_ingestion/prompts.py`)
- CFA cleaning wrapper (`src/data_ingestion/tools.py`)

## New Files

```
src/web/
    __init__.py
    app.py                  # FastAPI application, routes, SSE endpoint
    templates/
        chat.html           # Chat page Jinja template
        results.html        # Results page Jinja template
    static/
        style.css           # Dark theme styles
        chat.js             # SSE handling, chat UI logic, file upload
        results.js          # Results page rendering, collapsible sections
```

## Tech Stack

- FastAPI (already a project dependency preference from CLAUDE.md)
- Jinja2 (FastAPI built-in template support)
- Vanilla JavaScript (SSE via EventSource API, DOM manipulation)
- No build step, no npm, no React

## Dependencies to Add

- `fastapi`
- `uvicorn`
- `jinja2`
- `python-multipart` (for file uploads)

## What This Does NOT Cover

- Config editing UI (future enhancement — user edits ProblemConfig before cleaning)
- Data upload page as a separate page (upload is integrated into the chat page)
- Trace viewer as a separate page (trace is shown on results page)
- Authentication or multi-user support
- Deployment beyond localhost

"""FastAPI web application for the CFA Optimization Agent."""

import asyncio
import json
import traceback
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from src.models.problem_config import ProblemConfig, build_cfa_default_config
from src.orchestrator import Orchestrator, create_run_directory, save_trace

DEFAULT_CSV = Path("data/orders.csv")

WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"
RUNS_DIR = Path("runs")

app = FastAPI(title="CFA Optimization Agent")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# In-memory state for active runs
active_runs: dict[str, dict] = {}


@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request) -> HTMLResponse:
    """Serve the chat page."""
    return templates.TemplateResponse(request, "chat.html")


@app.get("/dev/skip-intake")
async def skip_intake(request: Request) -> JSONResponse:
    """Dev shortcut: skip intake, use default config, jump to cleaning."""
    run_dir = create_run_directory(base_dir=str(RUNS_DIR))
    run_id = Path(run_dir).name

    # Copy raw CSV into run dir
    csv_path = Path(run_dir) / "upload.csv"
    csv_path.write_bytes(DEFAULT_CSV.read_bytes())

    event_queue: asyncio.Queue = asyncio.Queue()
    config = build_cfa_default_config()

    async def event_callback(event: dict) -> None:
        await event_queue.put(event)

    orchestrator = Orchestrator(
        str(csv_path),
        base_dir=str(RUNS_DIR),
        callback=event_callback,
        interactive=False,
    )
    orchestrator.run_dir = run_dir

    active_runs[run_id] = {
        "run_dir": run_dir,
        "csv_path": str(csv_path),
        "queue": event_queue,
        "orchestrator": orchestrator,
        "config": config,
    }

    # Save config and start cleaning
    config_path = Path(run_dir) / "problem_config.json"
    config_path.write_text(config.model_dump_json(indent=2))

    asyncio.create_task(_run_cleaning(run_id))

    return JSONResponse({"run_id": run_id, "stream": f"/chat/stream?run_id={run_id}"})


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a CSV file and create a new run."""
    if not file.filename or not file.filename.endswith(".csv"):
        return JSONResponse({"error": "Please upload a CSV file"}, status_code=400)

    run_dir = create_run_directory(base_dir=str(RUNS_DIR))
    run_id = Path(run_dir).name

    csv_path = Path(run_dir) / "upload.csv"
    content = await file.read()
    csv_path.write_bytes(content)

    event_queue: asyncio.Queue = asyncio.Queue()

    active_runs[run_id] = {
        "run_dir": run_dir,
        "csv_path": str(csv_path),
        "queue": event_queue,
        "orchestrator": None,
        "config": None,
    }

    return JSONResponse({
        "run_id": run_id,
        "filename": file.filename,
        "size": len(content),
    })


@app.post("/chat/respond")
async def chat_respond(request: Request) -> JSONResponse:
    """User sends a message — start or continue the intake agent."""
    body = await request.json()
    run_id = body.get("run_id")
    message = body.get("message", "")

    if run_id not in active_runs:
        return JSONResponse({"error": "Run not found"}, status_code=404)

    run = active_runs[run_id]

    async def event_callback(event: dict) -> None:
        await run["queue"].put(event)

    if run["orchestrator"] is None:
        orchestrator = Orchestrator(
            run["csv_path"],
            base_dir=str(RUNS_DIR),
            callback=event_callback,
            interactive=False,
        )
        orchestrator.run_dir = run["run_dir"]
        run["orchestrator"] = orchestrator

        asyncio.create_task(_run_intake(run_id, message))
    else:
        # Follow-up message — feed it to the waiting agent
        run["orchestrator"].user_input_queue.put_nowait(message)

    return JSONResponse({"status": "ok"})


async def _run_intake(run_id: str, description: str) -> None:
    """Run the intake agent until config is saved and approved."""
    run = active_runs[run_id]
    orchestrator = run["orchestrator"]

    try:
        config = await orchestrator.run_intake(description)
        run["config"] = config
    except Exception as exc:
        traceback.print_exc()
        await run["queue"].put({"type": "error", "message": str(exc)})

    # Signal that intake is done — config_ready was already sent by the agent
    # The user will approve via the UI, which triggers /run/clean
    await run["queue"].put({"type": "__done__"})


@app.post("/run/clean")
async def run_cleaning(request: Request) -> JSONResponse:
    """Trigger the full pipeline after intake approval."""
    body = await request.json()
    run_id = body.get("run_id")

    if run_id not in active_runs:
        return JSONResponse({"error": "Run not found"}, status_code=404)

    run = active_runs[run_id]

    # Load config from disk if not in memory
    if run["config"] is None:
        config_path = Path(run["run_dir"]) / "problem_config.json"
        if config_path.exists():
            run["config"] = ProblemConfig.model_validate_json(config_path.read_text())
        else:
            return JSONResponse({"error": "No config available"}, status_code=400)

    asyncio.create_task(_run_pipeline(run_id))
    return JSONResponse({"status": "ok"})


async def _run_pipeline(run_id: str) -> None:
    """Run the full pipeline: cleaning -> modeling -> explanation."""
    run = active_runs[run_id]
    orchestrator = run["orchestrator"]
    queue = run["queue"]
    config = run["config"]

    # Stage 1: Data Cleaning (LLM Agent)
    cleaned_path = None
    try:
        cleaned_path, manifest = await orchestrator.run_cleaning(config)

        summary_path = Path(run["run_dir"]) / "cleaning_summary.txt"
        if summary_path.exists():
            await queue.put({
                "type": "stage_summary",
                "stage": "cleaning",
                "summary": summary_path.read_text(encoding="utf-8", errors="replace"),
            })
    except Exception as exc:
        traceback.print_exc()
        await queue.put({"type": "error", "message": f"Cleaning failed: {exc}"})
        await queue.put({"type": "__done__"})
        return

    if not cleaned_path:
        await queue.put({"type": "error", "message": "Cleaning produced no output"})
        await queue.put({"type": "__done__"})
        return

    # Stage 2: Modeling Agent (LLM — prediction + optimization)
    try:
        success = await orchestrator.run_modeling(config, cleaned_path)
        if not success:
            await queue.put({"type": "error", "message": "Modeling agent did not produce a summary"})
    except Exception as exc:
        traceback.print_exc()
        await queue.put({"type": "error", "message": f"Modeling failed: {exc}"})
        await queue.put({"type": "__done__"})
        return

    # Stage 3: Explanation Agent (LLM — final report)
    try:
        success = await orchestrator.run_explanation()
        if not success:
            await queue.put({"type": "error", "message": "Explanation agent did not produce a report"})
    except Exception as exc:
        traceback.print_exc()
        await queue.put({"type": "error", "message": f"Explanation failed: {exc}"})

    # Save final trace
    orchestrator.trace["activity_log"] = orchestrator.activity_log
    save_trace(orchestrator.run_dir, orchestrator.trace)

    await queue.put({"type": "__done__"})


@app.get("/chat/stream")
async def chat_stream(run_id: str) -> EventSourceResponse:
    """SSE endpoint streaming agent events to the browser."""
    if run_id not in active_runs:
        return EventSourceResponse(iter([]))

    run = active_runs[run_id]

    async def event_generator():
        while True:
            event = await run["queue"].get()
            if event.get("type") == "__done__":
                yield {"event": "done", "data": "{}"}
                break
            elif event.get("type") == "__waiting__":
                yield {"event": "waiting", "data": "{}"}
                break
            yield {"event": "message", "data": json.dumps(event, default=str)}

    return EventSourceResponse(event_generator())


@app.get("/run/{run_id}", response_class=HTMLResponse)
async def results_page(request: Request, run_id: str) -> HTMLResponse:
    """Serve the results page."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        return HTMLResponse("Run not found", status_code=404)
    return templates.TemplateResponse(
        request, "results.html", {"run_id": run_id}
    )


@app.get("/run/{run_id}/data")
async def run_data(run_id: str) -> JSONResponse:
    """Return run results as JSON — primarily the final report."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        return JSONResponse({"error": "Run not found"}, status_code=404)

    result: dict = {}

    # Final report from explanation agent
    report_path = run_dir / "final_report.md"
    if report_path.exists():
        result["report"] = report_path.read_text(encoding="utf-8", errors="replace")

    # Config for reference
    config_path = run_dir / "problem_config.json"
    if config_path.exists():
        result["config"] = json.loads(config_path.read_text())

    # Trace for timing stats
    trace_path = run_dir / "trace.json"
    if trace_path.exists():
        trace = json.loads(trace_path.read_text())
        result["trace"] = trace

    return JSONResponse(result)

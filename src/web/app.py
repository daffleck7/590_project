"""FastAPI web application for the CFA Optimization Agent."""

import asyncio
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from src.models.problem_config import ProblemConfig
from src.orchestrator import Orchestrator, create_run_directory

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
        )
        orchestrator.run_dir = run["run_dir"]
        run["orchestrator"] = orchestrator

        asyncio.create_task(_run_intake(run_id, message))

    return JSONResponse({"status": "ok"})


async def _run_intake(run_id: str, description: str) -> None:
    """Run the intake agent in background."""
    run = active_runs[run_id]
    orchestrator = run["orchestrator"]

    try:
        config = await orchestrator.run_intake(description)
        run["config"] = config
    except Exception as exc:
        await run["queue"].put({"type": "error", "message": str(exc)})

    await run["queue"].put({"type": "__done__"})


@app.post("/run/clean")
async def run_cleaning(request: Request) -> JSONResponse:
    """Trigger the data cleaning agent."""
    body = await request.json()
    run_id = body.get("run_id")

    if run_id not in active_runs:
        return JSONResponse({"error": "Run not found"}, status_code=404)

    run = active_runs[run_id]
    if run["config"] is None:
        return JSONResponse({"error": "No config available"}, status_code=400)

    asyncio.create_task(_run_cleaning(run_id))
    return JSONResponse({"status": "ok"})


async def _run_cleaning(run_id: str) -> None:
    """Run the data cleaning agent in background."""
    run = active_runs[run_id]
    orchestrator = run["orchestrator"]

    try:
        await orchestrator.run_cleaning(run["config"])
        orchestrator.trace["activity_log"] = orchestrator.activity_log
        from src.orchestrator import save_trace
        save_trace(orchestrator.run_dir, orchestrator.trace)
    except Exception as exc:
        await run["queue"].put({"type": "error", "message": str(exc)})

    await run["queue"].put({"type": "__done__"})


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
    """Return run results as JSON."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        return JSONResponse({"error": "Run not found"}, status_code=404)

    result: dict = {}

    config_path = run_dir / "problem_config.json"
    if config_path.exists():
        result["config"] = json.loads(config_path.read_text())

    manifest_path = run_dir / "data_manifest.json"
    if manifest_path.exists():
        result["manifest"] = json.loads(manifest_path.read_text())

    trace_path = run_dir / "trace.json"
    if trace_path.exists():
        trace = json.loads(trace_path.read_text())
        result["trace"] = trace
        result["activity_log"] = trace.get("activity_log", [])

    cleaned_csv = run_dir / "cleaned_data.csv"
    if not cleaned_csv.exists():
        cleaned_csv = run_dir / "cfa_cleaned.csv"

    if cleaned_csv.exists():
        df = pd.read_csv(cleaned_csv, nrows=20, low_memory=False)
        result["data_preview"] = df.to_dict(orient="records")
    else:
        result["data_preview"] = []

    return JSONResponse(result)

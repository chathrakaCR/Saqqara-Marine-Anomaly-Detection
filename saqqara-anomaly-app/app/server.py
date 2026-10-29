"""Web app: upload a video (or give a stream URL), watch progress, browse sightings, download results.

  uvicorn app.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .pipeline import Pipeline
from .runner import process_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("server")

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
UPLOADS = DATA_DIR / "uploads"
RESULTS = DATA_DIR / "results"
for d in (UPLOADS, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

_pipeline: Pipeline | None = None
_pipeline_error: str | None = None
_jobs: dict[str, dict] = {}
_queue: "queue.Queue[str]" = queue.Queue()
_lock = threading.Lock()


def _load_pipeline():
    global _pipeline, _pipeline_error
    try:
        _pipeline = Pipeline()
    except Exception as e:                       # shown on the page and at /api/health
        _pipeline_error = str(e)
        log.error("Pipeline failed to load: %s", e)


def _worker():
    while True:
        job_id = _queue.get()
        job = _jobs[job_id]
        if job["status"] == "cancelled":
            continue
        job.update(status="running", started=time.time())

        def progress(done, total, row, job=job):
            job["done"], job["total"] = done, total
            job["last"] = {k: row[k] for k in ("time_s", "label", "reason")}
            if row["label"] != "none":
                job["counts"][row["label"]] = job["counts"].get(row["label"], 0) + 1

        try:
            res = process_video(job["source"], RESULTS / job_id, _pipeline, job["every"], None, job["max_seconds"],
                                job["save_frames"], job["save_video"], progress=progress, stop_event=job["stop"])
            job.update(status="stopped" if job["stop"].is_set() else "done", summary=res["summary"], events=res["events"])
        except Exception as e:
            log.exception("Job %s failed", job_id)
            job.update(status="failed", error=str(e))
        finally:
            job["finished"] = time.time()
            if job.get("upload") and Path(job["upload"]).exists() and not job["keep_upload"]:
                Path(job["upload"]).unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=_load_pipeline, daemon=True).start()
    threading.Thread(target=_worker, daemon=True).start()
    yield


app = FastAPI(title="Marine anomaly detection", version="1.0.0", lifespan=lifespan)


def _public(job):
    keys = ("id", "name", "status", "done", "total", "counts", "last", "error", "created", "started", "finished",
            "summary", "events", "save_video")
    return {k: job.get(k) for k in keys}


@app.get("/api/health")
def health():
    return {"ready": _pipeline is not None, "error": _pipeline_error,
            "settings": _pipeline.describe() if _pipeline else None}


@app.post("/api/jobs")
async def create_job(file: UploadFile | None = File(None), url: str = Form(""), every: float = Form(1.0),
                     max_seconds: float = Form(0), save_frames: bool = Form(False), save_video: bool = Form(True),
                     keep_upload: bool = Form(False)):
    if _pipeline is None:
        raise HTTPException(503, _pipeline_error or "Models are still loading, try again in a moment")
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    upload = None
    if file is not None and file.filename:
        upload = UPLOADS / f"{job_id}_{Path(file.filename).name}"
        with open(upload, "wb") as f:
            shutil.copyfileobj(file.file, f)
        source, name = str(upload), file.filename
    elif url.strip():
        source, name = url.strip(), url.strip()
    else:
        raise HTTPException(400, "Upload a video file or give a stream URL")
    job = {"id": job_id, "name": name, "source": source, "upload": str(upload) if upload else None,
           "every": max(0.04, float(every)), "max_seconds": float(max_seconds) or None, "save_frames": save_frames,
           "save_video": save_video, "keep_upload": keep_upload, "status": "queued", "done": 0, "total": None,
           "counts": {}, "created": time.time(), "stop": threading.Event()}
    with _lock:
        _jobs[job_id] = job
    _queue.put(job_id)
    return _public(job)


@app.get("/api/jobs")
def list_jobs():
    return [_public(j) for j in sorted(_jobs.values(), key=lambda j: j["created"], reverse=True)]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "No such job")
    return _public(_jobs[job_id])


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str):
    job = _jobs.get(job_id) or {}
    if not job:
        raise HTTPException(404, "No such job")
    if job["status"] == "queued":
        job["status"] = "cancelled"
    job["stop"].set()
    return _public(job)


@app.get("/api/jobs/{job_id}/files/{path:path}")
def job_file(job_id: str, path: str):
    base = (RESULTS / job_id).resolve()
    target = (base / path).resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(404, "No such file")
    return FileResponse(target)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8"))

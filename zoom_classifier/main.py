# zoom-classifier/main.py
from __future__ import annotations

import asyncio
import base64
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas import RunRequest, RunResponse
from model import zoom_classifier

import sys
sys.path.append('..')
try:
    from artifact_io import fetch_artifact
except ImportError:
    # Fallback if not in expected structure
    async def fetch_artifact(uri: str) -> bytes:
        raise NotImplementedError("artifact_io not found - implement Azure fetch")

log = structlog.get_logger()

app = FastAPI(
    title="zoom-classifier",
    description="Classifies jewelry images into zoom levels 1-4",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

MIN_WORKERS = 1
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))
SCALE_UP_THRESHOLD = 0.8
SCALE_DOWN_IDLE_TIME = 30

_pool = ThreadPoolExecutor(max_workers=MIN_WORKERS)
_current_workers = MIN_WORKERS
_last_scale_time = 0
_active_tasks = 0
_task_lock = asyncio.Lock()


async def _infer(img_bytes: bytes) -> str:
    """Run inference in thread pool to avoid blocking."""
    global _active_tasks
    
    async with _task_lock:
        _active_tasks += 1
    
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(_pool, zoom_classifier.predict, img_bytes)
        return result
    finally:
        async with _task_lock:
            _active_tasks -= 1


async def _adjust_pool_size():
    """Dynamically adjust thread pool size based on load."""
    global _pool, _current_workers, _last_scale_time
    
    current_time = asyncio.get_event_loop().time()

    load_ratio = _active_tasks / _current_workers if _current_workers > 0 else 0

    if load_ratio >= SCALE_UP_THRESHOLD:
        if _current_workers < MAX_WORKERS:
            new_size = min(_current_workers + 2, MAX_WORKERS)
            old_pool = _pool
            _pool = ThreadPoolExecutor(max_workers=new_size)
            _current_workers = new_size
            _last_scale_time = current_time
            log.info("scaling_up", workers=new_size, active_tasks=_active_tasks)
    
    elif load_ratio < 0.3 and _active_tasks < _current_workers - 1:
        if _current_workers > MIN_WORKERS and (current_time - _last_scale_time) > SCALE_DOWN_IDLE_TIME:
            new_size = max(_current_workers - 1, MIN_WORKERS)
            old_pool = _pool
            _pool = ThreadPoolExecutor(max_workers=new_size)
            _current_workers = new_size
            _last_scale_time = current_time
            log.info("scaling_down", workers=new_size, active_tasks=_active_tasks)


def _find_artifact(node: Any) -> str | None:
    if isinstance(node, dict):
        if "uri" in node and str(node["uri"]).startswith("azure://"):
            return node["uri"]
        for v in node.values():
            uri = _find_artifact(v)
            if uri:
                return uri
    elif isinstance(node, list):
        for item in node:
            uri = _find_artifact(item)
            if uri:
                return uri
    return None


async def _extract_image_bytes(data: Dict[str, Any]) -> bytes:
    """Extract image bytes from various input formats."""
    if "image_bytes" in data:
        return base64.b64decode(data["image_bytes"])

    if "artifact" in data:
        return await fetch_artifact(data["artifact"])

    if isinstance(data, dict) and "uri" in data:
        return await fetch_artifact(data["uri"])

    uri = _find_artifact(data)
    if uri:
        return await fetch_artifact(uri)
    
    raise ValueError("No image_bytes or artifact found in request")



@app.post("/run", response_model=RunResponse)
async def run(req: RunRequest, raw_request: Request):
    """
    Classify image zoom level.
    
    Returns: {"zoom_level": "zoom_X"} where X is 1-4
    """
    request_id = req.meta.get("trace_id", "unknown")
    
    # Adjust pool size based on load
    await _adjust_pool_size()
    
    try:
        img_bytes = await _extract_image_bytes(req.data)
        
    except Exception as exc:
        log.error("image_extraction_failed", request_id=request_id, error=str(exc))
        raise HTTPException(422, f"Failed to extract image: {exc}")
    
    try:
        zoom_level = await _infer(img_bytes)
        
        log.info("inference_complete", 
                 request_id=request_id,
                 zoom_level=zoom_level,
                 workers=_current_workers,
                 active_tasks=_active_tasks)
        
        return RunResponse(zoom_level=zoom_level)
        
    except Exception as exc:
        log.error("inference_failed", 
                  request_id=request_id,
                  error=str(exc))
        raise HTTPException(500, f"Inference failed: {exc}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        from PIL import Image
        import io
        
        test_img = Image.new('RGB', (224, 224), color='white')
        buf = io.BytesIO()
        test_img.save(buf, format='PNG')
        buf.seek(0)
        
        zoom = await _infer(buf.getvalue())
        
        return JSONResponse({
            "ok": True,
            "workers": _current_workers,
            "max_workers": MAX_WORKERS,
            "active_tasks": _active_tasks,
            "model_loaded": True,
            "test_inference": zoom
        })
    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": str(e)},
            status_code=503
        )


@app.get("/metrics")
async def metrics():
    """Metrics endpoint for monitoring."""
    return JSONResponse({
        "current_workers": _current_workers,
        "max_workers": MAX_WORKERS,
        "min_workers": MIN_WORKERS,
        "active_tasks": _active_tasks,
        "pool_status": "healthy"
    })


# ────────────────────────────────────────────────
#  Startup & Shutdown
# ────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    """Pre-load model on startup."""
    global _last_scale_time
    _last_scale_time = asyncio.get_event_loop().time()
    
    log.info("service_starting", workers=MIN_WORKERS, max_workers=MAX_WORKERS)
    
    _ = zoom_classifier
    
    log.info("model_loaded")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    _pool.shutdown(wait=True)
    log.info("service_stopped")
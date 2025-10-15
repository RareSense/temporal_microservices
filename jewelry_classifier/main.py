from __future__ import annotations
"""Jewelry classifier micro-service with structured, verbose logging.
All existing behaviour is preserved; we only add richer log events so you can
see exactly what was detected when running under Uvicorn.
"""

import base64
import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas import RunRequest
from model import classifier

import sys
sys.path.append("..")
from artifact_io import fetch_artifact  # noqa: E402  (local import after sys.path tweak)

# ────────────────────────────────────────────────
#  Logging setup
# ────────────────────────────────────────────────
log = structlog.get_logger()

# ────────────────────────────────────────────────
#  FastAPI app
# ────────────────────────────────────────────────
app = FastAPI(title="jewelry-classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ThreadPool for synchronous model inference
auto_workers = int(os.getenv("WORKERS", "4"))
_pool = ThreadPoolExecutor(max_workers=auto_workers)


async def _infer(img: bytes, request_id: str) -> List[str]:
    """Runs classifier.predict in an executor and logs the result."""
    loop = asyncio.get_running_loop()
    labels: List[str] = await loop.run_in_executor(_pool, classifier.predict, img)

    # Detailed logging of inference output
    log.info("inference_complete", request_id=request_id, detected_labels=labels)
    return labels


def _find_artifact(node: Any) -> str | None:
    """Recursively search nested structures for an Azure-style artifact URI."""
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


@app.post("/run")
async def run(req: RunRequest, raw: Request):  # noqa: D401 (FastAPI handler)
    """Classify jewelry in the supplied image and return detected labels."""

    # Each request gets a correlation ID for coherent logs
    request_id = raw.headers.get("X-Request-ID", str(uuid.uuid4()))
    log.info("request_received", request_id=request_id)

    # ─── 1. Resolve image bytes ──────────────────────────────────────────────
    try:
        if "image_bytes" in req.data:
            img_bytes = base64.b64decode(req.data["image_bytes"])
            log.debug(
                "image_source", request_id=request_id, kind="inline_base64", size=len(img_bytes)
            )
        elif "artifact" in req.data:
            img_bytes = await fetch_artifact(req.data["artifact"])
            log.debug(
                "image_source", request_id=request_id, kind="artifact", uri=req.data["artifact"]
            )
        elif isinstance(req.data, dict) and "uri" in req.data:
            img_bytes = await fetch_artifact(req.data["uri"])
            log.debug("image_source", request_id=request_id, kind="uri", uri=req.data["uri"])
        else:
            uri = _find_artifact(req.data)
            if not uri:
                raise HTTPException(422, "no image_bytes or artifact found")
            img_bytes = await fetch_artifact(uri)
            log.debug("image_source", request_id=request_id, kind="nested_uri", uri=uri)

    except HTTPException:
        raise  # Propagate FastAPI semantics unchanged
    except Exception as exc:  # noqa: BLE001
        log.error("artifact_fetch_failed", request_id=request_id, err=str(exc))
        raise HTTPException(500, f"artifact fetch failed: {exc}")

    # ─── 2. Model inference ─────────────────────────────────────────────────
    try:
        labels = await _infer(img_bytes, request_id)

        # Log the final API response before returning
        log.info("response_ready", request_id=request_id, detected_labels=labels)
        return {"detected_jewelry": labels}

    except Exception as exc:  # noqa: BLE001
        log.error("inference_failed", request_id=request_id, err=str(exc))
        raise HTTPException(500, f"inference failed: {exc}")


@app.get("/health")
async def health():  # noqa: D401 (FastAPI handler)
    """Basic readiness probe."""
    return JSONResponse({"ok": True})

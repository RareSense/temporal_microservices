from __future__ import annotations

import base64
import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas import RunRequest, RunResponse
from model import upscaler  

import sys
sys.path.append("..")               
from artifact_io import fetch_artifact 

log = structlog.get_logger()

app = FastAPI(title="image-upscaler")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-bound work
_pool = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "4")))

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


async def _resolve_bytes(field: str,
                         req_data: Dict[str, Any],
                         request_id: str) -> bytes:
    """
    Resolve `field` (expects either `<field>_bytes` or `<field>_artifact` or nested uri)
    to raw bytes.
    """
    b64_key = f"{field}_bytes"
    art_key = f"{field}_artifact"

    # 1. Inline base64
    if b64_key in req_data:
        raw = base64.b64decode(req_data[b64_key])
        log.debug("image_source", request_id=request_id,
                  role=field, kind="inline_base64", size=len(raw))
        return raw

    # 2. Explicit artifact
    if art_key in req_data:
        raw = await fetch_artifact(req_data[art_key])
        log.debug("image_source", request_id=request_id,
                  role=field, kind="artifact", uri=req_data[art_key])
        return raw

    # 3. Nested uri
    uri = _find_artifact(req_data.get(field, {}))
    if uri:
        raw = await fetch_artifact(uri)
        log.debug("image_source", request_id=request_id,
                  role=field, kind="nested_uri", uri=uri)
        return raw

    raise HTTPException(422, f"missing {field} bytes or artifact")


async def _process(background: bytes, ghost: bytes,
                   request_id: str) -> str:
    loop = asyncio.get_running_loop()
    b64: str = await loop.run_in_executor(
        _pool, upscaler.process, background, ghost
    )
    log.info("inference_complete", request_id=request_id,
             output_size=len(b64))
    return b64

@app.post("/run", response_model=RunResponse)
async def run(req: RunRequest, raw: Request):  
    """
    Inputs:
      data.background_bytes | background_artifact
      data.ghost_bytes      | ghost_artifact
    Returns:
      { "composite_b64": <str> }
    """
    request_id = raw.headers.get("X-Request-ID", str(uuid.uuid4()))
    log.info("request_received", request_id=request_id)

    try:
        bg_bytes   = await _resolve_bytes("background", req.data, request_id)
        ghost_bytes = await _resolve_bytes("ghost", req.data, request_id)
    except HTTPException:
        raise
    except Exception as exc: 
        log.error("artifact_fetch_failed", request_id=request_id, err=str(exc))
        raise HTTPException(500, f"artifact fetch failed: {exc}")

    try:
        composite_b64 = await _process(bg_bytes, ghost_bytes, request_id)
        log.info("response_ready", request_id=request_id)
        return {"composite_b64": composite_b64}
    except Exception as exc: 
        log.error("processing_failed", request_id=request_id, err=str(exc))
        raise HTTPException(500, f"upscale/composite failed: {exc}")


@app.get("/health")
async def health():  
    """Readiness probe."""
    return JSONResponse({"ok": True})

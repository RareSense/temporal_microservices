from __future__ import annotations
import base64, asyncio, json, os
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas import RunRequest
from model import classifier
import sys
sys.path.append('..')
from artifact_io import fetch_artifact

log = structlog.get_logger()
app = FastAPI(title="jewelry-classifier")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_pool = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "4")))

async def _infer(img: bytes) -> List[str]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_pool, classifier.predict, img)


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


@app.post("/run", response_model=List[str])
async def run(req: RunRequest, raw: Request):
    """
    Returns a **plain list[str]** – jewellery names.
    Accepts any of:
      • data.image_bytes  – base-64 jpg/png
      • data.artifact     – 'azure://container/blob'
      • data.original_path.uri  (or any nested .uri)
      • data == Artifact dict itself
    """

    try:
        if "image_bytes" in req.data:                      
            img_bytes = base64.b64decode(req.data["image_bytes"])

        elif "artifact" in req.data:                   
            img_bytes = await fetch_artifact(req.data["artifact"])

        elif isinstance(req.data, dict) and "uri" in req.data: 
            img_bytes = await fetch_artifact(req.data["uri"])

        else:                                             
            uri = _find_artifact(req.data)
            if not uri:
                raise HTTPException(422, "no image_bytes or artifact found")
            img_bytes = await fetch_artifact(uri)

    except HTTPException:
        raise                                               
    except Exception as exc:
        log.error("artifact_fetch_failed", err=str(exc))
        raise HTTPException(500, f"artifact fetch failed: {exc}")

    try:
        labels = await _infer(img_bytes)
        return labels
    except Exception as exc:
        log.error("inference_failed", err=str(exc))
        raise HTTPException(500, f"inference failed: {exc}")


@app.get("/health")
async def health():
    return JSONResponse({"ok": True})

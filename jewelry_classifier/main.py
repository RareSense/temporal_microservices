from __future__ import annotations

"""
FastAPI entry-point for the jewelry-classifier micro-service
▸ Returns **just a list[str]** with the detected jewellery names – no wrapper object.
▸ Compatible with the existing schemas.RunRequest for input validation.
▸ Thread-pool off-loads Torch inference so the event-loop stays non-blocking.
"""

import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas import RunRequest  # unchanged input model
from model import classifier
from artifact_io import fetch_artifact

log = structlog.get_logger()
app = FastAPI(title="jewelry-classifier")

# ────────────────────────────────────────────────────────────────
#  CORS (open by default – lock down in prod)
# ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────────
#  CPU-bound inference ⇢ thread-pool
#    • Size defaults to 4; tweak via env var if needed.
# ────────────────────────────────────────────────────────────────
_pool = ThreadPoolExecutor(max_workers=4)


async def _infer(img_bytes: bytes) -> List[str]:
    """Run model.predict() in a worker thread and return label list."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_pool, classifier.predict, img_bytes)


@app.post("/run", response_model=List[str])
async def run(req: RunRequest):
    """Main inference endpoint – returns **list[str]** of jewellery names."""

    # 1) Decode image source
    try:
        if "image_bytes" in req.data:                       # inline base-64
            img_bytes = base64.b64decode(req.data["image_bytes"])
        elif "artifact" in req.data:                        # azure://…
            img_bytes = await fetch_artifact(req.data["artifact"])
        else:
            raise KeyError("expected 'image_bytes' or 'artifact'")
    except Exception as exc:
        raise HTTPException(422, f"bad image input: {exc}")

    # 2) Predict
    labels = await _infer(img_bytes)
    return labels


@app.get("/health")
async def health():
    """Liveness probe consumed by ServiceRegistry."""
    return JSONResponse({"ok": True})

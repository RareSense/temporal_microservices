# grounded_sam_segmenter/main.py
import asyncio, os, base64, structlog
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from schemas import RunRequest, RunResponse
import sys
sys.path.append('..')
from artifact_io import fetch_artifact, upload_artifact   # helper below
from model import segment

log = structlog.get_logger()
app = FastAPI(title="grounded-sam-segmenter")

_POOL = ProcessPoolExecutor(
    max_workers=int(os.getenv("MAX_WORKERS", "4")),
    mp_context=get_context("spawn"),
)

async def _run_segmentation(img: bytes, labels: list[str]) -> RunResponse:
    loop = asyncio.get_running_loop()
    masks = await loop.run_in_executor(_POOL, segment, img, labels)

    artifacts = {}
    for lab, png in masks.items():
        art = await upload_artifact(png, mime="image/png")   # ← Azure helper
        artifacts[lab] = art.model_dump()
    return RunResponse(labels=labels, artifacts=artifacts)

@app.post("/run", response_model=RunResponse)
async def run(req: RunRequest, raw: Request):
    # 1) get image bytes (identical logic to classifier)
    try:
        if "image_bytes" in req.data:
            img_bytes = base64.b64decode(req.data["image_bytes"])
        elif "artifact" in req.data:
            img_bytes = await fetch_artifact(req.data["artifact"])
        elif isinstance(req.data, dict) and "uri" in req.data:
            img_bytes = await fetch_artifact(req.data["uri"])
        else:
            raise HTTPException(422, "no image artifact or bytes")
    except Exception as exc:
        log.error("artifact_fetch_failed", err=str(exc))
        raise HTTPException(500, f"artifact fetch failed: {exc}")

    # 2) extract labels list
    labels = req.data.get("labels") or req.data.get("jewelry_labels")  # flexible
    if not labels or not isinstance(labels, list):
        raise HTTPException(422, "no labels list found")

    # 3) segmentation
    try:
        return await _run_segmentation(img_bytes, labels)
    except Exception as exc:
        log.error("segmentation_failed", err=str(exc))
        raise HTTPException(500, f"SAM failed: {exc}")

@app.get("/health")
async def health():  # liveness / readiness probe
    return JSONResponse({"ok": True})

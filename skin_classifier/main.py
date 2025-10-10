from __future__ import annotations

import asyncio
import base64
import os
import io
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List
from PIL import Image

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model import classifier
from schemas import RunRequest, RunResponse
import sys
sys.path.append('..')
from artifact_io import fetch_artifact

log = structlog.get_logger()

app = FastAPI(
    title="skin-tone-classifier",
    description="CLIP-based skin tone classification service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False
)

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
MIN_WORKERS = int(os.getenv("MIN_WORKERS", "1"))
_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

async def _infer(img_bytes: bytes) -> List[tuple[str, float]]:
    """Run inference in thread pool to avoid blocking."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_pool, classifier.predict, img_bytes)


async def _infer_single(img_bytes: bytes) -> str:
    """Get just the top-1 class name."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_pool, classifier.predict_single, img_bytes)

def _find_image_artifact(node: Any) -> str | None:
    if isinstance(node, dict):
        if "uri" in node and str(node["uri"]).startswith("azure://"):
            if "type" in node and "image" in str(node.get("type", "")):
                return node["uri"]
            if "type" not in node:
                return node["uri"]

        if "image" in node and isinstance(node["image"], dict):
            uri = _find_image_artifact(node["image"])
            if uri:
                return uri
        
        for v in node.values():
            uri = _find_image_artifact(v)
            if uri:
                return uri
                
    elif isinstance(node, list):
        for item in node:
            uri = _find_image_artifact(item)
            if uri:
                return uri
    
    return None

# ────────────────────────────────────────────────
#  Main endpoint
# ────────────────────────────────────────────────
@app.post("/run", response_model=RunResponse)
async def run(req: RunRequest) -> Dict[str, Any]:
    """
    Classify skin tone from image.
    Returns: {
        "skin_tone": "fair",  # top-1 prediction
        "skin_tone_predictions": [
            {"class": "fair", "confidence": 0.85},
            {"class": "light", "confidence": 0.12},
            {"class": "medium", "confidence": 0.03}
        ]
    }
    """
    try:
        img_bytes = None

        if "image_bytes" in req.data:
            img_bytes = base64.b64decode(req.data["image_bytes"])
        
        elif "artifact" in req.data:
            img_bytes = await fetch_artifact(req.data["artifact"])
        
        elif "image" in req.data and isinstance(req.data["image"], dict):
            if "uri" in req.data["image"]:
                img_bytes = await fetch_artifact(req.data["image"]["uri"])
        
        elif isinstance(req.data, dict) and "uri" in req.data:
            img_bytes = await fetch_artifact(req.data["uri"])

        else:
            uri = _find_image_artifact(req.data)
            if uri:
                img_bytes = await fetch_artifact(uri)
        
        if not img_bytes:
            log.error("no_image_found", data_keys=list(req.data.keys()))
            raise HTTPException(422, "No image_bytes, artifact, or image URI found")
    
    except HTTPException:
        raise
    except Exception as exc:
        log.error("artifact_fetch_failed", err=str(exc))
        raise HTTPException(500, f"Failed to fetch artifact: {exc}")
    
    try:
        predictions = await _infer(img_bytes)
        
        skin_tone_predictions = [
            {"class": cls, "confidence": round(conf, 4)}
            for cls, conf in predictions
        ]
        
        return {
            "skin_tone": predictions[0][0] if predictions else "unknown",
            "skin_tone_predictions": skin_tone_predictions
        }
    
    except Exception as exc:
        log.error("inference_failed", err=str(exc))
        raise HTTPException(500, f"Inference failed: {exc}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        test_img = Image.new("RGB", (224, 224), color="white")
        buf = io.BytesIO()
        test_img.save(buf, format="PNG")
        _ = await _infer_single(buf.getvalue())
        return JSONResponse({"status": "healthy", "workers": MAX_WORKERS})
    except Exception as e:
        log.error("health_check_failed", err=str(e))
        return JSONResponse(
            {"status": "unhealthy", "error": str(e)}, 
            status_code=503
        )


@app.on_event("startup")
async def startup():
    log.info("loading_model")
    try:
        from PIL import Image
        import io
        dummy = Image.new("RGB", (224, 224))
        buf = io.BytesIO()
        dummy.save(buf, format="PNG")
        _ = classifier.predict_single(buf.getvalue())
        log.info("model_loaded", classes=len(classifier.classes))
    except Exception as e:
        log.error("startup_failed", err=str(e))
        raise


@app.on_event("shutdown")
async def shutdown():
    """Cleanup thread pool."""
    _pool.shutdown(wait=True)
    log.info("shutdown_complete")
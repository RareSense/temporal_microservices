from __future__ import annotations
import asyncio, os, base64
from typing import Any, Dict, List
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import io

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from schemas import RunRequest, RunResponse, MaskResult
from model import grounded_sam
import sys
sys.path.append('..')
from artifact_io import fetch_artifact, upload_artifact

log = structlog.get_logger()

MIN_WORKERS = 1
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))
SCALE_UP_THRESHOLD = 0.8  
SCALE_DOWN_AFTER = 30  

class DynamicWorkerPool:
    def __init__(self, min_workers=MIN_WORKERS, max_workers=MAX_WORKERS):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=min_workers)
        self.current_workers = min_workers
        self.pending_tasks = 0
        self.last_busy_time = asyncio.get_event_loop().time()
        self._lock = asyncio.Lock()
        self._monitor_task = None
    
    async def submit(self, fn, *args):
        async with self._lock:
            self.pending_tasks += 1
            self.last_busy_time = asyncio.get_event_loop().time()
            
            load_ratio = self.pending_tasks / self.current_workers
            if load_ratio > SCALE_UP_THRESHOLD and self.current_workers < self.max_workers:
                new_workers = min(self.current_workers + 2, self.max_workers)
                log.info(f"Scaling up workers: {self.current_workers} -> {new_workers}")
                self.executor._max_workers = new_workers
                self.current_workers = new_workers
        
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, fn, *args)
        finally:
            async with self._lock:
                self.pending_tasks -= 1
    
    async def monitor_and_scale_down(self):
        """Background task to scale down idle workers"""
        while True:
            await asyncio.sleep(10)
            async with self._lock:
                idle_time = asyncio.get_event_loop().time() - self.last_busy_time
                if (self.pending_tasks == 0 and 
                    idle_time > SCALE_DOWN_AFTER and 
                    self.current_workers > self.min_workers):
                    
                    new_workers = max(self.min_workers, self.current_workers // 2)
                    log.info(f"Scaling down workers: {self.current_workers} -> {new_workers}")
                    self.executor._max_workers = new_workers
                    self.current_workers = new_workers
    
    async def start_monitoring(self):
        if not self._monitor_task:
            self._monitor_task = asyncio.create_task(self.monitor_and_scale_down())
    
    async def shutdown(self):
        if self._monitor_task:
            self._monitor_task.cancel()
        self.executor.shutdown(wait=True)

worker_pool = DynamicWorkerPool()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle"""
    log.info("Starting Grounded SAM service...")
    await worker_pool.start_monitoring()
    
    log.info("Pre-loading models...")
    _ = grounded_sam  
    
    yield
    
    log.info("Shutting down...")
    await worker_pool.shutdown()

app = FastAPI(
    title="grounded-sam-service",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

def _find_artifact(node: Any) -> str | None:
    """Recursively find artifact URI in nested structure"""
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

def _find_labels(data: Dict[str, Any]) -> List[str]:
    if "detected_jewelry" in data and isinstance(data["detected_jewelry"], list):
        return data["detected_jewelry"]

    if "labels" in data and isinstance(data["labels"], list):
        return data["labels"]
    
    # Backwards compatibility with old format
    if "jewelry-classify" in data:
        value = data["jewelry-classify"]
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [value]
    
    return []

@app.post("/run")
async def run(req: RunRequest, raw_req: Request):
    try:
        labels = _find_labels(req.data)
        if not labels:
            # If no explicit labels, check if we can infer from data structure
            log.warning("No labels found in input data, will attempt comprehensive detection")
            labels = ["jewelry"]  # Generic fallback
        
        log.info(f"Processing with labels: {labels}")
        log.info(f"Input data keys: {list(req.data.keys())}")
        
        if "image_bytes" in req.data:
            image_bytes = base64.b64decode(req.data["image_bytes"])
        else:
            uri = _find_artifact(req.data)
            if not uri:
                raise HTTPException(422, "No image_bytes or artifact found")
            image_bytes = await fetch_artifact(uri)
        
        def _inference(img_bytes: bytes, labels_list: List[str]):
            return grounded_sam.detect_and_segment(img_bytes, labels_list)
        
        results, overlay = await worker_pool.submit(_inference, image_bytes, labels)
        
        if not results:
            return RunResponse(
                masks=[],
                overlay_artifact=None,
                status=f"No jewelry detected from labels: {', '.join(labels)}"
            )
        
        unified_result = results[0]  

        mask_bytes = io.BytesIO()
        unified_result["mask"].save(mask_bytes, format="PNG")
        mask_artifact = await upload_artifact(
            mask_bytes.getvalue(), 
            "image/png"
        )
        
        mask_result = MaskResult(
            label=unified_result["label"],
            confidence=unified_result["confidence"],
            mask_artifact=mask_artifact,
            bbox=unified_result["bbox"]
        )

        overlay_artifact = None
        if overlay:
            overlay_bytes = io.BytesIO()
            overlay.save(overlay_bytes, format="PNG")
            overlay_artifact = await upload_artifact(
                overlay_bytes.getvalue(),
                "image/png"
            )

        num_objects = unified_result.get("num_objects", 1)
        detected_types = unified_result.get("detected_types", [])
        
        status_msg = f"✅ Found {num_objects} jewelry objects"
        if detected_types:
            status_msg += f" ({', '.join(detected_types)})"
        
        return RunResponse(
            masks=[mask_result], 
            overlay_artifact=overlay_artifact,
            status=status_msg
        )
        
    except HTTPException:
        raise
    except Exception as exc:
        log.error(f"Processing failed: {exc}", exc_info=True)
        raise HTTPException(500, f"Processing failed: {exc}")

@app.get("/health")
async def health():
    return JSONResponse({
        "ok": True,
        "workers": {
            "current": worker_pool.current_workers,
            "min": worker_pool.min_workers,
            "max": worker_pool.max_workers,
            "pending_tasks": worker_pool.pending_tasks
        }
    })

@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics"""
    return f"""# HELP worker_pool_size Current number of workers
# TYPE worker_pool_size gauge
worker_pool_size {worker_pool.current_workers}

# HELP pending_tasks Number of tasks in queue
# TYPE pending_tasks gauge
pending_tasks {worker_pool.pending_tasks}
"""

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "18007")),
        workers=1,  
        loop="asyncio",
        access_log=True,
        reload=os.getenv("ENV", "prod") == "dev"
    )
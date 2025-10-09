from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any
import asyncio, structlog
from .inference import predict

log = structlog.get_logger()
app = FastAPI(title="jewelry-classifier")

class RunRequest(BaseModel):
    data: Any = Field(..., description="Artifact dict or inline data-URL")
    meta: dict[str, Any] | None = None

@app.post("/run")
async def run(req: RunRequest):
    try:
        preds = await asyncio.get_running_loop().run_in_executor(
            None,                                      
            lambda: asyncio.run(predict(req.data)) 
        )
        return {"predictions": preds}
    except Exception as e:
        log.error("inference_err", err=str(e))
        raise HTTPException(500, "inference_failed")

@app.get("/health")
async def h():
    from .model import get_model
    try:
        get_model()
        return {"ok": True}
    except Exception as e:               
        log.error("health_err", err=str(e))
        raise HTTPException(503, "model_load_error")

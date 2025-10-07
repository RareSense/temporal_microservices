import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from .config import settings
from .schemas import (
    ClassifyRequest, 
    ClassifyResponse,
    HealthResponse
)
from .inference import init_engine, get_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint_path = Path(settings.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path.absolute()}\n"
            f"Set CHECKPOINT_PATH in .env or environment variable"
        )
    
    print(f"🔄 Loading model from {checkpoint_path}...")
    init_engine(str(checkpoint_path), settings.device)
    engine = get_engine()
    print(f"✅ Model loaded successfully")
    print(f"   Device: {settings.device}")
    print(f"   Classes: {engine.num_classes}")
    print(f"   Default threshold: {settings.default_threshold}")
    
    yield 
    
    print("🛑 Shutting down gracefully...")


app = FastAPI(
    title="Vision Classifier Service",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    engine = get_engine()
    return HealthResponse(
        status="healthy",
        device=settings.device,
        num_classes=engine.num_classes,
        classes=engine.get_class_names()
    )


@app.post("/run", response_model=ClassifyResponse, status_code=status.HTTP_200_OK)
async def classify_image(request: ClassifyRequest):
    try:
        image_b64 = request.data.get("image")
        if not image_b64:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing 'image' field in data object"
            )
        threshold = request.meta.get("threshold", settings.default_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Threshold must be between 0.0 and 1.0, got {threshold}"
            )

        engine = get_engine()
        predictions = await asyncio.to_thread(
            engine.predict,
            image_b64,
            threshold
        )
        
        detected_jewelry = [p["class"] for p in predictions]
        
        return ClassifyResponse(detected_jewelry=detected_jewelry)
        
    except HTTPException:
        raise 
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "type": type(exc).__name__
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        workers=settings.max_workers,
        log_level="info"
    )
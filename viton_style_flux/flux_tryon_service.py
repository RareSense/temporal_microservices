"""
Flux Try-On Service Wrapper for Temporal Pipeline
Handles artifact management and integrates with existing Flux Try-On API
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

import sys
sys.path.append('..')
from artifact_io import fetch_artifact, upload_artifact

# ===================== Configuration =====================
class Config:
    FLUX_TRYON_URL = "http://localhost:18010"  # Flux API
    SERVICE_PORT = 18008  # wrapper service port
    HTTP_TIMEOUT = 300
    LOG_LEVEL = logging.INFO

config = Config()

# ===================== Logging Setup =====================
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================== Request/Response Models =====================
class Payload(BaseModel):
    data: Dict[str, Any]
    meta: Dict[str, Any] = Field(default_factory=dict)

class TryOnServiceRequest(BaseModel):
    """Expected input from Temporal workflow"""
    # From grounded-sam
    masks: List[Dict[str, Any]] 
    
    # From jewelry-classify  
    detected_jewelry: List[str]
    
    # From root (original image)
    image: Optional[Dict[str, Any]] = None  
    
    # Additional fields that may be present
    class Config:
        extra = "allow"  # Allow additional fields

# ===================== Service Implementation =====================
class FluxTryOnWrapper:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=config.HTTP_TIMEOUT)
        
    async def process_tryon(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process try-on request by:
        1. Extracting image and mask from artifacts
        2. Determining jewelry type and getting garment
        3. Calling Flux Try-On API
        4. Uploading results as artifacts
        """
        try:
            # Extract the original image artifact
            image_artifact = data.get("image") or data.get("original_path")
            if not image_artifact or "uri" not in image_artifact:
                raise ValueError("Missing image artifact in request")
            
            # Download original image
            logger.info(f"Fetching original image from {image_artifact['uri']}")
            image_bytes = await fetch_artifact(image_artifact["uri"])
            image_b64 = base64.b64encode(image_bytes).decode()
            
            # Extract masks from grounded-sam results
            masks = data.get("masks") or data.get("grounded-sam", {}).get("masks", [])
            if not masks:
                raise ValueError("No masks found in request")

            first_mask = masks[0]
            mask_artifact = first_mask.get("mask_artifact")
            if not mask_artifact or "uri" not in mask_artifact:
                raise ValueError("No mask artifact found")
            
            logger.info(f"Fetching mask from {mask_artifact['uri']}")
            mask_bytes = await fetch_artifact(mask_artifact["uri"])
            mask_b64 = base64.b64encode(mask_bytes).decode()
            
            # Get jewelry type from detected_jewelry
            detected_jewelry = data.get("detected_jewelry", [])
            jewelry_type = detected_jewelry[0] if detected_jewelry else "necklace"
            logger.info(f"Processing try-on for jewelry type: {jewelry_type}")
            
            # Placeholder values for zoom and skin 
            zoom_level = "bust shot"  # TODO: Replace with zoom classifier service
            skin_shade = "medium"     # TODO: Replace with skin shade detector service
            
            # Prepare request for Flux Try-On API
            flux_request = {
                "image": image_b64,
                "mask": mask_b64,
                "num_variations": 1,
                "use_library": True,
                "zoom_level_override": zoom_level,
                "skin_shade_override": skin_shade,
                "jewelry_type_override": jewelry_type,
                "prompt": f"Two-panel image showing a person wearing {jewelry_type}",
                "size": [768, 1024],
                "num_steps": 30,
                "guidance_scale": 30.0
            }
            
            # Call Flux Try-On API
            logger.info("Calling Flux Try-On API...")
            response = await self.client.post(
                f"{config.FLUX_TRYON_URL}/tryon",
                json=flux_request
            )
            response.raise_for_status()
            result = response.json()
            
            # Process and upload results as artifacts
            output_artifacts = []
            
            # Upload variations
            for idx, variation_b64 in enumerate(result.get("variations", [])):
                variation_bytes = base64.b64decode(variation_b64)
                artifact = await upload_artifact(variation_bytes, mime="image/png")
                output_artifacts.append({
                    "tryon_result": artifact,
                    "variation_index": idx,
                    "jewelry_type": jewelry_type
                })
                logger.info(f"Uploaded variation {idx} as artifact: {artifact['uri']}")
            
            # Upload ghost image if present
            ghost_artifact = None
            if "ghost_image" in result:
                ghost_bytes = base64.b64decode(result["ghost_image"])
                ghost_artifact = await upload_artifact(ghost_bytes, mime="image/png")
                logger.info(f"Uploaded ghost image as artifact: {ghost_artifact['uri']}")
            
            # Return structured output
            return {
                "status": "success",
                "jewelry_type": jewelry_type,
                "zoom_level": zoom_level,
                "skin_shade": skin_shade,
                "tryon_results": output_artifacts,
                "ghost_image": ghost_artifact,
                "library_match": result.get("library_match"),
                "processing_time": result.get("processing_time", 0),
                "message": f"Successfully generated try-on for {jewelry_type}"
            }
            
        except Exception as e:
            logger.error(f"Error processing try-on: {e}")
            raise

# ===================== FastAPI Application =====================
app = FastAPI(
    title="Flux Try-On Temporal Service",
    description="Wrapper service for Flux Try-On integration with Temporal pipeline",
    version="1.0.0"
)

wrapper = FluxTryOnWrapper()

@app.post("/run")
async def process_tryon(request: Request):
    """
    Process try-on request from Temporal workflow
    Expects Payload format with data containing image, masks, and detected_jewelry
    """
    try:
        body = await request.json()
        
        # Handle Temporal Payload structure
        if "data" in body and "meta" in body:
            data = body["data"]
            meta = body.get("meta", {})
        else:
            data = body
            meta = {}
        
        logger.info(f"Processing try-on request with meta: {meta}")
        
        # Process the try-on
        result = await wrapper.process_tryon(data)
        
        # Return result maintaining Temporal structure if needed
        if "data" in body:
            return {"data": result, "meta": meta}
        else:
            return result
            
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        logger.error(f"Flux API error: {e}")
        raise HTTPException(status_code=502, detail=f"Flux API error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint for service registry"""
    # Check if Flux API is reachable
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{config.FLUX_TRYON_URL}/health")
            flux_healthy = response.status_code == 200
    except:
        flux_healthy = False
    
    return {
        "status": "healthy" if flux_healthy else "degraded",
        "service": "flux-tryon-wrapper",
        "flux_api": "healthy" if flux_healthy else "unhealthy"
    }

# ===================== Cleanup =====================
@app.on_event("shutdown")
async def shutdown():
    await wrapper.client.aclose()

# ===================== Main Entry Point =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.SERVICE_PORT,
        log_level="info"
    )
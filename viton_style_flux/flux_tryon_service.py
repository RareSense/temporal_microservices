"""
Flux Try-On Service Wrapper for Temporal Pipeline – v1.3

v1.3 ─ Emits **one dictionary per variation** so the orchestrator can
        fan-out down-stream tools (e.g. upscaler).  No other behaviour
        changed.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from PIL import Image

# Import shared artifact management utilities
import sys
sys.path.append('/home/nimra/temporal_microservices')
from artifact_io import fetch_artifact, upload_artifact  # type: ignore

# ────────────────────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────────────────────
class Config:
    FLUX_TRYON_URL = "http://localhost:18010"  # Underlying Flux API
    SERVICE_PORT   = 18008                     # Wrapper port
    HTTP_TIMEOUT   = 300
    LOG_LEVEL      = logging.DEBUG

config = Config()

# ────────────────────────────────────────────────────────────────
#  Logging
# ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
#  Pydantic models
# ────────────────────────────────────────────────────────────────
class Payload(BaseModel):
    data: Dict[str, Any]
    meta: Dict[str, Any] = Field(default_factory=dict)

# ────────────────────────────────────────────────────────────────
#  Core wrapper class
# ────────────────────────────────────────────────────────────────
class FluxTryOnWrapper:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=config.HTTP_TIMEOUT)

    # -------------------- helpers -------------------- #
    @staticmethod
    def _map_zoom_level(zoom_classifier_output: str) -> str:
        """Map zoom classifier output to Flux library zoom levels."""
        return {
            "zoom_1": "macro closeup shot",
            "zoom_2": "tight detail shot",
            "zoom_3": "bust shot",
            "zoom_4": "three quarter shot",
        }.get(zoom_classifier_output, "bust shot")

    @staticmethod
    def _select_best_jewelry_type(detected_jewelry: List[str]) -> str:
        """Priority: necklace > bracelet > ring > earring > watch."""
        priority_order = ["necklace", "bracelet", "ring", "earring", "watch"]
        for jt in priority_order:
            if jt in detected_jewelry:
                return jt
        return detected_jewelry[0] if detected_jewelry else "necklace"

    @staticmethod
    def _map_skin_tone(skin_tone: str) -> str:
        """Map skin classifier output to Flux library categories."""
        mapping = {
            "fair": "light",
            "light": "light",
            "medium": "medium",
            "tan": "medium",
            "latino_hispanic_light-medium": "light",
            "caucasian_light-medium": "dark",
            "east_asian_fair": "darkest",
            "dark": "dark",
            "deep": "deep dark",
            "deep dark": "deep dark",
            "darkest": "darkest",
        }
        return mapping.get(skin_tone.lower(), "medium")

    # -------------------- main entry -------------------- #
    async def process_tryon(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract inputs, call Flux API, upload results.
        Returns **list[dict]** – one envelope per generated variation.
        """

        # ───── Deep introspection for debug ─────
        logger.info("=" * 70)
        logger.info("📥 INCOMING DATA STRUCTURE:")
        logger.info(json.dumps(data, indent=2, default=str)[:1200])

        # 1️⃣ Locate image artifact
        image_art = data.get("image") or data.get("original_path")
        if not image_art or "uri" not in image_art:
            raise ValueError("Missing image artifact in request")
        image_bytes = await fetch_artifact(image_art["uri"])
        image_b64 = base64.b64encode(image_bytes).decode()

        # 2️⃣ Locate masks
        masks = (
            data.get("masks")
            or data.get("grounded-sam", {}).get("masks")
            or (
                data.get("grounded-sam", [{}])[0]
                if isinstance(data.get("grounded-sam"), list)
                else {}
            ).get("masks", [])
        )
        if not masks:
            raise ValueError("No masks found in request")
        mask_artifact = masks[0].get("mask_artifact")
        if not mask_artifact or "uri" not in mask_artifact:
            raise ValueError("No mask artifact found")
        mask_bytes = await fetch_artifact(mask_artifact["uri"])
        mask_b64 = base64.b64encode(mask_bytes).decode()

        # 3️⃣ Extract detected jewelry
        detected_jewelry = (
            data.get("detected_jewelry")
            or data.get("jewelry-classify", {}).get("detected_jewelry")
            or data.get("jewelry-classify", {}).get("data", {}).get(
                "detected_jewelry"
            )
            or (
                (
                    data.get("jewelry-classify", [{}])[0]
                    if isinstance(data.get("jewelry-classify"), list)
                    else {}
                ).get("detected_jewelry", [])
            )
        )
        if not isinstance(detected_jewelry, list):
            detected_jewelry = [detected_jewelry] if detected_jewelry else []
        jewelry_type = self._select_best_jewelry_type(detected_jewelry)

        # 4️⃣ Extract zoom level
        zoom_classifier_output = (
            data.get("zoom_level")
            or data.get("zoom-classifier", {}).get("zoom_level")
            or data.get("zoom-classifier", {}).get("data", {}).get("zoom_level")
            or (
                data.get("zoom-classifier", [{}])[0]
                if isinstance(data.get("zoom-classifier"), list)
                else {}
            ).get("zoom_level")
        )
        zoom_level = (
            self._map_zoom_level(zoom_classifier_output)
            if zoom_classifier_output
            else "bust shot"
        )

        # 5️⃣ Extract skin tone
        skin_tone_raw = (
            data.get("skin_tone")
            or data.get("skin-tone", {}).get("skin_tone")
            or data.get("skin-tone", {}).get("data", {}).get("skin_tone")
            or (
                data.get("skin-tone", [{}])[0]
                if isinstance(data.get("skin-tone"), list)
                else {}
            ).get("skin_tone")
            or "medium"
        )
        skin_shade = self._map_skin_tone(skin_tone_raw)

        # 6️⃣ Prepare Flux request
        num_vars = int(data.get("num_variations", 1))
        flux_request = {
            "image": image_b64,
            "mask": mask_b64,
            "num_variations": num_vars,
            "use_library": True,
            "zoom_level_override": zoom_level,
            "skin_shade_override": skin_shade,
            "jewelry_type_override": jewelry_type,
            "prompt": f"Two-panel image showing a person wearing {jewelry_type}",
            "size": [768, 1024],
            "num_steps": 30,
            "guidance_scale": 30.0,
        }
        logger.info(
            "Flux request params",
            extra=dict(zoom=zoom_level, skin=skin_shade, jewelry=jewelry_type),
        )

        # 7️⃣ Call Flux API
        res = await self.client.post(
            f"{config.FLUX_TRYON_URL}/tryon", json=flux_request
        )
        res.raise_for_status()
        flux_out = res.json()

        # 8️⃣ Upload variations + ghost, emit list
        outputs: List[Dict[str, Any]] = []
        ghost_art = None
        if "ghost_image" in flux_out:
            ghost_art = await upload_artifact(
                base64.b64decode(flux_out["ghost_image"]), mime="image/png"
            )

        for idx, var_b64 in enumerate(flux_out.get("variations", [])[:num_vars]):
            var_art = await upload_artifact(
                base64.b64decode(var_b64), mime="image/png"
            )
            outputs.append(
                {
                    "status": "success",
                    "jewelry_type": jewelry_type,
                    "detected_jewelry": detected_jewelry,
                    "zoom_level": zoom_level,
                    "zoom_classifier_output": zoom_classifier_output,
                    "skin_shade": skin_shade,
                    "skin_tone_raw": skin_tone_raw,
                    "tryon_results": [
                        {
                            "tryon_result": var_art,
                            "variation_index": idx,
                            "jewelry_type": jewelry_type,
                        }
                    ],
                    "ghost_image": ghost_art,
                    "library_match": flux_out.get("library_match"),
                    "processing_time": flux_out.get("processing_time", 0),
                    "message": (
                        f"Successfully generated try-on v{idx} for "
                        f"{jewelry_type} at {zoom_level} with {skin_shade} skin tone"
                    ),
                }
            )

        return outputs

# ────────────────────────────────────────────────────────────────
#  FastAPI application
# ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Flux Try-On Temporal Service",
    version="1.3.0",
    description="Wrapper with skin-tone integration",
)
wrapper = FluxTryOnWrapper()


@app.post("/run")
async def run(request: Request):
    body = await request.json()
    data = body["data"] if "data" in body else body
    try:
        return await wrapper.process_tryon(data)
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            res = await c.get(f"{config.FLUX_TRYON_URL}/health")
            status = "healthy" if res.status_code == 200 else "degraded"
    except Exception:
        status = "degraded"
    return {
        "status": status,
        "service": "flux-tryon-wrapper",
        "version": "1.3.0",
        "features": [
            "skin-tone-integration",
            "zoom-mapping",
            "jewelry-priority",
            "list-per-variation",
        ],
    }


@app.on_event("shutdown")
async def _shutdown():
    await wrapper.client.aclose()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host="0.0.0.0", port=config.SERVICE_PORT, log_level="debug"
    )

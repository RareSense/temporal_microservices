"""
Flux Try-On Service Wrapper for Temporal Pipeline – v1.4

v1.4 ─ Conditional library usage:
        • If either *zoom_level* **or** *skin_tone* is missing in the incoming
          data, the wrapper no longer asks Flux to pull a garment from its
          library. Instead, it calls **Gemini 2.5 Flash (image preview)** to
          generate an appropriate garment image on‑the‑fly, runs basic RGB/
          crop/resize sanitation, and sends that image to Flux with
          `use_library=False`.
        • All other behaviour remains **unchanged**.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from itertools import cycle
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from PIL import Image, ImageOps

# Google Gemini
from google import genai
from google.genai import types

# Import shared artifact management utilities
import sys
sys.path.append('/home/nimra/temporal_microservices')
from artifact_io import fetch_artifact, upload_artifact  # type: ignore

# ────────────────────────────────────────────────────────────────
#  Gemini helpers
# ────────────────────────────────────────────────────────────────
_API_KEYS = [
    "AIzaSyCVMTLdOX1xMI7Au4VJXrniZWMYs7H1-5I",
    "AIzaSyAaRID7ZHt5Z-rW0leZlvGEoEn7jaJmZ2o",
]
_KEY_CYCLE = cycle(_API_KEYS)
_MODEL_NAME = "gemini-2.5-flash-image-preview"
_PROMPT = (
    """
    Remove all jewelry and regenerate the image while keeping the same zoom level and gender, 
    professional young model, slim model, natural skin texture. Create a different face, pose, different outfit in differnt color, 
    and hairstyle, no jewelry, no unrealistic distortion, and place the subject in a professional studio background.
    """
)
_MAX_RETRIES = 10


def _safe_extract_image(resp) -> Optional[Image.Image]:
    """Extract `PIL.Image` from Gemini response (or *None*)."""
    try:
        if not resp or not getattr(resp, "candidates", None):
            return None
        parts = getattr(resp.candidates[0].content, "parts", None)
        if not parts:
            return None
        for part in parts:
            if getattr(part, "inline_data", None):
                try:
                    return Image.open(BytesIO(part.inline_data.data))
                except Exception:
                    continue
        return None
    except Exception:
        return None


def _gemini_generate(image: Image.Image) -> Image.Image:
    """Generate a garment image via Gemini.

    Raises `RuntimeError` if generation fails after retries.
    """
    last_err: str | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        api_key = next(_KEY_CYCLE)
        try:
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model=_MODEL_NAME,
                contents=[_PROMPT, image],
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"]
                ),
            )
        except Exception as e:  # noqa: BLE001
            last_err = f"API error: {e}"
            logging.warning("Gemini attempt %d failed (%s) – retrying…", attempt, api_key)
            time.sleep(min(attempt, 8))
            continue

        out_img = _safe_extract_image(resp)
        if out_img is not None:
            logging.info("Gemini succeeded on attempt %d via %s", attempt, api_key)
            return out_img

        time.sleep(min(attempt, 8))

    raise RuntimeError(f"Gemini generation failed: {last_err}")


# ────────────────────────────────────────────────────────────────
#  Basic image utilities
# ────────────────────────────────────────────────────────────────

def _pil_to_rgb(img: Image.Image) -> Image.Image:
    """Ensure output is **RGB**, stripping alpha if present."""
    if img.mode == "RGB":
        return img
    if img.mode in {"RGBA", "LA"}:
        return img.convert("RGBA").convert("RGB")
    return img.convert("RGB")


def _center_crop_and_resize(img: Image.Image, size: tuple[int, int] = (768, 1024)) -> Image.Image:
    """Center‑crop (if necessary) and resize to *size*."""
    width, height = img.size
    tgt_w, tgt_h = size
    left = (width - tgt_w) // 2 if width > tgt_w else 0
    top = (height - tgt_h) // 2 if height > tgt_h else 0
    right = left + min(width, tgt_w)
    bottom = top + min(height, tgt_h)
    img_cropped = img.crop((left, top, right, bottom))
    return img_cropped.resize(size, Image.LANCZOS)


# ────────────────────────────────────────────────────────────────
#  Configuration & logging
# ────────────────────────────────────────────────────────────────
class Config:
    FLUX_TRYON_URL = "http://localhost:18010"  # Underlying Flux API
    SERVICE_PORT = 18008  # Wrapper port
    HTTP_TIMEOUT = 300
    LOG_LEVEL = logging.DEBUG


auto_cfg = Config()

logging.basicConfig(
    level=auto_cfg.LOG_LEVEL,
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
        self.client = httpx.AsyncClient(timeout=auto_cfg.HTTP_TIMEOUT)

    # -------------------- helpers -------------------- #
    @staticmethod
    def _map_zoom_level(zoom_classifier_output: str | None) -> str:
        return {
            "zoom_1": "macro closeup shot",
            "zoom_2": "tight detail shot",
            "zoom_3": "bust shot",
            "zoom_4": "three quarter shot",
        }.get(zoom_classifier_output or "", "bust shot")

    @staticmethod
    def _select_best_jewelry_type(detected_jewelry: List[str]) -> str:
        priority = ["necklace", "bracelet", "ring", "earring", "watch"]
        for jt in priority:
            if jt in detected_jewelry:
                return jt
        return detected_jewelry[0] if detected_jewelry else "necklace"

    @staticmethod
    def _map_skin_tone(skin_tone: str | None) -> str:
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
        return mapping.get((skin_tone or "").lower(), "medium")

    # -------------------- main entry -------------------- #
    async def process_tryon(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract inputs, decide library vs Gemini, call Flux, upload results
        """
        logger.info("=" * 70)
        logger.info("📥 INCOMING DATA STRUCTURE:")
        logger.info(json.dumps(data, indent=2, default=str)[:1200])

        # 1️⃣ Locate image artifact
        image_art = data.get("image") or data.get("original_path")
        if not image_art or "uri" not in image_art:
            raise ValueError("Missing image artifact in request")
        image_bytes = await fetch_artifact(image_art["uri"])
        orig_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
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

        # 3️⃣ Jewelry detection
        detected_jewelry = (
            data.get("detected_jewelry")
            or data.get("jewelry-classify", {}).get("detected_jewelry")
            or data.get("jewelry-classify", {}).get("data", {}).get("detected_jewelry")
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

        # 4️⃣ Zoom level extraction (capture presence before fallback)
        zoom_raw = (
            data.get("zoom_level")
            or data.get("zoom-classifier", {}).get("zoom_level")
            or data.get("zoom-classifier", {}).get("data", {}).get("zoom_level")
            or (
                data.get("zoom-classifier", [{}])[0]
                if isinstance(data.get("zoom-classifier"), list)
                else {}
            ).get("zoom_level")
        )
        zoom_present = zoom_raw is not None
        zoom_level = self._map_zoom_level(zoom_raw) if zoom_present else None

        # 5️⃣ Skin‑tone extraction
        skin_raw = (
            data.get("skin_tone")
            or data.get("skin-tone", {}).get("skin_tone")
            or data.get("skin-tone", {}).get("data", {}).get("skin_tone")
            or (
                data.get("skin-tone", [{}])[0]
                if isinstance(data.get("skin-tone"), list)
                else {}
            ).get("skin_tone")
        )
        skin_present = skin_raw is not None
        skin_shade = self._map_skin_tone(skin_raw) if skin_present else None

        # 6️⃣ Decide: library vs Gemini
        garment_b64: str | None = None
        provided_garment = data.get("garment")
        if isinstance(provided_garment, dict) and "uri" in provided_garment:
            g_bytes = await fetch_artifact(provided_garment["uri"])
            garment_b64 = base64.b64encode(g_bytes).decode()

        # Decide: library vs Gemini, but skip Gemini if garment already provided
        use_library = (zoom_present and skin_present and garment_b64 is None)

        if not use_library:
            if garment_b64:
                logger.info("Using garment provided by caller – skipping Gemini.")
                # Provide safe defaults if zoom/skin missing
                if zoom_level is None:
                    zoom_level = "bust shot"
                if skin_shade is None:
                    skin_shade = "medium"
            else:
                logger.info("Missing zoom or skin and no garment provided – using Gemini to generate garment image…")
                gemini_img = await asyncio.get_event_loop().run_in_executor(
                    None, _gemini_generate, orig_pil
                )
                gemini_img = _center_crop_and_resize(_pil_to_rgb(gemini_img))
                buffered = BytesIO()
                gemini_img.save(buffered, format="PNG", quality=95)
                garment_b64 = base64.b64encode(buffered.getvalue()).decode()
                if zoom_level is None:
                    zoom_level = "bust shot"
                if skin_shade is None:
                    skin_shade = "medium"
        else:
            logger.info(
                "Using library garment (zoom=%s, skin=%s, jewelry=%s)",
                zoom_level, skin_shade, jewelry_type
            )

        # 7️⃣ Prepare Flux request
        num_vars = int(data.get("num_variations", 1))
        flux_req: Dict[str, Any] = {
            "image": image_b64,
            "mask": mask_b64,
            "num_variations": num_vars,
            "use_library": use_library,
            "zoom_level_override": zoom_level,
            "skin_shade_override": skin_shade,
            "jewelry_type_override": jewelry_type,
            "prompt": f"Two-panel image showing a person wearing {jewelry_type}",
            "size": [768, 1024],
            "num_steps": 30,
            "guidance_scale": 30.0,
        }
        if garment_b64:
            flux_req["garment"] = garment_b64

        logger.info(
            "Flux request prepared (use_library=%s)", use_library, extra=dict(zoom=zoom_level, skin=skin_shade)
        )

        # 8️⃣ Call Flux API
        res = await self.client.post(f"{auto_cfg.FLUX_TRYON_URL}/tryon", json=flux_req)
        res.raise_for_status()
        flux_out = res.json()

        import hashlib, base64
        def _h10(b: bytes) -> str:
            return hashlib.sha256(b).hexdigest()[:10]

        try:
            orig_hash = _h10(image_bytes)
            v0 = flux_out.get("variations", [])
            v0_hash = _h10(base64.b64decode(v0[0])) if v0 else "NONE"
            logger.info(f"[WRAPPER] hash input={orig_hash} var0={v0_hash} count={len(v0)}")
        except Exception as _e:
            logger.warning(f"[WRAPPER] hash check failed: {_e}")

        # 9️⃣ Upload variations + ghost, emit list
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
                    "zoom_classifier_output": zoom_raw,
                    "skin_shade": skin_shade,
                    "skin_tone_raw": skin_raw,
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
                        f"Generated try‑on v{idx} for {jewelry_type} at {zoom_level} "
                        f"with {skin_shade} skin tone (library={use_library})"
                    ),
                }
            )

        return outputs

# ────────────────────────────────────────────────────────────────
#  FastAPI application
# ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Flux Try‑On Temporal Service",
    version="1.4.0",
    description="Wrapper with conditional library selection & Gemini fallback",
)
wrapper = FluxTryOnWrapper()


@app.post("/run")
async def run(request: Request):
    body = await request.json()
    data = body["data"] if "data" in body else body
    try:
        return await wrapper.process_tryon(data)
    except Exception as e:  # noqa: BLE001
        logger.error("Processing error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            res = await c.get(f"{auto_cfg.FLUX_TRYON_URL}/health")
            status = "healthy" if res.status_code == 200 else "degraded"
    except Exception:
        status = "degraded"
    return {
        "status": status,
        "service": "flux-tryon-wrapper",
        "version": "1.4.0",
        "features": [
            "skin-tone-integration",
            "zoom-mapping",
            "jewelry-priority",
            "gemini-garment-fallback",
            "list-per-variation",
        ],
    }


@app.on_event("shutdown")
async def _shutdown() -> None:
    await wrapper.client.aclose()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("flux_tryon_service:app", host="0.0.0.0", port=auto_cfg.SERVICE_PORT, log_level="debug")

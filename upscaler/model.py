from __future__ import annotations

import io
import threading
import base64
from typing import Tuple

from PIL import Image
from utils import intelligent_upscale_and_resize, logger 


class _Singleton(type):
    _inst = None
    _lock = threading.Lock()

    def __call__(cls, *a, **kw):
        with cls._lock:
            if cls._inst is None:
                cls._inst = super().__call__(*a, **kw)
        return cls._inst


class Upscaler(metaclass=_Singleton):
    """Stateless helper housing the upscale/composite routine."""

    def process(self, bg_bytes: bytes, ghost_bytes: bytes) -> str:
        """
        • Decode background & ghost images
        • Intelligently upscale background (SwinIR on CPU) until ≥ ghost size
        • Resize down (if needed) to **exact ghost size**
        • Alpha-composite ghost over background
        • Return final JPG (RGB) as base64
        """
        # Decode images
        bg_pil    = Image.open(io.BytesIO(bg_bytes)).convert("RGB")
        ghost_pil = Image.open(io.BytesIO(ghost_bytes)).convert("RGBA")

        target_size: Tuple[int, int] = ghost_pil.size
        logger.info(f"Upscaler: target_size={target_size}, "
                    f"bg_original={bg_pil.size}")

        # Upscale → resize
        bg_upscaled = intelligent_upscale_and_resize(bg_pil, target_size)

        # Composite
        if bg_upscaled.mode != "RGBA":
            bg_upscaled = bg_upscaled.convert("RGBA")
        final = Image.alpha_composite(bg_upscaled, ghost_pil).convert("RGB")

        buf = io.BytesIO()
        final.save(buf, format="JPEG", quality=95)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return encoded

upscaler = Upscaler()

from __future__ import annotations
from typing import Any, Dict

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    data: Dict[str, Any]


class RunResponse(BaseModel):
    composite_b64: str = Field(..., description="Final upscaled composite image (base64-encoded)")

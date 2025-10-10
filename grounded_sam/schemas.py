# grounded_sam_service/schemas.py
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

class RunRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Input data with labels and optional image")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Metadata")

class MaskResult(BaseModel):
    label: str
    confidence: float
    mask_artifact: Dict[str, Any]  # Artifact dict for mask image
    bbox: List[float]  # [x1, y1, x2, y2]

class RunResponse(BaseModel):
    masks: List[MaskResult]
    overlay_artifact: Optional[Dict[str, Any]] = None
    status: str
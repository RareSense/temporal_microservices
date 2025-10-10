from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

class RunRequest(BaseModel):
    """Input schema matching Temporal workflow format."""
    data: Dict[str, Any] = Field(
        ...,
        description="Contains image as 'image_bytes' (base64) or 'artifact' (Azure URI)"
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata like trace_id, request_id, etc."
    )

class RunResponse(BaseModel):
    """Output schema for zoom classification."""
    zoom_level: str = Field(
        ...,
        description="Zoom level: zoom_1, zoom_2, zoom_3, or zoom_4",
        pattern="^zoom_[1-4]$"
    )
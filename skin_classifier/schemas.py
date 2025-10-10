from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

class RunRequest(BaseModel):
    data: Dict[str, Any] = Field(
        ..., 
        description="Input data containing image (base64/artifact/uri)"
    )
    meta: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional metadata"
    )

class SkinTonePrediction(BaseModel):
    class_: str = Field(..., alias="class")
    confidence: float

class RunResponse(BaseModel):
    skin_tone: str = Field(..., description="Top-1 skin tone prediction")
    skin_tone_predictions: List[SkinTonePrediction] = Field(
        ..., 
        description="Top-k predictions with confidence scores"
    )
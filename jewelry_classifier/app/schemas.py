"""Request/Response schemas"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any


class ClassifyRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Must contain 'image' key with base64 string")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata")


class ClassifyResponse(BaseModel):
    detected_jewelry: List[str] = Field(..., description="List of detected jewelry names")


class HealthResponse(BaseModel):
    status: str
    device: str
    num_classes: int
    classes: List[str]
from pydantic import BaseModel, Field
from typing import Any, Dict

class RunRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Either {'image_bytes': <b64>}"
                                               " or {'artifact': <azure://...>}")

class RunResponse(BaseModel):
    predictions: list[str]

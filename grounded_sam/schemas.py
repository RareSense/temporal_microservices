# grounded_sam_segmenter/schemas.py
from typing import Any, Dict, List
from pydantic import BaseModel, Field

class RunRequest(BaseModel):
    data: Dict[str, Any]

class RunResponse(BaseModel):
    labels: List[str]
    artifacts: Dict[str, Dict[str, Any]]  # label → Artifact envelope

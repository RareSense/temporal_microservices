from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):

    checkpoint_path: str = Field(
        default="./best_model.pth",
        description="Path to model checkpoint"
    )
    device: str = Field(
        default="cpu",
        description="Device for inference (cpu/cuda)"
    )
    default_threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Default confidence threshold"
    )
    max_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of uvicorn workers"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
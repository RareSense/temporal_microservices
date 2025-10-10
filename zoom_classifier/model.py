from __future__ import annotations

import threading
import torch
import open_clip
from PIL import Image
from pathlib import Path
import os


CHECKPOINT_PATH = Path(
    os.getenv("ZOOM_CKPT", Path(__file__).parent.parent / "weights" / "zoom_classifier_best_97.pt")
)
DEVICE = "cpu" 

class _Singleton(type):
    _inst = None
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls._inst is None:
                cls._inst = super().__call__(*args, **kwargs)
        return cls._inst


class ZoomClassifier(metaclass=_Singleton):
    """
    Singleton zoom classifier - loads once, shared across all workers in process.
    Returns zoom level as string: "zoom_1", "zoom_2", "zoom_3", or "zoom_4"
    """
    
    def __init__(self):
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {CHECKPOINT_PATH}. "
                f"Set ZOOM_CKPT env var or check path."
            )
        
        print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
        
        # Load checkpoint
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        
        # Create model and transforms
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        
        # Classification head
        self.head = torch.nn.Linear(self.model.visual.output_dim, 4)
        
        # Load weights
        self.model.load_state_dict(ckpt["backbone"])
        self.head.load_state_dict(ckpt["head"])
        
        # Set to eval mode
        self.model.eval()
        self.head.eval()
        
        # Move to device
        self.model.to(DEVICE)
        self.head.to(DEVICE)
        
        print(f"Model loaded successfully")
    
    @torch.inference_mode()
    def predict(self, img_bytes: bytes) -> str:
        """
        Predict zoom level from image bytes.
        Returns: "zoom_1", "zoom_2", "zoom_3", or "zoom_4"
        """
        # Load and preprocess image
        from io import BytesIO
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img_tensor = self.preprocess(img).unsqueeze(0).to(DEVICE)
        
        # Forward pass
        features = self.model.encode_image(img_tensor)
        logits = self.head(features)
        
        # Get prediction (1-indexed)
        zoom_level = logits.argmax(1).item() + 1
        
        return f"zoom_{zoom_level}"

zoom_classifier = ZoomClassifier()
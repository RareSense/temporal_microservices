from __future__ import annotations

import io
import os
import threading
from typing import List, Tuple
from pathlib import Path

import torch
import open_clip
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent      
DEFAULT_CKPT = ROOT / "weights" / "skin_tone_classifier_best.pt"

CHECKPOINT_PATH = Path(os.getenv("SKIN_TONE_MODEL_PATH", DEFAULT_CKPT))

TOP_K = int(os.getenv("SKIN_TONE_TOP_K", "3"))
DEVICE = "cpu"


class _Singleton(type):
    _inst = None
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls._inst is None:
                cls._inst = super().__call__(*args, **kwargs)
        return cls._inst


class SkinToneClassifier(metaclass=_Singleton):
    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model.to(DEVICE)
        self.model.eval()

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        self.model.load_state_dict(checkpoint["backbone"])
        
        self.classes = checkpoint["classes"]
        self.head = torch.nn.Linear(
            self.model.visual.output_dim, 
            len(self.classes)
        ).to(DEVICE)
        self.head.load_state_dict(checkpoint["head"])
        self.head.eval()

    @torch.inference_mode()
    def predict(self, img_bytes: bytes) -> List[Tuple[str, float]]:
        """
        Returns top-k skin tone predictions.
        Format: [(class_name, confidence), ...]
        """
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = self.preprocess(img).unsqueeze(0).to(DEVICE)
        
        feats = self.model.encode_image(img_tensor)
        logits = self.head(feats)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        
        topk_probs, topk_idx = torch.topk(probs, min(TOP_K, len(self.classes)))
        
        results = []
        for prob, idx in zip(topk_probs, topk_idx):
            class_name = self.classes[idx.item()]
            confidence = prob.item()
            results.append((class_name, confidence))
        
        return results

    @torch.inference_mode()
    def predict_single(self, img_bytes: bytes) -> str:
        """Returns just the top-1 prediction class name."""
        predictions = self.predict(img_bytes)
        return predictions[0][0] if predictions else "unknown"


classifier = SkinToneClassifier()
from __future__ import annotations

import io
import os
import sys
import threading
import types
from pathlib import Path
from typing import List

import torch
import timm
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).parent.parent
CHECKPOINT_PATH = Path(
     os.getenv("JC_KPT", Path(__file__).parent.parent / "weights" / "best_model.pth")
    )
IMG_SIZE = 384
THRESHOLD = float(os.environ.get("JC_THRESHOLD", 0.55))
DEVICE = "cpu" 


class TrainingConfig:
    """Dummy holder – attributes are restored by torch.load."""
    def __init__(self, **kw):  
        for k, v in kw.items():
            setattr(self, k, v)


class LabelEncoder:
    """Multi-label encoder/decoder compatible with the checkpoint."""
    def __init__(self, all_labels):
        self.classes = sorted(set(all_labels))
        self.num_classes = len(self.classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

    def labels_to_vector(self, labels):
        v = torch.zeros(self.num_classes, dtype=torch.float32)
        for lab in labels:
            if lab in self.class_to_idx:
                v[self.class_to_idx[lab]] = 1.0
        return v


for alias in ("__main__", "__mp_main__"):
    if alias not in sys.modules:
        sys.modules[alias] = types.ModuleType(alias)
    sys.modules[alias].LabelEncoder = LabelEncoder
    sys.modules[alias].TrainingConfig = TrainingConfig

class MultiLabelViT(torch.nn.Module):
    def __init__(self, model_name, num_classes, img_size=(384, 384), dropout=0.2):
        super().__init__()
        if "swin" in model_name:
            self.backbone = timm.create_model(
                model_name, pretrained=False, num_classes=0, img_size=img_size
            )
        else:
            self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)

        self.feat_dim = getattr(self.backbone, "num_features", 768)

        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(self.feat_dim),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.feat_dim, num_classes * 4),
            torch.nn.GELU(),
            torch.nn.LayerNorm(num_classes * 4),
            torch.nn.Dropout(dropout / 2),
            torch.nn.Linear(num_classes * 4, num_classes * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout / 4),
            torch.nn.Linear(num_classes * 2, num_classes),
        )

    def forward(self, x):
        feat = self.backbone(x)          

        if feat.ndim == 3:              
            feat = feat.mean(dim=1)
        elif feat.ndim == 4:           
            feat = feat.mean(dim=[2, 3])

        logits = self.classifier(feat)
        return logits


class _PadToSquare:
    def __init__(self, size=IMG_SIZE):
        self.s = size

    def __call__(self, img):
        w, h = img.size
        scale = min(self.s / w, self.s / h)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGB", (self.s, self.s), (0, 0, 0))
        canvas.paste(img, ((self.s - nw) // 2, (self.s - nh) // 2))
        return canvas


_transform = transforms.Compose(
    [
        _PadToSquare(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]
)


class _Singleton(type):
    _inst = None
    _lock = threading.Lock()

    def __call__(cls, *a, **kw):
        with cls._lock:
            if cls._inst is None:
                cls._inst = super().__call__(*a, **kw)
        return cls._inst


class Classifier(metaclass=_Singleton):
    def __init__(self):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        self.le: LabelEncoder = ckpt["label_encoder"]  
        self.model = MultiLabelViT(
            "swin_base_patch4_window12_384", self.le.num_classes, (IMG_SIZE, IMG_SIZE)
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(DEVICE).eval()

    @torch.inference_mode()
    def predict(self, img_bytes: bytes) -> List[str]:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = _transform(img).unsqueeze(0).to(DEVICE)
        probs = torch.sigmoid(self.model(tensor))[0]
        idxs = (probs > THRESHOLD).nonzero().flatten().tolist()
        return [self.le.idx_to_class[i] for i in idxs]


classifier = Classifier()

import torch
import base64
import io
import sys
import pickle
from PIL import Image
from torchvision import transforms
from typing import List, Dict
import threading

from .model import MultiLabelViT

class LabelEncoder:
    """
    Label encoder for multi-label classification.
    Must match the class used during training for checkpoint loading.
    """
    def __init__(self, all_labels):
        self.classes = sorted(list(set(all_labels)))
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
    def labels_to_vector(self, labels):
        vector = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in labels:
            if label in self.class_to_idx:
                vector[self.class_to_idx[label]] = 1.0
        return vector


class TrainingConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class RenameUnpickler(pickle.Unpickler):
    """
    Custom unpickler that redirects __main__ module references to current module.
    """
    def find_class(self, module, name):
        if module == '__main__':
            module = __name__
        return super().find_class(module, name)


def load_checkpoint_with_redirect(checkpoint_path: str, map_location):
    """
    Load PyTorch checkpoint with module name redirection.
    """
    with open(checkpoint_path, 'rb') as f:
        unpickler = RenameUnpickler(f)
        checkpoint = unpickler.load()
        if isinstance(checkpoint, dict):
            return checkpoint
        else:
            raise ValueError(f"Expected checkpoint to be a dict, got {type(checkpoint)}")


class ScaleAndPadToSquare:
    """Transform: scale and pad image to square"""
    def __init__(self, target_size: int = 384):
        self.target_size = target_size
    
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = min(self.target_size / w, self.target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        square_img = Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))
        
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        square_img.paste(img_resized, (paste_x, paste_y))
        
        return square_img

class InferenceEngine:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self._lock = threading.Lock()  
        
        self._register_classes_for_unpickling()
        
        # Load checkpoint with custom unpickler
        checkpoint = self._load_checkpoint(checkpoint_path)
        
        self.label_encoder = checkpoint['label_encoder']
        self.num_classes = self.label_encoder.num_classes
        
        self.model = MultiLabelViT(
            model_name='swin_base_patch4_window12_384',
            num_classes=self.num_classes,
            img_size=(384, 384),
            dropout=0.2
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval() 
        
        self.transform = transforms.Compose([
            ScaleAndPadToSquare(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        del checkpoint
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def _register_classes_for_unpickling(self):
        """
        Register classes in sys.modules so pickle can find them.
        """
        current_module = sys.modules[__name__]
        sys.modules['__main__'].LabelEncoder = LabelEncoder
        sys.modules['__main__'].TrainingConfig = TrainingConfig
    
    def _load_checkpoint(self, checkpoint_path: str) -> dict:
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False
            )
            return checkpoint
        except AttributeError as e:
            if "LabelEncoder" in str(e) or "TrainingConfig" in str(e):
                print("⚠️  Using custom unpickler for legacy checkpoint...")
                checkpoint = load_checkpoint_with_redirect(checkpoint_path, self.device)
                return checkpoint
            raise
    
    @torch.no_grad()  
    def predict(self, image_b64: str, threshold: float = 0.55) -> List[Dict[str, float]]:

        img_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        logits = self.model(image_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]  
        
        predictions = [
            {
                'class': self.label_encoder.idx_to_class[idx],
                'confidence': float(prob)
            }
            for idx, prob in enumerate(probs)
            if prob > threshold
        ]

        predictions.sort(key=lambda x: x['confidence'], reverse=True)

        return predictions
    
    def get_class_names(self) -> List[str]:
        return self.label_encoder.classes

_engine: InferenceEngine | None = None


def init_engine(checkpoint_path: str, device: str = "cpu") -> None:
    """Initialize singleton engine"""
    global _engine
    if _engine is not None:
        raise RuntimeError("Engine already initialized")
    _engine = InferenceEngine(checkpoint_path, device)


def get_engine() -> InferenceEngine:
    """Get singleton engine instance"""
    if _engine is None:
        raise RuntimeError("Engine not initialized. Call init_engine() first.")
    return _engine
from PIL import Image
import timm
import torch.nn as nn

class ScaleAndPadToSquare:
    """Scale and pad image to square"""
    def __init__(self, target_size=384, fill_color=(0, 0, 0)):
        self.target_size = target_size
        self.fill_color = fill_color
    
    def __call__(self, img):
        w, h = img.size
        scale = min(self.target_size / w, self.target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        square_img = Image.new('RGB', (self.target_size, self.target_size), self.fill_color)
        
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        square_img.paste(img_resized, (paste_x, paste_y))
        
        return square_img
    
class LabelEncoder:
    """Encoder for multi-label classification"""
    
    def __init__(self, all_labels):
        self.classes = sorted(list(set(all_labels)))
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
    def labels_to_vector(self, labels):
        """Convert list of labels to binary vector"""
        import torch
        vector = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in labels:
            if label in self.class_to_idx:
                vector[self.class_to_idx[label]] = 1.0
        return vector
    
class MultiLabelViT(nn.Module):
    """Vision Transformer for multi-label classification"""
    
    def __init__(self, model_name, num_classes, img_size=(384, 384), dropout=0.2):
        super().__init__()
        self.img_size = img_size
        
        if 'swin' in model_name:
            if isinstance(img_size, int):
                swin_img_size = (img_size, img_size)
            elif isinstance(img_size, tuple):
                swin_img_size = (img_size[0], img_size[0]) if img_size[0] == img_size[1] else img_size
            else:
                swin_img_size = (384, 384)
            
            self.backbone = timm.create_model(
                model_name, pretrained=False, num_classes=0, img_size=swin_img_size
            )
        else:
            self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        
        if hasattr(self.backbone, "num_features"):
            self.feat_dim = self.backbone.num_features
        else:
            self.feat_dim = 768
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, num_classes * 4),
            nn.GELU(),
            nn.LayerNorm(num_classes * 4),
            nn.Dropout(dropout / 2),
            nn.Linear(num_classes * 4, num_classes * 2),
            nn.GELU(),
            nn.Dropout(dropout / 4),
            nn.Linear(num_classes * 2, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        if len(features.shape) == 3:
            features = features.mean(dim=1)
        elif len(features.shape) == 4:
            features = features.mean(dim=[2, 3])
        
        logits = self.classifier(features)
        return logits
import torch
import torch.nn as nn
import timm


class MultiLabelViT(nn.Module):
    
    def __init__(self, model_name: str, num_classes: int, img_size: tuple = (384, 384), dropout: float = 0.2):
        super().__init__()
        self.img_size = img_size
        
        if 'swin' in model_name:
            swin_img_size = (img_size[0], img_size[0]) if isinstance(img_size, tuple) else (img_size, img_size)
            self.backbone = timm.create_model(
                model_name, pretrained=False, num_classes=0, img_size=swin_img_size
            )
        else:
            self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        
        self.feat_dim = getattr(self.backbone, "num_features", 768)
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() == 3:
            features = features.mean(dim=1)
        elif features.dim() == 4:
            features = features.mean(dim=[2, 3])
        
        return self.classifier(features)
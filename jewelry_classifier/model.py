from functools import lru_cache
from pathlib import Path
import torch, timm
from torch import nn
from .utils import LabelEncoder, MultiLabelViT  

CKPT  = Path.getenv('CKPT_PATH', '/data/checkpoints/best_model.pth')
DEVICE = 'cpu'      

@lru_cache(maxsize=1)
def get_model():
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    le: LabelEncoder = ckpt['label_encoder']
    model = MultiLabelViT(
        model_name='swin_base_patch4_window12_384',
        num_classes=le.num_classes,
        img_size=(384, 384),
        dropout=0.2,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(DEVICE)
    return model, le

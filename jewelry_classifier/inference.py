import io, httpx, numpy as np, torch
from PIL import Image
from torchvision import transforms
from .model import get_model
from .utils import ScaleAndPadToSquare

TX = transforms.Compose([
    ScaleAndPadToSquare(384),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

async def _load_image(inp: dict) -> Image.Image:
    """
    inp may be:
      • {"uri": "https://…"}  (Artifact SAS url)
      • {"path": "/data/uploads/abc.jpg"}
      • {"bytes_b64": "data:image/jpeg;base64,…"}
    """
    if 'uri' in inp:                # download
        async with httpx.AsyncClient() as cli:
            r = await cli.get(inp['uri'], timeout=30)
            r.raise_for_status()
            buf = io.BytesIO(r.content)
    elif 'path' in inp:
        buf = open(inp['path'], 'rb')
    else:                           # assume data URL already peeled in gateway
        import base64
        hdr, b64 = inp['bytes_b64'].split(',', 1)
        buf = io.BytesIO(base64.b64decode(b64))

    return Image.open(buf).convert('RGB')

async def predict(image_node: dict, threshold: float = 0.55):
    img = await _load_image(image_node)
    model, le = get_model()
    with torch.no_grad():
        x = TX(img).unsqueeze(0)
        logits = model(x)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    preds = [
        {"class": le.idx_to_class[i], "confidence": float(p)}
        for i, p in enumerate(probs) if p > threshold
    ]
    return sorted(preds, key=lambda x: x['confidence'], reverse=True)

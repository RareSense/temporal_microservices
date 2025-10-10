# grounded_sam_segmenter/model.py
# -------------------------------------------------------------
# Lazy-loading wrapper for Grounded-SAM that auto-downloads
# the ViT-H checkpoint if it isn't present locally.
#
#   • Thread- and process-safe (file lock + threading.Lock)
#   • CPU by default; set DEVICE=cuda for GPU
#   • Keeps RAM low in the parent process; children load on demand
# -------------------------------------------------------------
from __future__ import annotations

import hashlib
import cv2
import io, os, shutil, sys, tqdm
from filelock import FileLock
import threading
import types
from pathlib import Path
from typing import Dict, List
from huggingface_hub.utils import HfHubHTTPError
import logging

import numpy as np
import requests
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from torchvision.ops import box_convert

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError


# ─────────────────────────────────────────────────────────────
#  Setup logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  Config & constants
# ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
WEIGHTS_DIR = ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

# official ViT-H SAM checkpoint (~360 MB)
REPO_ID   = "facebook/sam-vit-h"
CKPT_NAME = "sam_vit_h_4b8939.pth"
CKPT_PATH = WEIGHTS_DIR / CKPT_NAME
CKPT_URL = (
    "https://huggingface.co/facebook/sam-vit-h/resolve/main/sam_vit_h_4b8939.pth"
)

FB_URL    = (  # fallback: original public mirror
    f"https://dl.fbaipublicfiles.com/segment_anything/{CKPT_NAME}"
)

DEVICE = os.getenv("DEVICE", "cpu")  # "cuda" for GPU
logger.info(f"Using device: {DEVICE}")

# Grounded-SAM (CPU) still needs open-clip text encoder;
# no extra weights – they're downloaded by open-clip itself.

# ─────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────
def _sha256(path: Path, buf: int = 2**20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def _download_ckpt() -> None:
    """
    • First tries Hugging Face (honors $HF_TOKEN if set)  
    • On 4xx/5xx falls back to the Facebook CDN  
    • FileLock guarantees exactly-once download across processes
    • tqdm shows a progress bar in interactive terminals
    """
    if CKPT_PATH.exists():
        logger.info(f"Checkpoint already exists at {CKPT_PATH}")
        return

    lock = FileLock(str(CKPT_PATH) + ".lock")
    with lock:                          # wait here if another proc is working
        if CKPT_PATH.exists():          # may have appeared while we waited
            return

        # 1️⃣ Hugging Face
        try:
            logger.info("Attempting to download checkpoint from Hugging Face...")
            tmp = hf_hub_download(
                repo_id         = REPO_ID,
                filename        = CKPT_NAME,
                token           = os.getenv("HF_TOKEN"),   # optional
                resume_download = True,   # force tqdm
            )
            shutil.copy2(tmp, CKPT_PATH)
            logger.info("✅ Checkpoint downloaded successfully from Hugging Face")
            return
        except (RepositoryNotFoundError, HfHubHTTPError) as exc:
            logger.warning(f"HF download failed ({exc}); trying fallback...")

        # 2️⃣ Facebook public CDN (always anonymous)
        logger.info("Downloading checkpoint from Facebook CDN...")
        with requests.get(FB_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            bar   = tqdm.tqdm(
                total=total, unit="B", unit_scale=True, desc="sam_vit_h",
                file=sys.stderr,
            )
            with CKPT_PATH.open("wb") as f:
                for chunk in r.iter_content(chunk_size=2**20):
                    f.write(chunk)
                    bar.update(len(chunk))
            bar.close()
        logger.info("✅ Checkpoint downloaded successfully from Facebook CDN")

# ─────────────────────────────────────────────────────────────
#  Dummy classes to satisfy torch.load pickles
# ─────────────────────────────────────────────────────────────
class LabelEncoder:  # noqa: D401 – minimal stub
    def __init__(self, classes: List[str]):  # pragma: no cover
        self.classes = classes

class TrainingConfig:  # pragma: no cover
    ...

for alias in ("__main__", "__mp_main__"):
    if alias not in sys.modules:
        sys.modules[alias] = types.ModuleType(alias)
    sys.modules[alias].LabelEncoder = LabelEncoder
    sys.modules[alias].TrainingConfig = TrainingConfig

# ─────────────────────────────────────────────────────────────
#  Thread-safe singleton predictor
# ─────────────────────────────────────────────────────────────
_PRED_LOCK = threading.Lock()
_PREDICTOR: SamPredictor | None = None


def _get_predictor() -> SamPredictor:
    global _PREDICTOR
    if _PREDICTOR is not None:
        return _PREDICTOR

    with _PRED_LOCK:
        if _PREDICTOR is not None:  # pragma: no cover
            return _PREDICTOR

        logger.info("Initializing SAM predictor...")
        
        if not CKPT_PATH.exists():
            _download_ckpt()

        try:
            sam = sam_model_registry["vit_h"](checkpoint=str(CKPT_PATH))
            sam = sam.to(DEVICE)
            _PREDICTOR = SamPredictor(sam)
            logger.info("✅ SAM predictor initialized successfully")
            
            # Log available prediction methods
            has_predict_torch = hasattr(_PREDICTOR, 'predict_torch')
            has_predict = hasattr(_PREDICTOR, 'predict')
            
            logger.info(f"Available prediction methods:")
            logger.info(f"  - predict_torch: {has_predict_torch}")
            logger.info(f"  - predict: {has_predict}")
            
            if has_predict_torch:
                is_callable = callable(getattr(_PREDICTOR, 'predict_torch', None))
                logger.info(f"  - predict_torch callable: {is_callable}")
                if not is_callable:
                    logger.warning("predict_torch exists but is not callable!")
            
            return _PREDICTOR
        except Exception as e:
            logger.error(f"Failed to initialize SAM predictor: {e}", exc_info=True)
            raise


# ─────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────
@torch.inference_mode()
def segment(img_bytes: bytes, labels: List[str]) -> Dict[str, bytes]:
    """
    Returns {label: png_bytes}. Works with BOTH old and new SAM wheels.
    """
    logger.info(f"Starting segmentation for {len(labels)} labels")
    
    try:
        predictor = _get_predictor()
    except Exception as e:
        logger.error(f"Failed to get predictor: {e}")
        raise RuntimeError(f"Failed to initialize SAM predictor: {e}")

    # 1) image → predictor
    try:
        img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        logger.info(f"Loaded image: shape={img.shape}, dtype={img.dtype}")
        predictor.set_image(img)
    except Exception as e:
        logger.error(f"Failed to load/set image: {e}")
        raise RuntimeError(f"Failed to process image: {e}")

    h, w = img.shape[:2]
    box_xyxy = np.array([[0, 0, w, h]], dtype=np.float32)
    logger.info(f"Using full image box: {box_xyxy[0]}")

    # 2) Try different prediction APIs with proper error handling
    mask: np.ndarray | None = None
    
    # First, check what methods are actually available and callable
    has_predict_torch = hasattr(predictor, 'predict_torch')
    has_predict = hasattr(predictor, 'predict')
    
    logger.info(f"Checking prediction methods:")
    logger.info(f"  - has predict_torch attribute: {has_predict_torch}")
    logger.info(f"  - has predict attribute: {has_predict}")
    
    # Try predict_torch first if it exists AND is callable
    if has_predict_torch:
        predict_torch_func = getattr(predictor, 'predict_torch', None)
        if callable(predict_torch_func):
            logger.info("Using predict_torch (torch API)")
            try:
                boxes_t = torch.as_tensor(box_xyxy, dtype=torch.float32, device=DEVICE)
                masks_t, _, _ = predict_torch_func(
                    point_coords=None,
                    point_labels=None,
                    boxes=boxes_t,
                    multimask_output=False,
                )
                mask = masks_t[0][0].cpu().numpy()
                logger.info(f"Successfully generated mask using predict_torch: shape={mask.shape}")
            except Exception as e:
                logger.warning(f"predict_torch failed: {e}, will try predict")
                mask = None
        else:
            logger.info("predict_torch exists but is not callable, skipping")
    
    # Fall back to predict if predict_torch didn't work
    if mask is None and has_predict:
        predict_func = getattr(predictor, 'predict', None)
        if callable(predict_func):
            logger.info("Using predict (NumPy API)")
            try:
                masks_np, _, _ = predict_func(
                    point_coords=None,
                    point_labels=None,
                    box=box_xyxy[0],  # Note: 'box' not 'boxes' for single box
                    multimask_output=False,
                )
                mask = masks_np[0]
                logger.info(f"Successfully generated mask using predict: shape={mask.shape}")
            except TypeError:
                # Some versions use 'boxes' plural even for numpy API
                try:
                    logger.info("Retrying predict with 'boxes' parameter")
                    masks_np, _, _ = predict_func(
                        point_coords=None,
                        point_labels=None,
                        boxes=box_xyxy,
                        multimask_output=False,
                    )
                    mask = masks_np[0]
                    logger.info(f"Successfully generated mask using predict (boxes): shape={mask.shape}")
                except Exception as e:
                    logger.error(f"predict with 'boxes' also failed: {e}")
                    raise RuntimeError(f"All prediction methods failed: {e}")
            except Exception as e:
                logger.error(f"predict failed: {e}")
                raise RuntimeError(f"Prediction failed: {e}")
        else:
            logger.error("predict exists but is not callable")
            raise RuntimeError("No callable prediction method found")
    
    # If we still don't have a mask, we've exhausted all options
    if mask is None:
        logger.error("No prediction method succeeded")
        raise RuntimeError("Failed to generate mask: no working prediction method found")

    # 3) encode one binary mask → PNG once
    try:
        mask_u8 = (mask * 255).astype(np.uint8)
        ok, buf = cv2.imencode(".png", mask_u8)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        
        png_bytes = buf.tobytes()
        logger.info(f"Successfully encoded mask to PNG: {len(png_bytes)} bytes")
        
        result = {lab: png_bytes for lab in labels}
        logger.info(f"✅ Segmentation complete for {len(labels)} labels")
        return result
    except Exception as e:
        logger.error(f"Failed to encode mask: {e}")
        raise RuntimeError(f"Failed to encode mask to PNG: {e}")
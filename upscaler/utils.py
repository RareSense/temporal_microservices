from typing import Tuple
import numpy as np
from PIL import Image
import logging
import warnings
import os
from pathlib import Path
from contextlib import nullcontext
from basicsr.archs.swinir_arch import SwinIR
from basicsr.utils import img2tensor, tensor2img
import torch
import cv2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
SWINIR_CKPT = Path(
     os.getenv("JC_KPT", Path(__file__).parent.parent / "weights" / "net_g_82500.pth")
    )
SWINIR_UPSAMPLER = None
DEVICE = "cpu" 

def upscale_tiled_bgr(
    img_bgr: np.ndarray,
    net: torch.nn.Module,
    device: str,
    tile: int = 192,
    pad: int = 16,
    wsize: int = 8,  # SwinIR window size (must divide H & W inside the model)
) -> np.ndarray:
    """
    SwinIR ×3 tiled inference with two kinds of padding:

    1.  context pad  (±`pad`) to avoid seams
    2.  window pad  (make H,W divisible by `wsize`) so SwinIR's reshape works

    We remove BOTH kinds of extra pixels before the tile is written back,
    so the destination slice and the SR slice always match.
    """
    h, w = img_bgr.shape[:2]
    scale = 3
    out = np.empty((h * scale, w * scale, 3), np.uint8)

    autocast_ctx = torch.cuda.amp.autocast if device.startswith("cuda") else nullcontext

    for y in range(0, h, tile):
        for x in range(0, w, tile):
            # context-padded coordinates in input
            i0, j0 = max(0, y - pad), max(0, x - pad)
            i1, j1 = min(h, y + tile + pad), min(w, x + tile + pad)

            patch = img_bgr[i0:i1, j0:j1]

            # ----------  window padding (divisible by 8) ----------
            ph_ctx, pw_ctx = patch.shape[:2]               # incl. context pad
            pad_b = (wsize - ph_ctx % wsize) % wsize
            pad_r = (wsize - pw_ctx % wsize) % wsize
            if pad_b or pad_r:
                patch = cv2.copyMakeBorder(
                    patch,
                    0, pad_b, 0, pad_r,
                    cv2.BORDER_REFLECT_101,
                )

            ph_full, pw_full = patch.shape[:2]             # after win-pad

            # ----------  run SwinIR ----------
            tensor = img2tensor(patch, bgr2rgb=True, float32=True) / 255.0
            tensor = tensor.unsqueeze(0).to(device)
            if device.startswith("cuda"):
                tensor = tensor.half()

            with autocast_ctx():
                sr = net(tensor)

            sr = tensor2img(sr, rgb2bgr=True)              # (H*3, W*3, 3)

            # ----------  strip window padding ----------
            ph_valid = ph_full - pad_b         # height w/out win-pad bottom
            pw_valid = pw_full - pad_r         # width  w/out win-pad right
            sr = sr[: ph_valid * scale, : pw_valid * scale]

            # ----------  coordinates for dst canvas ----------
            top    = y * scale
            left   = x * scale
            bottom = min(y + tile, h) * scale
            right  = min(x + tile, w) * scale

            # how deep inside sr to start (skip context pad top/left)
            pt_top = (y - i0) * scale
            pt_left = (x - j0) * scale
            pt_bot = pt_top + (bottom - top)
            pt_rgt = pt_left + (right - left)

            # copy
            out[top:bottom, left:right] = sr[pt_top:pt_bot, pt_left:pt_rgt]

    return out


def load_swinir_x3(ckpt_path: str, device: str = "cuda"):
    """SwinIR-x3 network weights → ready model (half precision if GPU)."""
    net = SwinIR(
        upscale=3, img_size=192, window_size=8,
        depths=[6]*6, embed_dim=60, num_heads=[6]*6,
        mlp_ratio=2, upsampler="pixelshuffle",
        img_range=1.0, in_chans=3
    )
    sd = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(sd.get("params", sd), strict=True)
    net = net.to(device).eval()
    if device.startswith("cuda"):
        net = net.half()  # fp16 for speed / memory
    return net


def get_swinir_upsampler():
    """Initialize SwinIR upsampler"""
    global SWINIR_UPSAMPLER
    if SWINIR_UPSAMPLER is None:
        try:
            if not os.path.exists(SWINIR_CKPT):
                logger.error(f"SwinIR checkpoint not found at: {SWINIR_CKPT}")
                return None
            
            SWINIR_UPSAMPLER = load_swinir_x3(SWINIR_CKPT, DEVICE)
            logger.info("SwinIR upsampler loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SwinIR upsampler: {e}")
            SWINIR_UPSAMPLER = None
    return SWINIR_UPSAMPLER

def intelligent_upscale_and_resize(
    bg_image: Image.Image,
    target_size: Tuple[int, int],
    max_passes: int = 2,
) -> Image.Image:
    """
    Make the input at least as large as `target_size` by applying
    SwinIR ×3 tiling passes and then
    resizing down with Lanczos to the exact dimensions.

    Parameters
    ----------
    bg_image : PIL.Image
        Original background (RGB or RGBA is fine – will be converted to RGB).
    target_size : (width, height)
        Final dimensions **of the ghost image** – the composite must match.
    max_passes : int
        Safety cap: how many 3× passes we allow (same as original code).

    Returns
    -------
    PIL.Image
        Upscaled (if needed) and resized background in **RGB** mode.
    """
    bg_image = bg_image.convert("RGB")
    target_w, target_h = target_size
    cur_w, cur_h = bg_image.size

    # ─── 1.  Calculate how much scale we need ──────────────────────────────
    # We must fully cover the ghost canvas, so take the *maximum* of the ratios
    scale_w = target_w / cur_w
    scale_h = target_h / cur_h
    required_scale = max(scale_w, scale_h)

    logger.info(
        f"[ups] target={target_size}  "
        f"orig={bg_image.size}  "
        f"req_scale={required_scale:0.2f}"
    )

    # ─── 2.  Try SwinIR ×3 passes (each pass = ×3) ─────────────────────────
    upsampler = get_swinir_upsampler()          
    swinir_scale = 3

    if required_scale > 1 and upsampler is not None:
        # Determine how many passes we *need* (but never > max_passes)
        passes = 1
        cumulative = swinir_scale
        while cumulative < required_scale and passes < max_passes:
            passes += 1
            cumulative *= swinir_scale
        logger.info(f"[ups] SwinIR passes = {passes}   cumulative≈{cumulative}×")

        # PIL → numpy BGR
        cv_img = cv2.cvtColor(np.array(bg_image), cv2.COLOR_RGB2BGR)

        for idx in range(passes):
            cv_img = upscale_tiled_bgr(cv_img, upsampler, DEVICE)
            logger.info(f"[ups] pass {idx+1}:   size={cv_img.shape[1]}×{cv_img.shape[0]}")

        # numpy BGR → PIL RGB
        bg_image = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    else:
        if upsampler is None:
            logger.warning("[ups] SwinIR model not available – falling back to Lanczos")
        else:
            logger.info("[ups] No upscaling required (background already ≥ target)")

    # ─── 3.  Final exact resize (down- or very slight up-sample) ────────────
    final = bg_image.resize(target_size, Image.Resampling.LANCZOS)
    logger.info(f"[ups] final size = {final.size}")
    return final


def alpha_composite_rgba(
    bg: Image.Image,
    ghost: Image.Image,
) -> Image.Image:
    """
    Alpha-composite a transparent-BG `ghost` onto `bg`.

    • Ensures both images are RGBA with identical sizes.
    • Returns **RGB** (opaque) result ready for JPEG encoding.
    """
    if bg.mode != "RGBA":
        bg = bg.convert("RGBA")
    if ghost.mode != "RGBA":
        ghost = ghost.convert("RGBA")

    if bg.size != ghost.size:
        raise ValueError(
            f"Background size {bg.size} must match ghost size {ghost.size}"
        )

    composite = Image.alpha_composite(bg, ghost)
    return composite.convert("RGB")

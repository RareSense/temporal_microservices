from __future__ import annotations
import os, io, logging, requests, shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import threading
from tqdm import tqdm

import torch
import torchvision
from torchvision.ops import box_convert
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# GroundingDINO imports
from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T

# SAM imports
from segment_anything import build_sam, SamPredictor

log = logging.getLogger(__name__)

# Model paths
ROOT = Path(__file__).parent
WEIGHTS_DIR = ROOT.parent / "weights"  # Use shared weights directory
WEIGHTS_DIR.mkdir(exist_ok=True)

DINO_CONFIG = WEIGHTS_DIR / "GroundingDINO_SwinT_OGC.py"
DINO_CKPT = WEIGHTS_DIR / "groundingdino_swint_ogc.pth"
SAM_CKPT = WEIGHTS_DIR / "sam_vit_h_4b8939.pth"

# Download URLs
DINO_CKPT_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
DINO_CONFIG_URL = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
SAM_CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

def download_file(url: str, dest: Path, desc: str = None):
    """Download file with progress bar"""
    if dest.exists():
        log.info(f"{desc or dest.name} already exists")
        return
    
    log.info(f"Downloading {desc or dest.name}...")
    
    # Create temp file first
    temp_file = dest.with_suffix('.tmp')
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(temp_file, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Move temp file to final destination
        shutil.move(temp_file, dest)
        log.info(f"Downloaded {desc or dest.name} successfully")
        
    except Exception as e:
        # Clean up temp file if exists
        if temp_file.exists():
            temp_file.unlink()
        raise RuntimeError(f"Failed to download {desc or dest.name}: {e}")

def ensure_weights():
    """Download model weights if not present"""
    # Download GroundingDINO checkpoint
    if not DINO_CKPT.exists():
        download_file(DINO_CKPT_URL, DINO_CKPT, "GroundingDINO checkpoint (~700MB)")
    
    # Download GroundingDINO config
    if not DINO_CONFIG.exists():
        download_file(DINO_CONFIG_URL, DINO_CONFIG, "GroundingDINO config")
    
    # Download SAM checkpoint
    if not SAM_CKPT.exists():
        download_file(SAM_CKPT_URL, SAM_CKPT, "SAM checkpoint (~2.6GB)")
    
    log.info("All model weights ready")

# Configuration
DEVICE = torch.device("cpu")

# Negative words to filter out non-jewelry detections
NEGATIVE_WORDS = ["hand", "face", "arm", "mouth", "lips", "teeth", "eye", "nails", "fingernail", "mole"]

def transform_image(image_pil: Image.Image) -> torch.Tensor:
    """Transform image for GroundingDINO (matching Space preprocessing)"""
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image_pil, None)
    return image_tensor

def get_grounding_output(model, image_tensor, caption, box_threshold, text_threshold):
    """Get GroundingDINO output (matching Space function)"""
    with torch.no_grad():
        outputs = model(image_tensor[None].to(DEVICE), captions=[caption])
    
    logits = outputs["pred_logits"].sigmoid()[0]  # (num_queries, num_classes)
    boxes = outputs["pred_boxes"][0]  # (num_queries, 4)
    
    # Filter by threshold
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]
    
    # Get phrases
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    
    # Build phrases from predictions
    phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, caption
        )
        phrases.append(pred_phrase)
        scores.append(logit.max().item())
    
    return boxes_filt, torch.tensor(scores), phrases

def get_phrases_from_posmap(posmap, tokenized, caption):
    """Extract phrases from position map (Space utility function)"""
    # Simplified version - in production you'd want the full implementation
    # from GroundingDINO utils
    if posmap.any():
        tokens = tokenized.tokens()
        return caption  # Simplified - return full caption
    return ""

class _Singleton(type):
    """Thread-safe singleton metaclass"""
    _inst = None
    _lock = threading.Lock()
    
    def __call__(cls, *a, **kw):
        with cls._lock:
            if cls._inst is None:
                cls._inst = super().__call__(*a, **kw)
        return cls._inst

class GroundedSAMModel(metaclass=_Singleton):
    def __init__(self):
        log.info("Initializing GroundedSAM models...")
        
        # Ensure weights are downloaded
        ensure_weights()
        
        # Load GroundingDINO
        log.info("Loading GroundingDINO model...")
        self.dino_model = load_model(str(DINO_CONFIG), str(DINO_CKPT))
        self.dino_model.to(DEVICE)
        self.dino_model.eval()
        
        # Load SAM
        log.info("Loading SAM model...")
        sam_model = build_sam(checkpoint=str(SAM_CKPT))
        sam_model.to(DEVICE)
        self.sam_predictor = SamPredictor(sam_model)
        
        log.info(f"Models loaded successfully on {DEVICE}")
    
    @torch.inference_mode()
    def detect_and_segment(
        self, 
        image_bytes: bytes, 
        labels: List[str]
    ) -> Tuple[List[dict], Image.Image]:
        """
        Detect objects and generate a SINGLE UNIFIED mask for all requested jewelry types.
        Returns: (list with single mask result, overlay image)
        """
        # Load image
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_image = np.array(pil_image)
        
        # Build comprehensive detection prompt with ALL jewelry types
        all_classes = set()
        requested_types = set()
        
        for label in labels:
            label_lower = label.lower()
            
            if "bracelet" in label_lower or "bangle" in label_lower:
                all_classes.update(["bracelet", "wrist band", "bangle"])
                requested_types.add("bracelet")
            elif "earring" in label_lower:
                all_classes.update(["earring", "earrings", "stud earring"])
                requested_types.add("earring")
            elif "ring" in label_lower and "earring" not in label_lower:
                all_classes.update(["wedding ring", "finger ring", "ring"])
                requested_types.add("ring")
            elif "watch" in label_lower:
                all_classes.update(["watch", "wristwatch", "smartwatch"])
                requested_types.add("watch")
            elif "necklace" in label_lower:
                all_classes.update(["necklace", "pendant", "chain"])
                requested_types.add("necklace")
        
        # If no specific classes found, use comprehensive mixed classes
        if not all_classes:
            all_classes = {"ring", "wedding ring", "bracelet", "wristwatch", "wrist band", 
                          "necklace", "earring", "stud earring", "jewelry"}
        
        # Always use mixed config thresholds for best detection
        box_threshold = 0.25
        text_threshold = 0.25
        nms_threshold = 0.5
        
        # Build text prompt with ". " separator
        text_prompt = ". ".join(sorted(all_classes))
        log.info(f"Detection prompt: {text_prompt}")
        log.info(f"Requested types: {requested_types}")
        log.info(f"Using thresholds - box: {box_threshold}, text: {text_threshold}, nms: {nms_threshold}")
        
        # Transform image for DINO
        img_tensor = transform_image(pil_image)
        
        # Run GroundingDINO detection
        boxes, scores, phrases = get_grounding_output(
            self.dino_model, img_tensor, text_prompt, box_threshold, text_threshold
        )
        
        if len(boxes) == 0:
            log.info("No detections found")
            return [], pil_image
        
        log.info(f"Found {len(boxes)} initial detections")
        log.info(f"Detected phrases: {phrases}")
        
        # Convert normalized boxes to pixel coordinates
        W, H = pil_image.size
        for i in range(boxes.size(0)):
            boxes[i] = boxes[i] * torch.tensor([W, H, W, H])
            boxes[i][:2] -= boxes[i][2:] / 2  # Convert center to top-left
            boxes[i][2:] += boxes[i][:2]      # Convert width/height to bottom-right
        
        # Filter negative words
        keep_idxs = []
        for i, phrase in enumerate(phrases):
            phrase_lower = phrase.lower()
            if any(neg in phrase_lower for neg in NEGATIVE_WORDS):
                log.info(f"Filtered out: {phrase}")
            else:
                keep_idxs.append(i)
        
        if not keep_idxs:
            log.info("All detections filtered by negative words")
            return [], pil_image
        
        boxes = boxes[keep_idxs]
        scores = scores[keep_idxs]
        phrases = [phrases[i] for i in keep_idxs]
        
        log.info(f"After filtering: {len(boxes)} detections")
        
        # Apply NMS
        keep_nms = torchvision.ops.nms(boxes, scores, nms_threshold).tolist()
        final_boxes = boxes[keep_nms]
        final_scores = scores[keep_nms]
        final_phrases = [phrases[i] for i in keep_nms]
        
        log.info(f"After NMS: {len(final_boxes)} detections")
        log.info(f"Final phrases: {final_phrases}")
        
        # Run SAM segmentation
        self.sam_predictor.set_image(np_image)
        
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            final_boxes, np_image.shape[:2]
        ).to(DEVICE)
        
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        log.info(f"Generated {masks.shape[0]} masks")
        
        # Create SINGLE UNIFIED mask combining all detections
        if masks.shape[0] > 0:
            # Merge all masks into one
            unified_mask = torch.any(masks.squeeze(1), dim=0).cpu().numpy().astype(np.uint8) * 255
            
            # CRITICAL: Ensure mask is same size as original image
            # SAM might output at different resolution, so resize to match input
            H_original, W_original = np_image.shape[:2]
            H_mask, W_mask = unified_mask.shape
            
            if (H_mask != H_original) or (W_mask != W_original):
                log.info(f"Resizing mask from {W_mask}x{H_mask} to {W_original}x{H_original}")
                unified_mask_pil = Image.fromarray(unified_mask, mode='L')
                unified_mask_pil = unified_mask_pil.resize((W_original, H_original), Image.NEAREST)
            else:
                unified_mask_pil = Image.fromarray(unified_mask, mode='L')
            
            # Verify final mask size matches input
            assert unified_mask_pil.size == pil_image.size, f"Mask size {unified_mask_pil.size} doesn't match input {pil_image.size}"
            
            # Collect all detected jewelry types
            detected_types = set()
            for phrase in final_phrases:
                phrase_lower = phrase.lower()
                if any(x in phrase_lower for x in ["bracelet", "wrist band", "bangle"]):
                    detected_types.add("bracelet")
                if any(x in phrase_lower for x in ["earring", "stud"]):
                    detected_types.add("earring")
                if any(x in phrase_lower for x in ["ring", "wedding"]) and "earring" not in phrase_lower:
                    detected_types.add("ring")
                if any(x in phrase_lower for x in ["watch", "wristwatch"]):
                    detected_types.add("watch")
                if any(x in phrase_lower for x in ["necklace", "pendant", "chain"]):
                    detected_types.add("necklace")
            
            # Create single result with unified mask
            unified_label = "jewelry_mask"  # Generic label for unified mask
            if detected_types:
                unified_label = "_".join(sorted(detected_types))
            
            # Calculate overall bounding box for all detections
            all_boxes = final_boxes.cpu().numpy()
            min_x = np.min(all_boxes[:, 0])
            min_y = np.min(all_boxes[:, 1])
            max_x = np.max(all_boxes[:, 2])
            max_y = np.max(all_boxes[:, 3])
            overall_bbox = [float(min_x), float(min_y), float(max_x), float(max_y)]
            
            # Average confidence
            avg_confidence = float(final_scores.mean())
            
            result = {
                "label": unified_label,
                "confidence": avg_confidence,
                "mask": unified_mask_pil,
                "bbox": overall_bbox,
                "num_objects": len(final_boxes),
                "detected_types": list(detected_types)
            }
            
            # Create overlay with all bounding boxes
            overlay = pil_image.copy()
            draw = ImageDraw.Draw(overlay)
            
            for box, score, phrase in zip(final_boxes, final_scores, final_phrases):
                bbox = box.cpu().tolist()
                draw.rectangle(bbox, outline="red", width=2)
                # Simplify label for display
                display_label = "jewelry"
                if "bracelet" in phrase.lower() or "wrist" in phrase.lower():
                    display_label = "bracelet"
                elif "earring" in phrase.lower():
                    display_label = "earring"
                elif "ring" in phrase.lower() and "earring" not in phrase.lower():
                    display_label = "ring"
                elif "watch" in phrase.lower():
                    display_label = "watch"
                elif "necklace" in phrase.lower():
                    display_label = "necklace"
                    
                draw.text((bbox[0], bbox[1] - 10), f"{display_label} ({score:.2f})", fill="red")
            
            log.info(f"Created unified mask for {len(detected_types)} jewelry types: {detected_types}")
            log.info(f"Requested but not found: {requested_types - detected_types}")
            
            return [result], overlay
        else:
            return [], pil_image

# Singleton instance
grounded_sam = GroundedSAMModel()
import asyncio
import time
import uuid
import json
import random
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from PIL import Image, ImageOps
import io
import base64
from torchvision import transforms

from diffusers import FluxFillPipeline
from diffusers.models.transformers import FluxTransformer2DModel

# ===================== Configuration =====================
class Config:
    MODEL_ID = "black-forest-labs/FLUX.1-Fill-dev"
    TRANSFORMER_DIR = "/home/nimra/temporal_microservices/weights/checkpoint-21000"
    
    # Library paths
    LIBRARY_JSON = "library.json"
    LIBRARY_FOLDER = "library_images"
    
    # Processing settings
    MAX_BATCH_SIZE = 4
    QUEUE_SIZE = 100
    MAX_WORKERS = 1
    REQUEST_TIMEOUT = 300
    
    # Model defaults
    DEFAULT_SIZE = (768, 1024)
    DEFAULT_STEPS = 30
    DEFAULT_GUIDANCE = 30.0
    DEFAULT_SEED = -1
    DEFAULT_PROMPT = "Two-panel image showing ; [IMAGE1]  A human body part; [IMAGE2] Same human body part with a precisely cut-out composited on it"
    
    # Logging
    LOG_LEVEL = logging.INFO
    
config = Config()

# ===================== Logging Setup =====================
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================== Library Manager =====================
class LibraryManager:
    """Manages the garment library and matching logic"""
    
    def __init__(self, json_path: str = config.LIBRARY_JSON, 
                 folder_path: str = config.LIBRARY_FOLDER):
        self.json_path = json_path
        self.folder_path = Path(folder_path)
        self.library = []
        self.load_library()
    
    def load_library(self):
        """Load library from JSON file"""
        try:
            with open(self.json_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.library.append(json.loads(line))
            logger.info(f"Loaded {len(self.library)} items from library")
        except Exception as e:
            logger.error(f"Failed to load library: {e}")
            self.library = []
    
    def find_matches(self, zoom_level: str, skin_shade: str, 
                    jewelry_type: str) -> List[Dict]:
        """Find exact matches based on all three criteria"""
        matches = []
        
        for item in self.library:
            zoom_match = item.get("zoom_level", "").lower() == zoom_level.lower()
            shade_match = item.get("skin_shade", "").lower() == skin_shade.lower()
            jewelry_match = item.get("jewelry", "").lower() == jewelry_type.lower()
            
            if zoom_match and shade_match and jewelry_match:
                matches.append(item)
        
        return matches
    
    def get_garment_image(self, zoom_level: str, skin_shade: str, 
                         jewelry_type: str) -> Optional[Image.Image]:
        """Get a matching garment image from library"""
        matches = self.find_matches(zoom_level, skin_shade, jewelry_type)
        
        if not matches:
            logger.warning(f"No match found for zoom={zoom_level}, "
                         f"shade={skin_shade}, jewelry={jewelry_type}")
            # Fallback: return a random image from library
            if self.library:
                selected = random.choice(self.library)
                logger.info(f"Using fallback image: {selected['image']}")
            else:
                return None
        else:
            selected = random.choice(matches)
            logger.info(f"Found {len(matches)} matches, selected: {selected['image']}")
        
        # Load and return the image
        image_path = self.folder_path / selected['image']
        if image_path.exists():
            return Image.open(image_path).convert("RGB")
        else:
            logger.error(f"Image file not found: {image_path}")
            return None

# ===================== External API Placeholders =====================
class ExternalClassifiers:
    """Placeholder for external classification APIs"""
    
    @staticmethod
    async def classify_zoom(image: Image.Image) -> str:
        """
        Placeholder for zoom level classification
        TODO: Replace with actual API call to zoom classifier
        """
        zoom_levels = ["bust shot", "tight detail shot", "macro closeup shot", "three quarter shot"]
        return random.choice(zoom_levels)
    
    @staticmethod
    async def detect_skin_shade(image: Image.Image) -> str:
        """
        Placeholder for skin shade detection
        TODO: Replace with actual API call to skin shade detector
        """
        shades = ["light", "medium", "dark", "deep dark", "darkest"]
        return random.choice(shades)
    
    @staticmethod
    async def classify_jewelry(image: Image.Image) -> str:
        """
        Placeholder for jewelry classification
        TODO: Replace with actual API call to jewelry classifier
        """
        jewelry_types = ["ring", "necklace", "bracelet", "earring", "watch", "no jewelry"]
        return random.choice(jewelry_types)

# ===================== Request/Response Models =====================
class TryOnRequest(BaseModel):
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Required fields
    image: str = Field(..., description="Base64 encoded person image")
    mask: str = Field(..., description="Base64 encoded mask image (will be inverted)")
    num_variations: int = Field(1, ge=1, le=10, description="Number of output variations")
    
    # Optional garment selection
    garment: Optional[str] = Field(None, description="Base64 encoded garment (if not using library)")
    use_library: bool = Field(True, description="Whether to select garment from library")
    
    # Manual overrides for library selection 
    zoom_level_override: Optional[str] = Field(None, description="Override detected zoom level")
    skin_shade_override: Optional[str] = Field(None, description="Override detected skin shade")
    jewelry_type_override: Optional[str] = Field(None, description="Override detected jewelry type")
    
    # Generation parameters
    prompt: Optional[str] = Field(config.DEFAULT_PROMPT, description="Custom prompt")
    size: Optional[Tuple[int, int]] = Field(config.DEFAULT_SIZE, description="Output size")
    num_steps: Optional[int] = Field(config.DEFAULT_STEPS, ge=1, le=100)
    guidance_scale: Optional[float] = Field(config.DEFAULT_GUIDANCE, ge=1.0, le=50.0)
    seed: Optional[int] = Field(config.DEFAULT_SEED, ge=-1)
    
    @validator('size')
    def validate_size(cls, v):
        if v and (v[0] % 8 != 0 or v[1] % 8 != 0):
            raise ValueError("Size dimensions must be divisible by 8")
        return v

class TryOnResponse(BaseModel):
    request_id: str
    status: str
    variations: List[str] = Field(..., description="Base64 encoded output images")
    ghost_image: str = Field(..., description="Base64 encoded ghost image with transparent background")
    processing_time: float
    message: Optional[str] = None
    metadata: Optional[Dict] = None
    library_match: Optional[Dict] = Field(None, description="Info about selected library item")

class QueueStatus(BaseModel):
    queue_size: int
    processing: int
    estimated_wait: float

# ===================== Image Processing Utilities =====================
class ImageProcessor:
    @staticmethod
    def decode_base64(b64_string: str) -> Image.Image:
        """Decode base64 string to PIL Image"""
        try:
            if ',' in b64_string:
                b64_string = b64_string.split(',')[1]
            
            img_bytes = base64.b64decode(b64_string)
            img = Image.open(io.BytesIO(img_bytes))
            return img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to decode image: {str(e)}")
    
    @staticmethod
    def encode_base64(image: Image.Image, format: str = "PNG", quality: int = 95) -> str:
        """Encode PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format=format, quality=quality)
        return base64.b64encode(buffered.getvalue()).decode()
    
    @staticmethod
    def create_ghost_image(original_image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Create ghost image: retain only the white areas of the original mask
        from the original image, with transparent background.
        
        Args:
            original_image: The original person image (RGB)
            mask: The original mask BEFORE inversion (L mode, white = keep)
        
        Returns:
            RGBA image with transparent background, showing only masked areas
        """
        # Ensure images are the same size
        if original_image.size != mask.size:
            mask = mask.resize(original_image.size, Image.LANCZOS)
        
        # Convert mask to L mode if not already
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        # Convert original image to RGBA
        if original_image.mode != 'RGBA':
            original_rgba = original_image.convert('RGBA')
        else:
            original_rgba = original_image.copy()
        
        # Create new RGBA image with transparent background
        ghost_image = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
        
        # Use the mask to composite: white areas (255) in mask will be visible
        # This keeps only the areas that are white in the original mask
        ghost_image = Image.composite(original_rgba, ghost_image, mask)
        
        logger.debug(f"Created ghost image with size {ghost_image.size}")
        return ghost_image
    
    @staticmethod
    def invert_mask(mask: Image.Image) -> Image.Image:
        """
        Invert the mask: black becomes white, white becomes black
        This is needed because the model expects the opposite convention
        """
        # Convert to grayscale if not already
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        # Invert the mask
        inverted = ImageOps.invert(mask)
        logger.debug("Mask inverted successfully")
        return inverted
    
    @staticmethod
    def prepare_tensors(image: Image.Image, mask: Image.Image, garment: Image.Image, 
                       size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare input tensors for the model"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # Resize images
        image = image.resize(size, Image.LANCZOS)
        mask = mask.resize(size, Image.LANCZOS)
        garment = garment.resize(size, Image.LANCZOS)
        
        # Convert to tensors
        image_tensor = transform(image)
        mask_tensor = mask_transform(mask)[:1]
        garment_tensor = transform(garment)
        
        # Create concatenated image and mask
        inpaint_image = torch.cat([garment_tensor, image_tensor], dim=2)
        garment_mask = torch.zeros_like(mask_tensor)
        extended_mask = torch.cat([garment_mask, mask_tensor], dim=2)
        
        return inpaint_image, extended_mask

# ===================== Model Manager =====================
class ModelManager:
    def __init__(self):
        self.pipe: Optional[FluxFillPipeline] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize the model pipeline"""
        async with self._lock:
            if self.pipe is not None:
                return
                
            logger.info("Initializing Flux model pipeline...")
            try:
                transformer = FluxTransformer2DModel.from_pretrained(
                    config.TRANSFORMER_DIR,
                    subfolder="transformer",
                    torch_dtype=self.dtype,
                ).to(self.device)
                
                self.pipe = FluxFillPipeline.from_pretrained(
                    config.MODEL_ID,
                    torch_dtype=self.dtype,
                    transformer=transformer,
                ).to(self.device)
                
                self.pipe.transformer.to(torch.bfloat16)
                
                # Memory optimizations
                self.pipe.enable_attention_slicing()
                self.pipe.vae.enable_tiling()
                self.pipe.vae.enable_slicing()
                
                logger.info("Model pipeline initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize model: {e}")
                raise
    
    def generate(self, inpaint_image: torch.Tensor, extended_mask: torch.Tensor,
                prompt: str, size: Tuple[int, int], num_steps: int,
                guidance_scale: float, seed: int) -> Tuple[Image.Image, Image.Image]:
        """Generate a single try-on result"""
        if self.pipe is None:
            raise RuntimeError("Model not initialized")
        
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
            
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        with torch.inference_mode():
            result = self.pipe(
                height=size[1],
                width=size[0] * 2,
                image=inpaint_image,
                mask_image=extended_mask,
                num_inference_steps=num_steps,
                generator=generator,
                max_sequence_length=512,
                guidance_scale=guidance_scale,
                prompt=prompt,
            ).images[0]
        
        width = size[0]
        garment_result = result.crop((0, 0, width, size[1]))
        tryon_result = result.crop((width, 0, width * 2, size[1]))
        
        return garment_result, tryon_result

# ===================== Request Queue Manager =====================
class QueueManager:
    def __init__(self, model_manager: ModelManager, library_manager: LibraryManager):
        self.model_manager = model_manager
        self.library_manager = library_manager
        self.classifiers = ExternalClassifiers()
        self.queue = asyncio.Queue(maxsize=config.QUEUE_SIZE)
        self.processing_count = 0
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
        self._stop_event = asyncio.Event()
        
    async def start(self):
        """Start the queue processor"""
        asyncio.create_task(self._process_queue())
        
    async def stop(self):
        """Stop the queue processor"""
        self._stop_event.set()
        
    async def add_request(self, request: TryOnRequest) -> asyncio.Future:
        """Add a request to the queue"""
        future = asyncio.get_event_loop().create_future()
        
        try:
            await asyncio.wait_for(
                self.queue.put((request, future)),
                timeout=5.0
            )
            return future
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service queue is full. Please try again later."
            )
    
    async def _process_queue(self):
        """Background task to process requests from the queue"""
        while not self._stop_event.is_set():
            try:
                try:
                    request, future = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                self.processing_count += 1
                
                try:
                    # Process with garment selection
                    result = await self._process_request_async(request)
                    future.set_result(result)
                    
                except Exception as e:
                    logger.error(f"Error processing request {request.request_id}: {e}")
                    future.set_exception(e)
                    
                finally:
                    self.processing_count -= 1
                    self.queue.task_done()
                    
            except Exception as e:
                logger.error(f"Queue processor error: {e}")
                await asyncio.sleep(1)
    
    async def _process_request_async(self, request: TryOnRequest) -> TryOnResponse:
        """Asynchronous request processing with library selection and ghost image"""
        start_time = time.time()
        logger.info(f"Processing request {request.request_id}")
        
        try:
            # Decode input images
            image = ImageProcessor.decode_base64(request.image)
            mask_original = ImageProcessor.decode_base64(request.mask)
            
            # Create ghost image BEFORE inverting the mask
            # Ghost image shows white areas of original mask with transparent bg
            ghost_image = ImageProcessor.create_ghost_image(image, mask_original)
            ghost_image_b64 = ImageProcessor.encode_base64(ghost_image, format="PNG")
            logger.info("Ghost image created with transparent background")
            
            # NOW invert the mask for processing
            mask_inverted = ImageProcessor.invert_mask(mask_original)
            logger.info("Mask inverted for processing")
            import numpy as np
            arr = np.array(mask_inverted)
            logger.info(f"[CORE] mask_inverted stats min={arr.min()} max={arr.max()} mean={arr.mean():.1f}")
            
            # Get or select garment
            library_info = None
            if request.use_library:
                # Use external classifiers or overrides
                zoom_level = request.zoom_level_override or await self.classifiers.classify_zoom(image)
                skin_shade = request.skin_shade_override or await self.classifiers.detect_skin_shade(image)
                jewelry_type = request.jewelry_type_override or await self.classifiers.classify_jewelry(image)
                
                logger.info(f"Classification: zoom={zoom_level}, shade={skin_shade}, jewelry={jewelry_type}")
                
                # Get garment from library
                garment = self.library_manager.get_garment_image(zoom_level, skin_shade, jewelry_type)
                
                if garment is None:
                    raise ValueError("Failed to get garment from library")
                
                library_info = {
                    "zoom_level": zoom_level,
                    "skin_shade": skin_shade,
                    "jewelry_type": jewelry_type
                }
            else:
                # Use provided garment
                if not request.garment:
                    raise ValueError("Garment image required when not using library")
                garment = ImageProcessor.decode_base64(request.garment)
            
            # Run generation in executor
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._generate_variations_sync,
                image, mask_inverted, garment, request
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Request {request.request_id} completed in {processing_time:.2f}s")
            
            return TryOnResponse(
                request_id=request.request_id,
                status="success",
                variations=result,
                ghost_image=ghost_image_b64,
                processing_time=processing_time,
                library_match=library_info,
                metadata={
                    "size": request.size,
                    "num_steps": request.num_steps,
                    "guidance_scale": request.guidance_scale
                }
            )
            
        except Exception as e:
            logger.error(f"Error in request {request.request_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Processing failed: {str(e)}"
            )
    
    def _generate_variations_sync(self, image: Image.Image, mask: Image.Image, 
                                  garment: Image.Image, request: TryOnRequest) -> List[str]:
        """Synchronous generation of variations"""
        # Prepare tensors
        inpaint_image, extended_mask = ImageProcessor.prepare_tensors(
            image, mask, garment, request.size
        )
        
        # Generate variations
        variations = []
        for i in range(request.num_variations):
            logger.debug(f"Generating variation {i+1}/{request.num_variations}")
            
            current_seed = request.seed
            if request.seed == -1:
                current_seed = -1
            elif request.num_variations > 1 and i > 0:
                current_seed = request.seed + i
            
            _, tryon_result = self.model_manager.generate(
                inpaint_image=inpaint_image,
                extended_mask=extended_mask,
                prompt=request.prompt,
                size=request.size,
                num_steps=request.num_steps,
                guidance_scale=request.guidance_scale,
                seed=current_seed
            )
            
            b64_image = ImageProcessor.encode_base64(tryon_result)
            variations.append(b64_image)
        
        return variations
    
    def get_status(self) -> QueueStatus:
        """Get current queue status"""
        queue_size = self.queue.qsize()
        estimated_wait = (queue_size + self.processing_count) * 30
        
        return QueueStatus(
            queue_size=queue_size,
            processing=self.processing_count,
            estimated_wait=estimated_wait
        )

# ===================== FastAPI Application =====================
model_manager = ModelManager()
library_manager = LibraryManager()
queue_manager = QueueManager(model_manager, library_manager)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting Flux Try-On API with Library Support and Ghost Images...")
    await model_manager.initialize()
    await queue_manager.start()
    logger.info("API ready to accept requests")
    logger.info(f"Library loaded with {len(library_manager.library)} items")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await queue_manager.stop()

app = FastAPI(
    title="Flux Virtual Try-On API with Ghost Images",
    description="Virtual try-on API with automatic garment selection and ghost image generation",
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== API Endpoints =====================
@app.post("/tryon", response_model=TryOnResponse)
async def virtual_tryon(request: TryOnRequest):
    """
    Generate virtual try-on images with automatic garment selection.
    
    Features:
    - Automatic mask inversion
    - Ghost image generation (transparent background)
    - Garment selection from library based on zoom/shade/jewelry
    - Multiple variation generation
    
    Returns:
    - variations: List of try-on result images
    - ghost_image: Original image areas under white mask with transparent background
    """
    logger.info(f"Received request {request.request_id}, use_library={request.use_library}")
    
    future = await queue_manager.add_request(request)
    
    try:
        result = await asyncio.wait_for(future, timeout=config.REQUEST_TIMEOUT)
        return result
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request processing timeout"
        )

@app.get("/library/info")
async def get_library_info():
    """Get information about the loaded library"""
    return {
        "total_items": len(library_manager.library),
        "zoom_levels": list(set(item.get("zoom_level", "") for item in library_manager.library)),
        "skin_shades": list(set(item.get("skin_shade", "") for item in library_manager.library)),
        "jewelry_types": list(set(item.get("jewelry", "") for item in library_manager.library))
    }

@app.get("/status", response_model=QueueStatus)
async def get_status():
    """Get current queue status"""
    return queue_manager.get_status()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model_manager.pipe is not None,
        "library_loaded": len(library_manager.library) > 0,
        "device": model_manager.device,
        "features": ["mask_inversion", "ghost_images", "library_selection"]
    }

# ===================== Main Entry Point =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=18010,
        workers=1,
        log_level="info",
        reload=False
    )
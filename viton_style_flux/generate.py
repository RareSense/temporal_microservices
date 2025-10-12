import requests
import base64
import json
from PIL import Image, ImageDraw
import io
from typing import Optional, Dict, Any

def encode_image(image_path: str) -> str:
    """Encode image file to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def encode_pil_image(img: Image.Image) -> str:
    """Encode PIL Image to base64"""
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()

def decode_and_save_image(b64_string: str, output_path: str, format: str = "PNG"):
    """Decode base64 and save as image"""
    img_data = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_data))

    if img.mode == "RGBA" and format.upper() == "PNG":
        img.save(output_path, format=format)
    else:
        if img.mode == "RGBA":
            img = img.convert("RGB")
        img.save(output_path, format=format if format.upper() != "PNG" else "JPEG", quality=95)
    return img

def create_checkered_background(size: tuple, square_size: int = 20) -> Image.Image:
    """Create a checkered background for previewing transparent images"""
    img = Image.new('RGB', size, (255, 255, 255))
    pixels = img.load()
    
    for i in range(0, size[0], square_size):
        for j in range(0, size[1], square_size):
            if (i // square_size + j // square_size) % 2:
                for x in range(i, min(i + square_size, size[0])):
                    for y in range(j, min(j + square_size, size[1])):
                        pixels[x, y] = (200, 200, 200)
    
    return img

def test_ghost_image_saving(ghost_image_b64: str, request_id: str):
    """Save ghost image with transparent background"""
    try:
        img_data = base64.b64decode(ghost_image_b64)
        ghost_img = Image.open(io.BytesIO(img_data))
        
        if ghost_img.mode == "RGBA":
            output_path = f"ghost_{request_id}.png"
            ghost_img.save(output_path, "PNG")
            print(f"   ✓ Ghost image saved: {output_path} (RGBA with transparency)")
            
            preview = create_checkered_background(ghost_img.size)
            preview.paste(ghost_img, (0, 0), ghost_img)
            preview_path = f"ghost_{request_id}_preview.png"
            preview.save(preview_path, "PNG")
            print(f"   ✓ Ghost preview saved: {preview_path} (with checkered background)")
        else:
            print(f"   ⚠ Ghost image is not RGBA: {ghost_img.mode}")
            
    except Exception as e:
        print(f"   ✗ Error saving ghost image: {e}")

def test_tryon_with_library(
    person_image_path: str = "person.jpg",
    mask_image_path: str = "mask.png",
    api_url: str = "http://localhost:18010",
    num_variations: int = 2,
    # Optional overrides for library selection
    zoom_override: Optional[str] = None,
    shade_override: Optional[str] = None,
    jewelry_override: Optional[str] = None
):
    """
    Test try-on using automatic garment selection from library.
    The mask will be automatically inverted by the API.
    Also generates a ghost image with transparent background.
    """
    print("=" * 60)
    print("TESTING WITH LIBRARY GARMENT SELECTION")
    print("=" * 60)
    
    # Encode images
    print(f"\n1. Loading images...")
    image_b64 = encode_image(person_image_path)
    mask_b64 = encode_image(mask_image_path)
    print(f"   ✓ Person image: {person_image_path}")
    print(f"   ✓ Mask image: {mask_image_path}")
    print(f"     → White areas will become ghost image")
    print(f"     → Mask will be inverted for processing")
    
    # Prepare request for library-based selection
    payload = {
        "image": image_b64,
        "mask": mask_b64,  # Will be inverted automatically
        "use_library": True,  # Use library for garment selection
        "num_variations": num_variations,
        "num_steps": 30,
        "guidance_scale": 30.0,
        "prompt": "High quality fashion photography, professional studio lighting"
    }
    
    if zoom_override:
        payload["zoom_level_override"] = zoom_override
        print(f"   → Zoom override: {zoom_override}")
    if shade_override:
        payload["skin_shade_override"] = shade_override
        print(f"   → Skin shade override: {shade_override}")
    if jewelry_override:
        payload["jewelry_type_override"] = jewelry_override
        print(f"   → Jewelry override: {jewelry_override}")
    
    print(f"\n2. Sending request to {api_url}/tryon...")
    print(f"   → Requesting {num_variations} variations")
    print(f"   → Using library for garment selection")
    print(f"   → Will generate ghost image")
    
    try:
        response = requests.post(f"{api_url}/tryon", json=payload, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n3. Success! Request ID: {result['request_id']}")
            print(f"   → Processing time: {result['processing_time']:.2f} seconds")
            
            # Show library match info if available
            if result.get('library_match'):
                match = result['library_match']
                print(f"\n4. Library Match Info:")
                print(f"   → Zoom level: {match.get('zoom_level', 'N/A')}")
                print(f"   → Skin shade: {match.get('skin_shade', 'N/A')}")
                print(f"   → Jewelry type: {match.get('jewelry_type', 'N/A')}")
            
            # Save ghost image
            print(f"\n5. Saving ghost image...")
            test_ghost_image_saving(result.get('ghost_image', ''), result['request_id'])
            
            # Save output images
            print(f"\n6. Saving {len(result['variations'])} output images...")
            for i, img_b64 in enumerate(result['variations']):
                output_path = f"output_library_{i}.png"
                decode_and_save_image(img_b64, output_path)
                print(f"   ✓ Saved: {output_path}")
                
        else:
            print(f"\n✗ Error: {response.status_code}")
            print(f"   Response: {response.json()}")
            
    except requests.exceptions.Timeout:
        print("\n✗ Request timed out (> 5 minutes)")
    except Exception as e:
        print(f"\n✗ Error: {e}")

def test_tryon_with_custom_garment(
    person_image_path: str = "person.jpg",
    mask_image_path: str = "mask.png",
    garment_image_path: str = "garment.jpg",
    api_url: str = "http://localhost:18010",
    num_variations: int = 1
):
    """
    Test try-on with a custom garment image (not from library).
    The mask will be automatically inverted by the API.
    Also generates a ghost image with transparent background.
    """
    print("\n" + "=" * 60)
    print("TESTING WITH CUSTOM GARMENT")
    print("=" * 60)
    
    # Encode images
    print(f"\n1. Loading images...")
    image_b64 = encode_image(person_image_path)
    mask_b64 = encode_image(mask_image_path)
    garment_b64 = encode_image(garment_image_path)
    print(f"   ✓ Person image: {person_image_path}")
    print(f"   ✓ Mask image: {mask_image_path}")
    print(f"     → White areas will become ghost image")
    print(f"     → Mask will be inverted for processing")
    print(f"   ✓ Garment image: {garment_image_path}")

    payload = {
        "image": image_b64,
        "mask": mask_b64,  
        "garment": garment_b64,
        "use_library": False,  # Don't use library
        "num_variations": num_variations,
        "num_steps": 30,
        "guidance_scale": 30.0
    }
    
    print(f"\n2. Sending request to {api_url}/tryon...")
    print(f"   → Requesting {num_variations} variations")
    print(f"   → Using custom garment")
    print(f"   → Will generate ghost image")
    
    try:
        response = requests.post(f"{api_url}/tryon", json=payload, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n3. Success! Request ID: {result['request_id']}")
            print(f"   → Processing time: {result['processing_time']:.2f} seconds")
            
            # Save ghost image
            print(f"\n4. Saving ghost image...")
            test_ghost_image_saving(result.get('ghost_image', ''), result['request_id'])
            
            # Save output images
            print(f"\n5. Saving {len(result['variations'])} output images...")
            for i, img_b64 in enumerate(result['variations']):
                output_path = f"output_custom_{i}.png"
                decode_and_save_image(img_b64, output_path)
                print(f"   ✓ Saved: {output_path}")
                
        else:
            print(f"\n✗ Error: {response.status_code}")
            print(f"   Response: {response.json()}")
            
    except requests.exceptions.Timeout:
        print("\n✗ Request timed out (> 5 minutes)")
    except Exception as e:
        print(f"\n✗ Error: {e}")

def test_ghost_image_feature(api_url: str = "http://localhost:18010"):
    """
    Specific test for ghost image generation feature.
    Creates a mask with specific patterns to test ghost image extraction.
    """
    print("\n" + "=" * 60)
    print("TESTING GHOST IMAGE FEATURE")
    print("=" * 60)
    
    print("\n1. Creating test images with specific mask pattern...")
    
    # Create a test person image with colored regions
    person_img = Image.new('RGB', (768, 1024), (100, 150, 200))
    # Add some distinct features
    draw = ImageDraw.Draw(person_img)
    # Draw a red rectangle (will be in ghost area)
    draw.rectangle([200, 300, 400, 500], fill=(255, 0, 0))
    # Draw a green circle (will be outside ghost area)
    draw.ellipse([400, 600, 600, 800], fill=(0, 255, 0))
    
    # Create mask with specific white areas
    mask_img = Image.new('L', (768, 1024), 0)  # Start with black
    draw_mask = ImageDraw.Draw(mask_img)
    # Make the red rectangle area white (will be in ghost)
    draw_mask.rectangle([200, 300, 400, 500], fill=255)
    # Add another white region
    draw_mask.ellipse([100, 100, 300, 300], fill=255)
    
    print("   ✓ Created person image with red rectangle and green circle")
    print("   ✓ Created mask with white areas over red rectangle")
    
    # Create a simple garment
    garment_img = Image.new('RGB', (768, 1024), (200, 100, 150))
    
    # Save test images for reference
    person_img.save("test_person.png")
    mask_img.save("test_mask.png")
    garment_img.save("test_garment.png")
    print("   ✓ Saved test images: test_person.png, test_mask.png, test_garment.png")
    
    # Properly encode images to base64
    image_b64 = encode_pil_image(person_img)
    mask_b64 = encode_pil_image(mask_img)
    garment_b64 = encode_pil_image(garment_img)
    
    # Prepare request
    payload = {
        "image": image_b64,
        "mask": mask_b64,
        "garment": garment_b64,
        "use_library": False,
        "num_variations": 1,
        "num_steps": 20,  # Fewer steps for testing
        "guidance_scale": 25.0
    }
    
    print("\n2. Sending request to test ghost image generation...")
    
    try:
        response = requests.post(f"{api_url}/tryon", json=payload, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n3. Success! Request ID: {result['request_id']}")
            
            # Analyze ghost image
            print(f"\n4. Analyzing ghost image...")
            ghost_b64 = result.get('ghost_image', '')
            if ghost_b64:
                ghost_data = base64.b64decode(ghost_b64)
                ghost_img = Image.open(io.BytesIO(ghost_data))
                
                print(f"   → Ghost image mode: {ghost_img.mode}")
                print(f"   → Ghost image size: {ghost_img.size}")
                
                if ghost_img.mode == "RGBA":
                    # Check transparency
                    alpha = ghost_img.split()[-1]
                    alpha_array = list(alpha.getdata())
                    transparent_pixels = sum(1 for p in alpha_array if p == 0)
                    opaque_pixels = sum(1 for p in alpha_array if p > 0)
                    
                    print(f"   → Transparent pixels: {transparent_pixels}")
                    print(f"   → Opaque/semi-opaque pixels: {opaque_pixels}")
                    print(f"   → Transparency ratio: {transparent_pixels / len(alpha_array) * 100:.1f}%")
                
                # Save ghost image
                test_ghost_image_saving(ghost_b64, "feature_test")
                
                print("\n   ✓ Ghost image should show only the red rectangle and circle areas")
                print("   ✓ Green circle area should be transparent")
                
        else:
            print(f"\n✗ Error: {response.status_code}")
            print(f"   Response: {response.json()}")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")

def check_library_info(api_url: str = "http://localhost:18010"):
    """Check what's available in the library"""
    print("\n" + "=" * 60)
    print("LIBRARY INFORMATION")
    print("=" * 60)
    
    try:
        response = requests.get(f"{api_url}/library/info", timeout=10)
        if response.status_code == 200:
            info = response.json()
            print(f"\nLibrary contains {info['total_items']} items")
            print(f"\nAvailable zoom levels:")
            for zoom in info['zoom_levels']:
                print(f"  - {zoom}")
            print(f"\nAvailable skin shades:")
            for shade in info['skin_shades']:
                print(f"  - {shade}")
            print(f"\nAvailable jewelry types:")
            for jewelry in info['jewelry_types']:
                print(f"  - {jewelry}")
        else:
            print(f"Failed to get library info: {response.status_code}")
    except Exception as e:
        print(f"Error checking library: {e}")

def check_api_health(api_url: str = "http://localhost:18010"):
    """Check API health status"""
    print("\n" + "=" * 60)
    print("API HEALTH CHECK")
    print("=" * 60)
    
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"\n✓ API Status: {health['status']}")
            print(f"✓ Model loaded: {health.get('model_loaded', False)}")
            print(f"✓ Library loaded: {health.get('library_loaded', False)}")
            print(f"✓ Device: {health.get('device', 'unknown')}")
            print(f"✓ Features: {', '.join(health.get('features', []))}")
            print(f"✓ Timestamp: {health.get('timestamp', 'N/A')}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to API: {e}")
        return False

if __name__ == "__main__":
    API_URL = "http://localhost:18010"
    
    # 1. Check API health
    if not check_api_health(API_URL):
        print("\n⚠️  Please ensure the API is running before testing!")
        exit(1)
    
    # 2. Check library information
    check_library_info(API_URL)
    
    # 3. Test ghost image feature specifically
    print("\n" + "=" * 60)
    print("TEST 1: Ghost Image Feature Test")
    print("=" * 60)
    test_ghost_image_feature(API_URL)
    
    # 4. Test with library selection 
    print("\n" + "=" * 60)
    print("TEST 2: Library-based garment selection with ghost image")
    print("=" * 60)
    
    # Uncomment these tests when you have real images
    # test_tryon_with_library(
    #     person_image_path="person.jpg",
    #     mask_image_path="mask.png",
    #     api_url=API_URL,
    #     num_variations=2
    # )
    
    # # 5. Test with library selection and overrides
    # print("\n" + "=" * 60)
    # print("TEST 3: Library selection with manual overrides")
    # print("=" * 60)
    
    # test_tryon_with_library(
    #     person_image_path="person.jpg",
    #     mask_image_path="mask.png",
    #     api_url=API_URL,
    #     num_variations=1,
    #     zoom_override="bust shot",
    #     shade_override="medium",
    #     jewelry_override="bracelet"
    # )
    
    # # 6. Test with custom garment
    # print("\n" + "=" * 60)
    # print("TEST 4: Custom garment with ghost image")
    # print("=" * 60)
    
    # test_tryon_with_custom_garment(
    #     person_image_path="person.jpg",
    #     mask_image_path="mask.png",
    #     garment_image_path="custom_garment.jpg",
    #     api_url=API_URL,
    #     num_variations=1
    # )
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("Check the generated ghost images:")
    print("  - ghost_*.png: Transparent background images")
    print("  - ghost_*_preview.png: Preview with checkered background")
    print("=" * 60)
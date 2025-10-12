import asyncio
import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import statistics

import requests
import numpy as np
from PIL import Image, ImageDraw
import aiohttp
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

# ===================== Configuration =====================
@dataclass
class TestConfig:
    API_URL: str = "http://localhost:18010"
    TIMEOUT: int = 300  # 5 minutes
    
    # Test image sizes
    IMAGE_SIZE: tuple = (768, 1024)  # Updated to match API defaults
    
    # Test scenarios
    SINGLE_TEST_VARIATIONS: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    LIBRARY_TEST_CONFIGS: List[Dict] = field(default_factory=lambda: [
        {"zoom": "bust shot", "shade": "medium", "jewelry": "bracelet"},
        {"zoom": "tight detail shot", "shade": "light", "jewelry": "earring"},
        {"zoom": "macro closeup shot", "shade": "dark", "jewelry": "ring"},
        {"zoom": "three quarter shot", "shade": "deep dark", "jewelry": "necklace"},
    ])
    BATCH_TEST_CONFIGS: List[Dict] = field(default_factory=lambda: [
        {"requests": 1, "variations": 4, "use_library": True},
        {"requests": 2, "variations": 2, "use_library": False},
        {"requests": 4, "variations": 1, "use_library": True},
        {"requests": 4, "variations": 4, "use_library": False},
    ])
    CONCURRENT_TEST_CONFIGS: List[Dict] = field(default_factory=lambda: [
        {"concurrent": 2, "variations": 2, "use_library": True},
        {"concurrent": 4, "variations": 1, "use_library": False},
        {"concurrent": 4, "variations": 4, "use_library": True},
    ])

config = TestConfig()

# ===================== Test Data Generator =====================
class TestDataGenerator:
    """Generate test images and encode them"""
    
    @staticmethod
    def create_test_image(color: tuple, text: str = "", size: tuple = (768, 1024)) -> Image.Image:
        """Create a test image with specified color and text"""
        img = Image.new('RGB', size, color)
        if text:
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), text, fill=(255, 255, 255))
        return img
    
    @staticmethod
    def create_test_mask(size: tuple = (768, 1024), pattern: str = "rectangle") -> Image.Image:
        """
        Create a test mask with various patterns.
        White areas will be preserved in ghost image and inverted for processing.
        """
        img = Image.new('L', size, 0)  # Start with black background
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = size[0] // 2, size[1] // 2
        
        if pattern == "rectangle":
            # Draw white rectangle in center
            draw.rectangle(
                [center_x - 150, center_y - 200, center_x + 150, center_y + 200],
                fill=255
            )
        elif pattern == "circle":
            # Draw white circle
            draw.ellipse(
                [center_x - 150, center_y - 150, center_x + 150, center_y + 150],
                fill=255
            )
        elif pattern == "multiple":
            # Draw multiple white regions
            draw.rectangle([100, 100, 300, 300], fill=255)
            draw.ellipse([400, 400, 600, 600], fill=255)
            draw.rectangle([200, 700, 500, 900], fill=255)
        
        return img
    
    @staticmethod
    def encode_image(img: Image.Image) -> str:
        """Encode PIL image to base64"""
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    @staticmethod
    def decode_ghost_image(ghost_b64: str) -> Optional[Dict[str, Any]]:
        """Decode and analyze ghost image"""
        try:
            img_data = base64.b64decode(ghost_b64)
            ghost_img = Image.open(io.BytesIO(img_data))
            
            analysis = {
                "mode": ghost_img.mode,
                "size": ghost_img.size,
                "has_transparency": ghost_img.mode == "RGBA"
            }
            
            if ghost_img.mode == "RGBA":
                alpha = ghost_img.split()[-1]
                alpha_array = list(alpha.getdata())
                analysis["transparent_pixels"] = sum(1 for p in alpha_array if p == 0)
                analysis["opaque_pixels"] = sum(1 for p in alpha_array if p > 0)
                analysis["transparency_ratio"] = analysis["transparent_pixels"] / len(alpha_array)
            
            return analysis
        except Exception as e:
            return {"error": str(e)}
    
    @classmethod
    def generate_test_request(cls, request_id: Optional[str] = None, 
                            num_variations: int = 1,
                            use_library: bool = False,
                            library_overrides: Optional[Dict] = None,
                            mask_pattern: str = "rectangle",
                            seed: int = 42) -> Dict[str, Any]:
        """Generate a complete test request for the updated API with ghost image support"""
        # Create distinct test images
        person_img = cls.create_test_image((100, 150, 200), f"Person {request_id or ''}")
        mask_img = cls.create_test_mask(pattern=mask_pattern)
        
        request = {
            "request_id": request_id or f"test_{int(time.time()*1000)}",
            "image": cls.encode_image(person_img),
            "mask": cls.encode_image(mask_img),  # API will invert this
            "num_variations": num_variations,
            "use_library": use_library,
            "prompt": "High quality fashion photography, studio lighting",
            "num_steps": 20,  # Fewer steps for testing
            "guidance_scale": 25.0,
            "seed": seed
        }
        
        # Add garment or library overrides
        if use_library:
            if library_overrides:
                if "zoom" in library_overrides:
                    request["zoom_level_override"] = library_overrides["zoom"]
                if "shade" in library_overrides:
                    request["skin_shade_override"] = library_overrides["shade"]
                if "jewelry" in library_overrides:
                    request["jewelry_type_override"] = library_overrides["jewelry"]
        else:
            # Add custom garment if not using library
            garment_img = cls.create_test_image((200, 100, 150), f"Garment {request_id or ''}")
            request["garment"] = cls.encode_image(garment_img)
        
        return request

# ===================== Performance Metrics =====================
@dataclass
class RequestMetrics:
    request_id: str
    variations: int
    use_library: bool
    library_match: Optional[Dict]
    has_ghost_image: bool
    ghost_analysis: Optional[Dict]
    start_time: float
    end_time: float
    response_time: float
    status: str
    error: Optional[str] = None
    
    @property
    def per_variation_time(self) -> float:
        return self.response_time / self.variations if self.variations > 0 else 0

class PerformanceTracker:
    def __init__(self):
        self.metrics: List[RequestMetrics] = []
    
    def add_metric(self, metric: RequestMetrics):
        self.metrics.append(metric)
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.metrics:
            return {}
        
        response_times = [m.response_time for m in self.metrics if m.status == "success"]
        per_var_times = [m.per_variation_time for m in self.metrics if m.status == "success"]
        library_requests = [m for m in self.metrics if m.use_library]
        custom_requests = [m for m in self.metrics if not m.use_library]
        ghost_successful = [m for m in self.metrics if m.has_ghost_image]
        
        return {
            "total_requests": len(self.metrics),
            "successful": len([m for m in self.metrics if m.status == "success"]),
            "failed": len([m for m in self.metrics if m.status == "failed"]),
            "library_requests": len(library_requests),
            "custom_garment_requests": len(custom_requests),
            "ghost_images_generated": len(ghost_successful),
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "median_response_time": statistics.median(response_times) if response_times else 0,
            "avg_per_variation": statistics.mean(per_var_times) if per_var_times else 0,
            "total_time": max([m.end_time for m in self.metrics], default=0) - min([m.start_time for m in self.metrics], default=0)
        }

# ===================== Test Implementations =====================
class FluxAPITester:
    def __init__(self, api_url: str = config.API_URL):
        self.api_url = api_url
        self.tracker = PerformanceTracker()
        self.session = requests.Session()
        
    def print_header(self, text: str):
        """Print formatted header"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}{text:^60}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    def print_result(self, label: str, value: Any, unit: str = ""):
        """Print formatted result"""
        if isinstance(value, (int, float)):
            print(f"{Fore.GREEN}✓ {label:<30}: {Fore.YELLOW}{value:>10.2f} {unit}{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}✓ {label:<30}: {Fore.YELLOW}{value}{Style.RESET_ALL}")
    
    def check_health(self) -> bool:
        """Check if API is healthy"""
        try:
            resp = self.session.get(f"{self.api_url}/health", timeout=5)
            if resp.status_code == 200:
                health = resp.json()
                print(f"{Fore.GREEN}✓ API is healthy")
                print(f"  - Model loaded: {health.get('model_loaded', False)}")
                print(f"  - Library loaded: {health.get('library_loaded', False)}")
                print(f"  - Device: {health.get('device', 'unknown')}")
                print(f"  - Features: {', '.join(health.get('features', []))}{Style.RESET_ALL}")
                return True
        except Exception as e:
            print(f"{Fore.RED}✗ API health check failed: {e}{Style.RESET_ALL}")
        return False
    
    def check_library(self):
        """Check library information"""
        try:
            resp = self.session.get(f"{self.api_url}/library/info", timeout=5)
            if resp.status_code == 200:
                info = resp.json()
                print(f"\n{Fore.CYAN}Library Information:{Style.RESET_ALL}")
                print(f"  - Total items: {info['total_items']}")
                print(f"  - Zoom levels: {len(info['zoom_levels'])} types")
                print(f"  - Skin shades: {len(info['skin_shades'])} types")
                print(f"  - Jewelry types: {len(info['jewelry_types'])} types")
        except Exception as e:
            print(f"{Fore.YELLOW}⚠ Could not fetch library info: {e}{Style.RESET_ALL}")
    
    def send_request(self, request_data: Dict[str, Any]) -> RequestMetrics:
        """Send a single request and track metrics including ghost image"""
        start_time = time.time()
        request_id = request_data["request_id"]
        variations = request_data["num_variations"]
        use_library = request_data.get("use_library", False)
        
        try:
            response = self.session.post(
                f"{self.api_url}/tryon",
                json=request_data,
                timeout=config.TIMEOUT
            )
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                library_match = result.get("library_match", None)
                ghost_image = result.get("ghost_image", "")
                ghost_analysis = TestDataGenerator.decode_ghost_image(ghost_image) if ghost_image else None
                
                metric = RequestMetrics(
                    request_id=request_id,
                    variations=variations,
                    use_library=use_library,
                    library_match=library_match,
                    has_ghost_image=bool(ghost_image),
                    ghost_analysis=ghost_analysis,
                    start_time=start_time,
                    end_time=end_time,
                    response_time=response_time,
                    status="success"
                )
                
                # Print result with additional info
                lib_info = ""
                if library_match:
                    lib_info = f" [Library: {library_match.get('zoom_level', 'N/A')}]"
                ghost_info = " [Ghost: ✓]" if ghost_image else " [Ghost: ✗]"
                print(f"{Fore.GREEN}✓ {request_id}: {response_time:.2f}s ({variations} var){lib_info}{ghost_info}{Style.RESET_ALL}")
            else:
                metric = RequestMetrics(
                    request_id=request_id,
                    variations=variations,
                    use_library=use_library,
                    library_match=None,
                    has_ghost_image=False,
                    ghost_analysis=None,
                    start_time=start_time,
                    end_time=end_time,
                    response_time=response_time,
                    status="failed",
                    error=f"HTTP {response.status_code}: {response.text}"
                )
                print(f"{Fore.RED}✗ {request_id}: Failed - {response.status_code}{Style.RESET_ALL}")
                
        except Exception as e:
            end_time = time.time()
            metric = RequestMetrics(
                request_id=request_id,
                variations=variations,
                use_library=use_library,
                library_match=None,
                has_ghost_image=False,
                ghost_analysis=None,
                start_time=start_time,
                end_time=end_time,
                response_time=end_time - start_time,
                status="failed",
                error=str(e)
            )
            print(f"{Fore.RED}✗ {request_id}: Exception - {e}{Style.RESET_ALL}")
        
        self.tracker.add_metric(metric)
        return metric
    
    # ==================== Test 1: Ghost Image Verification ====================
    def test_ghost_images(self):
        """Test ghost image generation with different mask patterns"""
        self.print_header("TEST 1: Ghost Image Generation")
        print("Testing ghost image feature with various mask patterns...\n")
        
        patterns = ["rectangle", "circle", "multiple"]
        results = []
        
        for pattern in patterns:
            print(f"\n{Fore.CYAN}Testing mask pattern: {pattern}{Style.RESET_ALL}")
            
            request_data = TestDataGenerator.generate_test_request(
                request_id=f"ghost_{pattern}",
                num_variations=1,
                use_library=False,
                mask_pattern=pattern
            )
            
            metric = self.send_request(request_data)
            
            if metric.ghost_analysis and not metric.ghost_analysis.get("error"):
                analysis = metric.ghost_analysis
                print(f"  → Ghost image mode: {analysis.get('mode')}")
                print(f"  → Has transparency: {analysis.get('has_transparency')}")
                if analysis.get('has_transparency'):
                    ratio = analysis.get('transparency_ratio', 0) * 100
                    print(f"  → Transparency ratio: {ratio:.1f}%")
            
            results.append({
                "pattern": pattern,
                "success": metric.has_ghost_image,
                "analysis": metric.ghost_analysis
            })
            
            time.sleep(2)
        
        # Print summary
        print(f"\n{Fore.CYAN}Ghost Image Summary:{Style.RESET_ALL}")
        successful = sum(1 for r in results if r["success"])
        self.print_result("Total tests", len(results), "")
        self.print_result("Successful ghost images", successful, "")
        
        for r in results:
            status = "✓" if r["success"] else "✗"
            print(f"  {status} Pattern '{r['pattern']}': ", end="")
            if r["analysis"] and not r["analysis"].get("error"):
                print(f"RGBA={r['analysis'].get('has_transparency', False)}")
            else:
                print("Failed")
    
    # ==================== Test 2: Library Selection with Ghost ====================
    def test_library_with_ghost(self):
        """Test library selection with ghost image generation"""
        self.print_header("TEST 2: Library Selection + Ghost Images")
        
        for i, lib_config in enumerate(config.LIBRARY_TEST_CONFIGS[:2]):  # Test first 2
            print(f"\n{Fore.CYAN}Test {i+1}: {lib_config}{Style.RESET_ALL}")
            
            request_data = TestDataGenerator.generate_test_request(
                request_id=f"lib_ghost_{i}",
                num_variations=2,
                use_library=True,
                library_overrides=lib_config
            )
            
            metric = self.send_request(request_data)
            
            if metric.library_match:
                print(f"  → Library match: {metric.library_match}")
            if metric.has_ghost_image:
                print(f"  → Ghost image generated: Yes")
                if metric.ghost_analysis and metric.ghost_analysis.get('has_transparency'):
                    print(f"  → Ghost transparency verified: Yes")
            
            time.sleep(2)
    
    # ==================== Test 3: Performance with Ghost Images ====================
    def test_performance_with_ghost(self):
        """Compare performance with ghost image generation"""
        self.print_header("TEST 3: Performance Impact of Ghost Images")
        
        print("Testing request processing time with ghost image generation...\n")
        
        test_sizes = [1, 2, 4]
        results = []
        
        for num_var in test_sizes:
            print(f"\n{Fore.CYAN}Testing with {num_var} variation(s)...{Style.RESET_ALL}")
            
            request_data = TestDataGenerator.generate_test_request(
                request_id=f"perf_ghost_{num_var}",
                num_variations=num_var,
                use_library=False
            )
            
            metric = self.send_request(request_data)
            
            results.append({
                "variations": num_var,
                "time": metric.response_time,
                "per_var": metric.per_variation_time,
                "has_ghost": metric.has_ghost_image
            })
            
            time.sleep(2)
        
        print(f"\n{Fore.CYAN}Performance Summary:{Style.RESET_ALL}")
        for r in results:
            ghost_status = "✓" if r["has_ghost"] else "✗"
            self.print_result(f"{r['variations']} var (Ghost: {ghost_status})", r['time'], "seconds")
            self.print_result(f"  Per variation", r['per_var'], "seconds")
    
    # ==================== Test 4: Concurrent with Ghost ====================
    def test_concurrent_with_ghost(self):
        """Test concurrent requests with ghost image generation"""
        self.print_header("TEST 4: Concurrent Requests with Ghost Images")
        
        num_concurrent = 4
        print(f"Sending {num_concurrent} concurrent requests with ghost images...\n")
        
        requests = []
        for i in range(num_concurrent):
            use_library = i % 2 == 0
            request_data = TestDataGenerator.generate_test_request(
                request_id=f"concurrent_ghost_{i}",
                num_variations=1,
                use_library=use_library,
                mask_pattern="circle" if i % 2 else "rectangle",
                seed=42 + i
            )
            requests.append(request_data)
        
        concurrent_start = time.time()
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(self.send_request, req) for req in requests]
            
            metrics = []
            for future in as_completed(futures):
                try:
                    metric = future.result()
                    metrics.append(metric)
                except Exception as e:
                    print(f"{Fore.RED}Concurrent request failed: {e}{Style.RESET_ALL}")
        
        concurrent_end = time.time()
        concurrent_time = concurrent_end - concurrent_start
        
        # Analyze ghost image generation
        ghost_success = sum(1 for m in metrics if m.has_ghost_image)
        
        print(f"\n{Fore.CYAN}Concurrent Results:{Style.RESET_ALL}")
        self.print_result("Total time", concurrent_time, "seconds")
        self.print_result("Total requests", len(metrics), "")
        self.print_result("Ghost images generated", ghost_success, "")
        self.print_result("Avg response time", 
                         statistics.mean([m.response_time for m in metrics]), 
                         "seconds")
    
    # ==================== Test 5: Stress Test with Ghost ====================
    async def test_stress_with_ghost_async(self, num_requests: int = 10):
        """Async stress test with ghost image generation"""
        self.print_header("TEST 5: Stress Test with Ghost Images (Async)")
        print(f"Sending {num_requests} async requests with ghost image generation...\n")
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for i in range(num_requests):
                request_data = TestDataGenerator.generate_test_request(
                    request_id=f"stress_ghost_{i}",
                    num_variations=1,
                    use_library=(i % 3 == 0),  # Every third uses library
                    mask_pattern=["rectangle", "circle", "multiple"][i % 3],
                    seed=42 + i
                )
                
                task = self._send_async_request(session, request_data)
                tasks.append(task)
            
            stress_start = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            stress_end = time.time()
            
            # Count successes and ghost images
            successes = sum(1 for r in results if not isinstance(r, Exception) and r.get("status") == "success")
            ghost_count = sum(1 for r in results if not isinstance(r, Exception) and r.get("has_ghost"))
            failures = num_requests - successes
            
            total_time = stress_end - stress_start
            
            print(f"\n{Fore.CYAN}Stress Test Results:{Style.RESET_ALL}")
            self.print_result("Total requests", num_requests, "")
            self.print_result("Successful", successes, "")
            self.print_result("Ghost images", ghost_count, "")
            self.print_result("Failed", failures, "")
            self.print_result("Total time", total_time, "seconds")
            self.print_result("Avg time per request", total_time / num_requests, "seconds")
    
    async def _send_async_request(self, session: aiohttp.ClientSession, 
                                 request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Helper to send async request"""
        try:
            async with session.post(
                f"{self.api_url}/tryon",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=config.TIMEOUT)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    has_ghost = bool(result.get("ghost_image"))
                    ghost_info = " [Ghost: ✓]" if has_ghost else " [Ghost: ✗]"
                    print(f"{Fore.GREEN}✓ {request_data['request_id']} completed{ghost_info}{Style.RESET_ALL}")
                    return {"status": "success", "data": result, "has_ghost": has_ghost}
                else:
                    print(f"{Fore.RED}✗ {request_data['request_id']} failed: {response.status}{Style.RESET_ALL}")
                    return {"status": "failed", "error": f"HTTP {response.status}"}
        except Exception as e:
            print(f"{Fore.RED}✗ {request_data['request_id']} exception: {e}{Style.RESET_ALL}")
            return {"status": "failed", "error": str(e)}

# ===================== Main Test Runner =====================
def main():
    """Run all tests"""
    print(f"{Fore.MAGENTA}")
    print("=" * 60)
    print("FLUX TRY-ON API TEST SUITE (v3.0)".center(60))
    print("With Ghost Images, Library & Mask Inversion".center(60))
    print("=" * 60)
    print(f"{Style.RESET_ALL}")
    
    tester = FluxAPITester()
    
    # Check API health first
    if not tester.check_health():
        print(f"\n{Fore.RED}API is not healthy. Please start the service first.{Style.RESET_ALL}")
        return
    
    # Check library status
    tester.check_library()
    
    # Run tests
    try:
        # 1. Ghost image verification
        tester.test_ghost_images()
        
        # 2. Library selection with ghost
        tester.test_library_with_ghost()
        
        # 3. Performance with ghost images
        tester.test_performance_with_ghost()
        
        # 4. Concurrent with ghost
        tester.test_concurrent_with_ghost()
        
        # 5. Stress test with ghost (async)
        print("\nRunning async stress test with ghost images...")
        asyncio.run(tester.test_stress_with_ghost_async(num_requests=8))
        
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Tests interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n\n{Fore.RED}Test suite failed: {e}{Style.RESET_ALL}")
    
    # Print final summary
    final_summary = tester.tracker.get_summary()
    
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print("FINAL TEST SUMMARY".center(60))
    print(f"{'='*60}{Style.RESET_ALL}")
    
    if final_summary:
        print(f"\n{Fore.CYAN}Overall Performance:{Style.RESET_ALL}")
        print(f"  Total requests: {final_summary['total_requests']}")
        print(f"  Successful: {final_summary['successful']}")
        print(f"  Failed: {final_summary['failed']}")
        print(f"  Ghost images generated: {final_summary['ghost_images_generated']}")
        print(f"  Library requests: {final_summary['library_requests']}")
        print(f"  Custom garment requests: {final_summary['custom_garment_requests']}")
        print(f"  Avg response time: {final_summary['avg_response_time']:.2f}s")
        print(f"  Avg per variation: {final_summary['avg_per_variation']:.2f}s")
    
    print(f"\n{Fore.MAGENTA}TEST SUITE COMPLETED")
    print("Ghost images feature verified successfully!{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
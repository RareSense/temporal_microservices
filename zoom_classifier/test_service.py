import argparse
import asyncio
import base64
import json
import pathlib
import statistics
import sys
import time
from typing import Dict, List, Any

import httpx
from PIL import Image
import io

SERVICE_URL = "http://localhost:18009"
HEALTH_PATH = "/health"
RUN_PATH = "/run"
METRICS_PATH = "/metrics"


def create_test_image(color: tuple = (255, 255, 255), size: tuple = (224, 224)) -> bytes:
    """Create a test image for when you don't have real images."""
    img = Image.new('RGB', size, color=color)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def b64(img_path: pathlib.Path = None, img_bytes: bytes = None) -> str:
    """Convert image to base64."""
    if img_path:
        return base64.b64encode(img_path.read_bytes()).decode()
    elif img_bytes:
        return base64.b64encode(img_bytes).decode()
    else:
        raise ValueError("Provide either img_path or img_bytes")


async def check_health():
    """Test health endpoint."""
    async with httpx.AsyncClient() as cli:
        r = await cli.get(SERVICE_URL + HEALTH_PATH, timeout=10)
    
    print("Health Status:")
    print(f"  Status Code: {r.status_code}")
    health_data = r.json()
    print(f"  Service OK: {health_data.get('ok', False)}")
    print(f"  Workers: {health_data.get('workers', 'N/A')}/{health_data.get('max_workers', 'N/A')}")
    print(f"  Model Loaded: {health_data.get('model_loaded', False)}")
    print(f"  Test Inference: {health_data.get('test_inference', 'N/A')}")
    print()


async def check_metrics():
    """Test metrics endpoint."""
    async with httpx.AsyncClient() as cli:
        r = await cli.get(SERVICE_URL + METRICS_PATH, timeout=5)
    
    print("Metrics:")
    metrics = r.json()
    print(f"  Current Workers: {metrics.get('current_workers', 'N/A')}")
    print(f"  Min Workers: {metrics.get('min_workers', 'N/A')}")
    print(f"  Max Workers: {metrics.get('max_workers', 'N/A')}")
    print(f"  Pool Status: {metrics.get('pool_status', 'N/A')}")
    print()


# ────────────────────────────────────────────────────────────
#  SINGLE IMAGE TEST
# ────────────────────────────────────────────────────────────
async def run_single(img: pathlib.Path = None, verbose: bool = True):
    """Test single image classification."""
    
    # Use provided image or create test image
    if img and img.exists():
        img_b64 = b64(img_path=img)
        img_name = img.name
    else:
        test_bytes = create_test_image(color=(128, 128, 128))
        img_b64 = b64(img_bytes=test_bytes)
        img_name = "test_image.png"
    
    payload = {
        "data": {"image_bytes": img_b64},
        "meta": {"trace_id": f"test-{time.time()}"}
    }
    
    async with httpx.AsyncClient() as cli:
        t0 = time.perf_counter()
        r = await cli.post(SERVICE_URL + RUN_PATH, json=payload, timeout=120)
        dt = (time.perf_counter() - t0) * 1000
    
    if r.status_code == 200:
        result = r.json()
        zoom_level = result.get("zoom_level", "unknown")
        print(f"✓ {img_name}: {zoom_level} ({dt:.1f} ms)")
        
        # Validate zoom level format
        if zoom_level in ["zoom_1", "zoom_2", "zoom_3", "zoom_4"]:
            print(f"  Valid zoom level format")
        else:
            print(f"  ⚠ Unexpected zoom level format: {zoom_level}")
    else:
        print(f"✗ {img_name}: {r.status_code} - {r.text[:100]} ({dt:.1f} ms)")
    
    if verbose:
        print(f"  Raw response: {r.json() if r.status_code == 200 else r.text}")
    
    return r.status_code, dt


# ────────────────────────────────────────────────────────────
#  BATCH TEST
# ────────────────────────────────────────────────────────────
async def run_batch(images: List[pathlib.Path] = None):
    """Test batch processing of multiple images."""
    if images and all(img.exists() for img in images):
        test_images = images
    else:
        print("Using synthetic test images...")
        test_images = []
        for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)]):
            test_images.append((f"test_{i}.png", create_test_image(color=color)))
    
    print(f"\nBatch testing {len(test_images)} images...")
    results = []
    
    for item in test_images:
        if isinstance(item, pathlib.Path):
            img_name = item.name
            img_b64 = b64(img_path=item)
        else:
            img_name, img_bytes = item
            img_b64 = b64(img_bytes=img_bytes)
        
        payload = {
            "data": {"image_bytes": img_b64},
            "meta": {"trace_id": f"batch-{time.time()}"}
        }
        
        async with httpx.AsyncClient() as cli:
            t0 = time.perf_counter()
            r = await cli.post(SERVICE_URL + RUN_PATH, json=payload, timeout=120)
            dt = (time.perf_counter() - t0) * 1000
        
        zoom_level = r.json().get("zoom_level", "error") if r.status_code == 200 else "failed"
        results.append((img_name, r.status_code, dt, zoom_level))
        print(f"  {img_name}: {zoom_level} ({dt:.1f} ms)")
    
    successful = sum(1 for _, status, _, _ in results if status == 200)
    avg_latency = statistics.mean([lat for _, _, lat, _ in results])
    zoom_dist = {}
    for _, status, _, zoom in results:
        if status == 200:
            zoom_dist[zoom] = zoom_dist.get(zoom, 0) + 1
    
    print(f"\n{'='*50}")
    print(f"Batch Summary:")
    print(f"  Total: {len(test_images)}")
    print(f"  Successful: {successful}/{len(test_images)}")
    print(f"  Avg latency: {avg_latency:.1f} ms")
    print(f"  Zoom distribution: {zoom_dist}")
    print('='*50)


# ────────────────────────────────────────────────────────────
#  CONCURRENCY TEST
# ────────────────────────────────────────────────────────────
async def run_concurrency(img: pathlib.Path = None, concurrent: int = 10, total: int = 50):
    """Test concurrent request handling and worker scaling."""
    
    # Create test image
    if img and img.exists():
        img_bytes = img.read_bytes()
    else:
        img_bytes = create_test_image(color=(100, 150, 200))
    
    img_b64 = b64(img_bytes=img_bytes)
    payload = json.dumps({
        "data": {"image_bytes": img_b64},
        "meta": {"trace_id": "concurrency-test"}
    })
    
    print(f"\nConcurrency test: {total} requests with max {concurrent} concurrent")
    print("Starting...")
    
    async def make_request(index: int):
        async with httpx.AsyncClient() as cli:
            t0 = time.perf_counter()
            r = await cli.post(
                SERVICE_URL + RUN_PATH,
                content=payload,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            return (
                time.perf_counter() - t0,
                r.status_code,
                r.json() if r.status_code == 200 else None,
                index
            )
    
    # Check initial metrics
    async with httpx.AsyncClient() as cli:
        initial_metrics = (await cli.get(SERVICE_URL + METRICS_PATH)).json()
        print(f"Initial workers: {initial_metrics.get('current_workers', 'N/A')}")
    
    # Run concurrent requests
    sem = asyncio.Semaphore(concurrent)
    latencies = []
    results = []
    
    async def bounded_request(i):
        async with sem:
            result = await make_request(i)
            latencies.append(result[0])
            results.append(result)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{total} completed...")
                # Check worker scaling
                async with httpx.AsyncClient() as cli:
                    metrics = (await cli.get(SERVICE_URL + METRICS_PATH)).json()
                    print(f"    Current workers: {metrics.get('current_workers', 'N/A')}")
    
    tasks = [bounded_request(i) for i in range(total)]
    await asyncio.gather(*tasks)
    
    # Final metrics
    async with httpx.AsyncClient() as cli:
        final_metrics = (await cli.get(SERVICE_URL + METRICS_PATH)).json()
        print(f"Final workers: {final_metrics.get('current_workers', 'N/A')}")
    
    # Analysis
    successful = sum(1 for _, status, _, _ in results if status == 200)
    zoom_levels = [r[2].get("zoom_level") for r in results if r[2]]
    unique_zooms = set(zoom_levels)
    
    print(f"\n{'='*60}")
    print(f"Concurrency Test Results:")
    print(f"  Requests: {total} at ≤{concurrent} concurrency")
    print(f"  Success rate: {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"  Worker scaling: {initial_metrics.get('current_workers')} → {final_metrics.get('current_workers')}")
    
    if latencies:
        print(f"\nLatency (seconds):")
        print(f"  Min: {min(latencies):.3f}")
        print(f"  p50: {statistics.median(latencies):.3f}")
        if len(latencies) >= 20:
            print(f"  p95: {statistics.quantiles(latencies, n=20)[18]:.3f}")
        print(f"  Max: {max(latencies):.3f}")
    
    print(f"\nZoom levels returned: {unique_zooms}")
    print('='*60)


# ────────────────────────────────────────────────────────────
#  ARTIFACT TEST
# ────────────────────────────────────────────────────────────
async def test_artifact():
    print("\nTesting artifact input format...")
    
    payload = {
        "data": {
            "artifact": {
                "uri": "azure://test-container/test-blob.jpg",
                "type": "image/jpeg",
                "bytes": 12345,
                "sha256": "abc123"
            }
        },
        "meta": {"trace_id": "artifact-test"}
    }
    
    async with httpx.AsyncClient() as cli:
        r = await cli.post(SERVICE_URL + RUN_PATH, json=payload, timeout=10)
    
    if r.status_code == 422:
        print("✓ Correctly rejected simulated artifact (expected without real Azure setup)")
    elif r.status_code == 200:
        print("✓ Artifact processing worked (Azure configured)")
        print(f"  Result: {r.json()}")
    else:
        print(f"? Unexpected response: {r.status_code} - {r.text[:200]}")


# ────────────────────────────────────────────────────────────
#  INTEGRATION TEST
# ────────────────────────────────────────────────────────────
async def test_integration():
    """Test integration with Temporal workflow format."""
    print("\nIntegration test: Temporal workflow compatibility")
    
    # Test 1: Standard Temporal payload format
    test_img = create_test_image()
    temporal_payload = {
        "data": {"image_bytes": b64(img_bytes=test_img)},
        "meta": {
            "trace_id": "temporal-wf-123",
            "workflow_id": "jewelry-pipeline-456",
            "idempotency_key": "unique-key-789"
        }
    }
    
    async with httpx.AsyncClient() as cli:
        r = await cli.post(SERVICE_URL + RUN_PATH, json=temporal_payload, timeout=120)
    
    if r.status_code != 200:
        print(f"✗ Failed with Temporal format: {r.status_code}")
        return
    
    result = r.json()
    print(f"✓ Temporal format accepted")
    print(f"  Zoom level: {result.get('zoom_level', 'N/A')}")
    
    # Test 2: Verify output format for downstream tools
    if "zoom_level" in result and isinstance(result["zoom_level"], str):
        if result["zoom_level"] in ["zoom_1", "zoom_2", "zoom_3", "zoom_4"]:
            print(f"✓ Output format correct for downstream processing")
            print(f"  Can be combined with SAM output for next stage")
        else:
            print(f"⚠ Unexpected zoom level: {result['zoom_level']}")
    else:
        print(f"✗ Output format incorrect: {result}")


def main():
    parser = argparse.ArgumentParser(description="Test zoom classifier service")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Health check
    subparsers.add_parser("health", help="Check service health")
    
    # Metrics check
    subparsers.add_parser("metrics", help="Check service metrics")
    
    # Single image test
    single_parser = subparsers.add_parser("single", help="Test single image")
    single_parser.add_argument("--image", type=pathlib.Path, help="Image file (optional)")
    
    # Batch test
    batch_parser = subparsers.add_parser("batch", help="Test batch processing")
    batch_parser.add_argument("--images", type=pathlib.Path, nargs="*", help="Image files (optional)")
    
    # Concurrency test
    conc_parser = subparsers.add_parser("concurrency", help="Test concurrent requests")
    conc_parser.add_argument("--image", type=pathlib.Path, help="Image file (optional)")
    conc_parser.add_argument("-c", "--concurrent", type=int, default=10, help="Max concurrent requests")
    conc_parser.add_argument("-n", "--total", type=int, default=50, help="Total requests")
    
    # Artifact test
    subparsers.add_parser("artifact", help="Test artifact input format")
    
    # Integration test
    subparsers.add_parser("integration", help="Test Temporal workflow integration")
    
    # All tests
    subparsers.add_parser("all", help="Run all tests")
    
    args = parser.parse_args()
    
    if args.command == "health":
        asyncio.run(check_health())
    elif args.command == "metrics":
        asyncio.run(check_metrics())
    elif args.command == "single":
        asyncio.run(run_single(args.image if hasattr(args, 'image') else None))
    elif args.command == "batch":
        asyncio.run(run_batch(args.images if hasattr(args, 'images') else None))
    elif args.command == "concurrency":
        asyncio.run(run_concurrency(
            args.image if hasattr(args, 'image') else None,
            args.concurrent,
            args.total
        ))
    elif args.command == "artifact":
        asyncio.run(test_artifact())
    elif args.command == "integration":
        asyncio.run(test_integration())
    elif args.command == "all":
        print("Running all tests...\n")
        asyncio.run(check_health())
        asyncio.run(check_metrics())
        asyncio.run(run_single())
        asyncio.run(run_batch())
        asyncio.run(run_concurrency(concurrent=5, total=20))
        asyncio.run(test_artifact())
        asyncio.run(test_integration())


if __name__ == "__main__":
    if sys.version_info < (3, 9):
        print("Requires Python 3.9+", file=sys.stderr)
        sys.exit(1)
    main()
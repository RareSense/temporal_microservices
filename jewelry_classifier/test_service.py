import argparse, asyncio, base64, json, time, statistics, sys, httpx, pathlib
from typing import Dict, List, Any

SERVICE_URL  = "http://localhost:18006"   
HEALTH_PATH  = "/health"
RUN_PATH     = "/run"

def b64(img_path: pathlib.Path) -> str:
    return base64.b64encode(img_path.read_bytes()).decode()

# ────────────────────────────────────────────────────────────
#  HEALTH CHECK
# ────────────────────────────────────────────────────────────
async def check_health():
    async with httpx.AsyncClient() as cli:
        r = await cli.get(SERVICE_URL + HEALTH_PATH, timeout=5)
    print("health:", r.status_code, r.json())

# ────────────────────────────────────────────────────────────
#  SINGLE IMAGE
# ────────────────────────────────────────────────────────────
async def run_single(img: pathlib.Path, verbose: bool = True):
    payload = {
        "data": {"image_bytes": b64(img)}
    }
    async with httpx.AsyncClient() as cli:
        t0 = time.perf_counter()
        r  = await cli.post(SERVICE_URL + RUN_PATH, json=payload, timeout=120)
        dt = (time.perf_counter() - t0) * 1000
    
    if r.status_code == 200:
        result = r.json()
        if isinstance(result, dict) and "detected_jewelry" in result:
            jewelry_list = result["detected_jewelry"]
            print(f"✓ {img.name}: Found {len(jewelry_list)} items: {jewelry_list} ({dt:.1f} ms)")
        # Handle old format for backwards compatibility
        elif isinstance(result, list):
            print(f"⚠ {img.name}: Old format - {result} ({dt:.1f} ms)")
            print("  Note: Service should return dict format {'detected_jewelry': [...]}")
        else:
            print(f"? {img.name}: Unexpected format - {result} ({dt:.1f} ms)")
    else:
        print(f"✗ {img.name}: {r.status_code} - {r.text[:100]} ({dt:.1f} ms)")
    
    if verbose:
        print(f"  Raw response: {r.json()}")
    
    return r.status_code, dt

# ────────────────────────────────────────────────────────────
#  BATCH TEST
# ────────────────────────────────────────────────────────────
async def run_batch(images: List[pathlib.Path]):
    print(f"\nBatch testing {len(images)} images...")
    results = []
    
    for img in images:
        status, latency = await run_single(img, verbose=False)
        results.append((img.name, status, latency))
    
    successful = sum(1 for _, status, _ in results if status == 200)
    avg_latency = statistics.mean([lat for _, _, lat in results])
    
    print(f"\n{'='*50}")
    print(f"Batch Summary:")
    print(f"  Total: {len(images)}")
    print(f"  Successful: {successful}/{len(images)}")
    print(f"  Avg latency: {avg_latency:.1f} ms")
    print('='*50)

# ────────────────────────────────────────────────────────────
#  FLOOD / CONCURRENCY BENCH
# ────────────────────────────────────────────────────────────
async def run_many(img: pathlib.Path, conc: int, repeat: int):
    payload = json.dumps({"data": {"image_bytes": b64(img)}})  # encode once
    
    print(f"\nFlood test: {repeat} requests with max {conc} concurrent")
    print("Starting...")

    async def _one(index: int):
        async with httpx.AsyncClient() as cli:
            t0 = time.perf_counter()
            r  = await cli.post(SERVICE_URL + RUN_PATH,
                                content=payload,
                                headers={"Content-Type": "application/json"},
                                timeout=120)
            return (time.perf_counter() - t0) * 1000, r.status_code, r.json() if r.status_code == 200 else None, index

    tasks, latencies, codes, responses = [], [], [], []
    sem = asyncio.Semaphore(conc)

    async def bound(i):
        async with sem:
            d, c, resp, idx = await _one(i)
            latencies.append(d)
            codes.append(c)
            responses.append(resp)
            
            if (idx + 1) % 10 == 0:
                print(f"  {idx + 1}/{repeat} completed...")

    for i in range(repeat):
        tasks.append(asyncio.create_task(bound(i)))

    await asyncio.gather(*tasks)
    
    ok = codes.count(200)

    format_types = {"dict": 0, "list": 0, "other": 0}
    for resp in responses:
        if resp:
            if isinstance(resp, dict) and "detected_jewelry" in resp:
                format_types["dict"] += 1
            elif isinstance(resp, list):
                format_types["list"] += 1
            else:
                format_types["other"] += 1
    
    print(f"\n{'='*60}")
    print(f"Flood Test Results:")
    print(f"  Requests: {repeat} at ≤{conc} concurrency")
    print(f"  Success rate: {ok}/{repeat} ({ok/repeat*100:.1f}%)")
    
    if latencies:
        print(f"\nLatency (ms):")
        print(f"  p50: {statistics.median(latencies):.1f}")
        if len(latencies) >= 20:
            print(f"  p95: {statistics.quantiles(latencies, n=20)[18]:.1f}")
        print(f"  max: {max(latencies):.1f}")
        print(f"  min: {min(latencies):.1f}")
    
    print(f"\nResponse formats:")
    print(f"  Correct (dict): {format_types['dict']}")
    print(f"  Old (list): {format_types['list']}")
    print(f"  Other: {format_types['other']}")
    print('='*60)

# ────────────────────────────────────────────────────────────
#  INTEGRATION TEST
# ────────────────────────────────────────────────────────────
async def test_integration(img: pathlib.Path):
    print(f"\nIntegration test: Simulating full pipeline")
    
    # Step 1: Call classifier
    payload = {"data": {"image_bytes": b64(img)}}
    
    async with httpx.AsyncClient() as cli:
        r = await cli.post(SERVICE_URL + RUN_PATH, json=payload, timeout=120)
    
    if r.status_code != 200:
        print(f"✗ Classifier failed: {r.status_code}")
        return
    
    classifier_output = r.json()
    print(f"✓ Classifier output: {classifier_output}")
    
    # Step 2: Prepare for Grounded SAM (simulated)
    if isinstance(classifier_output, dict) and "detected_jewelry" in classifier_output:
        sam_input = {
            "data": {
                "detected_jewelry": classifier_output["detected_jewelry"],
                "image_bytes": b64(img)  
            }
        }
        print(f"✓ Format correct for Grounded SAM input")
        print(f"  Would send labels: {classifier_output['detected_jewelry']}")
        print(f"  Expected: ONE unified mask covering all items")
    else:
        print(f"✗ Incorrect format - would cause multiple SAM calls!")
        print(f"  Got: {type(classifier_output)}")
        print(f"  Need: dict with 'detected_jewelry' field")

# ────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Test jewelry classifier service")
    sub = p.add_subparsers(dest="cmd", required=True, help="Command to run")

    # Health check
    sub.add_parser("health", help="Check service health")

    # Single image test
    s1 = sub.add_parser("single", help="Test single image")
    s1.add_argument("image", type=pathlib.Path, help="Image file path")

    # Batch test
    s2 = sub.add_parser("batch", help="Test multiple images")
    s2.add_argument("images", type=pathlib.Path, nargs="+", help="Image file paths")

    # Flood test
    s3 = sub.add_parser("flood", help="Load/concurrency test")
    s3.add_argument("image", type=pathlib.Path, help="Image to use for testing")
    s3.add_argument("-c", "--concurrency", type=int, default=8, help="Max concurrent requests")
    s3.add_argument("-n", "--repeat", type=int, default=32, help="Total requests to send")

    # Integration test
    s4 = sub.add_parser("integration", help="Test pipeline integration format")
    s4.add_argument("image", type=pathlib.Path, help="Image file path")

    args = p.parse_args()

    if args.cmd == "health":
        asyncio.run(check_health())
    elif args.cmd == "single":
        asyncio.run(run_single(args.image))
    elif args.cmd == "batch":
        asyncio.run(run_batch(args.images))
    elif args.cmd == "flood":
        asyncio.run(run_many(args.image, args.concurrency, args.repeat))
    elif args.cmd == "integration":
        asyncio.run(test_integration(args.image))

if __name__ == "__main__":
    if sys.version_info < (3, 9):
        print("Requires Python 3.9+", file=sys.stderr)
        sys.exit(1)
    main()
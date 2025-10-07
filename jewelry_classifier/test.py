import asyncio
import aiohttp
import requests
import base64
import time
import statistics
from pathlib import Path
from typing import List, Dict
import psutil
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


BASE_URL = "http://localhost:8001"
TEST_IMAGE = "sample.jpg" 

def load_test_image() -> str:
    """Load and encode test image once"""
    with open(TEST_IMAGE, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_memory_usage() -> float:
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


# ============================================================
# Test 1: Sequential Baseline
# ============================================================

def test_sequential(num_requests: int = 10):
    print("\n" + "="*60)
    print("TEST 1: Sequential Baseline")
    print("="*60)
    
    img_b64 = load_test_image()
    payload = {"data": {"image": img_b64}, "meta": {}}
    
    times = []
    start_total = time.time()
    
    for i in range(num_requests):
        start = time.time()
        response = requests.post(f"{BASE_URL}/run", json=payload)
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            times.append(elapsed)
            print(f"  Request {i+1}/{num_requests}: {elapsed:.3f}s -> {result['detected_jewelry']}")
        else:
            print(f"  Request {i+1}/{num_requests}: FAILED ({response.status_code})")
    
    total_time = time.time() - start_total
    
    print(f"\n📊 Results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Avg time per request: {statistics.mean(times):.3f}s")
    print(f"   Min: {min(times):.3f}s | Max: {max(times):.3f}s")
    print(f"   Throughput: {num_requests/total_time:.2f} req/s")


# ============================================================
# Test 2: Async Concurrent Requests
# ============================================================

async def send_request(session: aiohttp.ClientSession, img_b64: str, request_id: int) -> Dict:
    payload = {"data": {"image": img_b64}, "meta": {}}
    
    start = time.time()
    try:
        async with session.post(f"{BASE_URL}/run", json=payload) as resp:
            elapsed = time.time() - start
            result = await resp.json()
            return {
                "id": request_id,
                "status": resp.status,
                "elapsed": elapsed,
                "result": result
            }
    except Exception as e:
        return {
            "id": request_id,
            "status": "error",
            "elapsed": time.time() - start,
            "error": str(e)
        }


async def test_async_concurrent(num_requests: int = 20, concurrency: int = 10):
    print("\n" + "="*60)
    print(f"TEST 2: Async Concurrent ({concurrency} concurrent)")
    print("="*60)
    
    img_b64 = load_test_image()
    
    connector = aiohttp.TCPConnector(limit=concurrency)
    timeout = aiohttp.ClientTimeout(total=300)
    
    start_total = time.time()
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [send_request(session, img_b64, i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_total
    
    successful = [r for r in results if r["status"] == 200]
    failed = [r for r in results if r["status"] != 200]
    times = [r["elapsed"] for r in successful]
    
    print(f"\n📊 Results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Successful: {len(successful)}/{num_requests}")
    print(f"   Failed: {len(failed)}")
    print(f"   Avg time per request: {statistics.mean(times):.3f}s")
    print(f"   Min: {min(times):.3f}s | Max: {max(times):.3f}s")
    print(f"   Throughput: {len(successful)/total_time:.2f} req/s")
    print(f"   Speedup vs sequential: {num_requests * statistics.mean(times) / total_time:.2f}x")
    
    if failed:
        print(f"\n⚠️  Failed requests:")
        for r in failed[:5]:  # Show first 5 failures
            print(f"   Request {r['id']}: {r.get('error', r['status'])}")


# ============================================================
# Test 3: Thread Pool Concurrent
# ============================================================

def send_request_sync(args) -> Dict:
    img_b64, request_id = args
    payload = {"data": {"image": img_b64}, "meta": {}}
    
    start = time.time()
    try:
        response = requests.post(f"{BASE_URL}/run", json=payload, timeout=30)
        elapsed = time.time() - start
        return {
            "id": request_id,
            "status": response.status_code,
            "elapsed": elapsed,
            "result": response.json()
        }
    except Exception as e:
        return {
            "id": request_id,
            "status": "error",
            "elapsed": time.time() - start,
            "error": str(e)
        }


def test_thread_pool(num_requests: int = 20, num_workers: int = 10):
    print("\n" + "="*60)
    print(f"TEST 3: Thread Pool ({num_workers} workers)")
    print("="*60)
    
    img_b64 = load_test_image()
    
    start_total = time.time()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        args = [(img_b64, i) for i in range(num_requests)]
        results = list(executor.map(send_request_sync, args))
    
    total_time = time.time() - start_total
    
    successful = [r for r in results if r["status"] == 200]
    failed = [r for r in results if r["status"] != 200]
    times = [r["elapsed"] for r in successful]
    
    print(f"\n📊 Results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Successful: {len(successful)}/{num_requests}")
    print(f"   Failed: {len(failed)}")
    print(f"   Avg time per request: {statistics.mean(times):.3f}s")
    print(f"   Min: {min(times):.3f}s | Max: {max(times):.3f}s")
    print(f"   Throughput: {len(successful)/total_time:.2f} req/s")


# ============================================================
# Test 4: Sustained Load Test
# ============================================================

async def sustained_load_test(duration_seconds: int = 30, requests_per_second: int = 5):
    """Test sustained load over time"""
    print("\n" + "="*60)
    print(f"TEST 4: Sustained Load ({requests_per_second} req/s for {duration_seconds}s)")
    print("="*60)
    
    img_b64 = load_test_image()
    
    connector = aiohttp.TCPConnector(limit=50)
    timeout = aiohttp.ClientTimeout(total=300)
    
    results = []
    start_time = time.time()
    request_id = 0
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        while time.time() - start_time < duration_seconds:
            batch_start = time.time()
            
            tasks = [
                send_request(session, img_b64, request_id + i) 
                for i in range(requests_per_second)
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            request_id += requests_per_second
            
            elapsed = time.time() - batch_start
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)
            
            if request_id % 10 == 0:
                print(f"   Sent {request_id} requests...")
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful = [r for r in results if r["status"] == 200]
    failed = [r for r in results if r["status"] != 200]
    times = [r["elapsed"] for r in successful]
    
    print(f"\n📊 Results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Total requests: {len(results)}")
    print(f"   Successful: {len(successful)}")
    print(f"   Failed: {len(failed)}")
    print(f"   Actual throughput: {len(successful)/total_time:.2f} req/s")
    print(f"   Avg latency: {statistics.mean(times):.3f}s")
    print(f"   P50 latency: {statistics.median(times):.3f}s")
    print(f"   P95 latency: {sorted(times)[int(len(times)*0.95)]:.3f}s")
    print(f"   P99 latency: {sorted(times)[int(len(times)*0.99)]:.3f}s")


# ============================================================
# Test 5: Memory Leak Detection
# ============================================================

async def test_memory_leak(num_batches: int = 10, requests_per_batch: int = 20):
    """Test for memory leaks over multiple batches"""
    print("\n" + "="*60)
    print(f"TEST 5: Memory Leak Detection ({num_batches} batches)")
    print("="*60)
    
    img_b64 = load_test_image()
    memory_samples = []
    
    connector = aiohttp.TCPConnector(limit=20)
    timeout = aiohttp.ClientTimeout(total=300)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for batch in range(num_batches):
            mem_before = get_memory_usage()
            
            tasks = [send_request(session, img_b64, i) for i in range(requests_per_batch)]
            await asyncio.gather(*tasks)
            
            mem_after = get_memory_usage()
            memory_samples.append(mem_after)
            
            print(f"   Batch {batch+1}/{num_batches}: {mem_after:.1f} MB (Δ {mem_after-mem_before:+.1f} MB)")
            
            await asyncio.sleep(1)
    
    print(f"\n📊 Memory Analysis:")
    print(f"   Initial: {memory_samples[0]:.1f} MB")
    print(f"   Final: {memory_samples[-1]:.1f} MB")
    print(f"   Growth: {memory_samples[-1] - memory_samples[0]:+.1f} MB")
    print(f"   Avg per batch: {(memory_samples[-1] - memory_samples[0]) / num_batches:+.2f} MB")
    
    if memory_samples[-1] - memory_samples[0] > 100:
        print(f"   ⚠️  WARNING: Significant memory growth detected!")
    else:
        print(f"   ✅ Memory usage looks stable")



def main():
    print("\n" + "="*60)
    print("VISION CLASSIFIER CONCURRENCY TEST SUITE")
    print("="*60)
    
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5).json()
        print(f"✅ Service healthy: {health['device']}, {health['num_classes']} classes")
    except Exception as e:
        print(f"❌ Service not reachable: {e}")
        print(f"   Make sure the service is running on {BASE_URL}")
        return
    
    if not Path(TEST_IMAGE).exists():
        print(f"❌ Test image not found: {TEST_IMAGE}")
        print(f"   Update TEST_IMAGE variable in this script")
        return
    
    print(f"✅ Test image loaded: {TEST_IMAGE}")
    
    try:
        # Test 1: Sequential baseline
        test_sequential(num_requests=10)
        
        # Test 2: Async concurrent
        asyncio.run(test_async_concurrent(num_requests=20, concurrency=10))
        
        # Test 3: Thread pool
        test_thread_pool(num_requests=20, num_workers=10)
        
        # Test 4: Sustained load
        asyncio.run(sustained_load_test(duration_seconds=30, requests_per_second=5))
        
        # Test 5: Memory leak detection
        asyncio.run(test_memory_leak(num_batches=10, requests_per_batch=20))
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
"""
Usage examples:
  python test_skin_tone.py health
  python test_skin_tone.py single path/to/photo.jpg
  python test_skin_tone.py batch  imgs/*.jpg
  python test_skin_tone.py flood  path/to/photo.jpg -c 16 -n 100
"""

import argparse, asyncio, base64, json, pathlib, statistics, sys, time
from typing import Any, Dict, List

import httpx

SERVICE_URL = "http://localhost:18008"
HEALTH_PATH = "/health"
RUN_PATH = "/run"


def _b64(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _pretty_result(result: Dict[str, Any]) -> str:
    """
    Format classifier result:
      {"skin_tone": "fair",
       "skin_tone_predictions": [{"class": "fair", "confidence": 0.85}, ...]}
    """
    tone = result.get("skin_tone", "?")
    preds = ", ".join(
        f"{p['class']}: {p['confidence']:.2f}" for p in result.get("skin_tone_predictions", [])
    )
    return f"{tone}  ({preds})"


async def _post_json(client: httpx.AsyncClient, payload: Dict[str, Any]) -> httpx.Response:
    return await client.post(SERVICE_URL + RUN_PATH, json=payload, timeout=120)


async def check_health() -> None:
    async with httpx.AsyncClient() as client:
        r = await client.get(SERVICE_URL + HEALTH_PATH, timeout=5)
    print("health:", r.status_code, r.json())


async def run_single(img: pathlib.Path, verbose: bool = True):
    payload = {"data": {"image_bytes": _b64(img)}}

    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        r = await _post_json(client, payload)
        dt = (time.perf_counter() - t0) * 1000  # ms

    if r.status_code == 200:
        res = r.json()
        print(f"✓ {img.name}: {_pretty_result(res)}  [{dt:.1f} ms]")
        if verbose:
            print("  Raw:", res)
    else:
        print(f"✗ {img.name}: {r.status_code} – {r.text[:120]}  [{dt:.1f} ms]")

    return r.status_code, dt


async def run_batch(images: List[pathlib.Path]):
    print(f"\nBatch testing {len(images)} images …")
    latencies = []
    ok = 0

    for img in images:
        status, latency = await run_single(img, verbose=False)
        latencies.append(latency)
        ok += int(status == 200)

    print("\n" + "=" * 48)
    print("Batch summary:")
    print(f"  Total        : {len(images)}")
    print(f"  Successful   : {ok}/{len(images)}")
    if latencies:
        print(f"  Avg latency  : {statistics.mean(latencies):.1f} ms")
        print(f"  p95 latency  : {statistics.quantiles(latencies, n=20)[18]:.1f} ms")
    print("=" * 48)


# ────────────────────────────────────────────────────────────
#  Flood / concurrency test
# ────────────────────────────────────────────────────────────
async def run_flood(img: pathlib.Path, conc: int, repeat: int):
    print(f"\nFlood test → {repeat} requests, ≤{conc} concurrent")
    payload = json.dumps({"data": {"image_bytes": _b64(img)}})
    hdrs = {"Content-Type": "application/json"}

    sem = asyncio.Semaphore(conc)
    latencies, codes = [], []

    async def _one(i: int):
        async with sem, httpx.AsyncClient() as client:
            t0 = time.perf_counter()
            r = await client.post(SERVICE_URL + RUN_PATH, content=payload, headers=hdrs, timeout=120)
            latencies.append((time.perf_counter() - t0) * 1000)
            codes.append(r.status_code)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{repeat} …")

    await asyncio.gather(*[asyncio.create_task(_one(i)) for i in range(repeat)])

    ok = codes.count(200)
    print("\n" + "=" * 60)
    print("Flood test results:")
    print(f"  Requests     : {repeat}")
    print(f"  Success rate : {ok/repeat*100:.1f}%")
    if latencies:
        print(f"  p50 latency  : {statistics.median(latencies):.1f} ms")
        if len(latencies) >= 20:
            print(f"  p95 latency  : {statistics.quantiles(latencies, n=20)[18]:.1f} ms")
        print(f"  max / min    : {max(latencies):.1f} ms / {min(latencies):.1f} ms")
    print("=" * 60)



def main():
    p = argparse.ArgumentParser(description="Test client for skin-tone classifier")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health", help="Ping /health")

    s1 = sub.add_parser("single", help="Classify one image")
    s1.add_argument("image", type=pathlib.Path)

    s2 = sub.add_parser("batch", help="Classify multiple images")
    s2.add_argument("images", nargs="+", type=pathlib.Path)

    s3 = sub.add_parser("flood", help="Concurrency / load test")
    s3.add_argument("image", type=pathlib.Path)
    s3.add_argument("-c", "--concurrency", type=int, default=8)
    s3.add_argument("-n", "--repeat", type=int, default=32)

    args = p.parse_args()

    if args.cmd == "health":
        asyncio.run(check_health())
    elif args.cmd == "single":
        asyncio.run(run_single(args.image))
    elif args.cmd == "batch":
        asyncio.run(run_batch(args.images))
    elif args.cmd == "flood":
        asyncio.run(run_flood(args.image, args.concurrency, args.repeat))


if __name__ == "__main__":
    if sys.version_info < (3, 9):
        print("Requires Python 3.9+", file=sys.stderr)
        sys.exit(1)
    main()

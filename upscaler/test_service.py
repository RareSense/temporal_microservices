"""
USAGE
=====

1) health check
   python test_upscaler.py health

2) single request
   python test_upscaler.py single bg.jpg ghost.png [-o out.jpg]

3) batch (pairs)
   python test_upscaler.py batch bg1.jpg ghost1.png bg2.jpg ghost2.png -d results/

4) flood / concurrency
   python test_upscaler.py flood bg.jpg ghost.png -c 8 -n 50 [-d flood_out/]
"""

import argparse
import asyncio
import base64
import json
import statistics
import sys
import time
from pathlib import Path
from typing import List, Tuple

import httpx


# ────────────────────────────────────────────────────────────────
#  Config
# ────────────────────────────────────────────────────────────────
SERVICE_URL = "http://localhost:18012"
HEALTH_PATH = "/health"
RUN_PATH = "/run"


# ────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────
def b64(img_path: Path) -> str:
    return base64.b64encode(img_path.read_bytes()).decode()


def make_payload(bg: Path, ghost: Path) -> dict:
    return {"data": {"background_bytes": b64(bg), "ghost_bytes": b64(ghost)}}


def as_pairs(paths: List[Path]) -> List[Tuple[Path, Path]]:
    if len(paths) % 2 != 0:
        raise ValueError(
            "Batch mode requires an even number of paths: bg1 ghost1 bg2 ghost2 ..."
        )
    return [(paths[i], paths[i + 1]) for i in range(0, len(paths), 2)]


def decode_and_save(b64str: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(b64str))
    print(f"  ↳ saved → {out_path}")


# ────────────────────────────────────────────────────────────────
#  Health-check
# ────────────────────────────────────────────────────────────────
async def check_health():
    async with httpx.AsyncClient() as cli:
        r = await cli.get(SERVICE_URL + HEALTH_PATH, timeout=5)
    print("health:", r.status_code, r.json())


# ────────────────────────────────────────────────────────────────
#  Single request
# ────────────────────────────────────────────────────────────────
async def run_single(bg: Path, ghost: Path, out_file: Path | None):
    payload = make_payload(bg, ghost)

    async with httpx.AsyncClient() as cli:
        t0 = time.perf_counter()
        r = await cli.post(SERVICE_URL + RUN_PATH, json=payload, timeout=180)
        dt = (time.perf_counter() - t0) * 1000

    if r.status_code != 200:
        print(
            f"✗ {bg.name}+{ghost.name}: {r.status_code} – {r.text[:120]} "
            f"({dt:.1f} ms)"
        )
        return r.status_code, dt

    res = r.json()
    if not isinstance(res, dict) or "composite_b64" not in res:
        print(f"? Unexpected response: {res}")
        return 500, dt

    print(
        f"✓ {bg.name}+{ghost.name}: OK ({dt:.1f} ms) "
        f"[{len(res['composite_b64'])//1024} KB b64]"
    )

    if out_file:
        decode_and_save(res["composite_b64"], out_file)

    return 200, dt


# ────────────────────────────────────────────────────────────────
#  Batch test
# ────────────────────────────────────────────────────────────────
async def run_batch(pairs: List[Tuple[Path, Path]], out_dir: Path):
    print(f"\nBatch testing {len(pairs)} pairs…")
    stats = []

    for bg, ghost in pairs:
        fname = (
            f"composite_{bg.stem}__{ghost.stem}.jpg"
        )  # readable pairing
        code, lat = await run_single(
            bg, ghost, out_dir / fname if out_dir else None
        )
        stats.append((code, lat))

    ok = sum(1 for c, _ in stats if c == 200)
    lats = [lat for _, lat in stats]

    print("\n" + "=" * 50)
    print("Batch summary")
    print(f"  total      : {len(pairs)}")
    print(f"  successful : {ok}/{len(pairs)}")
    if lats:
        print(f"  avg latency: {statistics.mean(lats):.1f} ms")
    print("=" * 50)


# ────────────────────────────────────────────────────────────────
#  Flood / concurrency test
# ────────────────────────────────────────────────────────────────
async def run_flood(
    bg: Path, ghost: Path, conc: int, repeat: int, out_dir: Path | None
):
    payload_json = json.dumps(make_payload(bg, ghost))

    latencies, codes, images = [], [], []
    sem = asyncio.Semaphore(conc)

    async def one(idx: int):
        async with sem, httpx.AsyncClient() as cli:
            t0 = time.perf_counter()
            r = await cli.post(
                SERVICE_URL + RUN_PATH,
                content=payload_json,
                headers={"Content-Type": "application/json"},
                timeout=180,
            )
            latencies.append((time.perf_counter() - t0) * 1000)
            codes.append(r.status_code)
            if r.status_code == 200:
                images.append(r.json().get("composite_b64"))
            if (idx + 1) % 10 == 0:
                print(f"  {idx + 1}/{repeat} done")

    print(f"\nFlood test: {repeat} requests – max {conc} concurrent")
    await asyncio.gather(*(one(i) for i in range(repeat)))

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, b64str in enumerate(images):
            if b64str:
                decode_and_save(b64str, out_dir / f"flood_{i:04d}.jpg")

    ok = codes.count(200)
    print("\n" + "=" * 60)
    print("Flood summary")
    print(f"  success rate : {ok}/{repeat} ({ok/repeat*100:.1f}%)")
    if latencies:
        print("  latency (ms):")
        print(f"    p50 : {statistics.median(latencies):.1f}")
        if len(latencies) >= 20:
            print(f"    p95 : {statistics.quantiles(latencies, n=20)[18]:.1f}")
        print(f"    max : {max(latencies):.1f}")
        print(f"    min : {min(latencies):.1f}")
    print("=" * 60)


# ────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Test the upscaler micro-service")
    sub = p.add_subparsers(dest="cmd", required=True)

    # health
    sub.add_parser("health", help="Check /health endpoint")

    # single
    ps = sub.add_parser("single", help="Single request")
    ps.add_argument("background", type=Path)
    ps.add_argument("ghost", type=Path)
    ps.add_argument(
        "-o",
        "--out",
        type=Path,
        help="Where to save the composite (default: composite_<bg>.jpg)",
    )

    # batch
    pb = sub.add_parser("batch", help="Batch test (pairs)")
    pb.add_argument(
        "images",
        type=Path,
        nargs="+",
        help="bg1 ghost1 bg2 ghost2 … (even number of paths)",
    )
    pb.add_argument(
        "-d",
        "--dir",
        type=Path,
        default=Path("batch_out"),
        help="Directory to save composites",
    )

    # flood
    pf = sub.add_parser("flood", help="Concurrency / load test")
    pf.add_argument("background", type=Path)
    pf.add_argument("ghost", type=Path)
    pf.add_argument("-c", "--concurrency", type=int, default=8)
    pf.add_argument("-n", "--repeat", type=int, default=32)
    pf.add_argument(
        "-d",
        "--dir",
        type=Path,
        help="Directory to save each returned image (optional)",
    )

    args = p.parse_args()

    if args.cmd == "health":
        asyncio.run(check_health())

    elif args.cmd == "single":
        out = (
            args.out
            if args.out
            else Path(f"composite_{args.background.stem}.jpg")
        )
        asyncio.run(run_single(args.background, args.ghost, out))

    elif args.cmd == "batch":
        pairs = as_pairs(args.images)
        asyncio.run(run_batch(pairs, args.dir))

    elif args.cmd == "flood":
        asyncio.run(
            run_flood(
                args.background,
                args.ghost,
                args.concurrency,
                args.repeat,
                args.dir,
            )
        )


if __name__ == "__main__":
    if sys.version_info < (3, 9):
        sys.exit("Python 3.9+ required")
    main()

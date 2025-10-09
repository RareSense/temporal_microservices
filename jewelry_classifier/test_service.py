import argparse, asyncio, base64, json, time, statistics, sys, httpx, pathlib

SERVICE_URL  = "http://localhost:18006"     # adjust if running elsewhere
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
async def run_single(img: pathlib.Path):
    payload = {
        "data": {"image_bytes": b64(img)}
    }
    async with httpx.AsyncClient() as cli:
        t0 = time.perf_counter()
        r  = await cli.post(SERVICE_URL + RUN_PATH, json=payload, timeout=120)
        dt = (time.perf_counter() - t0) * 1000
    print(f"{img.name}: {r.status_code}  {r.json()}  ({dt:.1f} ms)")

# ────────────────────────────────────────────────────────────
#  FLOOD / CONCURRENCY BENCH
# ────────────────────────────────────────────────────────────
async def run_many(img: pathlib.Path, conc: int, repeat: int):
    payload = json.dumps({"data": {"image_bytes": b64(img)}})  # encode once

    async def _one(index: int):
        async with httpx.AsyncClient() as cli:
            t0 = time.perf_counter()
            r  = await cli.post(SERVICE_URL + RUN_PATH,
                                content=payload,
                                headers={"Content-Type": "application/json"},
                                timeout=120)
            return (time.perf_counter() - t0) * 1000, r.status_code, index

    tasks, latencies, codes = [], [], []
    sem = asyncio.Semaphore(conc)

    async def bound(i):
        async with sem:
            d, c, _ = await _one(i)
            latencies.append(d)
            codes.append(c)

    for i in range(repeat):
        tasks.append(asyncio.create_task(bound(i)))

    await asyncio.gather(*tasks)
    ok = codes.count(200)
    print(f"\nSent {repeat} requests at ≤{conc}-concurrency: "
          f"{ok}/{repeat} succeeded")
    print(f"latency ms  p50={statistics.median(latencies):.1f}  "
          f"p95={statistics.quantiles(latencies, n=20)[18]:.1f}  "
          f"max={max(latencies):.1f}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health")

    s1 = sub.add_parser("single")
    s1.add_argument("image", type=pathlib.Path)

    s2 = sub.add_parser("flood")
    s2.add_argument("image", type=pathlib.Path)
    s2.add_argument("-c", "--concurrency", type=int, default=8)
    s2.add_argument("-n", "--repeat",      type=int, default=32)

    args = p.parse_args()

    if args.cmd == "health":
        asyncio.run(check_health())
    elif args.cmd == "single":
        asyncio.run(run_single(args.image))
    else:
        asyncio.run(run_many(args.image, args.concurrency, args.repeat))

if __name__ == "__main__":
    if sys.version_info < (3, 9):
        print("Requires Python 3.9+", file=sys.stderr)
        sys.exit(1)
    main()

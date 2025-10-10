#!/usr/bin/env python
"""
grounded_sam – endpoint tester + mask downloader
================================================
• Supports local files **and remote image URLs**.
• Optional mask download (azure credentials required).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import unquote, urlparse

import httpx

# ──────────────────────────────────────────────────────────────
#  Azure helpers  (mask download + optional azure:// image arg)
# ──────────────────────────────────────────────────────────────
_ACC = os.getenv("AZURE_ACCOUNT_NAME")
_KEY = os.getenv("AZURE_ACCOUNT_KEY")
_CONN = os.getenv("AZURE_BLOB_CONN") or (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={_ACC};"
    f"AccountKey={_KEY};"
    f"EndpointSuffix=core.windows.net"
)
_CONTAINER = os.getenv("AZURE_CONTAINER", "agentic-artifacts")
_BLOB_CLI = None

async def _azure_client():
    global _BLOB_CLI
    if _BLOB_CLI is None:
        from azure.storage.blob.aio import BlobServiceClient
        if not (_ACC and _KEY):
            raise RuntimeError("Azure creds missing (AZURE_ACCOUNT_NAME / KEY)")
        _BLOB_CLI = BlobServiceClient.from_connection_string(_CONN)
    return _BLOB_CLI

async def _download_blob(uri: str) -> bytes:
    # uri = azure://container/blob_path w/ optional URL-encoded chars
    if not uri.startswith("azure://"):
        raise ValueError("only azure:// URIs")
    _, _, rest = uri.partition("azure://")
    cont, _, blob = rest.partition("/")
    blob = unquote(blob)
    cli = await _azure_client()
    stream = await cli.get_blob_client(cont, blob).download_blob()
    return await stream.readall()

# ──────────────────────────────────────────────────────────────
#  Input loader  (local path | http(s) | azure://)
# ──────────────────────────────────────────────────────────────
async def _load_image_bytes(src: str) -> bytes:
    parsed = urlparse(src)
    if parsed.scheme in ("http", "https"):             # remote URL
        async with httpx.AsyncClient(timeout=60) as cli:
            r = await cli.get(src)
            r.raise_for_status()
            return r.content
    elif parsed.scheme == "azure":
        return await _download_blob(src)
    else:                                              # assume local path
        path = Path(src).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        return path.read_bytes()

# ──────────────────────────────────────────────────────────────
#  Mask saver
# ──────────────────────────────────────────────────────────────
MASK_DIR = Path("./masks")
MASK_DIR.mkdir(exist_ok=True)

async def _save_masks(artifacts: Dict[str, Dict[str, str]]) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for label, art in artifacts.items():
        try:
            png = await _download_blob(art["uri"])
        except Exception as exc:
            print(f"✖ fetch {art['uri']}: {exc}")
            continue
        out = MASK_DIR / f"{ts}_{label}.png"
        out.write_bytes(png)
        print(f"✓ saved {out}")

# ──────────────────────────────────────────────────────────────
#  Request helper
# ──────────────────────────────────────────────────────────────
def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")

async def _fire(
    cli: httpx.AsyncClient,
    url: str,
    img_b64: str,
    labels: List[str],
    idx: int,
    save_masks: bool,
) -> Tuple[int, float]:
    payload = {"data": {"image_bytes": img_b64, "labels": labels}}
    t0 = time.perf_counter()
    r = await cli.post(url, json=payload)
    ms = (time.perf_counter() - t0) * 1000

    if idx == 0:
        body = r.json()
        print(json.dumps(body, indent=2)[:300] + " …")
        if save_masks and r.status_code == 200:
            await _save_masks(body["artifacts"])
    return r.status_code, ms

# ──────────────────────────────────────────────────────────────
#  Orchestrator
# ──────────────────────────────────────────────────────────────
async def _run(
    mode: str,
    img_src: str,
    labels: List[str],
    url: str,
    n: int,
    save_masks: bool,
) -> None:
    img_bytes = await _load_image_bytes(img_src)
    img_b64   = _b64(img_bytes)

    async with httpx.AsyncClient(timeout=120) as cli:
        if mode == "single":
            code, ms = await _fire(cli, url, img_b64, labels, 0, save_masks)
            print(f"{img_src}: {code}  ({ms:.1f} ms)")
        else:
            jobs = [
                _fire(cli, url, img_b64, labels, i, save_masks and i == 0)
                for i in range(n)
            ]
            t0 = time.perf_counter()
            res = await asyncio.gather(*jobs, return_exceptions=True)
            wall = time.perf_counter() - t0

            codes = [c for c, _ in res if isinstance(c, int)]
            lats  = [ms for _, ms in res if not isinstance(ms, Exception)]
            ok = codes.count(200)
            p50 = sorted(lats)[len(lats)//2] if lats else float("nan")
            avg = sum(lats)/len(lats) if lats else float("nan")
            print(
                f"{n} requests → {ok}/{n} OK  "
                f"p50={p50:.1f} ms  avg={avg:.1f} ms  wall={wall:.2f}s"
            )

    if _BLOB_CLI:
        await _BLOB_CLI.close()

# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["single", "multi"])
    ap.add_argument("image", help="local path, http(s) URL, or azure:// URI")
    ap.add_argument("-n", "--concurrency", type=int, default=10)
    ap.add_argument("--url", default="http://localhost:18007/run")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--save-masks", action="store_true")
    args = ap.parse_args()

    try:
        asyncio.run(
            _run(
                args.mode,
                args.image,
                args.labels,
                args.url,
                args.concurrency,
                args.save_masks,
            )
        )
    except KeyboardInterrupt:
        sys.exit(130)

from __future__ import annotations
import argparse, asyncio, base64, json, os, statistics, time, sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import aiofiles
import httpx
from PIL import Image


try:
    from azure.storage.blob.aio import BlobServiceClient
except ImportError:  # pragma: no cover
    BlobServiceClient = None  # type: ignore


# ═════════════════════════════════════════════
#  Tester
# ═════════════════════════════════════════════
class GroundedSAMTester:
    def __init__(self, service_url: str, output_dir: str):
        self.service_url = service_url.rstrip("/")
        self.base = Path(output_dir)
        for sub in ("inputs", "masks", "overlays"):
            (self.base / sub).mkdir(parents=True, exist_ok=True)

        self.azure_client = self._make_azure_client()

    @staticmethod
    def _make_azure_client():
        acc = os.getenv("AZURE_ACCOUNT_NAME")
        key = os.getenv("AZURE_ACCOUNT_KEY")
        if acc and key and BlobServiceClient:
            conn = (
                f"DefaultEndpointsProtocol=https;AccountName={acc};"
                f"AccountKey={key};EndpointSuffix=core.windows.net"
            )
            print(f"✓ Azure client configured for account: {acc}")
            return BlobServiceClient.from_connection_string(conn)
        print("⚠ Azure credentials not available – artifact download disabled")
        return None

    async def _download_artifact(self, art: Dict[str, Any]) -> bytes:
        if not self.azure_client:
            raise RuntimeError("Azure client not configured")
        uri = art["uri"].replace("azure://", "")
        container, blob = uri.split("/", 1)
        stream = await self.azure_client.get_blob_client(container, blob).download_blob()
        return await stream.readall()

    async def _download_url(self, url: str) -> bytes:
        async with httpx.AsyncClient(follow_redirects=True) as cli:
            r = await cli.get(url)
            r.raise_for_status()
            return r.content

    async def _load_image(self, src: str) -> Tuple[bytes, Image.Image]:
        if src.startswith(("http://", "https://")):
            raw = await self._download_url(src)
        else:
            async with aiofiles.open(src, "rb") as f:
                raw = await f.read()
        return raw, Image.open(BytesIO(raw))

    @staticmethod
    def _b64(data: bytes) -> str:
        return base64.b64encode(data).decode()

    async def test_single(self, img_src: str, labels: List[str], tag: str = "single"):
        raw, pil_img = await self._load_image(img_src)
        (self.base / "inputs" / f"{tag}.png").write_bytes(raw)

        payload = {
            "data": {"labels": labels, "image_bytes": self._b64(raw)},
            "meta": {"test_case": tag, "ts": datetime.utcnow().isoformat()},
        }

        async with httpx.AsyncClient(timeout=90.0) as cli:
            t0 = time.perf_counter()
            res = await cli.post(f"{self.service_url}/run", json=payload)
            latency = (time.perf_counter() - t0) * 1000

        if res.status_code != 200:
            print(f"✗ {tag}: {res.status_code} – {res.text[:120]}")
            return None

        out = res.json()
        print(f"✓ {tag}: {out['status']}  ({latency:.1f} ms)")

        if out.get("masks") and self.azure_client:
            m = out["masks"][0]
            mb = await self._download_artifact(m["mask_artifact"])
            Image.open(BytesIO(mb)).save(self.base / "masks" / f"{tag}_{m['label']}.png")

        if out.get("overlay_artifact") and self.azure_client:
            ob = await self._download_artifact(out["overlay_artifact"])
            Image.open(BytesIO(ob)).save(self.base / "overlays" / f"{tag}.png")

        return out

    async def test_batch(self, imgs: List[str], labels: List[str], tag: str = "batch"):
        results = await asyncio.gather(
            *[self.test_single(src, labels, f"{tag}_{i}") for i, src in enumerate(imgs)],
            return_exceptions=True,
        )
        ok = sum(isinstance(r, dict) for r in results)
        print(f"Batch finished – {ok}/{len(imgs)} succeeded")

    async def test_with_classifier_output(self, img_src: str):
        """Simulate the end-to-end flow *after* the jewelry-classifier."""
        raw, _ = await self._load_image(img_src)
        classifier_payload = {
            # NEW preferred field:
            "detected_jewelry": ["ring", "bracelet"],
            # …but the service still supports this legacy key:
            "jewelry-classify": ["ring", "bracelet"],
            "image_bytes": self._b64(raw),
        }

        async with httpx.AsyncClient(timeout=90.0) as cli:
            res = await cli.post(
                f"{self.service_url}/run",
                json={"data": classifier_payload, "meta": {"test_case": "classifier"}},
            )
        msg = res.json().get("status", "??")
        print(f"Classifier→SAM status: {msg}")

    # ─────────── load / flood test ───────────
    async def flood_test(
        self,
        img_src: str,
        labels: List[str],
        concurrency: int,
        repeat: int,
    ):
        raw, _ = await self._load_image(img_src)
        body = json.dumps(
            {"data": {"labels": labels, "image_bytes": self._b64(raw)}}
        ).encode()

        lat, codes = [], []
        sem = asyncio.Semaphore(concurrency)

        async def send(i: int):
            async with sem:
                t0 = time.perf_counter()
                async with httpx.AsyncClient(timeout=120.0) as cli:
                    r = await cli.post(
                        f"{self.service_url}/run",
                        content=body,
                        headers={"Content-Type": "application/json"},
                    )
                lat.append((time.perf_counter() - t0) * 1000)
                codes.append(r.status_code)
                if (i + 1) % 10 == 0:
                    print(f" … {i + 1}/{repeat} done", flush=True)

        print(f"Flood → {repeat} requests, ≤{concurrency} concurrent")
        await asyncio.gather(*[asyncio.create_task(send(i)) for i in range(repeat)])

        ok = codes.count(200)
        p50 = statistics.median(lat)
        p95 = statistics.quantiles(lat, n=20)[18] if len(lat) >= 20 else max(lat)
        print(
            f"\n{ok}/{repeat} succeeded   "
            f"p50={p50:.1f} ms  p95={p95:.1f} ms  max={max(lat):.1f} ms"
        )

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as cli:
                r = await cli.get(f"{self.service_url}/health")
            if r.status_code == 200:
                print("✓ service healthy", r.json().get("workers", {}))
                return True
        except Exception as exc:
            print("✗ health check failed:", exc)
        return False

async def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--service-url", default="http://localhost:18007")
    ap.add_argument("--output-dir", default="test_outputs")
    ap.add_argument(
        "--mode",
        choices=["single", "batch", "both", "flood"],
        default="single",
    )
    ap.add_argument(
        "--images",
        nargs="+",
        default=["https://images.unsplash.com/photo-1515562141207-7a88fb7ce338?w=800"],
    )
    ap.add_argument("--labels", nargs="+", default=["ring", "bracelet", "earring"])
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=32)
    args = ap.parse_args()

    tester = GroundedSAMTester(args.service_url, args.output_dir)
    if not await tester.health():
        print("Service unhealthy – aborting")
        return

    if args.mode == "flood":
        await tester.flood_test(args.images[0], args.labels, args.concurrency, args.repeat)
        return

    if args.mode in ("single", "both"):
        await tester.test_single(args.images[0], args.labels, "single")

    if args.mode in ("batch", "both"):
        if len(args.images) < 2:
            print("Batch mode needs ≥2 images (use --images …)")
        else:
            await tester.test_batch(args.images, args.labels)

    if args.mode == "both":
        await tester.test_with_classifier_output(args.images[0])


if __name__ == "__main__":
    if sys.version_info < (3, 9):
        sys.exit("Requires Python 3.9+")
    asyncio.run(_cli())

from __future__ import annotations
import os, re, hashlib, uuid
from datetime import datetime, timezone
from pydantic import BaseModel
from azure.storage.blob.aio import BlobServiceClient

class Artifact(BaseModel):
    uri: str
    type: str
    bytes: int | None = None
    sha256: str | None = None

    @property
    def container(self) -> str: return self.uri.split("/")[2]
    @property
    def blob(self) -> str:       return "/".join(self.uri.split("/")[3:])

# ─────────────────────────────────────────────────────────────
#  Azure connection
# ─────────────────────────────────────────────────────────────
_ACC  = os.getenv("AZURE_ACCOUNT_NAME")
_KEY  = os.getenv("AZURE_ACCOUNT_KEY")
_CONN = os.getenv("AZURE_BLOB_CONN") or (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={_ACC};"
    f"AccountKey={_KEY};"
    f"EndpointSuffix=core.windows.net"
)
_CONTAINER = os.getenv("AZURE_CONTAINER", "agentic-artifacts")

if not _ACC or not _KEY:
    raise RuntimeError("Azure creds missing: set AZURE_ACCOUNT_NAME & AZURE_ACCOUNT_KEY")

_ART_RE = re.compile(r"^azure://([^/]+)/(.+)$")
_blob_client: BlobServiceClient | None = None

async def _client() -> BlobServiceClient:
    global _blob_client
    if _blob_client is None:
        _blob_client = BlobServiceClient.from_connection_string(_CONN)
    return _blob_client

# ─────────────────────────────────────────────────────────────
#  Public helpers
# ─────────────────────────────────────────────────────────────
async def fetch_artifact(uri: str) -> bytes:
    m = _ART_RE.match(uri)
    if not m:
        raise ValueError(f"bad artifact URI: {uri}")
    cont, blob = m.groups()
    blob_cli = (await _client()).get_blob_client(cont, blob)
    stream = await blob_cli.download_blob()
    return await stream.readall()

async def upload_artifact(
    data: bytes,
    mime: str = "image/png",
    container: str = _CONTAINER,
) -> Artifact:
    cli = await _client()
    name = f"{datetime.now(timezone.utc)}/{uuid.uuid4().hex}.png"
    blob = cli.get_blob_client(container, name)
    await blob.upload_blob(data, overwrite=True, content_type=mime)
    return Artifact(
        uri=f"azure://{container}/{name}",
        type=mime,
        bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )

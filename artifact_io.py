from __future__ import annotations
from azure.storage.blob.aio import BlobServiceClient
from azure.storage.blob import BlobSasPermissions, generate_blob_sas
from datetime import datetime
import uuid
import hashlib
import os, re, asyncio

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
_cli: BlobServiceClient | None = None

async def _client() -> BlobServiceClient:
    global _cli
    if not _cli:
        _cli = BlobServiceClient.from_connection_string(_CONN)
    return _cli

async def fetch_artifact(uri: str) -> bytes:
    """Download artifact from Azure Blob Storage"""
    m = _ART_RE.match(uri)
    if not m:
        raise ValueError(f"bad artifact URI: {uri}")
    cont, blob = m.groups()
    blob_cli = (await _client()).get_blob_client(cont, blob)
    stream = await blob_cli.download_blob()
    return await stream.readall()

async def upload_artifact(data: bytes, mime: str = "application/octet-stream") -> dict:
    """Upload bytes to Azure and return artifact dict"""
    cli = await _client()
    blob_name = f"{datetime.utcnow().isoformat()}/{uuid.uuid4().hex}"
    blob = cli.get_blob_client(_CONTAINER, blob_name)
    await blob.upload_blob(data, overwrite=True, content_type=mime)
    
    return {
        "uri": f"azure://{_CONTAINER}/{blob_name}",
        "type": mime,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }

def fetch_artifact_sync(uri: str) -> bytes:
    """Sync wrapper for Streamlit. Use inside the UI thread."""
    return asyncio.run(fetch_artifact(uri))
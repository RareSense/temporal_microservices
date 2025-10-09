# Extremely thin wrapper to download an Artifact from Azure
from azure.storage.blob.aio import BlobServiceClient
import re, os

_ART_REGEX = re.compile(r"^azure://([^/]+)/(.+)$")
_CONN = os.getenv("AZURE_BLOB_CONN")
_client: BlobServiceClient | None = None

async def _cli():
    global _client
    if not _client:
        _client = BlobServiceClient.from_connection_string(_CONN)
    return _client

async def fetch_artifact(uri: str) -> bytes:
    m = _ART_REGEX.match(uri)
    if not m:
        raise ValueError("bad artifact URI")
    cont, blob = m.groups()
    blob_cli = (await _cli()).get_blob_client(cont, blob)
    stream = await blob_cli.download_blob()
    return await stream.readall()

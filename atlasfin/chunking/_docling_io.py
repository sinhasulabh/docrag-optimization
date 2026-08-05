from pathlib import Path

from docling_core.types.doc.document import DoclingDocument
from google.cloud import storage

from atlasfin_ingest import gcs as ingest_gcs


def _split_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"expected a gs:// URI, got {uri!r}")
    _, _, rest = uri.partition("gs://")
    bucket, _, blob_name = rest.partition("/")
    return bucket, blob_name


def md_uri_for(gcs_parsed_uri: str) -> str:
    """{doc_name}.docling.json -> the sibling {doc_name}.md Stage-3 also writes."""
    if not gcs_parsed_uri.endswith(".docling.json"):
        raise ValueError(f"expected a *.docling.json URI, got {gcs_parsed_uri!r}")
    return gcs_parsed_uri[: -len(".docling.json")] + ".md"


def load_docling_document(
    gcs_parsed_uri: str, gcs_client: storage.Client, scratch_dir: Path
) -> DoclingDocument:
    bucket, blob_name = _split_gs_uri(gcs_parsed_uri)
    local_path = scratch_dir / Path(blob_name).name
    ingest_gcs.download_blob_to_file(gcs_client, bucket, blob_name, local_path)
    try:
        return DoclingDocument.load_from_json(local_path)
    finally:
        local_path.unlink(missing_ok=True)


def load_markdown(gcs_md_uri: str, gcs_client: storage.Client, scratch_dir: Path) -> str:
    bucket, blob_name = _split_gs_uri(gcs_md_uri)
    local_path = scratch_dir / Path(blob_name).name
    ingest_gcs.download_blob_to_file(gcs_client, bucket, blob_name, local_path)
    try:
        return local_path.read_text(encoding="utf-8")
    finally:
        local_path.unlink(missing_ok=True)

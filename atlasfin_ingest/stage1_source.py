import json
import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .models import SourceRecord

logger = logging.getLogger("atlasfin_ingest.stage1")

REQUIRED_FIELDS = {"doc_name", "company", "gics_sector", "doc_type", "doc_period", "doc_link"}


@dataclass
class SourceFilter:
    doc_types: list[str] | None = None
    companies: list[str] | None = None
    doc_names: list[str] | None = None


def _iter_lines(source_uri: str):
    if source_uri.startswith("gs://"):
        from . import gcs

        yield from gcs.iter_blob_lines(source_uri)
        return
    if source_uri.startswith("http://") or source_uri.startswith("https://"):
        import requests

        resp = requests.get(source_uri, timeout=30)
        resp.raise_for_status()
        yield from resp.text.splitlines()
        return
    with Path(source_uri).open("r", encoding="utf-8") as f:
        yield from f


def _is_valid_doc_link(link: object) -> bool:
    if not isinstance(link, str) or not link:
        return False
    parsed = urlparse(link)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _matches_filter(rec: SourceRecord, flt: SourceFilter | None) -> bool:
    if flt is None:
        return True
    if flt.doc_types is not None and rec.doc_type not in flt.doc_types:
        return False
    if flt.companies is not None and rec.company not in flt.companies:
        return False
    if flt.doc_names is not None and rec.doc_name not in flt.doc_names:
        return False
    return True


def read_source(source_uri: str, flt: SourceFilter | None = None) -> list[SourceRecord]:
    by_name: dict[str, SourceRecord] = {}
    n_lines = 0

    for lineno, raw_line in enumerate(_iter_lines(source_uri), start=1):
        line = raw_line.strip()
        if not line:
            continue
        n_lines += 1

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("stage1: malformed JSON at line %d: %s", lineno, exc)
            continue

        if not isinstance(obj, dict) or not REQUIRED_FIELDS.issubset(obj):
            logger.warning("stage1: line %d missing required fields, skipping", lineno)
            continue

        if not _is_valid_doc_link(obj.get("doc_link")):
            logger.warning(
                "stage1: line %d (%s) has invalid/missing doc_link, dropping",
                lineno,
                obj.get("doc_name"),
            )
            continue

        try:
            rec = SourceRecord(
                doc_name=str(obj["doc_name"]),
                company=str(obj["company"]),
                gics_sector=str(obj["gics_sector"]),
                doc_type=str(obj["doc_type"]),
                doc_period=int(obj["doc_period"]),
                doc_link=str(obj["doc_link"]),
            )
        except (TypeError, ValueError) as exc:
            logger.warning("stage1: line %d failed validation: %s", lineno, exc)
            continue

        if rec.doc_name in by_name:
            logger.warning(
                "stage1: duplicate doc_name %r at line %d, last-wins", rec.doc_name, lineno
            )
        by_name[rec.doc_name] = rec

    records = [r for r in by_name.values() if _matches_filter(r, flt)]
    logger.info(
        "stage1: read %d lines, %d unique docs, %d after filter",
        n_lines,
        len(by_name),
        len(records),
    )
    return records

# TODO(phase-2): invert this dependency — move the canonical dataclass definitions here
# and turn atlasfin_ingest/models.py into the re-export shim instead. Not done now to avoid
# adding regression risk to the already-shipping ingestion package in the same change that
# builds five new components on top of it.
from atlasfin_ingest.models import ParsedObject, RawObject, SourceRecord, Status

__all__ = ["Status", "SourceRecord", "RawObject", "ParsedObject"]

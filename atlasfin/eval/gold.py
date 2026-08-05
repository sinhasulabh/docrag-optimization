import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GoldEvidence:
    doc_name: str
    evidence_page_num: int
    evidence_text: str


@dataclass(frozen=True)
class GoldRecord:
    financebench_id: str
    company: str
    doc_name: str
    question: str
    answer: str
    evidence: list[GoldEvidence]


def load_gold_set(path: str | Path, doc_names: list[str] | None = None) -> list[GoldRecord]:
    """Reads financebench_open_source.jsonl. If doc_names is given, keeps only questions with
    at least one evidence item grounded in one of those docs -- scoring recall@k against
    questions whose evidence can't possibly be in the indexed corpus would be meaningless
    noise, not a real recall measurement.
    """
    records: list[GoldRecord] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            evidence = [
                GoldEvidence(
                    doc_name=e["doc_name"],
                    evidence_page_num=e["evidence_page_num"],
                    evidence_text=e.get("evidence_text", ""),
                )
                for e in obj.get("evidence", [])
            ]
            if doc_names is not None:
                grounded = any(e.doc_name in doc_names for e in evidence) or obj["doc_name"] in doc_names
                if not grounded:
                    continue
            records.append(
                GoldRecord(
                    financebench_id=obj["financebench_id"],
                    company=obj["company"],
                    doc_name=obj["doc_name"],
                    question=obj["question"],
                    answer=obj["answer"],
                    evidence=evidence,
                )
            )
    return records

#!/usr/bin/env python3
"""Privacy-aware HOLiFOOD supply-chain dataset workflow.

This module replaces the former ingest/embed/index/anchor/score/label/export
scripts with one configurable program.  The normal path is::

    python supply_chain_workflow.py run \
        --input articles.json \
        --output-dir supply_chain_output \
        --output-format parquet

It creates a labelled master dataset and separate cereals, poultry and legumes
datasets.  FAISS indexing and semantic search are optional because they are not
required for supply-chain classification.

Core dependencies are pandas, NumPy, sentence-transformers and PyTorch. Strict
privacy mode additionally requires Presidio and its configured language model;
Parquet and FAISS outputs require pyarrow and faiss-cpu respectively.

Production privacy mode is fail-closed: Microsoft Presidio and a suitable NLP
model must be available.  Regex mode exists for deterministic development tests
but does not detect person names reliably.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import hmac
import json
import logging
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np
import pandas as pd


VERSION = "1.0.0"
LOGGER = logging.getLogger("supply_chain_workflow")
COMMODITIES = ("poultry", "cereals", "legumes")
COMPATIBLE_EXPORT_STEMS = {
    "poultry": "poultry_articles",
    "cereals": "cereal_articles",
    "legumes": "legume_articles",
}


DEFAULT_ANCHORS: dict[str, dict[str, list[str]]] = {
    "poultry": {
        "production": [
            "poultry farm",
            "broiler production",
            "chicken farming",
            "turkey farming",
            "poultry industry",
        ],
        "processing": [
            "poultry slaughterhouse",
            "chicken processing plant",
            "poultry meat processing facility",
            "chicken slaughter line",
        ],
        "products": [
            "chicken meat product",
            "poultry meat",
            "processed chicken meat",
            "egg production poultry",
        ],
        "disease": [
            "avian influenza outbreak poultry",
            "salmonella contamination poultry",
            "campylobacter poultry meat contamination",
            "poultry disease outbreak farm",
        ],
        "feed": [
            "poultry feed contamination",
            "soy meal poultry feed",
            "grain feed poultry farm",
            "animal feed contamination poultry",
        ],
    },
    "cereals": {
        "grain": [
            "wheat grain production",
            "maize corn crop farming",
            "barley cereal crop",
            "oat grain agriculture",
        ],
        "processing": [
            "flour milling wheat",
            "grain storage silo",
            "grain processing facility",
            "cereal processing plant",
        ],
        "products": [
            "wheat flour",
            "bread wheat food products",
            "cereal grain food production",
            "grain based food products",
        ],
        "contamination": [
            "grain mycotoxin contamination",
            "aflatoxin wheat contamination",
            "fumonisin maize contamination",
            "mycotoxin cereal contamination",
        ],
    },
    "legumes": {
        "crops": [
            "soybean crop farming",
            "lentil agriculture production",
            "bean crop cultivation",
            "chickpea farming production",
        ],
        "products": [
            "soy protein food product",
            "legume based food products",
            "plant protein soy product",
            "soybean food ingredient",
        ],
        "feed": [
            "soy meal animal feed",
            "legume feed ingredient livestock",
            "soybean feed supply chain",
        ],
        "contamination": [
            "soy pesticide residue food",
            "legume contamination food safety",
            "soybean contamination chemical residue",
        ],
    },
}


class WorkflowError(RuntimeError):
    """Raised when a workflow invariant is not satisfied."""


class PrivacyError(WorkflowError):
    """Raised when privacy-safe processing cannot be guaranteed."""


@dataclass(frozen=True)
class Finding:
    entity_type: str
    start: int
    end: int
    score: float


@dataclass
class WorkflowConfig:
    input_path: str
    output_dir: str
    output_format: str = "parquet"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    device: str = "auto"
    batch_size: int = 64
    max_chars: int = 6000
    min_similarity: float = 0.30
    min_margin: float = 0.02
    privacy_mode: str = "strict"
    language: str = "en"
    anchor_config: str | None = None
    source_allowlist: tuple[str, ...] = ()
    save_embeddings: bool = False
    build_faiss_index: bool = False


def validate_config(config: WorkflowConfig) -> None:
    if config.batch_size < 1:
        raise WorkflowError("--batch-size must be at least 1.")
    if config.max_chars < 200:
        raise WorkflowError("--max-chars must be at least 200.")
    if not -1.0 <= config.min_similarity <= 1.0:
        raise WorkflowError("--min-similarity must be between -1 and 1.")
    if not 0.0 <= config.min_margin <= 2.0:
        raise WorkflowError("--min-margin must be between 0 and 2.")


_SPACE_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")
_REPEAT_RE = re.compile(r"(.{30,120})(?:\s+\1){3,}", re.I)
_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "EMAIL_ADDRESS",
        re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])", re.I),
    ),
    (
        "IP_ADDRESS",
        re.compile(r"(?<!\d)(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?!\d)"),
    ),
    (
        "IBAN_CODE",
        re.compile(r"(?<![A-Z0-9])[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}(?![A-Z0-9])", re.I),
    ),
    (
        "PHONE_NUMBER",
        re.compile(r"(?<!\w)(?:\+|00)?\d{1,3}[\s()./-]*(?:\d[\s()./-]*){7,14}(?!\w)"),
    ),
    ("SOCIAL_HANDLE", re.compile(r"(?<![\w@])@[A-Z0-9_]{2,30}\b", re.I)),
    (
        "PRECISE_COORDINATE",
        re.compile(r"(?<!\d)-?\d{1,2}\.\d{4,}\s*[,;]\s*-?\d{1,3}\.\d{4,}(?!\d)"),
    ),
)


class PrivacyGuard:
    """PII detection/redaction with strict Presidio or regex-only modes."""

    DEFAULT_ENTITIES = (
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "IP_ADDRESS",
        "IBAN_CODE",
        "CREDIT_CARD",
        "CRYPTO",
        "MEDICAL_LICENSE",
        "US_SSN",
        "US_PASSPORT",
        "US_DRIVER_LICENSE",
        "US_BANK_NUMBER",
        "UK_NHS",
    )

    def __init__(self, mode: str = "strict", language: str = "en") -> None:
        if mode not in {"strict", "regex", "off"}:
            raise PrivacyError(f"Unsupported privacy mode: {mode}")
        self.mode = mode
        self.language = language
        self._analyzer = None
        if mode == "strict":
            try:
                from presidio_analyzer import AnalyzerEngine

                self._analyzer = AnalyzerEngine()
                self._analyzer.analyze(text="privacy startup check", language=language)
            except Exception as exc:
                raise PrivacyError(
                    "Strict privacy mode requires Microsoft Presidio and a "
                    "working NLP model. Export is blocked."
                ) from exc
        elif mode == "off":
            LOGGER.warning("Privacy mode is OFF; outputs are not dashboard-safe.")

    @staticmethod
    def _merge(findings: Iterable[Finding]) -> list[Finding]:
        ranked = sorted(
            findings,
            key=lambda f: (f.start, -(f.end - f.start), -f.score, f.entity_type),
        )
        accepted: list[Finding] = []
        for finding in ranked:
            if finding.start >= finding.end:
                continue
            if any(finding.start < old.end and old.start < finding.end for old in accepted):
                continue
            accepted.append(finding)
        return sorted(accepted, key=lambda f: f.start)

    def detect(self, text: str) -> list[Finding]:
        if self.mode == "off":
            return []
        value = text or ""
        findings: list[Finding] = []
        if self._analyzer is not None:
            results = self._analyzer.analyze(
                text=value,
                language=self.language,
                entities=list(self.DEFAULT_ENTITIES),
                score_threshold=0.55,
            )
            findings.extend(
                Finding(r.entity_type, int(r.start), int(r.end), float(r.score))
                for r in results
            )
        for entity_type, pattern in _PII_PATTERNS:
            findings.extend(
                Finding(entity_type, match.start(), match.end(), 1.0)
                for match in pattern.finditer(value)
            )
        return self._merge(findings)

    def redact(self, text: str) -> tuple[str, int]:
        value = text or ""
        findings = self.detect(value)
        for finding in reversed(findings):
            marker = f"[REDACTED_{finding.entity_type}]"
            value = value[: finding.start] + marker + value[finding.end :]
        return value, len(findings)

    def assert_safe(self, text: str, *, context: str) -> None:
        findings = self.detect(text or "")
        if findings:
            kinds = ", ".join(sorted({item.entity_type for item in findings}))
            raise PrivacyError(f"Residual PII at {context}: {kinds}")


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\x00", " ")
    return _SPACE_RE.sub(" ", text).strip()


def summary_is_bad(summary: str) -> bool:
    text = normalize_text(summary)
    if len(text) < 80:
        return True
    tokens = text.lower().split()
    if len(tokens) < 20:
        return True
    if len(set(tokens)) / len(tokens) < 0.35:
        return True
    return bool(_REPEAT_RE.search(text))


def clean_content(content: str) -> str:
    text = content or ""
    try:
        import trafilatura

        extracted = trafilatura.extract(
            text, include_comments=False, include_tables=False
        )
        if extracted and len(extracted) > 200:
            return normalize_text(extracted)
    except ImportError:
        pass
    if "<" in text and ">" in text:
        text = _TAG_RE.sub(" ", text)
    return normalize_text(text)


def first_value(record: Mapping[str, Any], names: Sequence[str], default: Any = "") -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return default


def source_domain(url_or_domain: str) -> str:
    value = normalize_text(url_or_domain)
    if not value:
        return ""
    parsed = urlsplit(value if "://" in value else f"https://{value}")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not re.fullmatch(r"[a-z0-9.-]+", host):
        return ""
    return host[4:] if host.startswith("www.") else host


def public_source_domain(value: str, allowlist: Sequence[str]) -> str:
    host = source_domain(value)
    approved = {source_domain(item) for item in allowlist if source_domain(item)}
    return host if host and host in approved else "withheld"


def stable_article_id(
    record: Mapping[str, Any], *, canonical_text: str, secret: str
) -> str:
    existing = normalize_text(first_value(record, ("Article_ID", "article_id")))
    # Legacy preprocessing used the raw URL itself as article_id. Do not carry
    # that value across the privacy boundary.
    existing_is_safe = bool(
        existing
        and len(existing) <= 128
        and "://" not in existing
        and "@" not in existing
        and re.fullmatch(r"[A-Za-z0-9_.-]+", existing)
    )
    if existing_is_safe:
        return existing
    if len(secret) < 32:
        raise PrivacyError(
            "ERI_PRIVACY_HMAC_KEY must contain at least 32 characters when "
            "input records do not already have privacy-safe Article_ID values."
        )
    url = normalize_text(first_value(record, ("URL", "url", "Source_URL")))
    identity = url or canonical_text
    digest = hmac.new(secret.encode(), identity.encode(), sha256).hexdigest()
    return f"eri_{digest[:24]}"


def parse_date(value: Any) -> str:
    if value is None or normalize_text(value) == "":
        return ""
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return normalize_text(value)
    return parsed.isoformat()


def load_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source).to_dict(orient="records")
    if suffix == ".csv":
        return pd.read_csv(source).to_dict(orient="records")
    if suffix in {".jsonl", ".ndjson"}:
        with source.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with source.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise WorkflowError("JSON input must contain a list of records.")
            return payload
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json_dump(payload: Any, path: str | Path, *, private: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        if private:
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        os.replace(temp_name, target)
        if private:
            os.chmod(target, 0o600)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def write_table(df: pd.DataFrame, path: str | Path, file_format: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if file_format == "parquet":
        try:
            df.to_parquet(target, index=False)
        except ImportError as exc:
            raise WorkflowError(
                "Parquet output requires pyarrow or fastparquet. Use --output-format "
                "csv/jsonl or install pyarrow."
            ) from exc
    elif file_format == "csv":
        df.to_csv(target, index=False)
    elif file_format == "jsonl":
        df.to_json(target, orient="records", lines=True, force_ascii=False, date_format="iso")
    else:
        raise WorkflowError(f"Unsupported output format: {file_format}")
    return target


def prepare_dataframe(
    records: Sequence[Mapping[str, Any]],
    *,
    guard: PrivacyGuard,
    hmac_secret: str,
    max_chars: int,
    source_allowlist: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        summary = normalize_text(first_value(record, ("Summary", "summary")))
        content = normalize_text(
            first_value(
                record,
                ("Redacted_Content", "redacted_content", "Content", "content", "Text", "text"),
            )
        )
        used_summary = bool(summary and not summary_is_bad(summary))
        selected = summary if used_summary else clean_content(content)
        if not selected:
            skipped += 1
            continue
        redacted, redaction_count = guard.redact(selected[:max_chars])
        canonical = normalize_text(redacted)
        if not canonical:
            skipped += 1
            continue
        guard.assert_safe(canonical, context="canonical_text")
        article_id = stable_article_id(record, canonical_text=canonical, secret=hmac_secret)
        raw_source = normalize_text(
            first_value(record, ("Source_Domain", "source_domain", "URL", "url"))
        )
        rows.append(
            {
                "article_id": article_id,
                "source_domain": public_source_domain(raw_source, source_allowlist),
                "scrape_date": parse_date(
                    first_value(record, ("Scrape_Date", "Scrape Date", "scrape_date", "date"))
                ),
                "canonical_text": canonical,
                "used_summary": used_summary,
                "privacy_redaction_count": int(redaction_count),
                "privacy_status": "PASSED" if guard.mode != "off" else "NOT_CHECKED",
            }
        )
    if not rows:
        raise WorkflowError("No usable text records remained after preprocessing.")
    df = pd.DataFrame(rows).drop_duplicates(subset=["article_id"], keep="first")
    LOGGER.info(
        "Prepared %d unique records; skipped %d empty records.", len(df), skipped
    )
    return df.reset_index(drop=True)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def encode_texts(
    texts: Sequence[str], *, model_name: str, device: str, batch_size: int
) -> tuple[np.ndarray, Any]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise WorkflowError(
            "sentence-transformers is required for embedding generation."
        ) from exc
    model = SentenceTransformer(model_name, device=resolve_device(device))
    vectors = model.encode(
        list(texts),
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise WorkflowError("Embedding model returned an unexpected shape.")
    if not np.isfinite(vectors).all():
        raise WorkflowError("Embedding matrix contains non-finite values.")
    return vectors, model


def load_anchor_config(path: str | None) -> dict[str, dict[str, list[str]]]:
    if not path:
        return DEFAULT_ANCHORS
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if set(payload) != set(COMMODITIES):
        raise WorkflowError(
            f"Anchor config must contain exactly: {', '.join(COMMODITIES)}"
        )
    for commodity, clusters in payload.items():
        if not isinstance(clusters, dict) or not clusters:
            raise WorkflowError(f"No anchor clusters for {commodity}.")
        for cluster, phrases in clusters.items():
            if not isinstance(phrases, list) or not phrases:
                raise WorkflowError(f"Empty anchor phrase list: {commodity}/{cluster}")
    return payload


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return values / norms


def encode_anchor_centroids(
    anchors: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    model: Any,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for commodity in COMMODITIES:
        result[commodity] = {}
        for cluster, phrases in anchors[commodity].items():
            vectors = model.encode(
                list(phrases),
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            centroid = np.asarray(vectors, dtype=np.float32).mean(axis=0, keepdims=True)
            result[commodity][cluster] = l2_normalize(centroid)[0]
    return result


def score_embeddings(
    embeddings: np.ndarray,
    anchor_vectors: Mapping[str, Mapping[str, np.ndarray]],
    *,
    min_similarity: float,
    min_margin: float,
) -> pd.DataFrame:
    docs = l2_normalize(embeddings)
    similarity_columns: dict[str, np.ndarray] = {}
    for commodity in COMMODITIES:
        clusters = np.stack(list(anchor_vectors[commodity].values())).astype(np.float32)
        clusters = l2_normalize(clusters)
        similarity_columns[commodity] = (docs @ clusters.T).max(axis=1)

    similarities = np.column_stack([similarity_columns[name] for name in COMMODITIES])
    order = np.argsort(similarities, axis=1)
    best_idx = order[:, -1]
    second_idx = order[:, -2]
    row_idx = np.arange(len(docs))
    best_similarity = similarities[row_idx, best_idx]
    second_similarity = similarities[row_idx, second_idx]
    margin = best_similarity - second_similarity
    accepted = (best_similarity >= min_similarity) & (margin >= min_margin)
    labels = np.array(COMMODITIES, dtype=object)[best_idx]
    labels = np.where(accepted, labels, "unclassified")

    output = {
        f"{name}_similarity": similarity_columns[name] for name in COMMODITIES
    }
    output.update(
        {
            "primary_supply_chain": labels,
            "supply_chain_similarity": best_similarity,
            "supply_chain_margin": margin,
            "supply_chain_accepted": accepted,
        }
    )
    return pd.DataFrame(output)


def build_faiss(embeddings: np.ndarray, path: Path) -> None:
    try:
        import faiss
    except ImportError as exc:
        raise WorkflowError("faiss is required when --build-faiss-index is used.") from exc
    vectors = np.ascontiguousarray(l2_normalize(embeddings), dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(path))


def extension_for(file_format: str) -> str:
    return {"parquet": ".parquet", "csv": ".csv", "jsonl": ".jsonl"}[file_format]


def run_pipeline(config: WorkflowConfig) -> dict[str, Any]:
    validate_config(config)
    started = datetime.now(timezone.utc)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    guard = PrivacyGuard(config.privacy_mode, config.language)
    hmac_secret = os.environ.get("ERI_PRIVACY_HMAC_KEY", "")
    records = load_records(config.input_path)
    df = prepare_dataframe(
        records,
        guard=guard,
        hmac_secret=hmac_secret,
        max_chars=config.max_chars,
        source_allowlist=config.source_allowlist,
    )
    embeddings, model = encode_texts(
        df["canonical_text"].tolist(),
        model_name=config.embedding_model,
        device=config.device,
        batch_size=config.batch_size,
    )
    anchors = load_anchor_config(config.anchor_config)
    anchor_vectors = encode_anchor_centroids(
        anchors, model=model, batch_size=config.batch_size
    )
    scores = score_embeddings(
        embeddings,
        anchor_vectors,
        min_similarity=config.min_similarity,
        min_margin=config.min_margin,
    )
    result = pd.concat([df, scores], axis=1)
    ext = extension_for(config.output_format)
    master_path = write_table(
        result, output_dir / f"supply_chain_labeled{ext}", config.output_format
    )

    output_paths: dict[str, str] = {"master": str(master_path)}
    counts: dict[str, int] = {}
    for commodity in COMMODITIES:
        subset = result[result["primary_supply_chain"] == commodity].copy()
        path = write_table(
            subset,
            # Preserve the filenames consumed by the existing dashboard flow.
            output_dir / f"{COMPATIBLE_EXPORT_STEMS[commodity]}{ext}",
            config.output_format,
        )
        output_paths[commodity] = str(path)
        counts[commodity] = int(len(subset))
    counts["unclassified"] = int(
        (result["primary_supply_chain"] == "unclassified").sum()
    )

    if config.save_embeddings or config.build_faiss_index:
        np.save(output_dir / "embeddings.npy", embeddings, allow_pickle=False)
        atomic_json_dump(result["article_id"].tolist(), output_dir / "article_ids.json")
        output_paths["embeddings"] = str(output_dir / "embeddings.npy")
        output_paths["article_ids"] = str(output_dir / "article_ids.json")
    if config.build_faiss_index:
        index_path = output_dir / "faiss.index"
        build_faiss(embeddings, index_path)
        output_paths["faiss_index"] = str(index_path)

    manifest = {
        "workflow": "HOLiFOOD supply-chain classification",
        "version": VERSION,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(Path(config.input_path).resolve()),
        "input_records": len(records),
        "output_records": len(result),
        "counts": counts,
        "config": asdict(config),
        "outputs": output_paths,
        "privacy_note": (
            "Strict mode uses Presidio plus deterministic patterns. Regex mode "
            "does not reliably detect person names."
        ),
    }
    atomic_json_dump(manifest, output_dir / "supply_chain_manifest.json")
    LOGGER.info("Supply-chain outputs: %s", counts)
    return manifest


def semantic_search(
    *,
    query: str,
    output_dir: str,
    model_name: str,
    device: str,
    k: int,
) -> list[dict[str, Any]]:
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise WorkflowError(
            "semantic search requires faiss and sentence-transformers."
        ) from exc
    root = Path(output_dir)
    manifest = json.loads((root / "supply_chain_manifest.json").read_text(encoding="utf-8"))
    master = Path(manifest["outputs"]["master"])
    df = pd.DataFrame(load_records(master))
    index = faiss.read_index(str(root / "faiss.index"))
    model = SentenceTransformer(model_name, device=resolve_device(device))
    vector = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    scores, positions = index.search(vector, min(k, len(df)))
    results: list[dict[str, Any]] = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue
        row = df.iloc[int(position)]
        results.append(
            {
                "article_id": row["article_id"],
                "supply_chain": row["primary_supply_chain"],
                "similarity": round(float(score), 6),
                "text_preview": str(row["canonical_text"])[:300],
            }
        )
    return results


def self_test() -> None:
    guard = PrivacyGuard("regex")
    redacted, count = guard.redact(
        "Contact jane@example.org or +36 30 123 4567 from 192.168.1.20."
    )
    assert count >= 3
    assert "jane@example.org" not in redacted

    test_records = [
        {
            "article_id": "https://example.org/Jane-Doe",
            "URL": "https://example.org/Jane-Doe?email=jane@example.org",
            "Summary": (
                "A sufficiently long food safety summary describing a poultry "
                "contamination event and the response by relevant authorities "
                "without repeating boilerplate in the analytical record."
            ),
            "Scrape Date": "2026-01-01",
        }
    ]
    prepared = prepare_dataframe(
        test_records,
        guard=guard,
        hmac_secret="a-test-secret-key-longer-than-thirty-two-characters",
        max_chars=6000,
        source_allowlist=(),
    )
    assert prepared.loc[0, "article_id"].startswith("eri_")
    assert "example.org" not in prepared.loc[0, "canonical_text"]
    assert prepared.loc[0, "source_domain"] == "withheld"

    vectors = np.eye(3, dtype=np.float32)
    docs = np.vstack([vectors, np.array([[1.0, 1.0, 1.0]], dtype=np.float32)])
    anchors = {
        "poultry": {"test": vectors[0]},
        "cereals": {"test": vectors[1]},
        "legumes": {"test": vectors[2]},
    }
    scores = score_embeddings(
        docs, anchors, min_similarity=0.30, min_margin=0.02
    )
    assert scores["primary_supply_chain"].tolist() == [
        "poultry",
        "cereals",
        "legumes",
        "unclassified",
    ]
    assert np.allclose(
        np.linalg.norm(l2_normalize(np.array([[3.0, 4.0]])), axis=1), [1.0]
    )
    print("Supply-chain workflow self-test: PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run preprocessing, embedding, scoring and export.")
    run.add_argument("--input", required=True, dest="input_path")
    run.add_argument("--output-dir", required=True)
    run.add_argument("--output-format", default="parquet", choices=("parquet", "csv", "jsonl"))
    run.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    run.add_argument("--device", default="auto")
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--max-chars", type=int, default=6000)
    run.add_argument("--min-similarity", type=float, default=0.30)
    run.add_argument("--min-margin", type=float, default=0.02)
    run.add_argument("--privacy-mode", default="strict", choices=("strict", "regex", "off"))
    run.add_argument("--language", default="en")
    run.add_argument("--anchor-config")
    run.add_argument("--allow-domain", action="append", default=[])
    run.add_argument("--save-embeddings", action="store_true")
    run.add_argument("--build-faiss-index", action="store_true")

    search = sub.add_parser("search", help="Search an optional saved FAISS index.")
    search.add_argument("--query", required=True)
    search.add_argument("--output-dir", required=True)
    search.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    search.add_argument("--device", default="auto")
    search.add_argument("-k", type=int, default=20)

    sub.add_parser("self-test", help="Run dependency-light deterministic tests.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        if args.command == "self-test":
            self_test()
        elif args.command == "search":
            results = semantic_search(
                query=args.query,
                output_dir=args.output_dir,
                model_name=args.embedding_model,
                device=args.device,
                k=args.k,
            )
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            config = WorkflowConfig(
                input_path=args.input_path,
                output_dir=args.output_dir,
                output_format=args.output_format,
                embedding_model=args.embedding_model,
                device=args.device,
                batch_size=args.batch_size,
                max_chars=args.max_chars,
                min_similarity=args.min_similarity,
                min_margin=args.min_margin,
                privacy_mode=args.privacy_mode,
                language=args.language,
                anchor_config=args.anchor_config,
                source_allowlist=tuple(args.allow_domain),
                save_embeddings=args.save_embeddings,
                build_faiss_index=args.build_faiss_index,
            )
            manifest = run_pipeline(config)
            print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    except (WorkflowError, ValueError, OSError, json.JSONDecodeError) as exc:
        LOGGER.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

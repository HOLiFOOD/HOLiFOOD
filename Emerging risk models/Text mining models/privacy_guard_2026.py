"""Privacy boundary for the ERI unstructured-data workflow.

The dashboard must only receive records produced by ``build_dashboard_record``.
PII detection uses Microsoft Presidio in production and supplements it with
deterministic recognizers for identifiers that should never reach the public
surface.  Production mode fails closed when the NER-based detector is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import os
import re
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit


class PrivacyConfigurationError(RuntimeError):
    """The privacy layer cannot safely start with the current configuration."""


class PrivacyViolation(RuntimeError):
    """Potential personal data was detected at the dashboard boundary."""


@dataclass(frozen=True)
class Finding:
    entity_type: str
    start: int
    end: int
    score: float


@dataclass(frozen=True)
class RedactionResult:
    text: str
    findings: tuple[Finding, ...]

    @property
    def redaction_count(self) -> int:
        return len(self.findings)


# LOCATION is deliberately excluded: country/region information is essential
# to food-risk analysis. Exact addresses and coordinates are handled by the
# deterministic recognizers and can be extended with project-specific rules.
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


_REGEX_RECOGNIZERS: tuple[tuple[str, re.Pattern[str]], ...] = (
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
    (
        "SOCIAL_HANDLE",
        re.compile(r"(?<![\w@])@[A-Z0-9_]{2,30}\b", re.I),
    ),
    (
        "PRECISE_COORDINATE",
        re.compile(r"(?<!\d)-?\d{1,2}\.\d{4,}\s*[,;]\s*-?\d{1,3}\.\d{4,}(?!\d)"),
    ),
)


DASHBOARD_FIELDS = frozenset(
    {
        "Article_ID",
        "Source_Domain",
        "Scrape_Date",
        "Assigned_Topic",
        "Topic_Probability",
        "Topic_Keywords",
        "Summary_Snippet",
        "Privacy_Status",
    }
)


class PrivacyGuard:
    """Redact personal data and enforce the public dashboard data contract."""

    def __init__(
        self,
        *,
        language: str = "en",
        score_threshold: float = 0.55,
        require_ner: bool = True,
        entities: Sequence[str] = DEFAULT_ENTITIES,
    ) -> None:
        self.language = language
        self.score_threshold = score_threshold
        self.entities = tuple(entities)
        self._analyzer = None
        try:
            from presidio_analyzer import AnalyzerEngine

            self._analyzer = AnalyzerEngine()
            # Force model initialization now so the service fails at startup,
            # not halfway through an export.
            self._analyzer.analyze(text="privacy startup check", language=language)
        except Exception as exc:  # dependency/model/configuration failure
            if require_ner:
                raise PrivacyConfigurationError(
                    "Production privacy mode requires a working Microsoft "
                    "Presidio Analyzer and NLP model. Dashboard export is blocked."
                ) from exc

    def detect(self, text: str) -> tuple[Finding, ...]:
        text = text or ""
        findings: list[Finding] = []

        if self._analyzer is not None:
            results = self._analyzer.analyze(
                text=text,
                language=self.language,
                entities=list(self.entities),
                score_threshold=self.score_threshold,
            )
            findings.extend(
                Finding(r.entity_type, int(r.start), int(r.end), float(r.score))
                for r in results
            )

        for entity_type, pattern in _REGEX_RECOGNIZERS:
            findings.extend(
                Finding(entity_type, match.start(), match.end(), 1.0)
                for match in pattern.finditer(text)
            )

        return tuple(self._merge_overlaps(findings))

    @staticmethod
    def _merge_overlaps(findings: Iterable[Finding]) -> list[Finding]:
        # Prefer longer and higher-confidence findings when recognizers overlap.
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

    def redact_text(self, text: str) -> RedactionResult:
        text = text or ""
        findings = self.detect(text)
        redacted = text
        for finding in reversed(findings):
            replacement = f"[REDACTED_{finding.entity_type}]"
            redacted = redacted[: finding.start] + replacement + redacted[finding.end :]
        return RedactionResult(redacted, findings)

    def assert_safe_text(self, text: str, *, context: str) -> None:
        findings = self.detect(text or "")
        if findings:
            kinds = ", ".join(sorted({f.entity_type for f in findings}))
            raise PrivacyViolation(f"Dashboard export blocked at {context}: {kinds}")

    def build_dashboard_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Return an allowlisted, redacted record suitable for the dashboard."""
        summary = self.redact_text(str(record.get("Summary", ""))).text
        snippet = summary[:500]

        raw_keywords = record.get("Topic_Keywords", []) or []
        safe_keywords = [
            self.redact_text(str(keyword)).text[:100] for keyword in raw_keywords[:10]
        ]

        public = {
            "Article_ID": str(record.get("Article_ID", "")),
            "Source_Domain": public_source_domain(
                str(record.get("Source_Domain", ""))
            ),
            "Scrape_Date": str(record.get("Scrape_Date", "")),
            "Assigned_Topic": int(record.get("Assigned_Topic", -1)),
            "Topic_Probability": float(record.get("Topic_Probability", 0.0)),
            "Topic_Keywords": safe_keywords,
            "Summary_Snippet": snippet,
            "Privacy_Status": "PASSED",
        }
        self.assert_dashboard_record(public)
        return public

    def assert_dashboard_record(self, record: Mapping[str, Any]) -> None:
        unexpected = set(record) - DASHBOARD_FIELDS
        missing = DASHBOARD_FIELDS - set(record)
        if unexpected or missing:
            raise PrivacyViolation(
                f"Dashboard schema rejected; unexpected={sorted(unexpected)}, "
                f"missing={sorted(missing)}"
            )

        for field in ("Source_Domain", "Summary_Snippet"):
            self.assert_safe_text(str(record[field]), context=field)
        for index, keyword in enumerate(record["Topic_Keywords"]):
            self.assert_safe_text(str(keyword), context=f"Topic_Keywords[{index}]")


def source_domain(url_or_domain: str) -> str:
    """Return only a normalized hostname; paths, query data and fragments vanish."""
    value = (url_or_domain or "").strip()
    parsed = urlsplit(value if "://" in value else f"https://{value}")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not re.fullmatch(r"[a-z0-9.-]+", host):
        return ""
    return host[4:] if host.startswith("www.") else host


def public_source_domain(url_or_domain: str) -> str:
    """Expose a source only when the operator explicitly approved its domain."""
    host = source_domain(url_or_domain)
    approved = {
        source_domain(item)
        for item in os.environ.get("ERI_PUBLIC_SOURCE_DOMAINS", "").split(",")
        if item.strip()
    }
    return host if host and host in approved else "withheld"


def article_id(url: str, *, secret: str | None = None) -> str:
    """Create a non-reversible public identifier using a secret HMAC key."""
    key = secret or os.environ.get("ERI_PRIVACY_HMAC_KEY", "")
    if len(key) < 32:
        raise PrivacyConfigurationError(
            "ERI_PRIVACY_HMAC_KEY must contain at least 32 characters."
        )
    digest = hmac.new(key.encode(), url.encode(), sha256).hexdigest()
    return f"eri_{digest[:24]}"

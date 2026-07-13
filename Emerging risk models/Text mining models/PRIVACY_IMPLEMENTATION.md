# ERI privacy boundary – implementation guide

## Objective

Prevent personal data contained in scraped news or produced by the summarizer
from reaching the dashboard. This package treats the dashboard as a public or
low-trust surface and applies a fail-closed export policy.

This is a technical safeguard, not a declaration of legal anonymity or a
replacement for a DPIA, retention policy, access controls, and DPO review.

## Privacy architecture

1. **Private ingestion zone** – `news_scraper_2025.py` stores source URLs in
   `private_food_safety_links.json` with owner-only file permissions.
2. **Redaction before persistence** – `article_scraper_json_2025.py` fetches a
   page into memory, detects PII, and persists only redacted content. It replaces
   the source URL with a keyed `Article_ID` and a hostname-only `Source_Domain`.
3. **Redaction after generation** – `summarizer_2025 (1).py` consumes only
   redacted content and independently scans the generated summary. This matters
   because a generative model may reproduce or infer identifying information.
4. **Privacy-safe modelling** – BERTopic is trained on already-redacted
   summaries. Its topic words and visualizations therefore derive from the
   privacy-safe corpus.
5. **Explicit dashboard allowlist** – `BERTopic_json_2025 (2).py` does not
   serialize its input DataFrame. Every public record is rebuilt with only the
   approved fields in `privacy_guard.DASHBOARD_FIELDS`.
6. **Final fail-closed gate** – each summary snippet, topic keyword and source
   domain is rescanned. A detected identifier raises `PrivacyViolation` and
   blocks the export.

## Public dashboard contract

Only the following fields may cross the boundary:

- `Article_ID`: secret-keyed HMAC identifier; it cannot be reversed without the
  private key.
- `Source_Domain`: hostname only, with path, parameters and fragments removed.
  It is exposed only when included in the operator-managed source allowlist;
  otherwise its value is `withheld`.
- `Scrape_Date`: collection timestamp.
- `Assigned_Topic`
- `Topic_Probability`
- `Topic_Keywords`: derived from redacted summaries and rescanned.
- `Summary_Snippet`: at most 500 characters, redacted and rescanned.
- `Privacy_Status`: must equal `PASSED`.

Raw URLs, raw article text, full redacted article text, error messages and model
operational metadata are not permitted in the dashboard export.

## Detection policy

Production mode requires Microsoft Presidio with an operational NER model. It
detects person names and common identifiers. Deterministic recognizers provide a
second layer for email addresses, telephone numbers, IP addresses, IBANs,
social-media handles and precise coordinates.

General locations are intentionally retained because country and regional
information is essential to food-risk identification. Precise coordinates are
removed. The project should add recognizers for national identifiers and exact
postal-address formats in every country covered by the source corpus.

Person names are removed even when the individual is a public figure. The
dashboard does not need a person's identity to represent an emerging food-risk
topic.

## Installation

Use the same managed Python environment as the ERI workflow:

```bash
pip install -r requirements_privacy.txt
python -m spacy download en_core_web_lg
```

Before starting the pipeline, provide a random secret of at least 32 characters:

```bash
export ERI_PRIVACY_HMAC_KEY="replace-with-a-random-secret-from-your-secret-manager"
export ERI_PUBLIC_SOURCE_DOMAINS="who.int,efsa.europa.eu,ecdc.europa.eu"
```

Do not put the key in source code, JSON outputs, logs, notebooks or the
dashboard configuration. Store it in the institution's secret manager and
restrict it to the ingestion service.

Maintain `ERI_PUBLIC_SOURCE_DOMAINS` as an explicit list of approved publisher
domains. This prevents a personal blog domain or identifying subdomain from
being surfaced merely because it appeared in the incoming feed.

The current EMM collector requests English content. Before processing other
languages, configure and validate a suitable Presidio NLP model for each
language. Do not silently fall back to regex-only mode in production.

## Execution order

```bash
python news_scraper_2025.py
python article_scraper_json_2025.py
python "summarizer_2025 (1).py"
python "BERTopic_json_2025 (2).py"
```

The first script is a long-running collector and does not launch the remaining
steps. In the present prototype, stop it after a collection cycle or run the
later stages through the existing scheduler. Full orchestration belongs to the
next workflow revision.

The only document-level dataset approved for dashboard ingestion is:

```text
dashboard_topics.json
```

The generated topic HTML files are also based on redacted summaries. They must
be published together with `dashboard_topics.json`, never with files whose names
begin with `private_`.

## Verification

Run the deterministic test suite:

```bash
python -m unittest -v test_privacy_guard.py
```

Before deployment, build a multilingual privacy test corpus containing:

- common and uncommon personal names;
- Hungarian and international phone/address formats;
- names next to organisations, locations, diseases and food products;
- usernames, email addresses, identifiers and precise coordinates;
- deliberately adversarial spellings and spacing;
- false-positive cases important to food-risk analysis.

Measure recall separately for each entity type and language. For a public
surface, missed identifiers are more harmful than unnecessary redaction, but
false positives must still be monitored because they can obscure risk signals.

## Operational controls still required

- Separate service accounts and storage for private ingestion and dashboard
  publication.
- No public web server access to private files or error logs.
- Encryption in transit and at rest.
- Short, documented retention for raw link and error data.
- Logs containing article IDs rather than URLs or extracted text.
- Human review and quarantine workflow for records blocked by the final gate.
- Dataset and model versioning, audit trails, incident response and deletion
  procedures.
- Periodic re-evaluation after source, language or model changes.
- A formal DPIA and review by the university's data protection officer.

## Design basis

- GDPR Article 5 establishes purpose limitation, data minimisation, storage
  limitation, integrity and confidentiality principles:
  https://eur-lex.europa.eu/eli/reg/2016/679/oj
- Microsoft Presidio provides analyzer and anonymization components but
  explicitly requires task-specific configuration and evaluation:
  https://microsoft.github.io/presidio/
- EDPB guidance distinguishes pseudonymisation from anonymisation; the keyed
  article ID should therefore be treated as pseudonymous operational data, not
  automatically as anonymous data:
  https://www.edpb.europa.eu/public-consultations/guidelines-012025-pseudonymisation_en

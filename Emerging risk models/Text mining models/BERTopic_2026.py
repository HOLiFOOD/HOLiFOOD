#!/usr/bin/env python3
"""BERTopic analysis with optional Emerging Risk Score (ERS).

The program consumes a privacy-safe dataset, performs BERTopic analysis, builds
representative-document and temporal outputs, and optionally calculates the
document/topic/fused ERS measures described in the UNIVET method note.

Typical usage::

    python topic_analysis.py run \
        --input supply_chain_output/poultry_articles.parquet \
        --output-dir topic_output/poultry \
        --ers

The program imports the privacy guard and file utilities from
``supply_chain_workflow.py``; keep both final scripts in the same directory.
Heavy ML and visualization dependencies are imported lazily so ``--help`` and
``self-test`` remain available in lightweight environments.

Core execution requires pandas, NumPy, sentence-transformers, PyTorch,
BERTopic, umap-learn, hdbscan and scikit-learn. Plotly is only required when
interactive visualizations are enabled.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from supply_chain_workflow import (
    PrivacyGuard,
    WorkflowError,
    atomic_json_dump,
    load_records,
    normalize_text,
    resolve_device,
    source_domain,
)


VERSION = "1.0.0"
LOGGER = logging.getLogger("topic_analysis")


@dataclass
class TopicConfig:
    input_path: str
    output_dir: str
    text_field: str = "auto"
    date_field: str = "auto"
    id_field: str = "auto"
    embedding_model: str = "intfloat/multilingual-e5-large-instruct"
    device: str = "auto"
    batch_size: int = 64
    n_neighbors: int = 15
    n_components: int = 5
    min_dist: float = 0.0
    min_cluster_size: int = 5
    min_samples: int | None = None
    vectorizer_min_df: int = 2
    top_n_words: int = 10
    keyword_count: int = 10
    reduce_topics: str | None = None
    calculate_probabilities: bool = False
    time_bin: str = "W"
    growth_window: int = 6
    dynamic_time_bins: int = 40
    ers: bool = True
    alpha: float = 0.6
    burst_threshold: float = 1.5
    tail_quantile: float = 0.25
    tail_min_size: int = 3
    outlier_neighbors: int = 10
    outlier_percentile: float = 35.0
    representative_count: int = 20
    privacy_mode: str = "strict"
    language: str = "en"
    include_text_preview: bool = False
    visualizations: bool = True
    save_model: bool = False


def validate_config(config: TopicConfig) -> None:
    positive = {
        "batch-size": config.batch_size,
        "n-neighbors": config.n_neighbors,
        "n-components": config.n_components,
        "min-cluster-size": config.min_cluster_size,
        "vectorizer-min-df": config.vectorizer_min_df,
        "top-n-words": config.top_n_words,
        "keyword-count": config.keyword_count,
        "representative-count": config.representative_count,
        "outlier-neighbors": config.outlier_neighbors,
    }
    for name, value in positive.items():
        if value < 1:
            raise WorkflowError(f"--{name} must be at least 1.")
    if config.n_neighbors < 2:
        raise WorkflowError("--n-neighbors must be at least 2.")
    if config.n_components < 2:
        raise WorkflowError("--n-components must be at least 2.")
    if config.min_cluster_size < 2:
        raise WorkflowError("--min-cluster-size must be at least 2.")
    if config.min_samples is not None and config.min_samples < 1:
        raise WorkflowError("--min-samples must be at least 1.")
    if config.growth_window < 3:
        raise WorkflowError("--growth-window must be at least 3.")
    if config.dynamic_time_bins < 2:
        raise WorkflowError("--dynamic-time-bins must be at least 2.")
    if not 0.0 <= config.alpha <= 1.0:
        raise WorkflowError("--alpha must be between 0 and 1.")
    if not 0.0 <= config.tail_quantile <= 1.0:
        raise WorkflowError("--tail-quantile must be between 0 and 1.")
    if config.tail_min_size < 1:
        raise WorkflowError("--tail-min-size must be at least 1.")
    if not 0.0 <= config.outlier_percentile <= 100.0:
        raise WorkflowError("--outlier-percentile must be between 0 and 100.")


TEXT_CANDIDATES = (
    "canonical_text",
    "Summary",
    "summary",
    "Redacted_Content",
    "redacted_content",
)
DATE_CANDIDATES = ("scrape_date", "Scrape_Date", "Scrape Date", "date")
ID_CANDIDATES = ("article_id", "Article_ID")
SOURCE_CANDIDATES = ("source_domain", "Source_Domain")
SUPPLY_CHAIN_CANDIDATES = (
    "primary_supply_chain",
    "supply_chain",
    "commodity",
)
_SAFE_ARTICLE_ID_RE = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")


def resolve_column(
    df: pd.DataFrame,
    requested: str,
    candidates: Sequence[str],
    *,
    required: bool,
    role: str,
) -> str | None:
    if requested != "auto":
        if requested not in df.columns:
            raise WorkflowError(f"Configured {role} column not found: {requested}")
        return requested
    for name in candidates:
        if name in df.columns:
            return name
    if required:
        raise WorkflowError(
            f"No {role} column found. Expected one of: {', '.join(candidates)}"
        )
    return None


def safe_zscore(values: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    result = np.zeros(array.shape, dtype=float)
    finite = np.isfinite(array)
    if not finite.any():
        return result
    fill = float(np.nanmedian(array[finite]))
    clean = np.where(finite, array, fill)
    std = float(clean.std(ddof=0))
    if std <= 1e-12:
        return result
    return (clean - float(clean.mean())) / std


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.where(norms > 0, norms, 1.0)


def prepare_input(
    records: Sequence[Mapping[str, Any]],
    *,
    config: TopicConfig,
    guard: PrivacyGuard,
) -> tuple[pd.DataFrame, str, str | None]:
    df = pd.DataFrame(records)
    if df.empty:
        raise WorkflowError("Topic input dataset is empty.")
    text_col = resolve_column(
        df, config.text_field, TEXT_CANDIDATES, required=True, role="text"
    )
    date_col = resolve_column(
        df, config.date_field, DATE_CANDIDATES, required=False, role="date"
    )
    id_col = resolve_column(
        df, config.id_field, ID_CANDIDATES, required=True, role="article ID"
    )
    source_col = resolve_column(
        df, "auto", SOURCE_CANDIDATES, required=False, role="source domain"
    )
    chain_col = resolve_column(
        df, "auto", SUPPLY_CHAIN_CANDIDATES, required=False, role="supply chain"
    )

    work = pd.DataFrame(
        {
            "article_id": df[id_col].map(normalize_text),
            "document": df[text_col].map(normalize_text),
            "source_domain": (
                df[source_col].map(normalize_text) if source_col else "withheld"
            ),
            "supply_chain": (
                df[chain_col].map(normalize_text) if chain_col else "unspecified"
            ),
        }
    )
    if date_col:
        work["scrape_date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    else:
        work["scrape_date"] = pd.NaT

    work = work[(work["article_id"] != "") & (work["document"] != "")].copy()
    work = work.drop_duplicates(subset=["article_id"], keep="first").reset_index(drop=True)
    if len(work) < max(5, config.min_cluster_size):
        raise WorkflowError(
            f"Only {len(work)} usable documents remain; at least "
            f"{max(5, config.min_cluster_size)} are required."
        )
    unsafe_ids = ~work["article_id"].map(
        lambda value: bool(_SAFE_ARTICLE_ID_RE.fullmatch(value))
        and "://" not in value
        and "@" not in value
    )
    if unsafe_ids.any():
        raise WorkflowError(
            "Input contains privacy-unsafe article IDs. Run the dataset through "
            "supply_chain_workflow.py first so URLs and identifiers are replaced "
            "with pseudonymous IDs."
        )
    work["source_domain"] = work["source_domain"].map(
        lambda value: "withheld"
        if value.lower() == "withheld"
        else (source_domain(value) or "withheld")
    )
    for index, text in enumerate(work["document"]):
        guard.assert_safe(text, context=f"input document {index}")
    for index, value in enumerate(work["source_domain"]):
        guard.assert_safe(value, context=f"source domain {index}")
    return work, text_col, date_col


def encode_documents(config: TopicConfig, documents: Sequence[str]) -> tuple[np.ndarray, Any]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise WorkflowError(
            "sentence-transformers is required for topic analysis."
        ) from exc
    model = SentenceTransformer(
        config.embedding_model, device=resolve_device(config.device)
    )
    embeddings = model.encode(
        list(documents),
        normalize_embeddings=True,
        batch_size=config.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(documents):
        raise WorkflowError("Embedding model returned an unexpected matrix shape.")
    if not np.isfinite(embeddings).all():
        raise WorkflowError("Document embeddings contain non-finite values.")
    return l2_normalize(embeddings), model


def parse_reduce_topics(value: str | None) -> int | str | None:
    if value is None or value.lower() in {"none", "false", "off"}:
        return None
    if value.lower() == "auto":
        return "auto"
    try:
        parsed = int(value)
    except ValueError as exc:
        raise WorkflowError("--reduce-topics must be an integer, auto, or none.") from exc
    if parsed < 2:
        raise WorkflowError("A reduced topic target must be at least 2.")
    return parsed


def fit_topic_model(
    config: TopicConfig,
    documents: Sequence[str],
    embeddings: np.ndarray,
    embedding_model: Any,
) -> tuple[Any, np.ndarray, np.ndarray | None]:
    try:
        from bertopic import BERTopic
        from bertopic.representation import KeyBERTInspired
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except ImportError as exc:
        raise WorkflowError(
            "BERTopic analysis requires bertopic, umap-learn and hdbscan."
        ) from exc

    if config.n_neighbors >= len(documents):
        effective_neighbors = max(2, len(documents) - 1)
    else:
        effective_neighbors = config.n_neighbors
    min_df = min(config.vectorizer_min_df, max(1, len(documents) // 2))
    umap_model = UMAP(
        n_neighbors=effective_neighbors,
        n_components=config.n_components,
        min_dist=config.min_dist,
        metric="cosine",
        random_state=42,
    )
    hdbscan_kwargs: dict[str, Any] = {
        "min_cluster_size": config.min_cluster_size,
        "metric": "euclidean",
        "cluster_selection_method": "eom",
        "prediction_data": True,
    }
    if config.min_samples is not None:
        hdbscan_kwargs["min_samples"] = config.min_samples
    hdbscan_model = HDBSCAN(**hdbscan_kwargs)
    vectorizer_model = CountVectorizer(
        stop_words="english", min_df=min_df, ngram_range=(1, 2)
    )
    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model={"KeyBERT": KeyBERTInspired()},
        top_n_words=config.top_n_words,
        calculate_probabilities=config.calculate_probabilities,
        verbose=True,
    )
    topics, probabilities = model.fit_transform(list(documents), embeddings)
    reduction = parse_reduce_topics(config.reduce_topics)
    if reduction is not None:
        model.reduce_topics(list(documents), nr_topics=reduction)
        topics = model.topics_
        probabilities = model.probabilities_
    topic_array = np.asarray(topics, dtype=int)
    probability_array = None if probabilities is None else np.asarray(probabilities)
    return model, topic_array, probability_array


def probabilities_to_membership(
    probabilities: np.ndarray | None, count: int
) -> np.ndarray:
    if probabilities is None:
        return np.full(count, np.nan, dtype=float)
    values = np.asarray(probabilities)
    if values.ndim == 1:
        result = values.astype(float)
    elif values.ndim == 2:
        result = np.nanmax(values.astype(float), axis=1)
    else:
        raise WorkflowError("Unexpected BERTopic probability array shape.")
    if len(result) != count:
        raise WorkflowError("BERTopic probabilities do not align with documents.")
    return result


def keywords_for_topic(model: Any, topic: int, top_n: int) -> list[str]:
    if int(topic) == -1:
        return []
    values = model.get_topic(int(topic))
    if not values or not isinstance(values, list):
        return []
    return [str(word) for word, _ in values[:top_n]]


def compute_centroid_distances(
    embeddings: np.ndarray, topics: Sequence[int]
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    vectors = l2_normalize(embeddings)
    topic_array = np.asarray(topics, dtype=int)
    distances = np.full(len(vectors), np.nan, dtype=float)
    centroids: dict[int, np.ndarray] = {}
    for topic in sorted(set(topic_array) - {-1}):
        indices = np.flatnonzero(topic_array == topic)
        centroid = l2_normalize(vectors[indices].mean(axis=0, keepdims=True))[0]
        centroids[int(topic)] = centroid
        distances[indices] = 1.0 - np.clip(vectors[indices] @ centroid, -1.0, 1.0)
    return distances, centroids


def rolling_slope(values: Sequence[float], window: int) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    slopes = np.zeros(len(y), dtype=float)
    for index in range(len(y)):
        start = max(0, index - window + 1)
        segment = y[start : index + 1]
        if len(segment) < 3:
            slopes[index] = 0.0
        else:
            x = np.arange(len(segment), dtype=float)
            slopes[index] = float(np.polyfit(x, segment, 1)[0])
    return slopes


def build_topic_timeseries(
    topics: Sequence[int],
    dates: Sequence[Any],
    *,
    frequency: str,
    growth_window: int,
) -> pd.DataFrame:
    topic_array = np.asarray(topics, dtype=int)
    parsed_dates = pd.to_datetime(pd.Series(dates), errors="coerce", utc=True)
    valid = parsed_dates.notna() & (topic_array != -1)
    if not valid.any():
        return pd.DataFrame(
            columns=(
                "Topic",
                "date",
                "count",
                "rate_change",
                "rate_z",
                "growth_slope",
                "growth_slope_z",
            )
        )
    frame = pd.DataFrame(
        {
            "Topic": topic_array[valid.to_numpy()],
            "date": parsed_dates[valid].dt.tz_convert(None),
        }
    )
    try:
        frame["period"] = frame["date"].dt.to_period(frequency)
    except ValueError as exc:
        raise WorkflowError(
            "Unsupported --time-bin. Use D, W, M, Q, or Y."
        ) from exc
    periods = pd.period_range(
        frame["period"].min(), frame["period"].max(), freq=frequency
    )
    rows: list[pd.DataFrame] = []
    for topic, group in frame.groupby("Topic"):
        counts = group.groupby("period").size().reindex(periods, fill_value=0).astype(float)
        rate_change = counts.diff().fillna(0.0).to_numpy()
        slopes = rolling_slope(counts.to_numpy(), growth_window)
        rows.append(
            pd.DataFrame(
                {
                    "Topic": int(topic),
                    "date": periods.to_timestamp(),
                    "count": counts.to_numpy(dtype=int),
                    "rate_change": rate_change,
                    "rate_z": safe_zscore(rate_change),
                    "growth_slope": slopes,
                    "growth_slope_z": safe_zscore(slopes),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def build_dynamic_topics(
    model: Any,
    df: pd.DataFrame,
    topics: Sequence[int],
    *,
    nr_bins: int,
) -> pd.DataFrame:
    """Build BERTopic's time-specific c-TF-IDF representations.

    This output is retained for interpretability and continuity with the prior
    workflow. ERS uses ``build_topic_timeseries`` instead because that function
    explicitly fills empty calendar bins before computing rate and slope.
    """
    valid = df["scrape_date"].notna().to_numpy()
    if valid.sum() < 2 or df.loc[valid, "scrape_date"].nunique() < 2:
        return pd.DataFrame()
    if nr_bins < 2:
        raise WorkflowError("--dynamic-time-bins must be at least 2.")
    unique_dates = int(df.loc[valid, "scrape_date"].nunique())
    bins = min(nr_bins, unique_dates)
    try:
        return model.topics_over_time(
            df.loc[valid, "document"].tolist(),
            df.loc[valid, "scrape_date"].tolist(),
            topics=np.asarray(topics, dtype=int)[valid].tolist(),
            nr_bins=bins,
            global_tuning=True,
            evolution_tuning=True,
        )
    except Exception as exc:
        LOGGER.warning("BERTopic dynamic representations were skipped: %s", exc)
        return pd.DataFrame()


def compute_ers(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    timeseries: pd.DataFrame,
    *,
    alpha: float,
    burst_threshold: float,
    tail_quantile: float,
    tail_min_size: int,
    outlier_neighbors: int,
    outlier_percentile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= alpha <= 1.0:
        raise WorkflowError("ERS alpha must be between 0 and 1.")
    scored = df.copy()
    topics = scored["assigned_topic"].to_numpy(dtype=int)
    distances, _ = compute_centroid_distances(embeddings, topics)
    finite_distances = distances[np.isfinite(distances)]
    distance_fill = float(np.median(finite_distances)) if len(finite_distances) else 0.0
    scored["novelty"] = np.where(np.isfinite(distances), distances, distance_fill)

    sizes = scored.loc[scored["assigned_topic"] != -1, "assigned_topic"].value_counts()
    if sizes.empty or len(sizes) < 4:
        # A lower-tail designation is not meaningful with fewer than four
        # observed topics; avoid marking a single/equal-size topic as a tail.
        size_threshold = -1
    else:
        size_threshold = tail_min_size
        size_threshold = max(
            tail_min_size, int(np.quantile(sizes.to_numpy(), tail_quantile))
        )
    scored["tail_topic"] = scored["assigned_topic"].map(
        lambda topic: bool(topic != -1 and sizes.get(topic, 0) <= size_threshold)
    )

    scored["outlier_microcluster"] = False
    outlier_indices = np.flatnonzero(topics == -1)
    if len(outlier_indices) >= 3:
        from sklearn.neighbors import NearestNeighbors

        neighbour_count = min(outlier_neighbors + 1, len(outlier_indices))
        model = NearestNeighbors(n_neighbors=neighbour_count, metric="cosine")
        outlier_vectors = l2_normalize(embeddings[outlier_indices])
        distances_knn, _ = model.fit(outlier_vectors).kneighbors(outlier_vectors)
        non_self = distances_knn[:, 1:] if distances_knn.shape[1] > 1 else distances_knn
        density = np.median(non_self, axis=1)
        threshold = float(np.percentile(density, outlier_percentile))
        scored.loc[outlier_indices, "outlier_microcluster"] = density <= threshold

    if timeseries.empty:
        latest = pd.DataFrame(
            columns=("Topic", "rate_z", "growth_slope", "growth_slope_z")
        )
    else:
        latest = (
            timeseries.sort_values("date").groupby("Topic", as_index=False).tail(1).copy()
        )
        # Compare latest growth momentum across topics; unlike the original code,
        # this does not z-score a constant value repeated within each topic.
        latest["growth_slope_z"] = safe_zscore(latest["growth_slope"])
    bursting = set(
        latest.loc[latest["rate_z"] >= burst_threshold, "Topic"].astype(int)
    )
    scored["burst_boost"] = scored["assigned_topic"].isin(bursting).astype(int)

    scored["novelty_z"] = safe_zscore(scored["novelty"])
    scored["topic_probability_z"] = safe_zscore(scored["topic_probability"])
    scored["ers_doc"] = (
        scored["novelty_z"]
        + 0.6 * scored["topic_probability_z"]
        + 0.8 * scored["tail_topic"].astype(int)
        + 0.8 * scored["outlier_microcluster"].astype(int)
        + 0.6 * scored["burst_boost"].astype(int)
    )

    if latest.empty:
        topic_score_map: dict[int, float] = {}
    else:
        latest["ers_topic"] = latest["rate_z"].fillna(0.0) + latest[
            "growth_slope_z"
        ].fillna(0.0)
        topic_score_map = dict(
            zip(latest["Topic"].astype(int), latest["ers_topic"].astype(float))
        )
    scored["ers_topic"] = scored["assigned_topic"].map(topic_score_map).fillna(0.0)
    scored["ers_final"] = alpha * scored["ers_doc"] + (1.0 - alpha) * scored[
        "ers_topic"
    ]
    return scored, latest


def build_topic_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rows: list[dict[str, Any]] = []
    for topic, group in df[df["assigned_topic"] != -1].groupby("assigned_topic"):
        keywords = group["topic_keywords"].iloc[0]
        rows.append(
            {
                "Topic": int(topic),
                "Count": int(len(group)),
                "Share_percent": round(len(group) / total * 100.0, 4),
                "Top_Keywords": ", ".join(keywords),
                "Mean_Probability": round(float(group["topic_probability"].mean()), 6),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=("Topic", "Count", "Share_percent", "Top_Keywords", "Mean_Probability")
        )
    return pd.DataFrame(rows).sort_values("Count", ascending=False).reset_index(drop=True)


def build_representatives(
    df: pd.DataFrame,
    *,
    count: int,
    include_text: bool,
) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = {}
    for topic, group in df[df["assigned_topic"] != -1].groupby("assigned_topic"):
        ranked = group.sort_values(
            ["topic_probability", "centroid_distance"],
            ascending=[False, True],
            na_position="last",
        ).head(count)
        docs: list[dict[str, Any]] = []
        for _, row in ranked.iterrows():
            item = {
                "article_id": row["article_id"],
                "source_domain": row["source_domain"],
                "supply_chain": row["supply_chain"],
                "topic_probability": round(float(row["topic_probability"]), 6),
                "centroid_distance": round(float(row["centroid_distance"]), 6),
            }
            if include_text:
                item["text_preview"] = str(row["document"])[:500]
            docs.append(item)
        output[int(topic)] = docs
    return output


def build_public_records(
    df: pd.DataFrame,
    *,
    guard: PrivacyGuard,
    include_text: bool,
    include_ers: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        keywords = [normalize_text(value) for value in row["topic_keywords"]]
        for keyword in keywords:
            guard.assert_safe(keyword, context="topic keyword")
        record: dict[str, Any] = {
            "Article_ID": row["article_id"],
            "Source_Domain": row["source_domain"],
            "Supply_Chain": row["supply_chain"],
            "Scrape_Date": (
                "" if pd.isna(row["scrape_date"]) else row["scrape_date"].isoformat()
            ),
            "Assigned_Topic": int(row["assigned_topic"]),
            "Topic_Probability": round(float(row["topic_probability"]), 6),
            "Topic_Keywords": keywords,
            "Centroid_Distance": (
                None
                if not np.isfinite(row["centroid_distance"])
                else round(float(row["centroid_distance"]), 6)
            ),
            "Privacy_Status": "PASSED" if guard.mode != "off" else "NOT_CHECKED",
        }
        if include_text:
            preview = str(row["document"])[:500]
            guard.assert_safe(preview, context="text preview")
            record["Text_Preview"] = preview
        if include_ers:
            record.update(
                {
                    "Novelty": round(float(row["novelty"]), 6),
                    "TailTopic": bool(row["tail_topic"]),
                    "OutlierMicrocluster": bool(row["outlier_microcluster"]),
                    "BurstBoost": int(row["burst_boost"]),
                    "ERS_doc": round(float(row["ers_doc"]), 6),
                    "ERS_topic": round(float(row["ers_topic"]), 6),
                    "ERS_final": round(float(row["ers_final"]), 6),
                }
            )
        records.append(record)
    return records


def generate_visualizations(
    model: Any,
    topic_summary: pd.DataFrame,
    timeseries: pd.DataFrame,
    dynamic_topics: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    outputs: list[str] = []

    def save(name: str, factory: Any) -> None:
        try:
            figure = factory()
            path = output_dir / name
            figure.write_html(path, include_plotlyjs=True)
            outputs.append(str(path))
        except Exception as exc:
            LOGGER.warning("Skipped %s: %s", name, exc)

    save("intertopic_distance.html", model.visualize_topics)
    save("topic_barchart.html", lambda: model.visualize_barchart(top_n_topics=20))
    save("topic_hierarchy.html", lambda: model.visualize_hierarchy(top_n_topics=20))
    try:
        import plotly.express as px
    except ImportError:
        LOGGER.warning("Plotly unavailable; summary charts were skipped.")
        return outputs
    if not topic_summary.empty:
        save(
            "topic_distribution_pie.html",
            lambda: px.pie(
                topic_summary,
                names="Topic",
                values="Count",
                hover_data=["Top_Keywords", "Share_percent", "Mean_Probability"],
                title="Topic distribution",
            ),
        )
        save(
            "total_topic_distribution_bar.html",
            lambda: px.bar(
                topic_summary,
                x="Topic",
                y="Count",
                hover_data=["Top_Keywords", "Share_percent", "Mean_Probability"],
                title="Topic distribution",
            ),
        )
    if not timeseries.empty:
        save(
            "topic_trends.html",
            lambda: px.line(
                timeseries,
                x="date",
                y="count",
                color=timeseries["Topic"].astype(str),
                title="Topic frequency over time",
                labels={"color": "Topic"},
            ),
        )
    if not dynamic_topics.empty:
        save(
            "topics_over_time.html",
            lambda: model.visualize_topics_over_time(
                dynamic_topics, top_n_topics=15
            ),
        )
    return outputs


def run_analysis(config: TopicConfig) -> dict[str, Any]:
    validate_config(config)
    started_at = datetime.now(timezone.utc)
    started_clock = time.time()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    guard = PrivacyGuard(config.privacy_mode, config.language)
    records = load_records(config.input_path)
    work, resolved_text_field, resolved_date_field = prepare_input(
        records, config=config, guard=guard
    )
    embeddings, embedding_model = encode_documents(config, work["document"].tolist())
    topic_model, topics, probabilities = fit_topic_model(
        config, work["document"].tolist(), embeddings, embedding_model
    )
    work["assigned_topic"] = topics
    work["topic_probability"] = probabilities_to_membership(probabilities, len(work))
    work["topic_probability"] = work["topic_probability"].fillna(
        work["topic_probability"].median()
    ).fillna(0.0)
    work["topic_keywords"] = [
        keywords_for_topic(topic_model, topic, config.keyword_count) for topic in topics
    ]
    centroid_distances, _ = compute_centroid_distances(embeddings, topics)
    work["centroid_distance"] = centroid_distances

    timeseries = build_topic_timeseries(
        topics,
        work["scrape_date"],
        frequency=config.time_bin,
        growth_window=config.growth_window,
    )
    dynamic_topics = build_dynamic_topics(
        topic_model,
        work,
        topics,
        nr_bins=config.dynamic_time_bins,
    )
    latest_topic_scores = pd.DataFrame()
    if config.ers:
        work, latest_topic_scores = compute_ers(
            work,
            embeddings,
            timeseries,
            alpha=config.alpha,
            burst_threshold=config.burst_threshold,
            tail_quantile=config.tail_quantile,
            tail_min_size=config.tail_min_size,
            outlier_neighbors=config.outlier_neighbors,
            outlier_percentile=config.outlier_percentile,
        )

    topic_summary = build_topic_summary(work)
    representatives = build_representatives(
        work,
        count=config.representative_count,
        include_text=config.include_text_preview,
    )
    public_records = build_public_records(
        work,
        guard=guard,
        include_text=config.include_text_preview,
        include_ers=config.ers,
    )
    atomic_json_dump(public_records, output_dir / "topic_analysis_records.json")
    atomic_json_dump(representatives, output_dir / "representative_documents.json")
    topic_summary.to_csv(output_dir / "topic_summary.csv", index=False)
    if not timeseries.empty:
        timeseries.to_csv(output_dir / "topic_timeseries.csv", index=False)
    if not dynamic_topics.empty:
        dynamic_topics.to_csv(
            output_dir / "bertopic_topics_over_time.csv", index=False
        )
    if config.ers:
        latest_topic_scores.to_csv(output_dir / "ers_latest_topic_scores.csv", index=False)

    clustered = int((work["assigned_topic"] != -1).sum())
    outliers = int(len(work) - clustered)
    topic_count = int(work.loc[work["assigned_topic"] != -1, "assigned_topic"].nunique())
    topic_sizes = work.loc[
        work["assigned_topic"] != -1, "assigned_topic"
    ].value_counts()
    analytics = {
        "total_records": len(work),
        "topic_count_excluding_outliers": topic_count,
        "clustered_records": clustered,
        "clustered_percent": round(clustered / len(work) * 100.0, 4),
        "outlier_records": outliers,
        "outlier_percent": round(outliers / len(work) * 100.0, 4),
        "average_topic_size": (
            round(float(topic_sizes.mean()), 4) if not topic_sizes.empty else 0.0
        ),
        "median_topic_size": (
            round(float(topic_sizes.median()), 4) if not topic_sizes.empty else 0.0
        ),
        "minimum_topic_size": int(topic_sizes.min()) if not topic_sizes.empty else 0,
        "maximum_topic_size": int(topic_sizes.max()) if not topic_sizes.empty else 0,
        "dated_records": int(work["scrape_date"].notna().sum()),
        "ers_enabled": config.ers,
        "ers_dashboard_status": "not_integrated",
    }
    atomic_json_dump(analytics, output_dir / "topic_analytics_summary.json")

    visualization_outputs: list[str] = []
    if config.visualizations:
        visualization_outputs = generate_visualizations(
            topic_model, topic_summary, timeseries, dynamic_topics, output_dir
        )
    if config.save_model:
        model_dir = output_dir / "bertopic_model"
        try:
            topic_model.save(
                model_dir,
                serialization="safetensors",
                save_ctfidf=True,
                save_embedding_model=config.embedding_model,
            )
        except TypeError:
            topic_model.save(model_dir)

    manifest = {
        "workflow": "HOLiFOOD BERTopic and Emerging Risk Score",
        "version": VERSION,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started_clock, 3),
        "input_path": str(Path(config.input_path).resolve()),
        "resolved_text_field": resolved_text_field,
        "resolved_date_field": resolved_date_field,
        "config": asdict(config),
        "analytics": analytics,
        "outputs": {
            "records": str(output_dir / "topic_analysis_records.json"),
            "topic_summary": str(output_dir / "topic_summary.csv"),
            "representatives": str(output_dir / "representative_documents.json"),
            "topic_timeseries": (
                str(output_dir / "topic_timeseries.csv") if not timeseries.empty else None
            ),
            "bertopic_topics_over_time": (
                str(output_dir / "bertopic_topics_over_time.csv")
                if not dynamic_topics.empty
                else None
            ),
            "ers_latest_topic_scores": (
                str(output_dir / "ers_latest_topic_scores.csv") if config.ers else None
            ),
            "visualizations": visualization_outputs,
        },
        "interpretation_note": (
            "ERS ranks emergence potential for expert review; it is not an "
            "automated risk or severity decision."
        ),
    }
    atomic_json_dump(manifest, output_dir / "topic_analysis_manifest.json")
    return manifest


def self_test() -> None:
    assert np.allclose(safe_zscore([2.0, 2.0, 2.0]), [0.0, 0.0, 0.0])
    dates = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-08",
            "2026-01-15",
            "2026-01-22",
            "2026-01-22",
            "2026-01-29",
            "2026-01-29",
            "2026-01-29",
        ],
        utc=True,
    )
    topics = np.array([0, 0, 0, 0, 0, 0, 0, -1])
    ts = build_topic_timeseries(topics, dates, frequency="W", growth_window=3)
    assert not ts.empty
    assert ts["growth_slope"].nunique() > 1
    assert np.isfinite(ts[["rate_z", "growth_slope_z"]].to_numpy()).all()

    rng = np.random.default_rng(42)
    embeddings = l2_normalize(rng.normal(size=(8, 6)).astype(np.float32))
    df = pd.DataFrame(
        {
            "article_id": [f"eri_{i}" for i in range(8)],
            "document": [f"privacy safe document {i}" for i in range(8)],
            "source_domain": ["withheld"] * 8,
            "supply_chain": ["poultry"] * 8,
            "scrape_date": dates,
            "assigned_topic": topics,
            "topic_probability": np.linspace(0.4, 0.9, 8),
            "topic_keywords": [["outbreak", "poultry"]] * 7 + [[]],
            "centroid_distance": [0.0] * 8,
        }
    )

    class FakeTopicModel:
        def topics_over_time(self, docs: Any, timestamps: Any, **kwargs: Any) -> pd.DataFrame:
            assert len(docs) == len(timestamps) == len(kwargs["topics"])
            assert kwargs["nr_bins"] >= 2
            return pd.DataFrame(
                {"Topic": [0], "Words": ["poultry"], "Frequency": [7], "Timestamp": [timestamps[0]]}
            )

    dynamic = build_dynamic_topics(
        FakeTopicModel(), df, topics, nr_bins=40
    )
    assert dynamic.loc[0, "Frequency"] == 7

    scored, latest = compute_ers(
        df,
        embeddings,
        ts,
        alpha=0.6,
        burst_threshold=1.5,
        tail_quantile=0.25,
        tail_min_size=3,
        outlier_neighbors=3,
        outlier_percentile=35,
    )
    assert np.isfinite(scored["ers_final"]).all()
    assert not bool(scored.loc[scored["assigned_topic"] == -1, "tail_topic"].iloc[0])
    assert not scored["tail_topic"].any()
    assert "ers_topic" in latest.columns

    public = build_public_records(
        scored,
        guard=PrivacyGuard("regex"),
        include_text=False,
        include_ers=True,
    )
    assert "document" not in public[0]
    assert "Text_Preview" not in public[0]
    assert public[0]["Privacy_Status"] == "PASSED"
    print("Topic analysis self-test: PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run BERTopic and optional ERS analysis.")
    run.add_argument("--input", required=True, dest="input_path")
    run.add_argument("--output-dir", required=True)
    run.add_argument("--text-field", default="auto")
    run.add_argument("--date-field", default="auto")
    run.add_argument("--id-field", default="auto")
    run.add_argument("--embedding-model", default="intfloat/multilingual-e5-large-instruct")
    run.add_argument("--device", default="auto")
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--n-neighbors", type=int, default=15)
    run.add_argument("--n-components", type=int, default=5)
    run.add_argument("--min-dist", type=float, default=0.0)
    run.add_argument("--min-cluster-size", type=int, default=5)
    run.add_argument("--min-samples", type=int)
    run.add_argument("--vectorizer-min-df", type=int, default=2)
    run.add_argument("--top-n-words", type=int, default=10)
    run.add_argument("--keyword-count", type=int, default=10)
    run.add_argument("--reduce-topics", default="none")
    run.add_argument("--calculate-probabilities", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--time-bin", default="W", choices=("D", "W", "M", "Q", "Y"))
    run.add_argument("--growth-window", type=int, default=6)
    run.add_argument("--dynamic-time-bins", type=int, default=40)
    run.add_argument("--ers", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--alpha", type=float, default=0.6)
    run.add_argument("--burst-threshold", type=float, default=1.5)
    run.add_argument("--tail-quantile", type=float, default=0.25)
    run.add_argument("--tail-min-size", type=int, default=3)
    run.add_argument("--outlier-neighbors", type=int, default=10)
    run.add_argument("--outlier-percentile", type=float, default=35.0)
    run.add_argument("--representative-count", type=int, default=20)
    run.add_argument("--privacy-mode", default="strict", choices=("strict", "regex", "off"))
    run.add_argument("--language", default="en")
    run.add_argument("--include-text-preview", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--visualizations", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=False)
    sub.add_parser("self-test", help="Run deterministic ERS and temporal tests.")
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
        else:
            config = TopicConfig(
                input_path=args.input_path,
                output_dir=args.output_dir,
                text_field=args.text_field,
                date_field=args.date_field,
                id_field=args.id_field,
                embedding_model=args.embedding_model,
                device=args.device,
                batch_size=args.batch_size,
                n_neighbors=args.n_neighbors,
                n_components=args.n_components,
                min_dist=args.min_dist,
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
                vectorizer_min_df=args.vectorizer_min_df,
                top_n_words=args.top_n_words,
                keyword_count=args.keyword_count,
                reduce_topics=args.reduce_topics,
                calculate_probabilities=args.calculate_probabilities,
                time_bin=args.time_bin,
                growth_window=args.growth_window,
                dynamic_time_bins=args.dynamic_time_bins,
                ers=args.ers,
                alpha=args.alpha,
                burst_threshold=args.burst_threshold,
                tail_quantile=args.tail_quantile,
                tail_min_size=args.tail_min_size,
                outlier_neighbors=args.outlier_neighbors,
                outlier_percentile=args.outlier_percentile,
                representative_count=args.representative_count,
                privacy_mode=args.privacy_mode,
                language=args.language,
                include_text_preview=args.include_text_preview,
                visualizations=args.visualizations,
                save_model=args.save_model,
            )
            manifest = run_analysis(config)
            print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    except (WorkflowError, ValueError, OSError, json.JSONDecodeError) as exc:
        LOGGER.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

LOGGER_NAME = __name__
DEFAULT_RETRIEVAL_CONFIG_PATH = Path("configs/retrieval.yaml")
ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::([^}]*))?\}")


@dataclass(slots=True, frozen=True)
class RetrievalConfig:
    """Configuration shared by the TV3 retrieval subgraph."""

    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    query_bonus_weight: float = 0.1
    top_k: int = 8
    bm25_top_k: int = 12
    vector_top_k: int = 12
    rerank_top_n: int = 4
    retrieval_score_threshold: float = 0.35
    max_retry_loops: int = 3
    article_only: bool = True
    article_first: bool = True
    allow_chunk_fallback: bool = False
    allow_article_backoff: bool = True
    enable_structured_legal_query_parse: bool = True
    enable_exact_legal_match_boost: bool = True
    article_match_bonus: float = 0.65
    article_code_match_bonus: float = 1.15
    title_match_bonus: float = 0.35
    law_id_match_bonus: float = 0.45
    article_name_match_bonus: float = 0.3
    structured_match_bonus: float = 0.75
    local_corpus_path: str = "data/processed/all_chunks.jsonl"
    indexing_config_path: str = "configs/indexing.yaml"
    cross_encoder_model: str = "BAAI/bge-reranker-v2-m3"
    cross_encoder_device: str = "cpu"
    min_valid_sources: int = 2
    min_unique_sources: int = 1
    min_top_rerank_score: float = 0.15
    require_structured_citation: bool = True
    context_max_docs: int = 4
    context_max_chars_per_doc: int = 1400
    max_queries_per_retrieve: int = 4
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    fast_path_enabled: bool = True
    article_first_fast: bool = True
    top_k_fast: int = 3
    bm25_top_k_fast: int = 3
    vector_top_k_fast: int = 3
    rerank_top_n_fast: int = 2
    enable_rerank_fast: bool = False
    max_fast_retries: int = 0
    fast_grounding_single_pass: bool = True
    fast_sources_limit: int = 2
    max_queries_fast: int = 2
    min_valid_sources_fast: int = 1
    min_unique_sources_fast: int = 1
    min_top_rerank_score_fast: float = 0.1
    context_max_docs_fast: int = 2
    context_max_chars_per_doc_fast: int = 900


def _load_yaml_module() -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load configs/retrieval.yaml.") from exc
    return yaml


def _substitute_env_placeholders(raw_text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(key, default)

    return ENV_PATTERN.sub(replace, raw_text)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def load_retrieval_config(config_path: str | Path | None = None) -> RetrievalConfig:
    """Load retrieval configuration from YAML with safe defaults."""

    resolved_path = Path(config_path or DEFAULT_RETRIEVAL_CONFIG_PATH).resolve()
    if not resolved_path.exists() or resolved_path.stat().st_size == 0:
        return RetrievalConfig()

    yaml = _load_yaml_module()
    raw_text = resolved_path.read_text(encoding="utf-8")
    config_data = yaml.safe_load(_substitute_env_placeholders(raw_text)) or {}
    if not isinstance(config_data, Mapping):
        raise ValueError(f"Invalid retrieval config structure in {resolved_path}")

    return RetrievalConfig(
        bm25_weight=float(config_data.get("bm25_weight") or 0.4),
        vector_weight=float(config_data.get("vector_weight") or 0.6),
        query_bonus_weight=float(config_data.get("query_bonus_weight") or 0.1),
        top_k=int(config_data.get("top_k") or 8),
        bm25_top_k=int(config_data.get("bm25_top_k") or 12),
        vector_top_k=int(config_data.get("vector_top_k") or 12),
        rerank_top_n=int(config_data.get("rerank_top_n") or 4),
        retrieval_score_threshold=float(config_data.get("retrieval_score_threshold") or 0.35),
        max_retry_loops=int(config_data.get("max_retry_loops") or 3),
        article_only=_coerce_bool(config_data.get("article_only"), True),
        article_first=_coerce_bool(config_data.get("article_first"), True),
        allow_chunk_fallback=_coerce_bool(config_data.get("allow_chunk_fallback"), False),
        allow_article_backoff=_coerce_bool(config_data.get("allow_article_backoff"), True),
        enable_structured_legal_query_parse=_coerce_bool(
            config_data.get("enable_structured_legal_query_parse"),
            True,
        ),
        enable_exact_legal_match_boost=_coerce_bool(config_data.get("enable_exact_legal_match_boost"), True),
        article_match_bonus=float(config_data.get("article_match_bonus") or 0.65),
        article_code_match_bonus=float(config_data.get("article_code_match_bonus") or 1.15),
        title_match_bonus=float(config_data.get("title_match_bonus") or 0.35),
        law_id_match_bonus=float(config_data.get("law_id_match_bonus") or 0.45),
        article_name_match_bonus=float(config_data.get("article_name_match_bonus") or 0.3),
        structured_match_bonus=float(config_data.get("structured_match_bonus") or 0.75),
        local_corpus_path=str(config_data.get("local_corpus_path") or "data/processed/all_chunks.jsonl"),
        indexing_config_path=str(config_data.get("indexing_config_path") or "configs/indexing.yaml"),
        cross_encoder_model=str(config_data.get("cross_encoder_model") or "BAAI/bge-reranker-v2-m3"),
        cross_encoder_device=str(config_data.get("cross_encoder_device") or "cpu").strip().lower() or "cpu",
        min_valid_sources=int(config_data.get("min_valid_sources") or 2),
        min_unique_sources=int(config_data.get("min_unique_sources") or 1),
        min_top_rerank_score=float(config_data.get("min_top_rerank_score") or 0.15),
        require_structured_citation=_coerce_bool(config_data.get("require_structured_citation"), True),
        context_max_docs=int(config_data.get("context_max_docs") or 4),
        context_max_chars_per_doc=int(config_data.get("context_max_chars_per_doc") or 1400),
        max_queries_per_retrieve=int(config_data.get("max_queries_per_retrieve") or 4),
        bm25_k1=float(config_data.get("bm25_k1") or 1.5),
        bm25_b=float(config_data.get("bm25_b") or 0.75),
        fast_path_enabled=_coerce_bool(config_data.get("fast_path_enabled"), True),
        article_first_fast=_coerce_bool(config_data.get("article_first_fast"), True),
        top_k_fast=int(config_data.get("top_k_fast") or 3),
        bm25_top_k_fast=int(config_data.get("bm25_top_k_fast") or 3),
        vector_top_k_fast=int(config_data.get("vector_top_k_fast") or 3),
        rerank_top_n_fast=int(config_data.get("rerank_top_n_fast") or 2),
        enable_rerank_fast=_coerce_bool(config_data.get("enable_rerank_fast"), False),
        max_fast_retries=int(config_data.get("max_fast_retries") or 0),
        fast_grounding_single_pass=_coerce_bool(config_data.get("fast_grounding_single_pass"), True),
        fast_sources_limit=int(config_data.get("fast_sources_limit") or 2),
        max_queries_fast=int(config_data.get("max_queries_fast") or 2),
        min_valid_sources_fast=int(config_data.get("min_valid_sources_fast") or 1),
        min_unique_sources_fast=int(config_data.get("min_unique_sources_fast") or 1),
        min_top_rerank_score_fast=float(config_data.get("min_top_rerank_score_fast") or 0.1),
        context_max_docs_fast=int(config_data.get("context_max_docs_fast") or 2),
        context_max_chars_per_doc_fast=int(config_data.get("context_max_chars_per_doc_fast") or 900),
    )


def resolve_execution_profile(state: Mapping[str, Any]) -> str:
    profile = str(state.get("execution_profile") or "").strip().lower()
    return "fast" if profile == "fast" else "full"


def default_retrieval_level(config: RetrievalConfig, *, execution_profile: str = "full") -> str:
    """Return the preferred retrieval level for the first pass."""

    if config.article_only:
        return "article"
    if execution_profile == "fast":
        return "article" if config.article_first_fast else "chunk"
    return "article" if config.article_first else "chunk"


def get_retrieval_limits(config: RetrievalConfig, execution_profile: str) -> dict[str, int | bool]:
    if execution_profile == "fast":
        return {
            "top_k": max(1, config.top_k_fast),
            "bm25_top_k": max(1, config.bm25_top_k_fast),
            "vector_top_k": max(1, config.vector_top_k_fast),
            "rerank_top_n": max(1, config.rerank_top_n_fast),
            "max_queries_per_retrieve": max(1, config.max_queries_fast),
            "context_max_docs": max(1, config.context_max_docs_fast),
            "context_max_chars_per_doc": max(200, config.context_max_chars_per_doc_fast),
            "sources_limit": max(1, config.fast_sources_limit),
            "enable_rerank": bool(config.enable_rerank_fast),
        }
    return {
        "top_k": max(1, config.top_k),
        "bm25_top_k": max(1, config.bm25_top_k),
        "vector_top_k": max(1, config.vector_top_k),
        "rerank_top_n": max(1, config.rerank_top_n),
        "max_queries_per_retrieve": max(1, config.max_queries_per_retrieve),
        "context_max_docs": max(1, config.context_max_docs),
        "context_max_chars_per_doc": max(200, config.context_max_chars_per_doc),
        "sources_limit": max(1, config.context_max_docs),
        "enable_rerank": True,
    }


def _safe_filters(filters: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in dict(filters or {}).items()
        if value not in (None, "", [], {})
    }


def decide_next_retrieval_step(state: Mapping[str, Any], config: RetrievalConfig) -> dict[str, Any]:
    """Decide whether retrieval should retry, relax filters, or fall back safely."""

    execution_profile = resolve_execution_profile(state)
    retrieval_debug = state.get("retrieval_debug") or {}
    retrieval_debug = dict(retrieval_debug) if isinstance(retrieval_debug, Mapping) else {}
    current_plan = dict(retrieval_debug.get("current_plan") or {})
    failure_reason = str(
        state.get("retrieval_failure_reason")
        or retrieval_debug.get("last_failure_reason")
        or retrieval_debug.get("evidence", {}).get("failure_reason")
        or ""
    ).strip()
    loop_count = int(state.get("loop_count") or 0)
    current_level = str(current_plan.get("level") or default_retrieval_level(config, execution_profile=execution_profile))
    filters = _safe_filters(current_plan.get("filters") or retrieval_debug.get("metadata_filters") or {})
    structured_query_used = bool(filters) or bool(retrieval_debug.get("legal_query_features", {}).get("is_structured_legal_query"))

    if execution_profile == "fast":
        return {
            "next_action": "escalate_to_full",
            "strategy": "escalate_fast_to_full",
            "retry_plan": {
                "level": "article",
                "top_k": max(config.top_k, config.top_k_fast),
                "filters": filters,
                "selected_query": str(current_plan.get("selected_query") or ""),
                "execution_profile": "full",
            },
        }

    if loop_count >= config.max_retry_loops:
        return {
            "next_action": "fallback",
            "strategy": "max_retry_exhausted",
            "retry_plan": None,
        }

    if structured_query_used and filters and failure_reason in {"no_results", "missing_structured_sources", "insufficient_results"}:
        return {
            "next_action": "retry",
            "strategy": "structured_query_fallback",
            "retry_plan": {
                "level": "article",
                "top_k": max(config.top_k, int(current_plan.get("top_k") or config.top_k)),
                "filters": {},
                "selected_query": str(current_plan.get("selected_query") or ""),
            },
        }

    if failure_reason in {"weak_evidence", "insufficient_results", "low_source_diversity", "no_exact_legal_hit"}:
        relaxed_filters = {} if config.allow_article_backoff else filters
        return {
            "next_action": "retry",
            "strategy": "broaden_article_search",
            "retry_plan": {
                "level": "article",
                "top_k": max(config.top_k + 2, config.rerank_top_n * 3),
                "filters": relaxed_filters,
                "selected_query": str(current_plan.get("selected_query") or ""),
            },
        }

    if not config.article_only and failure_reason in {"no_results", "missing_structured_sources"} and current_level == "article" and config.allow_chunk_fallback:
        return {
            "next_action": "retry",
            "strategy": "chunk_fallback",
            "retry_plan": {
                "level": "chunk",
                "top_k": max(config.top_k, config.top_k + 2),
                "filters": filters,
                "selected_query": str(current_plan.get("selected_query") or ""),
            },
        }

    return {
        "next_action": "fallback",
        "strategy": "fallback_safe_answer",
        "retry_plan": None,
    }


__all__ = [
    "DEFAULT_RETRIEVAL_CONFIG_PATH",
    "RetrievalConfig",
    "decide_next_retrieval_step",
    "default_retrieval_level",
    "get_retrieval_limits",
    "load_retrieval_config",
    "resolve_execution_profile",
]

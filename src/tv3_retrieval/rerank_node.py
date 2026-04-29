from __future__ import annotations

import logging
import os
from functools import lru_cache
from functools import wraps
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.tv3_retrieval.fallback_policy import (
    RetrievalConfig,
    get_retrieval_limits,
    load_retrieval_config,
    resolve_execution_profile,
)

LOGGER = logging.getLogger(__name__)

try:
    from langsmith import traceable as _langsmith_traceable
except Exception:  # pragma: no cover - optional dependency.
    _langsmith_traceable = None


def _langsmith_tracing_enabled() -> bool:
    tracing_flag = str(os.getenv("LANGSMITH_TRACING") or "").strip().lower() in {"1", "true", "yes", "on"}
    api_key = str(os.getenv("LANGSMITH_API_KEY") or "").strip()
    return tracing_flag and bool(api_key)


def _optional_traceable(*, name: str, run_type: str = "chain", tags: list[str] | None = None):
    def decorator(func):
        if _langsmith_traceable is None:
            @wraps(func)
            def no_trace(*args, **kwargs):
                kwargs.pop("langsmith_extra", None)
                return func(*args, **kwargs)

            return no_trace

        traced_func = _langsmith_traceable(name=name, run_type=run_type, tags=list(tags or []))(func)

        @wraps(func)
        def wrapped(*args, **kwargs):
            if not _langsmith_tracing_enabled():
                kwargs.pop("langsmith_extra", None)
                return func(*args, **kwargs)
            return traced_func(*args, **kwargs)

        return wrapped

    return decorator


def format_source(metadata: Mapping[str, Any]) -> str:
    """Build a citation-ready source string from legal metadata."""

    parts: list[str] = []
    article_code = str(metadata.get("article_code") or "").strip()
    article_name = str(metadata.get("article_name") or "").strip()
    law_id = str(metadata.get("law_id") or "").strip()
    title = str(metadata.get("title") or "").strip()
    issuer = str(metadata.get("issuer") or "").strip()
    effective_date = str(metadata.get("effective_date") or "").strip()

    if article_code:
        parts.append(article_code)
    if article_name:
        parts.append(article_name)
    if law_id:
        parts.append(law_id)
    elif title:
        parts.append(title)

    if issuer or effective_date:
        tail = ", ".join([item for item in (issuer, effective_date) if item])
        if tail:
            parts.append(tail)
    return " - ".join([part for part in parts if part])


@lru_cache(maxsize=8)
def _get_cached_cross_encoder_backend(model_name: str, device: str) -> Any:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("sentence-transformers is required for CrossEncoder reranking.") from exc
    resolved_device = (device or "cpu").strip().lower() or "cpu"
    return CrossEncoder(model_name, device=resolved_device)


def _load_cross_encoder_backend(model_name: str, device: str, logger: logging.Logger | None = None) -> Any:
    resolved_logger = logger or LOGGER
    resolved_device = (device or "cpu").strip().lower() or "cpu"
    cache_info_before = _get_cached_cross_encoder_backend.cache_info()
    backend = _get_cached_cross_encoder_backend(model_name, resolved_device)
    cache_info_after = _get_cached_cross_encoder_backend.cache_info()
    if cache_info_after.misses > cache_info_before.misses:
        resolved_logger.info("Loaded CrossEncoder reranker model=%s device=%s", model_name, resolved_device)
    else:
        resolved_logger.debug("Reusing cached CrossEncoder reranker model=%s device=%s", model_name, resolved_device)
    return backend


def _score_with_backend(backend: Any, query_text: str, docs: Sequence[Mapping[str, Any]]) -> list[float]:
    pairs = [(query_text, str(doc.get("content") or "")) for doc in docs]
    if hasattr(backend, "predict"):
        scores = backend.predict(pairs)
    elif callable(backend):
        scores = backend(pairs)
    else:
        raise TypeError("Unsupported reranker backend. Expected `.predict()` or a callable.")

    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    return [float(score) for score in scores]


def build_context(
    docs: Sequence[Mapping[str, Any]],
    *,
    max_docs: int,
    max_chars_per_doc: int,
) -> str:
    """Create a stable context block for TV5 reasoning nodes."""

    chunks: list[str] = []
    for index, doc in enumerate(docs[:max_docs], start=1):
        content = str(doc.get("content") or "").strip()
        clipped = content[:max_chars_per_doc].strip()
        chunks.append(f"[{index}] {clipped}")
    return "\n\n".join(chunks)


def _combined_score_fallback(docs: Sequence[Mapping[str, Any]], *, top_n: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reranked_docs: list[dict[str, Any]] = []
    for doc in docs:
        normalized = dict(doc)
        normalized["rerank_score"] = float(doc.get("combined_score") or 0.0)
        normalized["source"] = format_source(normalized.get("metadata") or {})
        reranked_docs.append(normalized)
    reranked_docs.sort(
        key=lambda item: (float(item.get("combined_score") or 0.0), float(item.get("vector_score") or 0.0)),
        reverse=True,
    )
    return reranked_docs[:top_n], {"reranker_used": "combined_score_fallback"}


def rerank_candidates(
    query_text: str,
    candidate_docs: Sequence[Mapping[str, Any]],
    *,
    config: RetrievalConfig,
    execution_profile: str,
    reranker_backend: Any | None = None,
    logger: logging.Logger | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rerank hybrid candidates using CrossEncoder when appropriate."""

    resolved_logger = logger or LOGGER
    docs = [dict(doc) for doc in candidate_docs]
    if not docs:
        return [], {"reranker_used": "none"}

    limits = get_retrieval_limits(config, execution_profile)
    top_n = int(limits["rerank_top_n"])
    if execution_profile == "fast" and not bool(limits["enable_rerank"]):
        return _combined_score_fallback(docs, top_n=top_n)

    docs_for_scoring = docs[: max(top_n, min(len(docs), top_n + 1))]
    backend = reranker_backend
    reranker_used = "combined_score_fallback"
    scores: list[float]
    if backend is None:
        try:
            backend = _load_cross_encoder_backend(
                config.cross_encoder_model,
                config.cross_encoder_device,
                logger=resolved_logger,
            )
        except Exception as exc:  # pragma: no cover - optional dependency/runtime model.
            resolved_logger.warning("CrossEncoder unavailable, falling back to combined_score sorting: %s", exc)
            backend = None

    if backend is not None:
        try:
            scores = _score_with_backend(backend, query_text, docs_for_scoring)
            reranker_used = f"{config.cross_encoder_model}@{config.cross_encoder_device}"
        except Exception as exc:  # pragma: no cover - optional dependency/runtime model.
            resolved_logger.warning("CrossEncoder rerank failed, using combined_score fallback: %s", exc)
            return _combined_score_fallback(docs_for_scoring, top_n=top_n)
    else:
        return _combined_score_fallback(docs_for_scoring, top_n=top_n)

    reranked_docs: list[dict[str, Any]] = []
    for doc, score in zip(docs_for_scoring, scores):
        doc["rerank_score"] = float(score)
        doc["source"] = format_source(doc.get("metadata") or {})
        reranked_docs.append(doc)

    reranked_docs.sort(
        key=lambda item: (float(item.get("rerank_score") or 0.0), float(item.get("combined_score") or 0.0)),
        reverse=True,
    )
    return reranked_docs[:top_n], {"reranker_used": reranker_used}


@_optional_traceable(name="tv3.rerank_node", run_type="chain", tags=["tv3", "retrieval"])
def rerank_node(
    state: Mapping[str, Any],
    *,
    retrieval_config: RetrievalConfig | None = None,
    config_path: str | Path | None = None,
    reranker_backend: Any | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """LangGraph-friendly rerank node for the TV3 retrieval subgraph."""

    resolved_retrieval_config = retrieval_config or load_retrieval_config(config_path)
    execution_profile = resolve_execution_profile(state)
    limits = get_retrieval_limits(resolved_retrieval_config, execution_profile)
    query_text = str(state.get("normalized_question") or state.get("question") or "").strip()
    candidate_docs = state.get("retrieved_docs") or []
    reranked_docs, debug_info = rerank_candidates(
        query_text,
        candidate_docs,
        config=resolved_retrieval_config,
        execution_profile=execution_profile,
        reranker_backend=reranker_backend,
        logger=logger,
    )

    context = build_context(
        reranked_docs,
        max_docs=int(limits["context_max_docs"]),
        max_chars_per_doc=int(limits["context_max_chars_per_doc"]),
    )
    sources = [str(doc.get("source") or format_source(doc.get("metadata") or {})) for doc in reranked_docs]
    sources = sources[: int(limits["sources_limit"])]

    previous_debug = state.get("retrieval_debug") or {}
    previous_debug = dict(previous_debug) if isinstance(previous_debug, Mapping) else {}
    previous_debug.update(debug_info)
    previous_debug["reranked_count"] = len(reranked_docs)
    previous_debug["execution_profile"] = execution_profile

    return {
        "reranked_docs": reranked_docs,
        "context": context,
        "sources": sources,
        "retrieval_debug": previous_debug,
    }


__all__ = [
    "build_context",
    "format_source",
    "rerank_candidates",
    "rerank_node",
]

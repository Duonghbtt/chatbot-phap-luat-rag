from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.tv2_index.build_qdrant_index import load_tv1_records, prepare_article_documents, prepare_chunk_documents
from src.tv2_index.search_with_filters import QdrantSearchService, result_matches_filters
from src.tv3_retrieval.fallback_policy import (
    RetrievalConfig,
    default_retrieval_level,
    get_retrieval_limits,
    load_retrieval_config,
    resolve_execution_profile,
)

LOGGER = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"[0-9A-Za-zÀ-ỹ]+")
_BM25_CACHE: dict[tuple[str, str, float, float], "LocalBM25Retriever"] = {}
_CORPUS_CACHE: dict[tuple[str, str], list[dict[str, Any]]] = {}


def normalize_legal_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def tokenize_legal_text(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(normalize_legal_text(text)) if token]


def _normalize_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _hash_content(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def _dedup_key(document: Mapping[str, Any]) -> str:
    metadata = document.get("metadata") or {}
    if isinstance(metadata, Mapping):
        if metadata.get("law_id") or metadata.get("article_code"):
            return "|".join(
                [
                    str(metadata.get("law_id") or ""),
                    str(metadata.get("article_code") or ""),
                    str(metadata.get("article_name") or ""),
                ]
            )
        if metadata.get("mapc"):
            return str(metadata["mapc"])
    return _hash_content(str(document.get("content") or ""))


@dataclass(slots=True)
class BM25Document:
    retrieval_text: str
    content: str
    metadata: dict[str, Any]
    tokens: list[str]


class LocalBM25Retriever:
    def __init__(self, docs: Sequence[dict[str, Any]], *, k1: float, b: float) -> None:
        self.k1 = k1
        self.b = b
        self.docs: list[BM25Document] = []
        self.doc_freq: Counter[str] = Counter()
        self.term_freqs: list[Counter[str]] = []
        self.doc_lengths: list[int] = []

        for row in docs:
            retrieval_text = str(row.get("retrieval_text") or row.get("content") or "").strip()
            content = str(row.get("content") or "").strip()
            metadata = dict(row.get("metadata") or {})
            tokens = tokenize_legal_text(retrieval_text)
            bm25_doc = BM25Document(
                retrieval_text=retrieval_text,
                content=content,
                metadata=metadata,
                tokens=tokens,
            )
            self.docs.append(bm25_doc)
            term_freq = Counter(tokens)
            self.term_freqs.append(term_freq)
            self.doc_lengths.append(len(tokens))
            for token in term_freq:
                self.doc_freq[token] += 1

        self.document_count = len(self.docs)
        self.avg_doc_length = (sum(self.doc_lengths) / self.document_count) if self.document_count else 0.0

    def _idf(self, token: str) -> float:
        df = self.doc_freq.get(token, 0)
        if df == 0 or self.document_count == 0:
            return 0.0
        return math.log(1 + (self.document_count - df + 0.5) / (df + 0.5))

    def _score_doc(self, query_tokens: Sequence[str], doc_index: int) -> float:
        if not query_tokens or not self.avg_doc_length:
            return 0.0

        score = 0.0
        term_freq = self.term_freqs[doc_index]
        doc_length = self.doc_lengths[doc_index]
        for token in query_tokens:
            frequency = term_freq.get(token, 0)
            if frequency == 0:
                continue
            numerator = frequency * (self.k1 + 1)
            denominator = frequency + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            score += self._idf(token) * (numerator / denominator)
        return score

    def search(
        self,
        query_text: str,
        *,
        level: str,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query_tokens = tokenize_legal_text(query_text)
        scored_docs: list[dict[str, Any]] = []
        for index, doc in enumerate(self.docs):
            result_doc = {"content": doc.content, "metadata": doc.metadata}
            if filters and not result_matches_filters(result_doc, filters):
                continue
            score = self._score_doc(query_tokens, index)
            if score <= 0:
                continue
            scored_docs.append(
                {
                    "content": doc.content,
                    "metadata": dict(doc.metadata),
                    "retrieval_text": doc.retrieval_text,
                    "bm25_score": float(score),
                    "vector_score": 0.0,
                    "semantic_score": 0.0,
                    "combined_score": 0.0,
                    "level": level,
                    "retrieval_channel": "bm25",
                    "matched_filters": {},
                    "exact_hit_fields": [],
                    "ranking_components": {"bm25_score": float(score)},
                }
            )
        scored_docs.sort(key=lambda item: item["bm25_score"], reverse=True)
        return scored_docs[: max(1, top_k)]


def load_local_corpus_documents(corpus_path: str | Path, *, level: str) -> list[dict[str, Any]]:
    resolved_path = Path(corpus_path).resolve()
    cache_key = (str(resolved_path), level)
    if cache_key in _CORPUS_CACHE:
        return _CORPUS_CACHE[cache_key]

    records = load_tv1_records(resolved_path)
    if level == "article":
        index_docs = prepare_article_documents(records)
    elif level == "chunk":
        index_docs = prepare_chunk_documents(records)
    else:
        raise ValueError(f"Unsupported retrieval level: {level}")

    corpus_docs = [
        {
            "content": str(doc.payload.get("content") or ""),
            "retrieval_text": str(doc.payload.get("retrieval_text") or doc.content or ""),
            "metadata": dict(doc.payload.get("metadata") or {}),
        }
        for doc in index_docs
    ]
    _CORPUS_CACHE[cache_key] = corpus_docs
    return corpus_docs


def get_local_bm25_retriever(config: RetrievalConfig, *, level: str) -> LocalBM25Retriever:
    cache_key = (str(Path(config.local_corpus_path).resolve()), level, config.bm25_k1, config.bm25_b)
    if cache_key not in _BM25_CACHE:
        docs = load_local_corpus_documents(config.local_corpus_path, level=level)
        _BM25_CACHE[cache_key] = LocalBM25Retriever(docs, k1=config.bm25_k1, b=config.bm25_b)
    return _BM25_CACHE[cache_key]


def _matches_exact(metadata: Mapping[str, Any], field: str, expected: str) -> bool:
    actual = _normalize_match_text(str(metadata.get(field) or ""))
    target = _normalize_match_text(expected)
    if not actual or not target:
        return False
    return actual == target


def _compute_legal_exact_bonus(
    metadata: Mapping[str, Any],
    metadata_filters: Mapping[str, Any],
    config: RetrievalConfig,
) -> dict[str, Any]:
    if not config.enable_exact_legal_match_boost:
        return {"bonus": 0.0, "matched_filters": {}, "exact_hit_fields": [], "ranking_components": {}}

    exact_hit_fields: list[str] = []
    matched_filters: dict[str, str] = {}
    ranking_components: dict[str, float] = {}
    bonus = 0.0
    field_bonus_pairs = (
        ("article_code", float(config.article_code_match_bonus)),
        ("article", float(config.article_match_bonus)),
        ("law_id", float(config.law_id_match_bonus)),
        ("title", float(config.title_match_bonus)),
        ("article_name", float(config.article_name_match_bonus)),
    )
    for field, field_bonus in field_bonus_pairs:
        expected = str(metadata_filters.get(field) or "").strip()
        if not expected or not _matches_exact(metadata, field, expected):
            continue
        exact_hit_fields.append(field)
        matched_filters[field] = expected
        ranking_components[f"{field}_match_bonus"] = field_bonus
        bonus += field_bonus

    if "article_code" in exact_hit_fields or (
        "article" in exact_hit_fields and ("title" in exact_hit_fields or "law_id" in exact_hit_fields)
    ):
        ranking_components["structured_match_bonus"] = float(config.structured_match_bonus)
        bonus += float(config.structured_match_bonus)

    return {
        "bonus": round(bonus, 6),
        "matched_filters": matched_filters,
        "exact_hit_fields": exact_hit_fields,
        "ranking_components": ranking_components,
    }


def _resolve_retrieval_plan(state: Mapping[str, Any], config: RetrievalConfig) -> dict[str, Any]:
    debug = state.get("retrieval_debug") or {}
    debug = dict(debug) if isinstance(debug, Mapping) else {}
    plan = dict(debug.get("next_retrieval_plan") or debug.get("current_plan") or {})
    base_filters = state.get("metadata_filters") or debug.get("metadata_filters") or {}
    metadata_filters = dict(plan.get("filters") or base_filters or {})
    legal_query_features = state.get("legal_query_features") or debug.get("legal_query_features") or {}
    execution_profile = resolve_execution_profile(state)
    limits = get_retrieval_limits(config, execution_profile)

    base_query = str(state.get("normalized_question") or state.get("question") or "").strip()
    rewritten_queries = [str(item).strip() for item in state.get("rewritten_queries", []) if str(item).strip()]
    queries: list[str] = []
    for query in [plan.get("selected_query"), base_query, *rewritten_queries]:
        cleaned = str(query or "").strip()
        if cleaned and cleaned not in queries:
            queries.append(cleaned)

    if config.article_only:
        level = "article"
    else:
        level = str(plan.get("level") or default_retrieval_level(config, execution_profile=execution_profile))

    if not config.enable_structured_legal_query_parse:
        metadata_filters = {}
        legal_query_features = {}

    return {
        "execution_profile": execution_profile,
        "level": level,
        "top_k": int(plan.get("top_k") or limits["top_k"]),
        "bm25_top_k": int(plan.get("bm25_top_k") or limits["bm25_top_k"]),
        "vector_top_k": int(plan.get("vector_top_k") or limits["vector_top_k"]),
        "queries": queries[: max(1, int(limits["max_queries_per_retrieve"]))],
        "filters": metadata_filters,
        "legal_query_features": dict(legal_query_features),
        "selected_query": str(plan.get("selected_query") or (queries[0] if queries else "")),
        "strategies_tried": list(debug.get("strategies_tried") or []),
    }


def _vector_search(
    query_text: str,
    *,
    level: str,
    top_k: int,
    filters: Mapping[str, Any] | None,
    vector_search_service: Any,
) -> list[dict[str, Any]]:
    if level == "article":
        raw_results = vector_search_service.search_article_level(query_text, filters=filters, top_k=top_k)
    elif level == "chunk":
        raw_results = vector_search_service.search_chunk_level(query_text, filters=filters, top_k=top_k)
    else:
        raise ValueError(f"Unsupported retrieval level: {level}")

    normalized_results: list[dict[str, Any]] = []
    for row in raw_results or []:
        normalized_results.append(
            {
                "content": str(row.get("content") or "").strip(),
                "metadata": dict(row.get("metadata") or {}),
                "bm25_score": 0.0,
                "vector_score": float(row.get("score") or row.get("vector_score") or 0.0),
                "semantic_score": float(row.get("semantic_score") or row.get("score") or 0.0),
                "combined_score": 0.0,
                "level": level,
                "retrieval_channel": "vector",
                "point_id": row.get("point_id", ""),
                "matched_filters": dict(row.get("matched_filters") or {}),
                "exact_hit_fields": list(row.get("exact_hit_fields") or []),
                "ranking_components": dict(row.get("ranking_components") or {}),
            }
        )
    return normalized_results


def _search_hybrid_once(
    query_text: str,
    *,
    level: str,
    filters: Mapping[str, Any],
    bm25_top_k: int,
    vector_top_k: int,
    bm25_backend: LocalBM25Retriever,
    vector_backend: QdrantSearchService,
) -> dict[str, Any]:
    return {
        "query": query_text,
        "filters": dict(filters),
        "bm25": bm25_backend.search(query_text, level=level, top_k=bm25_top_k, filters=filters),
        "vector": _vector_search(
            query_text,
            level=level,
            top_k=vector_top_k,
            filters=filters,
            vector_search_service=vector_backend,
        ),
    }


def merge_hybrid_results(
    query_traces: Sequence[dict[str, Any]],
    *,
    bm25_weight: float,
    vector_weight: float,
    query_bonus_weight: float,
    metadata_filters: Mapping[str, Any],
    config: RetrievalConfig,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for trace in query_traces:
        query_text = str(trace.get("query") or "")
        for source_name, rows in (("bm25", trace.get("bm25", [])), ("vector", trace.get("vector", []))):
            for row in rows or []:
                key = _dedup_key(row)
                payload = merged.setdefault(
                    key,
                    {
                        "content": str(row.get("content") or "").strip(),
                        "metadata": dict(row.get("metadata") or {}),
                        "bm25_score": 0.0,
                        "vector_score": 0.0,
                        "semantic_score": 0.0,
                        "combined_score": 0.0,
                        "matched_queries": [],
                        "retrieval_channels": set(),
                        "level": row.get("level") or "",
                        "matched_filters": {},
                        "exact_hit_fields": set(),
                        "ranking_components": {},
                    },
                )
                payload["content"] = payload["content"] or str(row.get("content") or "").strip()
                if not payload["metadata"]:
                    payload["metadata"] = dict(row.get("metadata") or {})
                payload["bm25_score"] = max(float(payload["bm25_score"]), float(row.get("bm25_score") or 0.0))
                payload["vector_score"] = max(float(payload["vector_score"]), float(row.get("vector_score") or 0.0))
                payload["semantic_score"] = max(float(payload["semantic_score"]), float(row.get("semantic_score") or 0.0))
                if query_text and query_text not in payload["matched_queries"]:
                    payload["matched_queries"].append(query_text)
                payload["retrieval_channels"].add(source_name)
                payload["matched_filters"].update(dict(row.get("matched_filters") or {}))
                payload["exact_hit_fields"].update(list(row.get("exact_hit_fields") or []))
                payload["ranking_components"].update(dict(row.get("ranking_components") or {}))

    max_bm25 = max((float(doc["bm25_score"]) for doc in merged.values()), default=0.0)
    max_vector = max((float(doc["vector_score"]) for doc in merged.values()), default=0.0)
    max_query_hits = max((len(doc["matched_queries"]) for doc in merged.values()), default=1)

    merged_docs: list[dict[str, Any]] = []
    for doc in merged.values():
        normalized_bm25 = (doc["bm25_score"] / max_bm25) if max_bm25 > 0 else 0.0
        normalized_vector = (doc["vector_score"] / max_vector) if max_vector > 0 else 0.0
        normalized_query_hits = len(doc["matched_queries"]) / max_query_hits if max_query_hits > 0 else 0.0
        exact_signals = _compute_legal_exact_bonus(doc["metadata"], metadata_filters, config)
        doc["matched_filters"].update(exact_signals["matched_filters"])
        doc["exact_hit_fields"] = sorted(set(doc["exact_hit_fields"]) | set(exact_signals["exact_hit_fields"]))
        doc["ranking_components"].update(exact_signals["ranking_components"])
        doc["ranking_components"].update(
            {
                "normalized_bm25": round(normalized_bm25, 6),
                "normalized_vector": round(normalized_vector, 6),
                "normalized_query_hits": round(normalized_query_hits, 6),
            }
        )
        doc["exact_match_bonus"] = exact_signals["bonus"]
        doc["combined_score"] = (
            bm25_weight * normalized_bm25
            + vector_weight * normalized_vector
            + query_bonus_weight * normalized_query_hits
            + exact_signals["bonus"]
        )
        doc["retrieval_channels"] = sorted(doc["retrieval_channels"])
        merged_docs.append(doc)

    merged_docs.sort(
        key=lambda item: (
            float(item["combined_score"]),
            len(item.get("exact_hit_fields") or []),
            float(item.get("semantic_score") or 0.0),
            float(item.get("bm25_score") or 0.0),
        ),
        reverse=True,
    )
    return merged_docs


def retrieve_node(
    state: Mapping[str, Any],
    *,
    retrieval_config: RetrievalConfig | None = None,
    config_path: str | Path | None = None,
    vector_search_service: Any | None = None,
    bm25_retriever: Any | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run article-level hybrid retrieval using local BM25 + Qdrant vector search."""

    resolved_logger = logger or LOGGER
    resolved_retrieval_config = retrieval_config or load_retrieval_config(config_path)
    plan = _resolve_retrieval_plan(state, resolved_retrieval_config)
    execution_profile = str(plan["execution_profile"])
    limits = get_retrieval_limits(resolved_retrieval_config, execution_profile)
    level = "article" if resolved_retrieval_config.article_only else str(plan["level"])
    top_k = int(plan["top_k"])
    filters = dict(plan["filters"])
    queries = plan["queries"]
    legal_query_features = dict(plan["legal_query_features"])

    if not queries:
        return {
            "retrieved_docs": [],
            "retrieval_debug": {
                "execution_profile": execution_profile,
                "metadata_filters": filters,
                "legal_query_features": legal_query_features,
                "current_plan": {"level": level, "top_k": top_k, "filters": filters, "selected_query": ""},
                "candidate_counts": {"bm25": 0, "vector": 0, "merged": 0},
            },
        }

    vector_backend = vector_search_service or QdrantSearchService(
        config_path=resolved_retrieval_config.indexing_config_path,
        retrieval_config=resolved_retrieval_config,
        logger=resolved_logger,
    )
    bm25_backend = bm25_retriever or get_local_bm25_retriever(resolved_retrieval_config, level=level)

    query_traces: list[dict[str, Any]] = []
    total_bm25 = 0
    total_vector = 0
    structured_filter_no_hit = False
    structured_filter_fallback_used = False
    for query_text in queries:
        trace = _search_hybrid_once(
            query_text,
            level=level,
            filters=filters,
            bm25_top_k=int(plan["bm25_top_k"]),
            vector_top_k=int(plan["vector_top_k"]),
            bm25_backend=bm25_backend,
            vector_backend=vector_backend,
        )
        if filters and not trace["bm25"] and not trace["vector"]:
            structured_filter_no_hit = True
            resolved_logger.info(
                "Structured filter produced no hit. Falling back to unfiltered article search. query=%s filters=%s",
                query_text,
                filters,
            )
            trace = _search_hybrid_once(
                query_text,
                level=level,
                filters={},
                bm25_top_k=int(plan["bm25_top_k"]),
                vector_top_k=int(plan["vector_top_k"]),
                bm25_backend=bm25_backend,
                vector_backend=vector_backend,
            )
            trace["structured_filter_fallback"] = True
            structured_filter_fallback_used = True
        total_bm25 += len(trace["bm25"])
        total_vector += len(trace["vector"])
        query_traces.append(trace)

    merged_docs = merge_hybrid_results(
        query_traces,
        bm25_weight=resolved_retrieval_config.bm25_weight,
        vector_weight=resolved_retrieval_config.vector_weight,
        query_bonus_weight=resolved_retrieval_config.query_bonus_weight,
        metadata_filters=filters,
        config=resolved_retrieval_config,
    )

    candidate_limit = max(top_k, int(limits["rerank_top_n"]))
    if execution_profile == "full":
        candidate_limit = max(candidate_limit, int(limits["rerank_top_n"]) * 3)
    merged_docs = merged_docs[:candidate_limit]

    top_exact_hit = bool(merged_docs and merged_docs[0].get("exact_hit_fields"))
    if filters and not top_exact_hit and merged_docs:
        resolved_logger.info(
            "Article retrieval returned results but no exact legal hit at top. query=%s filters=%s top_metadata=%s",
            queries[0],
            filters,
            merged_docs[0].get("metadata"),
        )

    resolved_logger.info(
        "Hybrid retrieval profile=%s level=%s queries=%s bm25=%s vector=%s merged=%s filters=%s exact_top_hit=%s",
        execution_profile,
        level,
        len(queries),
        total_bm25,
        total_vector,
        len(merged_docs),
        bool(filters),
        top_exact_hit,
    )

    previous_debug = state.get("retrieval_debug") or {}
    previous_debug = dict(previous_debug) if isinstance(previous_debug, Mapping) else {}
    current_plan = {
        "execution_profile": execution_profile,
        "level": level,
        "top_k": top_k,
        "bm25_top_k": int(plan["bm25_top_k"]),
        "vector_top_k": int(plan["vector_top_k"]),
        "filters": filters,
        "selected_query": plan["selected_query"] or (queries[0] if queries else ""),
    }
    previous_debug.update(
        {
            "execution_profile": execution_profile,
            "metadata_filters": filters,
            "legal_query_features": legal_query_features,
            "current_plan": current_plan,
            "next_retrieval_plan": None,
            "candidate_counts": {"bm25": total_bm25, "vector": total_vector, "merged": len(merged_docs)},
            "query_traces": [
                {
                    "query": trace["query"],
                    "filters": trace.get("filters") or {},
                    "structured_filter_fallback": bool(trace.get("structured_filter_fallback")),
                }
                for trace in query_traces
            ],
            "strategies_tried": plan["strategies_tried"],
            "structured_filter_no_hit": structured_filter_no_hit,
            "structured_filter_fallback_used": structured_filter_fallback_used,
            "top_exact_hit": top_exact_hit,
        }
    )
    return {
        "retrieved_docs": merged_docs,
        "retrieval_debug": previous_debug,
        "metadata_filters": filters,
        "legal_query_features": legal_query_features,
        "execution_profile": execution_profile,
    }


__all__ = [
    "LocalBM25Retriever",
    "load_local_corpus_documents",
    "merge_hybrid_results",
    "normalize_legal_text",
    "retrieve_node",
    "tokenize_legal_text",
]

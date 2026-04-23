from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.tv3_retrieval.fallback_policy import (
    RetrievalConfig,
    decide_next_retrieval_step,
    load_retrieval_config,
    resolve_execution_profile,
)

LOGGER = logging.getLogger(__name__)


def _is_structured_legal_doc(doc: Mapping[str, Any]) -> bool:
    metadata = doc.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        return False
    return bool(metadata.get("article_code") or metadata.get("law_id") or metadata.get("title"))


def _source_key(doc: Mapping[str, Any]) -> str:
    metadata = doc.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        return str(doc.get("source") or "")
    return "|".join(
        [
            str(metadata.get("law_id") or ""),
            str(metadata.get("article_code") or ""),
            str(metadata.get("article_name") or ""),
            str(metadata.get("mapc") or ""),
        ]
    )


def evaluate_evidence(
    reranked_docs: Sequence[Mapping[str, Any]],
    *,
    config: RetrievalConfig,
    question: str = "",
    execution_profile: str = "full",
) -> dict[str, Any]:
    """Compute evidence sufficiency signals for the retrieval loop."""

    valid_docs = [doc for doc in reranked_docs if str(doc.get("content") or "").strip()]
    structured_docs = [doc for doc in valid_docs if _is_structured_legal_doc(doc)]
    unique_sources = {_source_key(doc) for doc in structured_docs if _source_key(doc)}
    top_doc = valid_docs[0] if valid_docs else {}
    top_score = float(top_doc.get("rerank_score") or top_doc.get("combined_score") or 0.0)

    question_lower = (question or "").lower()
    requires_strong_citation = any(
        token in question_lower
        for token in ("điều", "luật số", "nghị định", "thông tư", "quyền", "nghĩa vụ", "xử phạt")
    )

    if execution_profile == "fast":
        min_valid_sources = config.min_valid_sources_fast
        min_unique_sources = config.min_unique_sources_fast
        min_top_score = config.min_top_rerank_score_fast
    else:
        min_valid_sources = config.min_valid_sources
        min_unique_sources = config.min_unique_sources
        min_top_score = config.min_top_rerank_score

    if not valid_docs:
        failure_reason = "no_results"
    elif len(valid_docs) < min_valid_sources:
        failure_reason = "insufficient_results"
    elif config.require_structured_citation and len(structured_docs) < 1:
        failure_reason = "missing_structured_sources"
    elif len(unique_sources) < min_unique_sources:
        failure_reason = "low_source_diversity"
    elif top_score < min_top_score:
        failure_reason = "weak_evidence"
    elif requires_strong_citation and not structured_docs:
        failure_reason = "missing_structured_sources"
    else:
        failure_reason = ""

    if execution_profile == "fast" and not failure_reason and structured_docs:
        failure_reason = ""

    retrieval_ok = not failure_reason
    return {
        "retrieval_ok": retrieval_ok,
        "failure_reason": failure_reason,
        "valid_doc_count": len(valid_docs),
        "structured_doc_count": len(structured_docs),
        "unique_source_count": len(unique_sources),
        "top_rerank_score": top_score,
    }


def retrieval_check_node(
    state: Mapping[str, Any],
    *,
    retrieval_config: RetrievalConfig | None = None,
    config_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Evaluate retrieval evidence and decide whether TV3 should retry."""

    resolved_logger = logger or LOGGER
    resolved_retrieval_config = retrieval_config or load_retrieval_config(config_path)
    execution_profile = resolve_execution_profile(state)
    question = str(state.get("normalized_question") or state.get("question") or "").strip()
    reranked_docs = state.get("reranked_docs") or []
    evidence = evaluate_evidence(
        reranked_docs,
        config=resolved_retrieval_config,
        question=question,
        execution_profile=execution_profile,
    )

    previous_debug = state.get("retrieval_debug") or {}
    previous_debug = dict(previous_debug) if isinstance(previous_debug, Mapping) else {}
    previous_debug["evidence"] = evidence
    previous_debug["last_failure_reason"] = evidence["failure_reason"]
    previous_debug["execution_profile"] = execution_profile

    if evidence["retrieval_ok"]:
        resolved_logger.info(
            "Retrieval evidence sufficient profile=%s valid=%s unique=%s top_score=%.4f",
            execution_profile,
            evidence["valid_doc_count"],
            evidence["unique_source_count"],
            evidence["top_rerank_score"],
        )
        return {
            "retrieval_ok": True,
            "retrieval_failure_reason": "",
            "next_action": "proceed",
            "retrieval_debug": previous_debug,
        }

    decision = decide_next_retrieval_step(state, resolved_retrieval_config)
    strategy = decision.get("strategy", "")
    retry_plan = decision.get("retry_plan")
    strategies_tried = list(previous_debug.get("strategies_tried") or [])
    if strategy and strategy not in strategies_tried:
        strategies_tried.append(str(strategy))
    previous_debug["strategies_tried"] = strategies_tried
    previous_debug["next_retrieval_plan"] = retry_plan

    next_action = str(decision.get("next_action") or "fallback")
    updates: dict[str, Any] = {
        "retrieval_ok": False,
        "retrieval_failure_reason": evidence["failure_reason"],
        "next_action": next_action,
        "retrieval_debug": previous_debug,
    }
    if next_action == "retry":
        updates["loop_count"] = int(state.get("loop_count") or 0) + 1

    resolved_logger.info(
        "Retrieval evidence insufficient profile=%s reason=%s next_action=%s strategy=%s",
        execution_profile,
        evidence["failure_reason"],
        next_action,
        strategy,
    )
    return updates


__all__ = [
    "evaluate_evidence",
    "retrieval_check_node",
]

from __future__ import annotations

import json
import logging
import os
import re
from functools import wraps
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.tv3_retrieval.fallback_policy import resolve_execution_profile
from src.tv5_reasoning.citation_critic import inspect_citations
from src.tv5_reasoning.generate_draft_node import (
    DEFAULT_LLM_CONFIG_PATH,
    LLMConfig,
    OllamaChatClient,
    load_llm_config,
)
from src.tv5_reasoning.prompt_library import (
    DEFAULT_PROMPTS_CONFIG_PATH,
    PromptLibrary,
    get_prompt_library,
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
TOKEN_PATTERN = re.compile(r"[0-9A-Za-zÀ-ỹ]+")
DISCLAIMER_PATTERN = re.compile(
    r"(chưa đủ căn cứ|cần làm rõ thêm|không đủ dữ liệu|chỉ có thể xác định|tham khảo)",
    re.IGNORECASE,
)


def _split_claims(answer: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", (answer or "").strip())
    raw_parts = re.split(r"(?:\n+|(?<=[.!?;:]))\s+", normalized)
    claims = [part.strip("- ").strip() for part in raw_parts if part.strip()]
    return [claim for claim in claims if len(claim) >= 18]


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text or "") if len(token) >= 2}


def _doc_support_texts(reranked_docs: Sequence[Mapping[str, Any]]) -> list[str]:
    support_texts: list[str] = []
    for doc in reranked_docs or []:
        content = str(doc.get("content") or "").strip()
        metadata = dict(doc.get("metadata") or {})
        citation_hint = " ".join(
            part
            for part in (
                str(metadata.get("article_code") or metadata.get("article") or "").strip(),
                str(metadata.get("article_name") or "").strip(),
                str(metadata.get("law_id") or metadata.get("title") or "").strip(),
            )
            if part
        )
        combined = f"{content}\n{citation_hint}".strip()
        if combined:
            support_texts.append(combined)
    return support_texts


def _claim_supported(claim: str, support_texts: Sequence[str], *, citation_hints: Sequence[str]) -> bool:
    if DISCLAIMER_PATTERN.search(claim):
        return True
    claim_tokens = _tokenize(claim)
    if not claim_tokens:
        return True

    best_overlap = 0.0
    for support_text in support_texts:
        support_tokens = _tokenize(support_text)
        if not support_tokens:
            continue
        overlap = len(claim_tokens & support_tokens) / max(len(claim_tokens), 1)
        best_overlap = max(best_overlap, overlap)
        if overlap >= 0.42:
            return True

    normalized_claim = claim.lower()
    if any(hint.lower() in normalized_claim for hint in citation_hints if hint):
        return True
    return best_overlap >= 0.28 and len(claim_tokens) <= 8


def _decide_next_action(
    *,
    grounding_score: float,
    unsupported_claims: Sequence[str],
    missing_evidence: Sequence[str],
    context: str,
    risk_level: str,
    citation_ok: bool,
    execution_profile: str,
) -> str:
    if execution_profile == "fast":
        if grounding_score >= 0.7 and not unsupported_claims and citation_ok:
            return "proceed"
        return "escalate_to_full"

    if not str(context).strip():
        return "retrieve_again"
    if str(risk_level).lower() == "high" and (grounding_score < 0.8 or unsupported_claims or not citation_ok):
        return "human_review"
    if grounding_score >= 0.78 and not unsupported_claims and citation_ok:
        return "proceed"
    if len(unsupported_claims) >= 2 and not str(context).strip():
        return "retrieve_again"
    if missing_evidence and grounding_score < 0.45:
        return "retrieve_again"
    return "revise"


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _has_strong_exact_legal_hit(reranked_docs: Sequence[Mapping[str, Any]] | None) -> bool:
    for doc in reranked_docs or []:
        exact_hit_fields = {
            str(item).strip().lower()
            for item in (doc.get("exact_hit_fields") or [])
            if str(item).strip()
        }
        matched_filters = {
            str(key).strip().lower(): str(value).strip()
            for key, value in dict(doc.get("matched_filters") or {}).items()
            if str(value).strip()
        }
        if "article_code" in exact_hit_fields:
            return True
        if "article" in exact_hit_fields and ({"title", "law_id"} & exact_hit_fields):
            return True
        if "article_code" in matched_filters:
            return True
        if "article" in matched_filters and ("title" in matched_filters or "law_id" in matched_filters):
            return True
    return False


def _is_low_risk_definition_overview(
    *,
    question: str,
    intent: str,
    risk_level: str,
    sources: Sequence[str] | None,
    reranked_docs: Sequence[Mapping[str, Any]] | None,
) -> bool:
    if str(risk_level).strip().lower() not in {"low", "medium"}:
        return False
    if str(intent).strip().lower() != "hoi_dinh_nghia":
        return False
    normalized_question = str(question or "").strip().lower()
    if any(marker in normalized_question for marker in ("tôi", "em", "gia đình", "công ty", "trường hợp", "nếu")):
        return False
    source_count = len([item for item in (sources or []) if str(item).strip()])
    doc_count = len([doc for doc in (reranked_docs or []) if doc])
    return source_count >= 2 and doc_count >= 2


def _resolve_low_risk_exact_hit_next_action(
    *,
    question: str,
    intent: str,
    risk_level: str,
    current_next_action: str,
    grounding_score: float,
    unsupported_claims: Sequence[str],
    sources: Sequence[str] | None,
    reranked_docs: Sequence[Mapping[str, Any]] | None,
) -> str:
    normalized_risk = str(risk_level).strip().lower()
    if normalized_risk == "high":
        return current_next_action
    if current_next_action != "human_review":
        return current_next_action

    if _has_strong_exact_legal_hit(reranked_docs):
        if unsupported_claims:
            return "revise"
        if grounding_score >= 0.5:
            return "proceed"
        return "revise"

    if _is_low_risk_definition_overview(
        question=question,
        intent=intent,
        risk_level=normalized_risk,
        sources=sources,
        reranked_docs=reranked_docs,
    ):
        if unsupported_claims:
            return "revise"
        if grounding_score >= 0.75:
            return "proceed"
        return "revise"

    return current_next_action


def rule_based_grounding_check(
    *,
    draft_answer: str,
    context: str,
    sources: Sequence[str] | None,
    reranked_docs: Sequence[Mapping[str, Any]] | None,
    risk_level: str,
    execution_profile: str = "full",
) -> dict[str, Any]:
    """Run a deterministic grounding check using evidence overlap + citation inspection."""

    claims = _split_claims(draft_answer)
    support_texts = _doc_support_texts(reranked_docs or [])
    citation_report = inspect_citations(draft_answer, sources, reranked_docs or [])
    citation_hints = citation_report["expected_citations"] + citation_report["normalized_citations"]

    unsupported_claims: list[str] = []
    for claim in claims:
        if not _claim_supported(claim, support_texts or [context], citation_hints=citation_hints):
            unsupported_claims.append(claim)

    missing_evidence: list[str] = []
    if not str(context).strip():
        missing_evidence.append("Không có context/evidence để kiểm tra grounding.")
    if citation_report["missing_citations"]:
        missing_evidence.extend(citation_report["missing_citations"])
    if citation_report["weak_citations"]:
        missing_evidence.extend(citation_report["weak_citations"])

    total_claims = max(len(claims), 1)
    unsupported_ratio = len(unsupported_claims) / total_claims
    citation_penalty = 0.15 if not citation_report["citation_ok"] else 0.0
    grounding_score = max(0.0, min(1.0, 1.0 - unsupported_ratio - citation_penalty))
    next_action = _decide_next_action(
        grounding_score=grounding_score,
        unsupported_claims=unsupported_claims,
        missing_evidence=missing_evidence,
        context=context,
        risk_level=risk_level,
        citation_ok=bool(citation_report["citation_ok"]),
        execution_profile=execution_profile,
    )
    grounding_ok = next_action == "proceed"
    return {
        "grounding_ok": grounding_ok,
        "grounding_score": round(grounding_score, 4),
        "unsupported_claims": unsupported_claims,
        "missing_evidence": missing_evidence,
        "next_action": next_action,
        "citation_findings": citation_report,
        "claim_count": total_claims,
    }


def _llm_grounding_check(
    *,
    question: str,
    intent: str,
    risk_level: str,
    draft_answer: str,
    context: str,
    sources: Sequence[str] | None,
    llm_config: LLMConfig,
    prompt_library: PromptLibrary,
    client: OllamaChatClient,
) -> dict[str, Any] | None:
    prompt = prompt_library.get_grounding_prompt(
        intent,
        risk_level,
        question=question,
        context=context,
        sources=sources,
        draft_answer=draft_answer,
    )
    raw = client.generate_with_retry(prompt=prompt, system_prompt=prompt_library.get_system_prompt())
    return _extract_json_object(raw)

@_optional_traceable(name="tv5.grounding_check_node", run_type="chain", tags=["tv5", "reasoning"])
def grounding_check_node(
    state: Mapping[str, Any],
    *,
    llm_config: LLMConfig | None = None,
    llm_config_path: str | Path | None = None,
    prompts_path: str | Path | None = None,
    prompt_library: PromptLibrary | None = None,
    client: OllamaChatClient | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Check whether the draft answer is grounded in retrieved legal evidence."""

    resolved_logger = logger or LOGGER
    question = str(state.get("normalized_question") or state.get("question") or "").strip()
    intent = str(state.get("intent") or "hoi_tinh_huong_thuc_te").strip()
    risk_level = str(state.get("risk_level") or "medium").strip().lower()
    draft_answer = str(state.get("draft_answer") or "").strip()
    context = str(state.get("context") or "").strip()
    sources = list(state.get("sources") or [])
    reranked_docs = list(state.get("reranked_docs") or [])
    execution_profile = resolve_execution_profile(state)

    rule_based = rule_based_grounding_check(
        draft_answer=draft_answer,
        context=context,
        sources=sources,
        reranked_docs=reranked_docs,
        risk_level=risk_level,
        execution_profile=execution_profile,
    )

    resolved_llm_config = llm_config or load_llm_config(llm_config_path or DEFAULT_LLM_CONFIG_PATH)
    resolved_prompt_library = prompt_library or get_prompt_library(
        config_path=prompts_path or DEFAULT_PROMPTS_CONFIG_PATH,
        logger=resolved_logger,
    )
    resolved_client = client or OllamaChatClient(resolved_llm_config, logger=resolved_logger)

    llm_result: dict[str, Any] | None = None
    if execution_profile != "fast":
        try:
            llm_result = _llm_grounding_check(
                question=question,
                intent=intent,
                risk_level=risk_level,
                draft_answer=draft_answer,
                context=context,
                sources=sources,
                llm_config=resolved_llm_config,
                prompt_library=resolved_prompt_library,
                client=resolved_client,
            )
        except Exception as exc:  # pragma: no cover - runtime-dependent.
            resolved_logger.warning("LLM grounding critic failed, keeping rule-based result: %s", exc)

    if llm_result:
        llm_score = _safe_float(llm_result.get("grounding_score"), rule_based["grounding_score"])
        llm_unsupported = [str(item).strip() for item in llm_result.get("unsupported_claims", []) if str(item).strip()]
        llm_missing = [str(item).strip() for item in llm_result.get("missing_evidence", []) if str(item).strip()]
        merged_unsupported = list(dict.fromkeys(rule_based["unsupported_claims"] + llm_unsupported))
        merged_missing = list(dict.fromkeys(rule_based["missing_evidence"] + llm_missing))
        merged_score = round((rule_based["grounding_score"] + llm_score) / 2.0, 4)
        merged_next_action = str(llm_result.get("next_action") or rule_based["next_action"])
        if risk_level == "high" and merged_next_action == "proceed" and merged_score < 0.8:
            merged_next_action = "human_review"
        merged_next_action = _resolve_low_risk_exact_hit_next_action(
            question=question,
            intent=intent,
            risk_level=risk_level,
            current_next_action=merged_next_action,
            grounding_score=merged_score,
            unsupported_claims=merged_unsupported,
            sources=sources,
            reranked_docs=reranked_docs,
        )
        rule_based.update(
            {
                "grounding_score": merged_score,
                "unsupported_claims": merged_unsupported,
                "missing_evidence": merged_missing,
                "next_action": merged_next_action,
                "grounding_ok": merged_next_action == "proceed",
                "reasoning_notes": {
                    "grounding_critic_model": resolved_llm_config.model_name,
                    "llm_notes": str(llm_result.get("notes") or "").strip(),
                    "execution_profile": execution_profile,
                },
            }
        )

    human_review_required = False
    review_note = ""
    next_action = _resolve_low_risk_exact_hit_next_action(
        question=question,
        intent=intent,
        risk_level=risk_level,
        current_next_action=str(rule_based["next_action"] or ""),
        grounding_score=float(rule_based["grounding_score"] or 0.0),
        unsupported_claims=rule_based["unsupported_claims"],
        sources=sources,
        reranked_docs=reranked_docs,
    )
    rule_based["next_action"] = next_action
    rule_based["grounding_ok"] = next_action == "proceed"

    if execution_profile == "fast":
        human_review_required = False
        review_note = ""
        if next_action not in {"proceed", "escalate_to_full"}:
            next_action = "escalate_to_full"
        rule_based["next_action"] = next_action
        rule_based["grounding_ok"] = next_action == "proceed"
    elif next_action == "human_review":
        human_review_required = True
        review_note = review_note or (
            "Grounding chưa đủ mạnh cho câu hỏi rủi ro cao; nên có human review trước khi trả lời chính thức."
            if risk_level == "high"
            else "Grounding hoặc trích dẫn chưa đủ chắc chắn; nên có human review trước khi trả lời chính thức."
        )

    return {
        "grounding_ok": bool(rule_based["grounding_ok"]),
        "grounding_score": rule_based["grounding_score"],
        "unsupported_claims": rule_based["unsupported_claims"],
        "missing_evidence": rule_based["missing_evidence"],
        "citation_findings": rule_based["citation_findings"],
        "reasoning_notes": rule_based.get("reasoning_notes", {"execution_profile": execution_profile}),
        "human_review_required": human_review_required,
        "review_note": review_note,
        "next_action": rule_based["next_action"],
    }


__all__ = ["grounding_check_node", "rule_based_grounding_check"]

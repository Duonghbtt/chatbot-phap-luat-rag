from __future__ import annotations

import json
import logging
import re
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


def _trim_text(text: str, max_chars: int = 550) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    return normalized[:max_chars].strip()


def _unique_sources(sources: Sequence[str] | None) -> list[str]:
    normalized: list[str] = []
    for source in sources or []:
        cleaned = str(source).strip()
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def _fallback_final_answer(
    *,
    question: str,
    context: str,
    sources: Sequence[str] | None,
    reranked_docs: Sequence[Mapping[str, Any]] | None,
    risk_level: str,
    grounding_score: float,
    unsupported_claims: Sequence[str] | None,
) -> tuple[str, str]:
    normalized_sources = _unique_sources(sources)
    if not str(context).strip():
        final_answer = (
            "1. Trả lời ngắn gọn\n"
            "Dữ liệu hiện có chưa đủ căn cứ để kết luận chắc chắn cho câu hỏi này.\n\n"
            "2. Căn cứ pháp lý\n"
            "Chưa có evidence đủ mạnh trong ngữ cảnh hiện tại.\n\n"
            "3. Lưu ý\n"
            "Bạn nên truy xuất thêm điều luật liên quan hoặc cung cấp thêm bối cảnh cụ thể."
        )
        return final_answer, "Thiếu context/evidence nên chưa thể đưa ra kết luận pháp lý chắc chắn."

    top_doc = dict((reranked_docs or [{}])[0])
    top_content = _trim_text(str(top_doc.get("content") or context))
    source_lines = "\n".join(f"- {item}" for item in normalized_sources[:3]) if normalized_sources else "- Chưa có nguồn chuẩn hóa"
    note = "Câu hỏi có rủi ro cao; nên human review trước khi sử dụng kết quả." if str(risk_level).lower() == "high" else ""
    if unsupported_claims:
        note = (
            "Một phần nội dung ban đầu chưa được context hỗ trợ đầy đủ; câu trả lời đã được thu hẹp theo evidence hiện có."
            if not note
            else f"{note} Một phần nội dung ban đầu chưa được context hỗ trợ đầy đủ."
        )
    if not note and grounding_score < 0.7:
        note = "Mức độ chắc chắn còn hạn chế do evidence hiện có chưa thật mạnh."
    if not note:
        note = "Câu trả lời đã được giới hạn trong phạm vi context truy xuất."

    final_answer = (
        "1. Trả lời ngắn gọn\n"
        f"Từ dữ liệu hiện có, có thể xác định rằng: {top_content}\n\n"
        "2. Căn cứ pháp lý\n"
        f"{source_lines}\n\n"
        "3. Lưu ý\n"
        f"{note}"
    )
    review_note = note if str(risk_level).lower() == "high" or grounding_score < 0.75 else ""
    return final_answer, review_note


def revise_answer_node(
    state: Mapping[str, Any],
    *,
    llm_config: LLMConfig | None = None,
    llm_config_path: str | Path | None = None,
    prompts_path: str | Path | None = None,
    prompt_library: PromptLibrary | None = None,
    client: OllamaChatClient | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Revise a draft answer so it only contains grounded, well-cited legal content."""

    resolved_logger = logger or LOGGER
    question = str(state.get("normalized_question") or state.get("question") or "").strip()
    intent = str(state.get("intent") or "hoi_tinh_huong_thuc_te").strip()
    risk_level = str(state.get("risk_level") or "medium").strip().lower()
    draft_answer = str(state.get("draft_answer") or state.get("final_answer") or "").strip()
    context = str(state.get("context") or "").strip()
    sources = list(state.get("sources") or [])
    reranked_docs = list(state.get("reranked_docs") or [])
    unsupported_claims = list(state.get("unsupported_claims") or [])
    missing_evidence = list(state.get("missing_evidence") or [])
    grounding_score = float(state.get("grounding_score") or 0.0)
    execution_profile = resolve_execution_profile(state)

    resolved_llm_config = llm_config or load_llm_config(llm_config_path or DEFAULT_LLM_CONFIG_PATH)
    resolved_prompt_library = prompt_library or get_prompt_library(
        config_path=prompts_path or DEFAULT_PROMPTS_CONFIG_PATH,
        logger=resolved_logger,
    )
    resolved_client = client or OllamaChatClient(resolved_llm_config, logger=resolved_logger)

    prompt = resolved_prompt_library.get_revision_prompt(
        intent,
        risk_level,
        question=question,
        context=context,
        sources=sources,
        draft_answer=draft_answer,
        unsupported_claims=unsupported_claims,
        missing_evidence=missing_evidence,
    )
    system_prompt = resolved_prompt_library.get_system_prompt()

    payload: dict[str, Any] | None = None
    if execution_profile != "fast":
        try:
            raw_response = resolved_client.generate_with_retry(prompt=prompt, system_prompt=system_prompt)
            payload = _extract_json_object(raw_response)
        except Exception as exc:  # pragma: no cover - runtime-dependent.
            resolved_logger.warning("Revision via Ollama failed, using fallback answer synthesis: %s", exc)

    if payload and str(payload.get("final_answer") or "").strip():
        final_answer = str(payload.get("final_answer") or "").strip()
        grounding_ok = bool(payload.get("grounding_ok"))
        review_note = str(payload.get("review_note") or "").strip()
    else:
        final_answer, review_note = _fallback_final_answer(
            question=question,
            context=context,
            sources=sources,
            reranked_docs=reranked_docs,
            risk_level=risk_level,
            grounding_score=grounding_score,
            unsupported_claims=unsupported_claims,
        )
        grounding_ok = grounding_score >= 0.7 and not unsupported_claims

    citation_report = inspect_citations(final_answer, sources, reranked_docs)
    if not citation_report["citation_ok"] and sources:
        source_tail = "\n".join(f"- {item}" for item in _unique_sources(sources[:3]))
        final_answer = (
            f"{final_answer}\n\nBổ sung căn cứ pháp lý\n{source_tail}"
            if "Bổ sung căn cứ pháp lý" not in final_answer
            else final_answer
        )
        citation_report = inspect_citations(final_answer, sources, reranked_docs)

    human_review_required = bool(state.get("human_review_required") or False)
    if execution_profile == "fast":
        human_review_required = False
        review_note = ""
    elif risk_level == "high" or grounding_score < 0.65 or not citation_report["citation_ok"]:
        human_review_required = True
        if not review_note:
            review_note = (
                "Nội dung nên được human review vì câu hỏi có rủi ro cao hoặc grounding/citation chưa thật mạnh."
            )

    return {
        "final_answer": final_answer,
        "grounding_ok": bool(grounding_ok and citation_report["citation_ok"]),
        "human_review_required": human_review_required,
        "review_note": review_note,
        "citation_findings": citation_report,
        "reasoning_notes": {"execution_profile": execution_profile},
    }


__all__ = ["revise_answer_node"]

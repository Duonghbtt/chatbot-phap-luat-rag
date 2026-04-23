from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Mapping

from src.tv4_router.clarify_detector import detect_clarify_need
from src.tv4_router.intent_classifier import (
    RoutingConfig,
    classify_intent,
    load_routing_config,
    normalize_question,
)
from src.tv4_router.risk_tagger import tag_risk

LOGGER = logging.getLogger(__name__)

HUMAN_REVIEW_QUESTIONS = [
    "Câu hỏi này có thể ảnh hưởng đến quyết định pháp lý thực tế. Bạn muốn hệ thống chỉ phân tích căn cứ pháp luật chung, hay tiếp tục theo tình huống cụ thể của bạn?",
    "Để tránh trả lời vượt quá dữ liệu hiện có, bạn muốn mình trình bày thông tin pháp luật tổng quát hay tiếp tục theo trường hợp cá nhân?",
]
HIGH_RISK_HINTS = ["khởi kiện", "kiện", "tranh chấp", "bồi thường", "ly hôn", "thừa kế", "thu hồi đất"]
CLARIFY_CRITICAL_SLOTS = {
    "hanh_vi_vi_pham",
    "chu_the",
    "loai_tranh_chap",
    "van_ban_phap_luat",
    "tham_chieu_dieu_luat",
}
ARTICLE_REFERENCE_PATTERN = re.compile(r"điều\s+\d+[0-9a-z.\-]*", re.IGNORECASE)
LAW_REFERENCE_PATTERN = re.compile(r"(luật số|nghị định số|thông tư số|bộ luật|theo luật)", re.IGNORECASE)
COMPLEX_QUESTION_HINTS = ["nên", "có nên", "khởi kiện", "tranh chấp", "bồi thường", "phải làm sao", "thủ tục nào"]


def detect_unsupported_query(
    question: str,
    *,
    config: RoutingConfig,
    intent: str = "",
    intent_score: float = 0.0,
) -> dict[str, Any]:
    """Detect whether the query is outside the current legal QA system scope."""

    normalized = normalize_question(question).lower()
    unsupported_patterns = [item.lower() for item in config.unsupported_patterns]
    if unsupported_patterns and any(pattern in normalized for pattern in unsupported_patterns):
        return {
            "unsupported_query": True,
            "reason": "Câu hỏi có dấu hiệu nằm ngoài phạm vi hỗ trợ pháp luật của hệ thống.",
        }

    legal_indicators = [
        "luật",
        "nghị định",
        "thông tư",
        "điều",
        "khoản",
        "phạt",
        "thủ tục",
        "quy định",
        "khởi kiện",
        "khiếu nại",
        "nghĩa vụ",
        "quyền",
        "tranh chấp",
        "trách nhiệm",
        "đối tượng",
    ]
    if _is_direct_legal_question(question):
        return {"unsupported_query": False, "reason": ""}
    if intent and intent != "hoi_tinh_huong_thuc_te" and intent_score >= config.confidence_threshold:
        return {"unsupported_query": False, "reason": ""}

    if not any(token in normalized for token in legal_indicators) and len(normalized.split()) >= 4:
        return {
            "unsupported_query": True,
            "reason": "Câu hỏi chưa cho thấy ngữ cảnh pháp luật rõ ràng hoặc nằm ngoài phạm vi Bộ pháp điển điện tử.",
        }

    return {"unsupported_query": False, "reason": ""}


def _is_direct_legal_question(question: str) -> bool:
    normalized = normalize_question(question).lower()
    if ARTICLE_REFERENCE_PATTERN.search(normalized) or LAW_REFERENCE_PATTERN.search(normalized):
        return True
    return any(
        pattern in normalized
        for pattern in (
            "là gì",
            "khái niệm",
            "định nghĩa",
            "giải thích từ ngữ",
            "ai có trách nhiệm",
            "ai được",
            "đối tượng nào",
            "được hiểu là gì",
        )
    )


def _should_use_fast_path(question: str, *, intent: str, risk_level: str, config: RoutingConfig) -> bool:
    if not config.route_policy_flags.get("enable_fast_path", True):
        return False
    if risk_level != "low":
        return False

    normalized = normalize_question(question).lower()
    if any(token in normalized for token in COMPLEX_QUESTION_HINTS):
        return False

    fast_patterns = [item.lower() for item in config.fast_path_patterns]
    if intent == "hoi_dinh_nghia" and (not fast_patterns or any(pattern in normalized for pattern in fast_patterns)):
        return True

    return _is_direct_legal_question(question) and len(normalized.split()) <= 18


def _build_human_review_question(question: str, *, risk_level: str, intent: str) -> str:
    normalized = normalize_question(question).lower()
    if risk_level == "high" and any(token in normalized for token in ("khởi kiện", "kiện", "tranh chấp", "bồi thường")):
        return HUMAN_REVIEW_QUESTIONS[0]
    if intent == "hoi_tinh_huong_thuc_te":
        return HUMAN_REVIEW_QUESTIONS[1]
    return HUMAN_REVIEW_QUESTIONS[0]


def _is_local_high_risk(question: str) -> bool:
    normalized = normalize_question(question).lower()
    return any(token in normalized for token in HIGH_RISK_HINTS)


def route_node(
    state: Mapping[str, Any],
    *,
    routing_config: RoutingConfig | None = None,
    config_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """TV4 + TV6 routing node for the LangGraph agentic workflow."""

    resolved_logger = logger or LOGGER
    resolved_config = routing_config or load_routing_config(config_path)
    question = str(state.get("normalized_question") or state.get("question") or "").strip()

    intent_result = {
        "intent": str(state.get("intent") or "").strip(),
        "score": float(state.get("intent_score") or 0.0),
        "top_labels": list(state.get("top_labels") or state.get("top_intents") or []),
    }
    if not intent_result["intent"]:
        intent_result = classify_intent(question, config=resolved_config, logger=resolved_logger)

    clarify_result = {
        "need_clarify": bool(state.get("need_clarify") or False),
        "reason": str(state.get("clarify_reason") or ""),
        "clarify_question": str(state.get("clarify_question") or ""),
        "missing_slots": list(state.get("missing_slots") or []),
    }
    if not clarify_result["need_clarify"] and not clarify_result["reason"]:
        clarify_result = detect_clarify_need(question, config=resolved_config, logger=resolved_logger)

    risk_result = {
        "risk_level": str(state.get("risk_level") or "").strip(),
        "risk_reason": str(state.get("risk_reason") or ""),
        "human_review_recommended": bool(state.get("human_review_required") or False),
    }
    if not risk_result["risk_level"]:
        risk_result = tag_risk(
            question,
            intent=intent_result["intent"],
            config=resolved_config,
            logger=resolved_logger,
        )
    if _is_local_high_risk(question) and risk_result["risk_level"] != "high":
        risk_result = {
            "risk_level": "high",
            "risk_reason": "Câu hỏi liên quan tranh chấp hoặc quyết định pháp lý thực tế nên cần xác nhận phạm vi trả lời.",
            "human_review_recommended": True,
        }

    unsupported_result = detect_unsupported_query(
        question,
        config=resolved_config,
        intent=intent_result["intent"],
        intent_score=float(intent_result["score"]),
    )

    next_route = "legal-agent-path"
    route_reason = "Câu hỏi pháp luật hợp lệ, cần đi qua legal retrieval và reasoning path."
    human_review_required = bool(risk_result["human_review_recommended"])
    status = ""
    resume_kind = ""
    resume_question = ""
    execution_profile = "full"
    fast_path_enabled = False

    clarify_missing_slots = set(clarify_result.get("missing_slots") or [])

    if clarify_result["need_clarify"] and bool(clarify_missing_slots & CLARIFY_CRITICAL_SLOTS):
        next_route = "clarify-path"
        route_reason = "Câu hỏi còn mơ hồ hoặc thiếu dữ kiện pháp lý quan trọng để truy xuất chính xác."
        status = "waiting_user_input"
        resume_kind = "clarify"
        resume_question = str(clarify_result.get("clarify_question") or "").strip()
    elif unsupported_result["unsupported_query"]:
        next_route = "unsupported-path"
        route_reason = unsupported_result["reason"]
    elif risk_result["risk_level"] == "high" and resolved_config.route_policy_flags.get(
        "prefer_human_review_for_high_risk", True
    ):
        next_route = "human-review-path"
        route_reason = "Câu hỏi có rủi ro pháp lý cao, cần xác nhận phạm vi trả lời với người dùng trước khi tiếp tục."
        human_review_required = True
        status = "waiting_user_input"
        resume_kind = "human_review"
        resume_question = _build_human_review_question(
            question,
            risk_level=risk_result["risk_level"],
            intent=intent_result["intent"],
        )
    elif _should_use_fast_path(
        question,
        intent=intent_result["intent"],
        risk_level=risk_result["risk_level"],
        config=resolved_config,
    ):
        next_route = "fast-path"
        route_reason = "Câu hỏi định nghĩa hoặc tra cứu trực tiếp đơn giản, rủi ro thấp, có thể xử lý bằng fast-path."
        execution_profile = "fast"
        fast_path_enabled = True

    unsupported_query = unsupported_result["unsupported_query"] if next_route == "unsupported-path" else False

    return {
        "intent": intent_result["intent"],
        "intent_score": intent_result["score"],
        "top_intents": intent_result["top_labels"],
        "need_clarify": clarify_result["need_clarify"],
        "clarify_reason": clarify_result["reason"],
        "clarify_question": clarify_result["clarify_question"],
        "missing_slots": list(clarify_result.get("missing_slots") or []),
        "risk_level": risk_result["risk_level"],
        "risk_reason": risk_result["risk_reason"],
        "next_route": next_route,
        "human_review_required": human_review_required,
        "unsupported_query": unsupported_query,
        "route_reason": route_reason,
        "status": status,
        "response_status": status,
        "resume_kind": resume_kind,
        "resume_question": resume_question,
        "execution_profile": execution_profile,
        "fast_path_enabled": fast_path_enabled,
    }


__all__ = ["route_node", "detect_unsupported_query"]

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from src.tv4_router.intent_classifier import RoutingConfig, load_routing_config, normalize_question, tokenize_question

LOGGER = logging.getLogger(__name__)

AMBIGUOUS_REFERENCES = [
    "quy định này",
    "điều này",
    "khoản này",
    "trường hợp này",
    "việc này",
    "cái này",
    "như vậy",
]

ACTION_HINTS = [
    "vượt đèn đỏ",
    "đỗ xe",
    "đậu xe",
    "xây dựng",
    "chuyển nhượng",
    "mua bán",
    "đăng ký",
    "xin cấp",
    "khởi kiện",
    "khiếu nại",
    "ly hôn",
    "trộm cắp",
    "đánh nhau",
    "lấn chiếm",
    "vi phạm",
    "xử phạt",
]

SUBJECT_HINTS = [
    "tôi",
    "em",
    "cá nhân",
    "công dân",
    "người lao động",
    "người sử dụng lao động",
    "doanh nghiệp",
    "công ty",
    "hộ gia đình",
    "vợ",
    "chồng",
    "cha mẹ",
    "con",
]

DISPUTE_HINTS = [
    "tranh chấp đất đai",
    "tranh chấp lao động",
    "tranh chấp hợp đồng",
    "tranh chấp thừa kế",
    "ly hôn",
    "bồi thường",
    "đòi nợ",
    "ranh giới",
    "đất đai",
    "nhà ở",
]

LEGAL_DOCUMENT_HINTS = [
    "luật",
    "bộ luật",
    "nghị định",
    "thông tư",
    "quyết định",
    "nghị quyết",
    "bộ pháp điển",
]

ARTICLE_REFERENCE_PATTERN = re.compile(r"\b(điều|khoản|điểm)\s+\w+", re.IGNORECASE)

AMBIGUOUS_RULE_REFERENCE_PHRASES = [
    "quy định này",
    "điều này",
    "khoản này",
]

DEFINITION_LOOKUP_PHRASES = [
    "là gì",
    "khái niệm",
    "định nghĩa",
    "quyền của",
    "nghĩa vụ của",
    "trách nhiệm của",
    "ai có trách nhiệm",
    "đối tượng nào",
    "được hiểu là gì",
]

SLOT_QUESTIONS = {
    "hanh_vi_vi_pham": "Bạn muốn hỏi về hành vi vi phạm nào cụ thể?",
    "chu_the": "Quy định này đang áp dụng cho chủ thể nào cụ thể, ví dụ cá nhân, doanh nghiệp hay hộ gia đình?",
    "loai_tranh_chap": "Bạn có thể nói rõ loại tranh chấp hoặc vấn đề pháp lý mà bạn đang gặp phải không?",
    "van_ban_phap_luat": "Bạn đang muốn hỏi theo luật hoặc văn bản pháp luật nào cụ thể?",
    "tham_chieu_dieu_luat": "Bạn đang muốn hỏi theo điều hoặc khoản nào cụ thể?",
    "muc_tieu_tra_loi": "Bạn muốn hệ thống tập trung vào nội dung nào, ví dụ mức phạt, thủ tục, điều kiện hay căn cứ pháp luật?",
}

PRIORITY_ORDER = [
    "hanh_vi_vi_pham",
    "loai_tranh_chap",
    "van_ban_phap_luat",
    "tham_chieu_dieu_luat",
    "chu_the",
    "muc_tieu_tra_loi",
]


def _contains_any(text: str, candidates: list[str]) -> bool:
    lowered = text.lower()
    return any(candidate in lowered for candidate in candidates)


def _detect_missing_slots(question: str, config: RoutingConfig) -> list[str]:
    normalized = normalize_question(question).lower()
    tokens = tokenize_question(normalized)
    missing_slots: list[str] = []

    is_too_short = len(normalized) < config.clarify_min_length or len(tokens) < 3

    asks_penalty = any(phrase in normalized for phrase in ("mức phạt", "phạt bao nhiêu", "xử phạt bao nhiêu"))
    asks_ambiguous_rule_reference = any(phrase in normalized for phrase in AMBIGUOUS_RULE_REFERENCE_PHRASES)
    asks_definition_lookup = any(phrase in normalized for phrase in DEFINITION_LOOKUP_PHRASES)
    asks_should_sue = any(phrase in normalized for phrase in ("có nên kiện", "có nên khởi kiện", "nên khởi kiện"))
    asks_procedure = any(phrase in normalized for phrase in ("thủ tục", "hồ sơ", "cách làm", "trình tự"))

    has_action = _contains_any(normalized, ACTION_HINTS)
    has_subject = _contains_any(normalized, SUBJECT_HINTS)
    has_dispute = _contains_any(normalized, DISPUTE_HINTS)
    has_legal_document = _contains_any(normalized, LEGAL_DOCUMENT_HINTS)
    has_article_reference = bool(ARTICLE_REFERENCE_PATTERN.search(normalized))
    has_ambiguous_reference = _contains_any(normalized, AMBIGUOUS_REFERENCES)

    if asks_penalty and not has_action:
        missing_slots.append("hanh_vi_vi_pham")

    if asks_should_sue and not has_dispute:
        missing_slots.append("loai_tranh_chap")

    if asks_ambiguous_rule_reference and not has_legal_document:
        missing_slots.append("van_ban_phap_luat")

    if has_ambiguous_reference and not has_article_reference and has_legal_document:
        missing_slots.append("tham_chieu_dieu_luat")

    if asks_procedure and not has_subject:
        missing_slots.append("chu_the")

    has_specific_gap = bool(missing_slots)
    has_any_legal_anchor = any(
        (
            asks_penalty,
            asks_ambiguous_rule_reference,
            asks_definition_lookup,
            asks_should_sue,
            asks_procedure,
            has_action,
            has_dispute,
            has_legal_document,
            has_article_reference,
        )
    )
    if not has_specific_gap and (is_too_short or not has_any_legal_anchor):
        missing_slots.append("muc_tieu_tra_loi")

    # Deduplicate while keeping stable priority.
    ordered_missing = [slot for slot in PRIORITY_ORDER if slot in missing_slots]
    return ordered_missing


def _build_clarify_question(missing_slots: list[str]) -> str:
    if not missing_slots:
        return "Bạn có thể bổ sung thêm bối cảnh pháp lý hoặc nêu rõ vấn đề cụ thể mà bạn muốn hỏi không?"
    focus_slot = missing_slots[0]
    return SLOT_QUESTIONS.get(
        focus_slot,
        "Bạn có thể bổ sung thêm bối cảnh pháp lý hoặc nêu rõ vấn đề cụ thể mà bạn muốn hỏi không?",
    )


def detect_clarify_need(
    question: str,
    *,
    config: RoutingConfig | None = None,
    config_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Detect whether the user question needs clarification before routing."""

    resolved_logger = logger or LOGGER
    resolved_config = config or load_routing_config(config_path)
    normalized = normalize_question(question)
    missing_slots = _detect_missing_slots(normalized, resolved_config)
    reason = missing_slots[0] if missing_slots else ""

    if missing_slots:
        resolved_logger.info("clarify_detector missing_slots=%s question=%s", missing_slots, normalized)

    return {
        "need_clarify": bool(missing_slots),
        "reason": reason,
        "clarify_question": _build_clarify_question(missing_slots) if missing_slots else "",
        "missing_slots": missing_slots,
    }


__all__ = ["detect_clarify_need"]

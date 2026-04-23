from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.tv4_router.intent_classifier import RoutingConfig, load_routing_config, normalize_question

LOGGER = logging.getLogger(__name__)

HIGH_RISK_DEFAULT_PATTERNS = [
    "khởi kiện",
    "tranh chấp",
    "tố tụng",
    "khiếu nại",
    "tố cáo",
    "bồi thường",
    "bắt giam",
    "đi tù",
    "thu hồi đất",
    "sa thải",
    "ly hôn",
    "thừa kế",
    "trốn thuế",
]
MEDIUM_RISK_PATTERNS = [
    "xử phạt",
    "mức phạt",
    "phạt bao nhiêu",
    "nghĩa vụ",
    "thủ tục",
    "đăng ký",
    "trách nhiệm",
]
CERTAINTY_PATTERNS = [
    "chắc chắn",
    "cam kết",
    "khẳng định",
    "100%",
    "có bị",
]


def tag_risk(
    question: str,
    intent: str | None = None,
    *,
    config: RoutingConfig | None = None,
    config_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Tag the legal risk level of a Vietnamese user question."""

    _ = logger or LOGGER
    resolved_config = config or load_routing_config(config_path)
    normalized = normalize_question(question).lower()
    configured_high_risk = [item.lower() for item in resolved_config.high_risk_patterns]
    high_patterns = configured_high_risk or HIGH_RISK_DEFAULT_PATTERNS

    risk_level = "low"
    reason = "Câu hỏi pháp lý thông thường, có thể xử lý tự động."
    human_review_recommended = False

    if any(pattern in normalized for pattern in high_patterns):
        risk_level = "high"
        reason = "Câu hỏi liên quan tranh chấp, xử lý pháp lý nghiêm trọng hoặc quyền lợi pháp lý nhạy cảm."
        human_review_recommended = True
    elif any(pattern in normalized for pattern in CERTAINTY_PATTERNS) and any(
        pattern in normalized for pattern in MEDIUM_RISK_PATTERNS
    ):
        risk_level = "high"
        reason = "Câu hỏi yêu cầu kết luận chắc chắn trong vấn đề pháp lý có thể ảnh hưởng lớn tới quyền/lợi ích."
        human_review_recommended = True
    elif any(pattern in normalized for pattern in MEDIUM_RISK_PATTERNS):
        risk_level = "medium"
        reason = "Câu hỏi có yếu tố thủ tục, nghĩa vụ hoặc chế tài nên cần thận trọng khi trả lời."
    elif (intent or "") in {"hoi_tinh_huong_thuc_te", "hoi_muc_phat"}:
        risk_level = "medium"
        reason = "Câu hỏi thuộc nhóm tình huống thực tế hoặc mức phạt, cần bám sát nguồn pháp lý."

    return {
        "risk_level": risk_level,
        "risk_reason": reason,
        "human_review_recommended": human_review_recommended,
    }


__all__ = ["tag_risk"]

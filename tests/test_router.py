from __future__ import annotations

from src.tv4_router.clarify_detector import detect_clarify_need
from src.tv4_router.intent_classifier import classify_intent
from src.tv4_router.risk_tagger import tag_risk
from src.tv4_router.route_node import route_node


def test_intent_classifier_basic_penalty() -> None:
    result = classify_intent("Mức phạt khi đỗ xe sai nơi quy định là bao nhiêu?")
    assert result["intent"] == "hoi_muc_phat"
    assert result["score"] > 0.5


def test_clarify_detector_for_ambiguous_question() -> None:
    result = detect_clarify_need("Đúng không?")
    assert result["need_clarify"] is True
    assert result["reason"] in {"too_short", "ambiguous_reference"}
    assert result["clarify_question"]


def test_risk_tagger_high_risk_question() -> None:
    result = tag_risk("Tôi có nên khởi kiện tranh chấp đất đai với hàng xóm không?", intent="hoi_tinh_huong_thuc_te")
    assert result["risk_level"] == "high"
    assert result["human_review_recommended"] is True


def test_route_to_clarify_path() -> None:
    state = {
        "question": "Mức phạt là bao nhiêu?",
        "normalized_question": "Mức phạt là bao nhiêu?",
    }
    result = route_node(state)
    assert result["next_route"] == "clarify-path"
    assert result["need_clarify"] is True


def test_route_to_legal_agent_path() -> None:
    state = {
        "question": "Theo Luật Thanh niên, thanh niên là gì?",
        "normalized_question": "Theo Luật Thanh niên, thanh niên là gì?",
    }
    result = route_node(state)
    assert result["next_route"] in {"legal-agent-path", "fast-path"}
    assert result["unsupported_query"] is False


def test_route_to_unsupported_path() -> None:
    state = {
        "question": "Hôm nay thời tiết ở Hà Nội thế nào?",
        "normalized_question": "Hôm nay thời tiết ở Hà Nội thế nào?",
    }
    result = route_node(state)
    assert result["next_route"] == "unsupported-path"
    assert result["unsupported_query"] is True


def test_route_to_human_review_path() -> None:
    state = {
        "question": "Tôi có nên khởi kiện tranh chấp đất đai và yêu cầu bồi thường không?",
        "normalized_question": "Tôi có nên khởi kiện tranh chấp đất đai và yêu cầu bồi thường không?",
    }
    result = route_node(state)
    assert result["next_route"] == "human-review-path"
    assert result["human_review_required"] is True


def test_route_uses_existing_state_fields() -> None:
    state = {
        "question": "Thủ tục đăng ký khai sinh cần những gì?",
        "normalized_question": "Thủ tục đăng ký khai sinh cần những gì?",
        "intent": "hoi_thu_tuc_hanh_chinh",
        "intent_score": 0.88,
        "need_clarify": False,
        "risk_level": "medium",
    }
    result = route_node(state)
    assert result["intent"] == "hoi_thu_tuc_hanh_chinh"
    assert result["intent_score"] == 0.88
    assert result["next_route"] == "legal-agent-path"

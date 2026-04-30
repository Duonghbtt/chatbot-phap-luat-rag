from __future__ import annotations

import json
from typing import Any

import pytest

import src.tv4_router.intent_classifier as intent_classifier_module
from src.tv4_router.clarify_detector import detect_clarify_need
from src.tv4_router.intent_classifier import (
    OllamaLLMIntentClassifier,
    RoutingConfig,
    classify_intent,
    get_intent_classifier,
)
from src.tv4_router.risk_tagger import tag_risk
from src.tv4_router.route_node import route_node


class _FakeHTTPResponse:
    def __init__(self, payload: str) -> None:
        self._payload = payload.encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


@pytest.fixture()
def rule_config() -> RoutingConfig:
    return RoutingConfig(model_type="rule_based")


@pytest.fixture()
def llm_config() -> RoutingConfig:
    return RoutingConfig(
        model_type="llm_based",
        llm_provider="ollama",
        llm_model="qwen2.5:7b",
        llm_base_url="http://localhost:11434",
        llm_timeout_seconds=20,
        llm_temperature=0.0,
        llm_fallback_to_rule_based=True,
    )


def _mock_ollama_chat(monkeypatch: pytest.MonkeyPatch, message_content: str) -> None:
    response_body = json.dumps({"message": {"content": message_content}}, ensure_ascii=False)

    def fake_urlopen(request: Any, timeout: int = 0) -> _FakeHTTPResponse:
        return _FakeHTTPResponse(response_body)

    monkeypatch.setattr(intent_classifier_module.urllib.request, "urlopen", fake_urlopen)


def test_intent_classifier_basic_penalty(rule_config: RoutingConfig) -> None:
    result = classify_intent("Mức phạt khi đỗ xe sai nơi quy định là bao nhiêu?", config=rule_config)
    assert result["intent"] == "hoi_muc_phat"
    assert result["score"] > 0.5
    assert result["classifier_type"] == "rule_based"


def test_llm_classifier_parses_valid_json(
    monkeypatch: pytest.MonkeyPatch,
    llm_config: RoutingConfig,
) -> None:
    _mock_ollama_chat(
        monkeypatch,
        '{"intent":"hoi_dinh_nghia","score":0.9,"top_labels":[{"label":"hoi_dinh_nghia","score":0.9}]}',
    )
    classifier = OllamaLLMIntentClassifier(llm_config)

    result = classifier.classify("Điều 1 Luật Thanh niên quy định gì?")

    assert result["intent"] == "hoi_dinh_nghia"
    assert result["classifier_type"] == "llm_based"
    assert result["top_labels"][0]["label"] == "hoi_dinh_nghia"


def test_llm_classifier_parses_json_inside_markdown(
    monkeypatch: pytest.MonkeyPatch,
    llm_config: RoutingConfig,
) -> None:
    _mock_ollama_chat(
        monkeypatch,
        '```json\n{"intent":"hoi_dinh_nghia","score":0.91,"top_labels":[{"label":"hoi_dinh_nghia","score":0.91}]}\n```',
    )
    classifier = OllamaLLMIntentClassifier(llm_config)

    result = classifier.classify("Điều 1 Luật Thanh niên quy định gì?")

    assert result["intent"] == "hoi_dinh_nghia"
    assert result["classifier_type"] == "llm_based"


def test_llm_classifier_fallbacks_on_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
    llm_config: RoutingConfig,
) -> None:
    _mock_ollama_chat(monkeypatch, "toi se tra loi sau")
    classifier = OllamaLLMIntentClassifier(llm_config)

    result = classifier.classify("Mức phạt khi đỗ xe sai nơi quy định là bao nhiêu?")

    assert result["intent"] == "hoi_muc_phat"
    assert result["classifier_type"] == "llm_fallback_rule_based"


def test_llm_classifier_fallbacks_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
    llm_config: RoutingConfig,
) -> None:
    def fake_urlopen(request: Any, timeout: int = 0) -> _FakeHTTPResponse:
        raise TimeoutError("timed out")

    monkeypatch.setattr(intent_classifier_module.urllib.request, "urlopen", fake_urlopen)
    classifier = OllamaLLMIntentClassifier(llm_config)

    result = classifier.classify("Mức phạt khi đỗ xe sai nơi quy định là bao nhiêu?")

    assert result["intent"] == "hoi_muc_phat"
    assert result["classifier_type"] == "llm_fallback_rule_based"


def test_get_intent_classifier_llm_based(llm_config: RoutingConfig) -> None:
    classifier = get_intent_classifier(llm_config)
    assert isinstance(classifier, OllamaLLMIntentClassifier)


def test_clarify_detector_for_ambiguous_question() -> None:
    result = detect_clarify_need("Đúng không?")
    assert result["need_clarify"] is True
    assert result["reason"] in {"too_short", "ambiguous_reference", "muc_tieu_tra_loi"}
    assert result["clarify_question"]


def test_risk_tagger_high_risk_question() -> None:
    result = tag_risk(
        "Tôi có nên khởi kiện tranh chấp đất đai với hàng xóm không?",
        intent="hoi_tinh_huong_thuc_te",
    )
    assert result["risk_level"] == "high"
    assert result["human_review_recommended"] is True


def test_route_to_clarify_path(rule_config: RoutingConfig) -> None:
    state = {
        "question": "Mức phạt là bao nhiêu?",
        "normalized_question": "Mức phạt là bao nhiêu?",
    }
    result = route_node(state, routing_config=rule_config)
    assert result["next_route"] == "clarify-path"
    assert result["need_clarify"] is True


def test_route_to_legal_agent_path(rule_config: RoutingConfig) -> None:
    state = {
        "question": "Theo Luật Thanh niên, thanh niên là gì?",
        "normalized_question": "Theo Luật Thanh niên, thanh niên là gì?",
    }
    result = route_node(state, routing_config=rule_config)
    assert result["next_route"] in {"legal-agent-path", "fast-path"}
    assert result["unsupported_query"] is False


def test_definition_question_without_explicit_law_does_not_force_clarify(rule_config: RoutingConfig) -> None:
    state = {
        "question": "Quyền của thanh niên là gì?",
        "normalized_question": "Quyền của thanh niên là gì?",
    }
    result = route_node(state, routing_config=rule_config)
    assert result["next_route"] in {"legal-agent-path", "fast-path"}
    assert result["need_clarify"] is False
    assert result["clarify_question"] == ""


def test_route_to_unsupported_path(rule_config: RoutingConfig) -> None:
    state = {
        "question": "Hôm nay thời tiết ở Hà Nội thế nào?",
        "normalized_question": "Hôm nay thời tiết ở Hà Nội thế nào?",
    }
    result = route_node(state, routing_config=rule_config)
    assert result["next_route"] == "unsupported-path"
    assert result["unsupported_query"] is True
    assert result["need_clarify"] is False
    assert result["clarify_question"] == ""


def test_route_to_human_review_path(rule_config: RoutingConfig) -> None:
    state = {
        "question": "Tôi có nên khởi kiện tranh chấp đất đai và yêu cầu bồi thường không?",
        "normalized_question": "Tôi có nên khởi kiện tranh chấp đất đai và yêu cầu bồi thường không?",
    }
    result = route_node(state, routing_config=rule_config)
    assert result["next_route"] == "human-review-path"
    assert result["human_review_required"] is True


def test_route_uses_existing_state_fields(rule_config: RoutingConfig) -> None:
    state = {
        "question": "Thủ tục đăng ký khai sinh cần những gì?",
        "normalized_question": "Thủ tục đăng ký khai sinh cần những gì?",
        "intent": "hoi_thu_tuc_hanh_chinh",
        "intent_score": 0.88,
        "need_clarify": False,
        "clarify_reason": "preserved_from_state",
        "risk_level": "medium",
    }
    result = route_node(state, routing_config=rule_config)
    assert result["intent"] == "hoi_thu_tuc_hanh_chinh"
    assert result["intent_score"] == 0.88
    assert result["next_route"] == "legal-agent-path"

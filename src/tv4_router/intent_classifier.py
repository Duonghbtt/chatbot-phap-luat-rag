from __future__ import annotations

import json
import logging
import os
import pickle
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

LOGGER = logging.getLogger(__name__)
DEFAULT_ROUTING_CONFIG_PATH = Path("configs/routing.yaml")
ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::([^}]*))?\}")
TOKEN_PATTERN = re.compile(r"[0-9A-Za-zÀ-ỹ]+")

DEFAULT_INTENT_LABELS = [
    "hoi_dinh_nghia",
    "hoi_muc_phat",
    "hoi_thu_tuc_hanh_chinh",
    "hoi_so_sanh_luat",
    "hoi_tinh_huong_thuc_te",
]

DEFAULT_RULE_KEYWORDS: dict[str, list[str]] = {
    "hoi_dinh_nghia": [
        "là gì",
        "khái niệm",
        "định nghĩa",
        "giải thích từ ngữ",
        "quy định là gì",
        "hiểu thế nào",
    ],
    "hoi_muc_phat": [
        "mức phạt",
        "phạt bao nhiêu",
        "xử phạt",
        "bị phạt",
        "chế tài",
        "vi phạm",
    ],
    "hoi_thu_tuc_hanh_chinh": [
        "thủ tục",
        "hồ sơ",
        "đăng ký",
        "xin cấp",
        "cần làm gì",
        "trình tự",
        "cách thực hiện",
    ],
    "hoi_so_sanh_luat": [
        "khác nhau",
        "so sánh",
        "phân biệt",
        "giống nhau",
        "hay",
        "với",
    ],
    "hoi_tinh_huong_thuc_te": [
        "trường hợp",
        "nếu",
        "tôi",
        "em",
        "gia đình",
        "có bị",
        "phải làm sao",
        "thì sao",
    ],
}

LLM_INTENT_SYSTEM_PROMPT = """Bạn là bộ phân loại intent cho hệ thống hỏi đáp pháp luật Việt Nam.
Nhiệm vụ của bạn chỉ là phân loại câu hỏi người dùng, không trả lời nội dung pháp luật.

Chỉ được chọn một trong các intent sau:
- hoi_dinh_nghia: hỏi khái niệm, định nghĩa, điều luật quy định gì, đối tượng/quyền/nghĩa vụ là gì.
- hoi_muc_phat: hỏi mức phạt, xử phạt, chế tài, bị phạt bao nhiêu, vi phạm bị xử lý thế nào.
- hoi_thu_tuc_hanh_chinh: hỏi thủ tục, hồ sơ, trình tự, đăng ký, xin cấp, điều kiện thực hiện.
- hoi_so_sanh_luat: hỏi so sánh, phân biệt, khác nhau, giống nhau giữa quy định/khái niệm/văn bản.
- hoi_tinh_huong_thuc_te: hỏi theo tình huống cá nhân/thực tế, có từ như tôi/em/gia đình/công ty/nếu/trường hợp/phải làm sao/có bị không.

Quy tắc phân loại:
- Nếu câu hỏi hỏi trực tiếp "Điều X ... quy định gì", "Luật ... quy định gì", "quyền là gì", "nghĩa vụ là gì" thì chọn hoi_dinh_nghia.
- Nếu câu hỏi có "mức phạt", "phạt bao nhiêu", "xử phạt", "vi phạm bị xử lý thế nào" thì chọn hoi_muc_phat.
- Nếu câu hỏi có "thủ tục", "hồ sơ", "trình tự", "đăng ký", "xin cấp", "điều kiện thực hiện" thì chọn hoi_thu_tuc_hanh_chinh.
- Nếu câu hỏi có "khác nhau", "so sánh", "phân biệt", "giống nhau" thì chọn hoi_so_sanh_luat.
- Nếu câu hỏi gắn với tình huống cá nhân, ví dụ "tôi", "em", "gia đình tôi", "công ty tôi", "nếu", "trường hợp", "phải làm sao", "có bị không", thì chọn hoi_tinh_huong_thuc_te.
- Nếu câu hỏi vừa có tình huống cá nhân vừa hỏi mức phạt, ưu tiên hoi_muc_phat.
- Nếu câu hỏi vừa có tình huống cá nhân vừa hỏi thủ tục, ưu tiên hoi_thu_tuc_hanh_chinh.
- Không suy diễn ngoài câu hỏi.
- Không trả lời nội dung pháp luật.

Chỉ trả về JSON hợp lệ, không markdown, không giải thích ngoài JSON:
{
  "intent": "...",
  "score": 0.0,
  "top_labels": [
    {"label": "...", "score": 0.0},
    {"label": "...", "score": 0.0},
    {"label": "...", "score": 0.0}
  ],
  "reason": "lý do rất ngắn"
}"""


@dataclass(slots=True, frozen=True)
class RoutingConfig:
    """Configuration shared by TV4 router-related modules."""

    intent_labels: list[str] = field(default_factory=lambda: list(DEFAULT_INTENT_LABELS))
    confidence_threshold: float = 0.55
    clarify_min_length: int = 12
    unsupported_patterns: list[str] = field(default_factory=list)
    high_risk_patterns: list[str] = field(default_factory=list)
    fast_path_patterns: list[str] = field(default_factory=list)
    route_policy_flags: dict[str, bool] = field(default_factory=dict)
    rule_keywords: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_RULE_KEYWORDS))
    model_path: str = ""
    model_type: str = "rule_based"
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:7b"
    llm_base_url: str = "http://localhost:11434"
    llm_timeout_seconds: int = 20
    llm_temperature: float = 0.0
    llm_fallback_to_rule_based: bool = True


def _load_yaml_module() -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("PyYAML is required to load configs/routing.yaml.") from exc
    return yaml


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _substitute_env_placeholders(raw_text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(key, default)

    return ENV_PATTERN.sub(replace, raw_text)


def normalize_question(question: str) -> str:
    """Normalize whitespace for legal Vietnamese user questions."""

    return re.sub(r"\s+", " ", (question or "").strip())


def tokenize_question(question: str) -> list[str]:
    """Tokenize Vietnamese legal text for heuristic matching."""

    normalized = normalize_question(question).lower()
    return [token for token in TOKEN_PATTERN.findall(normalized) if token]


def load_routing_config(config_path: str | Path | None = None) -> RoutingConfig:
    """Load routing config from YAML, with safe defaults."""

    yaml = _load_yaml_module()
    resolved_path = Path(config_path or DEFAULT_ROUTING_CONFIG_PATH).resolve()
    if not resolved_path.exists():
        return RoutingConfig()

    raw_text = resolved_path.read_text(encoding="utf-8")
    config_data = yaml.safe_load(_substitute_env_placeholders(raw_text)) or {}
    if not isinstance(config_data, Mapping):
        raise ValueError(f"Invalid routing config structure in {resolved_path}")

    route_policy_flags = {
        str(key): _coerce_bool(value)
        for key, value in dict(config_data.get("route_policy_flags") or {}).items()
    }
    raw_keywords = dict(config_data.get("rule_keywords") or {})
    merged_keywords = {
        label: [str(keyword) for keyword in raw_keywords.get(label, DEFAULT_RULE_KEYWORDS.get(label, []))]
        for label in DEFAULT_INTENT_LABELS
    }

    return RoutingConfig(
        intent_labels=[str(item) for item in config_data.get("intent_labels", DEFAULT_INTENT_LABELS)],
        confidence_threshold=float(config_data.get("confidence_threshold", 0.55)),
        clarify_min_length=int(config_data.get("clarify_min_length", 12)),
        unsupported_patterns=[str(item) for item in config_data.get("unsupported_patterns", [])],
        high_risk_patterns=[str(item) for item in config_data.get("high_risk_patterns", [])],
        fast_path_patterns=[str(item) for item in config_data.get("fast_path_patterns", [])],
        route_policy_flags=route_policy_flags,
        rule_keywords=merged_keywords,
        model_path=str(config_data.get("model_path") or ""),
        model_type=str(config_data.get("model_type") or "rule_based"),
        llm_provider=str(config_data.get("llm_provider") or "ollama"),
        llm_model=str(config_data.get("llm_model") or "qwen2.5:7b"),
        llm_base_url=str(config_data.get("llm_base_url") or "http://localhost:11434"),
        llm_timeout_seconds=int(config_data.get("llm_timeout_seconds", 20)),
        llm_temperature=float(config_data.get("llm_temperature", 0.0)),
        llm_fallback_to_rule_based=_coerce_bool(config_data.get("llm_fallback_to_rule_based", True)),
    )


def _strip_code_fences(text: str) -> str:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _extract_first_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fences(text)
    if not cleaned:
        raise ValueError("LLM returned empty content for intent classification.")

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, Mapping):
        return dict(parsed)

    in_string = False
    escaped = False
    depth = 0
    start_index: int | None = None
    for index, char in enumerate(cleaned):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
            continue
        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_index is not None:
                candidate = cleaned[start_index : index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    start_index = None
                    continue
                if isinstance(parsed, Mapping):
                    return dict(parsed)
    raise ValueError("Unable to parse a valid JSON object from LLM response.")


def _coerce_score(value: Any, *, default: float = 0.0) -> float:
    try:
        return min(0.99, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _coerce_top_labels(
    raw_top_labels: Any,
    *,
    allowed_labels: Sequence[str],
    fallback_intent: str,
    fallback_score: float,
) -> list[dict[str, float | str]]:
    allowed = {str(label) for label in allowed_labels}
    normalized_items: list[dict[str, float | str]] = []
    seen_labels: set[str] = set()

    if isinstance(raw_top_labels, Sequence) and not isinstance(raw_top_labels, (str, bytes, bytearray)):
        for item in raw_top_labels:
            if not isinstance(item, Mapping):
                continue
            label = str(item.get("label") or "").strip()
            if label not in allowed or label in seen_labels:
                continue
            score = round(_coerce_score(item.get("score"), default=0.0), 4)
            normalized_items.append({"label": label, "score": score})
            seen_labels.add(label)
            if len(normalized_items) >= 3:
                break

    if fallback_intent not in seen_labels:
        normalized_items.insert(0, {"label": fallback_intent, "score": round(fallback_score, 4)})
        normalized_items = normalized_items[:3]

    if not normalized_items:
        return [{"label": fallback_intent, "score": round(fallback_score, 4)}]
    return normalized_items


class BaseIntentClassifier(ABC):
    """Stable intent classification interface for TV4."""

    def __init__(self, config: RoutingConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or LOGGER

    @abstractmethod
    def classify(self, question: str) -> dict[str, Any]:
        """Classify a legal Vietnamese question into one of the supported intent labels."""


class RuleBasedIntentClassifier(BaseIntentClassifier):
    """Keyword and pattern-based baseline classifier that always works locally."""

    def _score_label(self, question: str, label: str) -> float:
        normalized = normalize_question(question).lower()
        score = 0.0
        for keyword in self.config.rule_keywords.get(label, []):
            keyword_lower = keyword.lower()
            if keyword_lower in normalized:
                score += 1.0
        if label == "hoi_dinh_nghia" and normalized.endswith("là gì?"):
            score += 1.2
        if label == "hoi_muc_phat" and any(token in normalized for token in ("bao nhiêu", "mức phạt", "xử phạt")):
            score += 1.2
        if label == "hoi_thu_tuc_hanh_chinh" and any(
            token in normalized for token in ("thủ tục", "hồ sơ", "đăng ký", "xin")
        ):
            score += 1.2
        if label == "hoi_so_sanh_luat" and "khác nhau" in normalized:
            score += 1.2
        if label == "hoi_tinh_huong_thuc_te" and any(
            token in normalized for token in ("nếu", "trường hợp", "tôi", "gia đình", "có bị")
        ):
            score += 1.0
        return score

    def classify(self, question: str) -> dict[str, Any]:
        normalized = normalize_question(question)
        label_scores = [(label, self._score_label(normalized, label)) for label in self.config.intent_labels]
        label_scores.sort(key=lambda item: item[1], reverse=True)
        best_label, best_score = label_scores[0]
        total_score = sum(score for _, score in label_scores)

        if best_score <= 0:
            best_label = "hoi_tinh_huong_thuc_te"
            normalized_score = 0.4
        else:
            normalized_score = min(0.99, max(0.35, best_score / max(total_score, best_score)))

        top_labels = [
            {
                "label": label,
                "score": round(min(0.99, max(0.0, score / max(total_score or 1.0, 1.0))), 4),
            }
            for label, score in label_scores[:3]
        ]
        return {
            "intent": best_label,
            "score": round(normalized_score, 4),
            "top_labels": top_labels,
            "classifier_type": "rule_based",
        }


class PickleModelIntentClassifier(BaseIntentClassifier):
    """Optional model-based classifier loaded from a local pickle file."""

    def __init__(self, config: RoutingConfig, logger: logging.Logger | None = None) -> None:
        super().__init__(config=config, logger=logger)
        model_path = Path(config.model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Intent model not found: {model_path}")
        with model_path.open("rb") as handle:
            self.model = pickle.load(handle)
        self.labels = getattr(self.model, "classes_", config.intent_labels)

    def classify(self, question: str) -> dict[str, Any]:
        normalized = normalize_question(question)
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba([normalized])[0]
            scored = sorted(
                ((str(label), float(score)) for label, score in zip(self.labels, probabilities)),
                key=lambda item: item[1],
                reverse=True,
            )
            best_label, best_score = scored[0]
            top_labels = [{"label": label, "score": round(score, 4)} for label, score in scored[:3]]
            return {
                "intent": best_label,
                "score": round(best_score, 4),
                "top_labels": top_labels,
                "classifier_type": "model_based",
            }

        prediction = self.model.predict([normalized])[0]
        return {
            "intent": str(prediction),
            "score": 0.7,
            "top_labels": [{"label": str(prediction), "score": 0.7}],
            "classifier_type": "model_based",
        }


class OllamaLLMIntentClassifier(BaseIntentClassifier):
    """LLM-first intent classifier backed by Ollama chat completions."""

    def __init__(self, config: RoutingConfig, logger: logging.Logger | None = None) -> None:
        super().__init__(config=config, logger=logger)
        provider = str(config.llm_provider or "").strip().lower()
        if provider != "ollama":
            raise ValueError(f"Unsupported llm_provider for intent classification: {config.llm_provider}")
        self._fallback_classifier = RuleBasedIntentClassifier(config=config, logger=logger)

    def _build_user_prompt(self, question: str) -> str:
        return (
            "Câu hỏi người dùng:\n"
            f"\"{normalize_question(question)}\"\n\n"
            "Hãy phân loại intent của câu hỏi trên.\n"
            "Chỉ trả về JSON hợp lệ."
        )

    def _post_ollama_chat(self, question: str) -> dict[str, Any]:
        payload = {
            "model": self.config.llm_model,
            "messages": [
                {"role": "system", "content": LLM_INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_prompt(question)},
            ],
            "stream": False,
            "options": {
                "temperature": self.config.llm_temperature,
            },
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        base_url = str(self.config.llm_base_url or "http://localhost:11434").rstrip("/")
        request = urllib.request.Request(
            url=f"{base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.llm_timeout_seconds) as response:
            raw_response = response.read().decode("utf-8")
        parsed_response = json.loads(raw_response)
        if not isinstance(parsed_response, Mapping):
            raise ValueError("Ollama response must be a JSON object.")
        return dict(parsed_response)

    def _normalize_llm_result(self, parsed_json: Mapping[str, Any]) -> dict[str, Any]:
        intent = str(parsed_json.get("intent") or "").strip()
        if intent not in self.config.intent_labels:
            raise ValueError(f"LLM returned unsupported intent label: {intent!r}")
        score = round(_coerce_score(parsed_json.get("score"), default=0.0), 4)
        top_labels = _coerce_top_labels(
            parsed_json.get("top_labels"),
            allowed_labels=self.config.intent_labels,
            fallback_intent=intent,
            fallback_score=score,
        )
        return {
            "intent": intent,
            "score": score,
            "top_labels": top_labels,
            "classifier_type": "llm_based",
        }

    def _fallback_or_raise(self, question: str, exc: Exception) -> dict[str, Any]:
        if not self.config.llm_fallback_to_rule_based:
            raise exc
        self.logger.warning(
            "LLM intent classification failed, falling back to rule-based classifier: %s",
            exc,
        )
        fallback_result = self._fallback_classifier.classify(question)
        fallback_result["classifier_type"] = "llm_fallback_rule_based"
        return fallback_result

    def classify(self, question: str) -> dict[str, Any]:
        normalized_question = normalize_question(question)
        try:
            response_json = self._post_ollama_chat(normalized_question)
            message_payload = dict(response_json.get("message") or {})
            raw_content = str(message_payload.get("content") or "").strip()
            if not raw_content:
                raise ValueError("Ollama response missing message.content.")
            parsed_json = _extract_first_json_object(raw_content)
            return self._normalize_llm_result(parsed_json)
        except Exception as exc:
            return self._fallback_or_raise(normalized_question, exc)


def get_intent_classifier(
    config: RoutingConfig | None = None,
    *,
    config_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> BaseIntentClassifier:
    """Return the active intent classifier, falling back safely to rule-based mode."""

    resolved_config = config or load_routing_config(config_path)
    resolved_logger = logger or LOGGER
    model_type = str(resolved_config.model_type or "rule_based").strip().lower()

    if model_type == "llm_based":
        try:
            return OllamaLLMIntentClassifier(resolved_config, logger=resolved_logger)
        except Exception as exc:
            if resolved_config.llm_fallback_to_rule_based:
                resolved_logger.warning(
                    "LLM intent classifier unavailable, falling back to rule-based: %s",
                    exc,
                )
                return RuleBasedIntentClassifier(resolved_config, logger=resolved_logger)
            raise

    if model_type == "model_based" and resolved_config.model_path:
        try:
            return PickleModelIntentClassifier(resolved_config, logger=resolved_logger)
        except Exception as exc:  # pragma: no cover - optional local model.
            resolved_logger.warning(
                "Model-based intent classifier unavailable, falling back to rule-based: %s",
                exc,
            )
    return RuleBasedIntentClassifier(resolved_config, logger=resolved_logger)


def classify_intent(
    question: str,
    *,
    config: RoutingConfig | None = None,
    config_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Stable public interface for intent classification."""

    classifier = get_intent_classifier(config, config_path=config_path, logger=logger)
    return classifier.classify(question)


__all__ = [
    "DEFAULT_ROUTING_CONFIG_PATH",
    "LLM_INTENT_SYSTEM_PROMPT",
    "OllamaLLMIntentClassifier",
    "RoutingConfig",
    "_extract_first_json_object",
    "_strip_code_fences",
    "classify_intent",
    "get_intent_classifier",
    "load_routing_config",
    "normalize_question",
    "tokenize_question",
]

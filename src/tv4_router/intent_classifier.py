from __future__ import annotations

import logging
import os
import pickle
import re
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
    )


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
        if label == "hoi_thu_tuc_hanh_chinh" and any(token in normalized for token in ("thủ tục", "hồ sơ", "đăng ký", "xin")):
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


def get_intent_classifier(
    config: RoutingConfig | None = None,
    *,
    config_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> BaseIntentClassifier:
    """Return the active intent classifier, falling back safely to rule-based mode."""

    resolved_config = config or load_routing_config(config_path)
    resolved_logger = logger or LOGGER

    if resolved_config.model_type.lower() == "model_based" and resolved_config.model_path:
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
    "RoutingConfig",
    "classify_intent",
    "get_intent_classifier",
    "load_routing_config",
    "normalize_question",
    "tokenize_question",
]

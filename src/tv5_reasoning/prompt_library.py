from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

LOGGER = logging.getLogger(__name__)
DEFAULT_PROMPTS_CONFIG_PATH = Path("configs/prompts.yaml")
ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::([^}]*))?\}")


DEFAULT_PROMPTS: dict[str, Any] = {
    "shared": {
        "system": (
            "Bạn là trợ lý pháp lý tiếng Việt cho hệ thống hỏi đáp văn bản pháp luật. "
            "Chỉ được suy luận từ ngữ cảnh đã cung cấp. Không bịa điều luật, không viện dẫn ngoài evidence. "
            "Nếu dữ liệu hiện có chưa đủ căn cứ, phải nói rõ là chưa đủ căn cứ."
        ),
        "answer_style": (
            "Câu trả lời phải gồm 3 phần nếu phù hợp: "
            "1. Trả lời ngắn gọn. 2. Căn cứ pháp lý. 3. Lưu ý/giới hạn."
        ),
        "citation_rule": (
            "Khi có trong ngữ cảnh, phải ưu tiên nêu Điều/Khoản/Điểm, article_code hoặc law_id/tên văn bản. "
            "Không được tạo citation nếu không thấy trong context."
        ),
        "human_review_note": (
            "Nếu câu hỏi rủi ro cao hoặc evidence yếu, phải dùng ngôn ngữ thận trọng và nêu cần human review."
        ),
    },
    "draft": {
        "default": (
            "Nhiệm vụ: soạn bản nháp câu trả lời pháp lý grounded.\n"
            "{answer_style}\n"
            "{citation_rule}\n"
            "Câu hỏi: {question}\n"
            "Intent: {intent}\n"
            "Mức rủi ro: {risk_level}\n"
            "Nguồn đã truy xuất:\n{sources_block}\n"
            "Ngữ cảnh:\n{context}\n\n"
            "Yêu cầu xuất JSON với các khóa:\n"
            "- draft_answer: chuỗi\n"
            "- draft_citations: mảng chuỗi\n"
            "- draft_confidence: số từ 0 đến 1\n"
            "Nếu context thiếu căn cứ thì draft_answer phải nêu rõ điều đó."
        ),
        "hoi_dinh_nghia": (
            "Ưu tiên giải thích ngắn gọn khái niệm pháp lý bằng tiếng Việt dễ hiểu, "
            "sau đó nêu căn cứ pháp lý từ context."
        ),
        "hoi_muc_phat": (
            "Ưu tiên nêu rõ mức phạt/chế tài trong phạm vi context. "
            "Nếu context không nêu chính xác mức phạt, phải nói chưa đủ căn cứ."
        ),
        "hoi_thu_tuc_hanh_chinh": (
            "Ưu tiên trình bày theo bước hoặc điều kiện/hồ sơ nếu evidence có đề cập."
        ),
        "hoi_so_sanh_luat": (
            "Ưu tiên đối chiếu các điểm giống/khác chỉ trong phạm vi evidence đã truy xuất."
        ),
        "hoi_tinh_huong_thuc_te": (
            "Ưu tiên áp dụng quy định vào tình huống nhưng không kết luận vượt quá evidence."
        ),
        "high_risk_appendix": (
            "Đây là câu hỏi rủi ro cao. Không được đưa ra kết luận chắc chắn nếu evidence chưa thật mạnh. "
            "Nếu cần, nhấn mạnh đây chỉ là thông tin tham khảo và nên human review."
        ),
    },
    "grounding": {
        "default": (
            "Nhiệm vụ: kiểm tra grounding của câu trả lời pháp lý.\n"
            "{citation_rule}\n"
            "Câu hỏi: {question}\n"
            "Intent: {intent}\n"
            "Mức rủi ro: {risk_level}\n"
            "Draft answer:\n{draft_answer}\n\n"
            "Nguồn đã truy xuất:\n{sources_block}\n"
            "Ngữ cảnh:\n{context}\n\n"
            "Hãy trả JSON với các khóa:\n"
            "- grounding_ok: bool\n"
            "- grounding_score: số 0..1\n"
            "- unsupported_claims: mảng chuỗi\n"
            "- missing_evidence: mảng chuỗi\n"
            "- next_action: proceed|revise|retrieve_again|human_review\n"
            "- notes: chuỗi ngắn"
        ),
        "high_risk_appendix": (
            "Vì đây là câu hỏi rủi ro cao, chỉ đánh dấu proceed nếu evidence thực sự rõ và citations đủ mạnh."
        ),
    },
    "revision": {
        "default": (
            "Nhiệm vụ: viết lại câu trả lời sao cho grounded hơn, ngắn gọn hơn, và trung thực hơn.\n"
            "{answer_style}\n"
            "{citation_rule}\n"
            "{human_review_note}\n"
            "Câu hỏi: {question}\n"
            "Intent: {intent}\n"
            "Mức rủi ro: {risk_level}\n"
            "Draft hiện tại:\n{draft_answer}\n\n"
            "Unsupported claims:\n{unsupported_claims_block}\n"
            "Missing evidence:\n{missing_evidence_block}\n"
            "Nguồn đã truy xuất:\n{sources_block}\n"
            "Ngữ cảnh:\n{context}\n\n"
            "Hãy trả JSON với các khóa:\n"
            "- final_answer: chuỗi\n"
            "- grounding_ok: bool\n"
            "- review_note: chuỗi"
        ),
        "high_risk_appendix": (
            "Nếu còn nghi ngờ, final_answer phải nêu giới hạn và review_note phải yêu cầu human review."
        ),
    },
    "human_review": {
        "default": (
            "Câu hỏi thuộc nhóm cần human review. Hãy trả lời thận trọng, nêu giới hạn evidence, "
            "và khuyến nghị rà soát thêm bởi người có chuyên môn pháp lý."
        )
    },
}


@dataclass(slots=True, frozen=True)
class PromptConfig:
    """In-memory prompt configuration for TV5 reasoning."""

    shared: dict[str, str] = field(default_factory=dict)
    draft: dict[str, str] = field(default_factory=dict)
    grounding: dict[str, str] = field(default_factory=dict)
    revision: dict[str, str] = field(default_factory=dict)
    human_review: dict[str, str] = field(default_factory=dict)


def _load_yaml_module() -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("PyYAML is required to load configs/prompts.yaml.") from exc
    return yaml


def _substitute_env_placeholders(raw_text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(key, default)

    return ENV_PATTERN.sub(replace, raw_text)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            merged[key] = _deep_merge(dict(base[key]), dict(value))
        else:
            merged[key] = value
    return merged


def _safe_dict(mapping: Mapping[str, Any] | None) -> dict[str, str]:
    return {str(key): str(value) for key, value in dict(mapping or {}).items()}


def load_prompt_config(config_path: str | Path | None = None) -> PromptConfig:
    """Load prompt templates from YAML, falling back to safe defaults."""

    resolved_path = Path(config_path or DEFAULT_PROMPTS_CONFIG_PATH).resolve()
    merged_config = dict(DEFAULT_PROMPTS)
    if resolved_path.exists() and resolved_path.stat().st_size > 0:
        yaml = _load_yaml_module()
        raw_text = resolved_path.read_text(encoding="utf-8")
        loaded = yaml.safe_load(_substitute_env_placeholders(raw_text)) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError(f"Invalid prompts config structure in {resolved_path}")
        merged_config = _deep_merge(DEFAULT_PROMPTS, loaded)

    return PromptConfig(
        shared=_safe_dict(merged_config.get("shared")),
        draft=_safe_dict(merged_config.get("draft")),
        grounding=_safe_dict(merged_config.get("grounding")),
        revision=_safe_dict(merged_config.get("revision")),
        human_review=_safe_dict(merged_config.get("human_review")),
    )


def _coerce_sources(sources: Sequence[str] | None) -> list[str]:
    return [str(item).strip() for item in (sources or []) if str(item).strip()]


class PromptLibrary:
    """Centralized prompt access for TV5 reasoning nodes."""

    def __init__(
        self,
        config: PromptConfig | None = None,
        *,
        config_path: str | Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or LOGGER
        self.config = config or load_prompt_config(config_path)

    def _sources_block(self, sources: Sequence[str] | None) -> str:
        normalized = _coerce_sources(sources)
        return "\n".join(f"- {item}" for item in normalized) if normalized else "- Không có nguồn truy xuất rõ ràng"

    def _list_block(self, items: Sequence[str] | None, *, empty_message: str) -> str:
        normalized = [str(item).strip() for item in (items or []) if str(item).strip()]
        return "\n".join(f"- {item}" for item in normalized) if normalized else f"- {empty_message}"

    def _render(self, template: str, **variables: Any) -> str:
        merged_variables = {
            "answer_style": self.config.shared.get("answer_style", ""),
            "citation_rule": self.config.shared.get("citation_rule", ""),
            "human_review_note": self.config.shared.get("human_review_note", ""),
            **variables,
        }
        return template.format(**merged_variables).strip()

    def get_draft_prompt(
        self,
        intent: str,
        risk_level: str,
        *,
        question: str = "",
        context: str = "",
        sources: Sequence[str] | None = None,
    ) -> str:
        """Return the rendered draft-generation prompt."""

        template = self.config.draft.get("default", DEFAULT_PROMPTS["draft"]["default"])
        appendix = self.config.draft.get(intent, "")
        if str(risk_level).lower() == "high":
            appendix = f"{appendix}\n{self.config.draft.get('high_risk_appendix', '')}".strip()
        base_prompt = self._render(
            template,
            question=question,
            intent=intent,
            risk_level=risk_level,
            context=context,
            sources_block=self._sources_block(sources),
        )
        return f"{base_prompt}\n\nBổ sung theo intent/risk:\n{appendix}".strip()

    def get_grounding_prompt(
        self,
        intent: str,
        risk_level: str,
        *,
        question: str = "",
        context: str = "",
        sources: Sequence[str] | None = None,
        draft_answer: str = "",
    ) -> str:
        """Return the rendered grounding-check prompt."""

        template = self.config.grounding.get("default", DEFAULT_PROMPTS["grounding"]["default"])
        appendix = ""
        if str(risk_level).lower() == "high":
            appendix = self.config.grounding.get("high_risk_appendix", "")
        base_prompt = self._render(
            template,
            question=question,
            intent=intent,
            risk_level=risk_level,
            context=context,
            sources_block=self._sources_block(sources),
            draft_answer=draft_answer,
        )
        return f"{base_prompt}\n\nBổ sung theo risk:\n{appendix}".strip()

    def get_revision_prompt(
        self,
        intent: str,
        risk_level: str,
        *,
        question: str = "",
        context: str = "",
        sources: Sequence[str] | None = None,
        draft_answer: str = "",
        unsupported_claims: Sequence[str] | None = None,
        missing_evidence: Sequence[str] | None = None,
    ) -> str:
        """Return the rendered answer-revision prompt."""

        template = self.config.revision.get("default", DEFAULT_PROMPTS["revision"]["default"])
        appendix = ""
        if str(risk_level).lower() == "high":
            appendix = self.config.revision.get("high_risk_appendix", "")
        base_prompt = self._render(
            template,
            question=question,
            intent=intent,
            risk_level=risk_level,
            context=context,
            sources_block=self._sources_block(sources),
            draft_answer=draft_answer,
            unsupported_claims_block=self._list_block(
                unsupported_claims,
                empty_message="Không phát hiện unsupported claim cụ thể, nhưng cần tăng độ thận trọng.",
            ),
            missing_evidence_block=self._list_block(
                missing_evidence,
                empty_message="Không xác định được thiếu evidence cụ thể.",
            ),
        )
        return f"{base_prompt}\n\nBổ sung theo risk:\n{appendix}".strip()

    def get_human_review_prompt(self, *, question: str = "", context: str = "", sources: Sequence[str] | None = None) -> str:
        """Return the human-review-mode prompt."""

        template = self.config.human_review.get("default", DEFAULT_PROMPTS["human_review"]["default"])
        return self._render(
            template,
            question=question,
            context=context,
            sources_block=self._sources_block(sources),
        )

    def get_system_prompt(self) -> str:
        """Return the shared system prompt."""

        return self.config.shared.get("system", DEFAULT_PROMPTS["shared"]["system"]).strip()


def get_prompt_library(
    config: PromptConfig | None = None,
    *,
    config_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> PromptLibrary:
    """Factory for a prompt library instance."""

    return PromptLibrary(config=config, config_path=config_path, logger=logger)


__all__ = [
    "DEFAULT_PROMPTS_CONFIG_PATH",
    "PromptConfig",
    "PromptLibrary",
    "get_prompt_library",
    "load_prompt_config",
]

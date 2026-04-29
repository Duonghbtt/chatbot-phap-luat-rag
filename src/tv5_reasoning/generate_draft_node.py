from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib import error, request

from src.tv3_retrieval.fallback_policy import load_retrieval_config, resolve_execution_profile
from src.tv5_reasoning.citation_critic import inspect_citations
from src.tv5_reasoning.prompt_library import (
    DEFAULT_PROMPTS_CONFIG_PATH,
    PromptLibrary,
    get_prompt_library,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_LLM_CONFIG_PATH = Path("configs/llm.yaml")
ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::([^}]*))?\}")

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


@dataclass(slots=True, frozen=True)
class LLMConfig:
    """Runtime configuration for local Ollama inference."""

    provider: str = "ollama"
    model_name: str = "qwen2.5:7b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 768
    timeout_seconds: int = 120
    retry_count: int = 2


def _load_yaml_module() -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("PyYAML is required to load configs/llm.yaml.") from exc
    return yaml


def _substitute_env_placeholders(raw_text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(key, default)

    return ENV_PATTERN.sub(replace, raw_text)


def load_llm_config(config_path: str | Path | None = None) -> LLMConfig:
    """Load local LLM config for TV5 from YAML."""

    resolved_path = Path(config_path or DEFAULT_LLM_CONFIG_PATH).resolve()
    if not resolved_path.exists() or resolved_path.stat().st_size == 0:
        return LLMConfig()

    yaml = _load_yaml_module()
    raw_text = resolved_path.read_text(encoding="utf-8")
    config_data = yaml.safe_load(_substitute_env_placeholders(raw_text)) or {}
    if not isinstance(config_data, Mapping):
        raise ValueError(f"Invalid llm config structure in {resolved_path}")

    return LLMConfig(
        provider=str(config_data.get("provider") or "ollama"),
        model_name=str(config_data.get("model_name") or "qwen2.5:7b"),
        base_url=str(config_data.get("base_url") or "http://localhost:11434"),
        temperature=float(config_data.get("temperature") or 0.1),
        max_tokens=int(config_data.get("max_tokens") or 768),
        timeout_seconds=int(config_data.get("timeout_seconds") or 120),
        retry_count=int(config_data.get("retry_count") or 2),
    )


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


def _coerce_list(items: Any) -> list[str]:
    if items in (None, "", []):
        return []
    if isinstance(items, list):
        return [str(item).strip() for item in items if str(item).strip()]
    return [str(items).strip()]


class OllamaChatClient:
    """Minimal local Ollama wrapper used by TV5 nodes."""

    def __init__(self, config: LLMConfig, logger: logging.Logger | None = None) -> None:
        if config.provider.lower() != "ollama":
            raise ValueError("TV5 only supports provider=ollama as the primary path.")
        self.config = config
        self.logger = logger or LOGGER

    def generate(self, *, prompt: str, system_prompt: str = "") -> str:
        endpoint = f"{self.config.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.config.model_name,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        encoded = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            endpoint,
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(http_request, timeout=self.config.timeout_seconds) as response:
            body = response.read().decode("utf-8")
        response_data = json.loads(body)
        return str(response_data.get("response") or "").strip()

    def generate_with_retry(self, *, prompt: str, system_prompt: str = "") -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.config.retry_count + 2):
            try:
                return self.generate(prompt=prompt, system_prompt=system_prompt)
            except error.URLError as exc:
                last_error = exc
            except error.HTTPError as exc:
                last_error = exc
            except Exception as exc:  # pragma: no cover - runtime-dependent.
                last_error = exc
            sleep_seconds = min(2.0 * attempt, 5.0)
            self.logger.warning(
                "Ollama call failed on attempt %s/%s: %s. Retrying in %.1fs.",
                attempt,
                self.config.retry_count + 1,
                last_error,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
        raise RuntimeError(f"Ollama generation failed after retries: {last_error}")


def _normalize_context(context: str) -> str:
    return re.sub(r"\s+\n", "\n", (context or "").strip())


def _fallback_confidence(context: str, sources: Sequence[str] | None) -> float:
    source_count = len([item for item in (sources or []) if str(item).strip()])
    context_length = len(_normalize_context(context))
    base = 0.35 + min(source_count, 4) * 0.1 + min(context_length / 4000.0, 0.2)
    return round(min(base, 0.85), 3)


def _build_rule_based_draft(
    *,
    question: str,
    context: str,
    sources: Sequence[str] | None,
    risk_level: str,
    execution_profile: str,
) -> dict[str, Any]:
    normalized_context = _normalize_context(context)
    normalized_sources = [str(item).strip() for item in (sources or []) if str(item).strip()]

    if not normalized_context:
        draft_answer = (
            "Dữ liệu hiện có chưa đủ căn cứ để trả lời chắc chắn câu hỏi này. "
            "Bạn nên bổ sung thêm ngữ cảnh hoặc truy xuất thêm căn cứ pháp lý liên quan."
        )
        return {
            "draft_answer": draft_answer,
            "draft_citations": [],
            "draft_confidence": 0.2,
        }

    first_block = normalized_context.split("\n\n")[0].strip()
    summary_sentence = first_block[:600].strip()
    if execution_profile == "fast":
        lead = f"Theo dữ liệu truy xuất, {summary_sentence}"
        if normalized_sources:
            lead = f"{lead} Căn cứ chính: {normalized_sources[0]}."
        return {
            "draft_answer": lead.strip(),
            "draft_citations": normalized_sources[:2],
            "draft_confidence": _fallback_confidence(normalized_context, normalized_sources[:2]),
        }

    source_lines = "\n".join(f"- {item}" for item in normalized_sources[:3]) if normalized_sources else "- Chưa có nguồn chuẩn hóa"
    note = (
        "Do câu hỏi thuộc nhóm rủi ro cao, nội dung dưới đây chỉ nên xem là thông tin tham khảo và cần rà soát thêm."
        if str(risk_level).lower() == "high"
        else "Câu trả lời được tóm lược từ ngữ cảnh truy xuất hiện có."
    )
    draft_answer = (
        "1. Trả lời ngắn gọn\n"
        f"Từ ngữ cảnh được truy xuất, có thể xác định rằng: {summary_sentence}\n\n"
        "2. Căn cứ pháp lý\n"
        f"{source_lines}\n\n"
        "3. Lưu ý\n"
        f"{note}"
    )
    return {
        "draft_answer": draft_answer,
        "draft_citations": normalized_sources[:3],
        "draft_confidence": _fallback_confidence(normalized_context, normalized_sources),
    }


def _build_draft_prompt(
    *,
    prompt_library: PromptLibrary,
    intent: str,
    risk_level: str,
    execution_profile: str,
    question: str,
    context: str,
    sources: Sequence[str] | None,
) -> str:
    prompt = prompt_library.get_draft_prompt(
        intent,
        risk_level,
        question=question,
        context=_normalize_context(context),
        sources=sources,
    )
    if execution_profile != "fast":
        return prompt

    shared_fast_style = prompt_library.config.shared.get(
        "fast_answer_style",
        "Trả lời cực ngắn, trực tiếp, tối đa 3 câu. Chỉ sử dụng 1-2 nguồn mạnh nhất và không suy diễn ngoài context.",
    )
    draft_fast_style = prompt_library.config.draft.get(
        "fast_default",
        "Đây là fast-path. Ưu tiên nêu kết luận trực tiếp theo điều luật hoặc khái niệm được truy xuất. Không viết 3 mục dài.",
    )
    return f"{prompt}\n\nYêu cầu bổ sung cho fast-path:\n{shared_fast_style}\n{draft_fast_style}".strip()


def generate_draft(
    *,
    question: str,
    context: str,
    sources: Sequence[str] | None,
    intent: str,
    risk_level: str,
    execution_profile: str = "full",
    llm_config: LLMConfig | None = None,
    prompt_library: PromptLibrary | None = None,
    client: OllamaChatClient | None = None,
    prompts_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Generate a grounded legal draft answer from retrieved evidence."""

    resolved_logger = logger or LOGGER
    resolved_llm_config = llm_config or load_llm_config()
    resolved_prompt_library = prompt_library or get_prompt_library(config_path=prompts_path, logger=resolved_logger)
    resolved_client = client or OllamaChatClient(resolved_llm_config, logger=resolved_logger)
    normalized_sources = list(sources or [])
    if execution_profile == "fast":
        fast_limits = load_retrieval_config().fast_sources_limit
        normalized_sources = normalized_sources[:fast_limits]

    prompt = _build_draft_prompt(
        prompt_library=resolved_prompt_library,
        intent=intent,
        risk_level=risk_level,
        execution_profile=execution_profile,
        question=question,
        context=context,
        sources=normalized_sources,
    )
    system_prompt = resolved_prompt_library.get_system_prompt()

    try:
        raw_response = resolved_client.generate_with_retry(prompt=prompt, system_prompt=system_prompt)
        payload = _extract_json_object(raw_response) or {}
    except Exception as exc:  # pragma: no cover - runtime-dependent.
        resolved_logger.warning("Draft generation via Ollama failed, using rule-based fallback: %s", exc)
        payload = {}

    draft_answer = str(payload.get("draft_answer") or "").strip()
    draft_citations = _coerce_list(payload.get("draft_citations"))
    draft_confidence = _safe_float(payload.get("draft_confidence"), default=_fallback_confidence(context, normalized_sources))

    if not draft_answer:
        fallback = _build_rule_based_draft(
            question=question,
            context=context,
            sources=normalized_sources,
            risk_level=risk_level,
            execution_profile=execution_profile,
        )
        draft_answer = fallback["draft_answer"]
        draft_citations = fallback["draft_citations"]
        draft_confidence = fallback["draft_confidence"]

    citation_report = inspect_citations(draft_answer, normalized_sources, [])
    if not draft_citations and citation_report["normalized_citations"]:
        draft_citations = citation_report["normalized_citations"]

    if execution_profile == "fast":
        draft_citations = draft_citations[:2]

    return {
        "draft_answer": draft_answer,
        "draft_citations": draft_citations,
        "draft_confidence": round(max(0.0, min(1.0, draft_confidence)), 4),
        "reasoning_notes": {
            "draft_model": resolved_llm_config.model_name,
            "used_fallback": not bool(payload.get("draft_answer")),
            "execution_profile": execution_profile,
        },
    }


@_optional_traceable(name="tv5.generate_draft_node", run_type="chain", tags=["tv5", "reasoning"])
def generate_draft_node(
    state: Mapping[str, Any],
    *,
    llm_config: LLMConfig | None = None,
    llm_config_path: str | Path | None = None,
    prompts_path: str | Path | None = None,
    prompt_library: PromptLibrary | None = None,
    client: OllamaChatClient | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """LangGraph-friendly TV5 node that generates a grounded draft answer."""

    question = str(state.get("normalized_question") or state.get("question") or "").strip()
    context = str(state.get("context") or "").strip()
    sources = list(state.get("sources") or [])
    intent = str(state.get("intent") or "hoi_tinh_huong_thuc_te").strip()
    risk_level = str(state.get("risk_level") or "medium").strip().lower()
    execution_profile = resolve_execution_profile(state)

    resolved_llm_config = llm_config or load_llm_config(llm_config_path)
    result = generate_draft(
        question=question,
        context=context,
        sources=sources,
        intent=intent,
        risk_level=risk_level,
        execution_profile=execution_profile,
        llm_config=resolved_llm_config,
        prompt_library=prompt_library,
        client=client,
        prompts_path=prompts_path or DEFAULT_PROMPTS_CONFIG_PATH,
        logger=logger,
    )

    if risk_level == "high" and execution_profile != "fast":
        result["human_review_required"] = True
        result["review_note"] = (
            "Câu hỏi thuộc nhóm rủi ro pháp lý cao; nên có bước human review trước khi sử dụng kết quả."
        )
    return result


__all__ = [
    "DEFAULT_LLM_CONFIG_PATH",
    "LLMConfig",
    "OllamaChatClient",
    "generate_draft",
    "generate_draft_node",
    "load_llm_config",
]

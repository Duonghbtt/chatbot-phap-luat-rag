from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.graph.checkpointing import BaseCheckpointStore, create_checkpoint_store
from src.graph.human_review_node import human_review_node, manual_human_review_node
from src.graph.state import (
    AgentState,
    append_history,
    clone_state,
    create_initial_state,
    merge_state,
    normalize_user_text,
)
from src.graph.subgraphs import build_legal_agent_subgraph, build_review_subgraph
from src.tv3_retrieval.rerank_node import rerank_node
from src.tv3_retrieval.retrieve_node import retrieve_node
from src.tv3_retrieval.retrieval_check_node import retrieval_check_node
from src.tv3_retrieval.rewrite_query_node import rewrite_query_node
from src.tv4_router.route_node import route_node
from src.tv5_reasoning.citation_critic import inspect_citations
from src.tv5_reasoning.generate_draft_node import generate_draft_node
from src.tv5_reasoning.grounding_check_node import grounding_check_node
from src.tv5_reasoning.revise_answer_node import revise_answer_node

LOGGER = logging.getLogger(__name__)
DEFAULT_APP_CONFIG_PATH = Path("configs/app.yaml")
ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::([^}]*))?\}")
WAITING_USER_INPUT_STATUS = "waiting_user_input"


@dataclass(slots=True, frozen=True)
class AppConfig:
    """Runtime configuration for the TV6 orchestration/backend layer."""

    app_name: str = "Legal QA Agentic Backend"
    host: str = "127.0.0.1"
    port: int = 8000
    enable_streaming: bool = True
    enable_resume: bool = True
    checkpoint_backend: str = "local_json"
    checkpoint_dir: str = ".checkpoints"
    default_timeout_seconds: int = 120
    max_reasoning_loops: int = 2
    max_retrieval_rounds: int = 3


@dataclass(slots=True)
class GraphDependencies:
    """Dependency-injectable node registry for TV6 orchestration."""

    route_node: Callable[..., dict[str, Any]] = route_node
    rewrite_query_node: Callable[..., dict[str, Any]] = rewrite_query_node
    retrieve_node: Callable[..., dict[str, Any]] = retrieve_node
    rerank_node: Callable[..., dict[str, Any]] = rerank_node
    retrieval_check_node: Callable[..., dict[str, Any]] = retrieval_check_node
    generate_draft_node: Callable[..., dict[str, Any]] = generate_draft_node
    grounding_check_node: Callable[..., dict[str, Any]] = grounding_check_node
    revise_answer_node: Callable[..., dict[str, Any]] = revise_answer_node
    human_review_node: Callable[..., dict[str, Any]] = human_review_node


def _load_yaml_module() -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("PyYAML is required to load configs/app.yaml.") from exc
    return yaml


def _substitute_env_placeholders(raw_text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(key, default)

    return ENV_PATTERN.sub(replace, raw_text)


def load_app_config(config_path: str | Path | None = None) -> AppConfig:
    """Load TV6 app/orchestration config from YAML."""

    resolved_path = Path(config_path or DEFAULT_APP_CONFIG_PATH).resolve()
    if not resolved_path.exists() or resolved_path.stat().st_size == 0:
        return AppConfig()

    yaml = _load_yaml_module()
    raw_text = resolved_path.read_text(encoding="utf-8")
    payload = yaml.safe_load(_substitute_env_placeholders(raw_text)) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid app config structure in {resolved_path}")

    return AppConfig(
        app_name=str(payload.get("app_name") or "Legal QA Agentic Backend"),
        host=str(payload.get("host") or "127.0.0.1"),
        port=int(payload.get("port") or 8000),
        enable_streaming=bool(payload.get("enable_streaming", True)),
        enable_resume=bool(payload.get("enable_resume", True)),
        checkpoint_backend=str(payload.get("checkpoint_backend") or "local_json"),
        checkpoint_dir=str(payload.get("checkpoint_dir") or ".checkpoints"),
        default_timeout_seconds=int(payload.get("default_timeout_seconds") or 120),
        max_reasoning_loops=int(payload.get("max_reasoning_loops") or 2),
        max_retrieval_rounds=int(payload.get("max_retrieval_rounds") or 3),
    )


def analyze_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Initialize and normalize the top-level graph state for one question."""

    question = normalize_user_text(str(state.get("question") or state.get("normalized_question") or ""))
    session_id = str(state.get("session_id") or "")
    thread_id = str(state.get("thread_id") or "")
    initialized = create_initial_state(
        question=question,
        session_id=session_id or None,
        thread_id=thread_id or None,
        extra=state,
    )
    initialized["response_status"] = ""
    initialized["status"] = ""
    initialized["interrupt_payload"] = None
    initialized["resume_kind"] = ""
    initialized["resume_question"] = ""
    initialized["execution_profile"] = str(state.get("execution_profile") or "")
    initialized["fast_path_enabled"] = bool(state.get("fast_path_enabled") or False)
    return initialized


def _build_clarify_interrupt_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    clarify_question = str(
        state.get("resume_question")
        or state.get("clarify_question")
        or "Bạn có thể nói rõ thêm câu hỏi pháp lý này không?"
    ).strip()
    return {
        "kind": "clarify",
        "thread_id": str(state.get("thread_id") or ""),
        "session_id": str(state.get("session_id") or ""),
        "question": str(state.get("normalized_question") or state.get("question") or ""),
        "clarify_question": clarify_question,
        "clarify_reason": str(state.get("clarify_reason") or ""),
        "missing_slots": list(state.get("missing_slots") or []),
    }


def _build_human_review_interrupt_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    stage = (
        "post_reasoning"
        if str(state.get("draft_answer") or state.get("final_answer") or state.get("context") or "").strip()
        else "pre_retrieval"
    )
    answer_preview = str(state.get("final_answer") or state.get("draft_answer") or "").strip()
    return {
        "kind": "human_review",
        "stage": stage,
        "thread_id": str(state.get("thread_id") or ""),
        "session_id": str(state.get("session_id") or ""),
        "question": str(state.get("normalized_question") or state.get("question") or ""),
        "review_note": str(state.get("review_note") or ""),
        "answer_preview": answer_preview,
        "sources": list(state.get("sources") or []),
    }


def _default_human_review_question(state: Mapping[str, Any]) -> str:
    existing = str(state.get("resume_question") or "").strip()
    if existing:
        return existing

    question = str(state.get("normalized_question") or state.get("question") or "").lower()
    review_note = str(state.get("review_note") or "").strip()
    if any(token in question for token in ("khởi kiện", "kiện", "tranh chấp", "bồi thường", "đất đai")):
        return (
            "Câu hỏi này có thể ảnh hưởng đến quyết định pháp lý thực tế. "
            "Bạn muốn hệ thống chỉ phân tích căn cứ pháp luật chung, hay tiếp tục theo tình huống cụ thể của bạn?"
        )
    if review_note:
        return (
            f"{review_note} Bạn muốn hệ thống chỉ phân tích căn cứ pháp luật chung, "
            "hay tiếp tục theo tình huống cụ thể của bạn?"
        )
    return (
        "Để tránh trả lời vượt quá dữ liệu hiện có, bạn muốn mình trình bày thông tin pháp luật tổng quát "
        "hay tiếp tục theo trường hợp cá nhân?"
    )


def _build_waiting_user_input_updates(state: Mapping[str, Any]) -> dict[str, Any]:
    interrupt_payload = dict(state.get("interrupt_payload") or {})
    resume_kind = str(state.get("resume_kind") or "").strip()
    if not resume_kind:
        interrupt_kind = str(interrupt_payload.get("kind") or "").strip()
        if interrupt_kind == "clarify" or bool(state.get("need_clarify")):
            resume_kind = "clarify"
        elif interrupt_kind == "human_review" or bool(state.get("human_review_required")):
            resume_kind = "human_review"

    resume_question = str(state.get("resume_question") or "").strip()
    if resume_kind == "clarify" and not resume_question:
        resume_question = str(
            state.get("clarify_question")
            or interrupt_payload.get("clarify_question")
            or "Bạn có thể nói rõ thêm câu hỏi pháp lý này không?"
        ).strip()
    if resume_kind == "human_review" and not resume_question:
        resume_question = _default_human_review_question(state)

    if not resume_kind or not resume_question:
        return {}

    if not interrupt_payload:
        interrupt_payload = (
            _build_clarify_interrupt_payload(state)
            if resume_kind == "clarify"
            else _build_human_review_interrupt_payload(state)
        )

    return {
        "status": WAITING_USER_INPUT_STATUS,
        "response_status": WAITING_USER_INPUT_STATUS,
        "resume_kind": resume_kind,
        "resume_question": resume_question,
        "interrupt_payload": interrupt_payload,
        "final_answer": "",
    }


def _apply_clarify_response(state: Mapping[str, Any], clarify_response: str) -> dict[str, Any]:
    merged_question = normalize_user_text(
        f"{state.get('question') or state.get('normalized_question')}\nThông tin bổ sung: {clarify_response}"
    )
    return {
        "question": merged_question,
        "normalized_question": merged_question,
        "need_clarify": False,
        "clarify_question": "",
        "clarify_reason": "",
        "missing_slots": [],
        "interrupt_payload": None,
        "next_action": "resume_after_clarify",
        "status": "",
        "response_status": "",
        "resume_kind": "",
        "resume_question": "",
        "history": append_history(
            state,
            role="user",
            content=clarify_response,
            kind="clarify_response",
        ),
    }


def clarify_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Pause the graph for clarification or apply a clarification response."""

    clarify_response = str(state.get("clarify_response") or "").strip()
    if clarify_response:
        return _apply_clarify_response(state, clarify_response)

    payload = _build_clarify_interrupt_payload(state)
    resume_value = interrupt(payload)
    if isinstance(resume_value, Mapping):
        clarify_response = str(resume_value.get("clarify_response") or resume_value.get("note") or "").strip()
    else:
        clarify_response = str(resume_value or "").strip()
    if not clarify_response:
        return {
            "interrupt_payload": payload,
            "status": WAITING_USER_INPUT_STATUS,
            "response_status": WAITING_USER_INPUT_STATUS,
            "resume_kind": "clarify",
            "resume_question": str(payload.get("clarify_question") or ""),
            "next_action": "await_clarify",
        }
    return clarify_node(merge_state(state, {"clarify_response": clarify_response}))


def manual_clarify_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Manual-runtime equivalent of the clarify node without LangGraph interrupt."""

    clarify_response = str(state.get("clarify_response") or "").strip()
    if clarify_response:
        return _apply_clarify_response(state, clarify_response)

    payload = _build_clarify_interrupt_payload(state)
    return {
        "interrupt_payload": payload,
        "status": WAITING_USER_INPUT_STATUS,
        "response_status": WAITING_USER_INPUT_STATUS,
        "resume_kind": "clarify",
        "resume_question": str(payload.get("clarify_question") or ""),
        "next_action": "await_clarify",
    }


def unsupported_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Build a safe response for unsupported/out-of-scope queries."""

    answer = (
        "Câu hỏi này hiện nằm ngoài phạm vi hỗ trợ của hệ thống hỏi đáp văn bản pháp luật "
        "dựa trên Bộ pháp điển điện tử. Bạn vui lòng nêu rõ vấn đề pháp lý hoặc tên văn bản/điều luật liên quan."
    )
    return {
        "final_answer": answer,
        "response_status": "unsupported",
        "status": "unsupported",
        "next_action": "proceed",
    }


def retrieval_fallback_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return a safe fallback answer when retrieval cannot find enough evidence."""

    risk_level = str(state.get("risk_level") or "medium").strip().lower()
    review_note = ""
    human_review_required = bool(state.get("human_review_required") or False)
    if risk_level == "high":
        human_review_required = True
        review_note = (
            "Dữ liệu truy xuất chưa đủ mạnh cho câu hỏi rủi ro cao; nên có human review hoặc bổ sung thêm căn cứ pháp lý."
        )

    final_answer = (
        "1. Trả lời ngắn gọn\n"
        "Dữ liệu hiện có chưa đủ căn cứ để đưa ra câu trả lời pháp lý chắc chắn.\n\n"
        "2. Căn cứ pháp lý\n"
        "Hệ thống chưa truy xuất được nguồn phù hợp hoặc evidence còn yếu.\n\n"
        "3. Lưu ý\n"
        "Bạn có thể nêu rõ hơn tên luật, điều luật, chủ thể hoặc tình huống để hệ thống truy xuất lại chính xác hơn."
    )
    return {
        "final_answer": final_answer,
        "grounding_ok": False,
        "human_review_required": human_review_required,
        "review_note": review_note,
    }


def citation_format_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize citation display without changing the legal substance of the answer."""

    sources: list[str] = []
    for source in state.get("sources") or []:
        cleaned = normalize_user_text(str(source))
        if cleaned and cleaned not in sources:
            sources.append(cleaned)

    citation_findings = dict(state.get("citation_findings") or {})
    if not citation_findings:
        citation_findings = inspect_citations(
            str(state.get("final_answer") or state.get("draft_answer") or ""),
            sources,
            list(state.get("reranked_docs") or []),
        )

    normalized_citations = [
        normalize_user_text(str(item))
        for item in citation_findings.get("normalized_citations", [])
        if normalize_user_text(str(item))
    ]
    if normalized_citations:
        sources = list(dict.fromkeys(normalized_citations + sources))
    if str(state.get("execution_profile") or "").strip().lower() == "fast":
        sources = sources[:2]
    return {
        "sources": sources,
        "citation_findings": citation_findings,
    }


def final_answer_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Finalize the answer payload and ensure a safe response always exists."""

    waiting_updates = _build_waiting_user_input_updates(state)
    if waiting_updates:
        return waiting_updates

    final_answer = str(state.get("final_answer") or "").strip()
    draft_answer = str(state.get("draft_answer") or "").strip()
    if not final_answer and draft_answer:
        final_answer = draft_answer

    response_status = str(state.get("response_status") or state.get("status") or "").strip()
    if not response_status:
        if state.get("unsupported_query"):
            response_status = "unsupported"
        else:
            response_status = "ok"

    if not final_answer:
        final_answer = (
            "Dữ liệu hiện có chưa đủ căn cứ để kết luận chắc chắn. "
            "Bạn vui lòng bổ sung thêm bối cảnh pháp lý hoặc thử lại với câu hỏi cụ thể hơn."
        )

    history = list(state.get("history") or [])
    if final_answer and response_status in {"ok", "unsupported"}:
        history = append_history(
            state,
            role="assistant",
            content=final_answer,
            kind="final_answer",
            metadata={"status": response_status},
        )

    return {
        "final_answer": final_answer,
        "response_status": response_status,
        "status": response_status,
        "history": history,
    }


class LegalQAGraphRuntime:
    """Production-style orchestration runtime that mirrors the LangGraph topology."""

    def __init__(
        self,
        *,
        app_config: AppConfig,
        checkpoint_store: BaseCheckpointStore,
        dependencies: GraphDependencies | None = None,
        compiled_graph: Any | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.app_config = app_config
        self.checkpoint_store = checkpoint_store
        self.dependencies = dependencies or GraphDependencies()
        self.compiled_graph = compiled_graph
        self.logger = logger or LOGGER

    def _manual_human_review_step(self, state: Mapping[str, Any]) -> dict[str, Any]:
        review_handler = self.dependencies.human_review_node
        if review_handler is human_review_node:
            updates = manual_human_review_node(state)
        else:
            updates = review_handler(state)

        if not updates:
            return {}

        merged_state = merge_state(state, updates)
        waiting_updates = _build_waiting_user_input_updates(merged_state)
        if waiting_updates:
            merged_updates = dict(updates)
            merged_updates.update(waiting_updates)
            return merged_updates
        return dict(updates)

    def _start_timing(self, state: Mapping[str, Any], *, phase: str) -> AgentState:
        return self._apply(
            state,
            {
                "timing_started_at": time.perf_counter(),
                "total_elapsed_ms": 0.0,
                "timing_debug": {
                    "phase": phase,
                    "steps": [],
                    "step_count": 0,
                    "total_elapsed_ms": 0.0,
                },
            },
        )

    def _log_step(
        self,
        step_name: str,
        started_at: float,
        state: Mapping[str, Any],
        **extra: Any,
    ) -> AgentState:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 3)
        timing_debug = dict(state.get("timing_debug") or {})
        steps = [dict(item) for item in list(timing_debug.get("steps") or [])]
        step_payload = {
            "step": step_name,
            "elapsed_ms": elapsed_ms,
            "route": str(state.get("next_route") or ""),
            "next_action": str(state.get("next_action") or ""),
            "loop_count": int(state.get("loop_count") or 0),
            "retrieval_ok": bool(state.get("retrieval_ok") or False),
            "grounding_ok": bool(state.get("grounding_ok") or False),
        }
        for key, value in extra.items():
            if value not in (None, "", [], {}):
                step_payload[str(key)] = value
        steps.append(step_payload)
        timing_debug["steps"] = steps
        timing_debug["step_count"] = len(steps)
        updated_state = self._apply(state, {"timing_debug": timing_debug})
        self.logger.info(
            "step=%s thread=%s route=%s next_action=%s loop=%s retrieval_ok=%s grounding_ok=%s elapsed_ms=%.1f",
            step_name,
            updated_state.get("thread_id"),
            updated_state.get("next_route"),
            updated_state.get("next_action"),
            updated_state.get("loop_count"),
            updated_state.get("retrieval_ok"),
            updated_state.get("grounding_ok"),
            elapsed_ms,
        )
        return updated_state

    def _apply(self, state: Mapping[str, Any], updates: Mapping[str, Any] | None) -> AgentState:
        return merge_state(state, updates)

    def _emit(self, emitter: Callable[[dict[str, Any]], None] | None, event: str, **payload: Any) -> None:
        if emitter is None:
            return
        emitter({"event": event, "data": payload})

    def _build_timing_payload(self, state: Mapping[str, Any]) -> dict[str, Any]:
        timing_debug = dict(state.get("timing_debug") or {})
        steps = [dict(item) for item in list(timing_debug.get("steps") or [])]
        return {
            "phase": str(timing_debug.get("phase") or ""),
            "step_count": int(timing_debug.get("step_count") or len(steps)),
            "total_elapsed_ms": round(float(state.get("total_elapsed_ms") or timing_debug.get("total_elapsed_ms") or 0.0), 3),
            "steps": steps,
        }

    def _finalize_timing(
        self,
        state: Mapping[str, Any],
        *,
        emitter: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentState:
        started_at = float(state.get("timing_started_at") or 0.0)
        if started_at <= 0:
            return self._apply(state, {"total_elapsed_ms": 0.0})

        total_elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 3)
        timing_debug = dict(state.get("timing_debug") or {})
        steps = [dict(item) for item in list(timing_debug.get("steps") or [])]
        timing_debug["steps"] = steps
        timing_debug["step_count"] = len(steps)
        timing_debug["total_elapsed_ms"] = total_elapsed_ms
        updated_state = self._apply(
            state,
            {
                "timing_debug": timing_debug,
                "total_elapsed_ms": total_elapsed_ms,
            },
        )
        self.logger.info(
            "timing_summary thread=%s phase=%s total_elapsed_ms=%.1f step_count=%s",
            updated_state.get("thread_id"),
            timing_debug.get("phase"),
            total_elapsed_ms,
            timing_debug.get("step_count"),
        )
        self._emit(
            emitter,
            "timing_summary",
            timing=self._build_timing_payload(updated_state),
            total_elapsed_ms=total_elapsed_ms,
            thread_id=str(updated_state.get("thread_id") or ""),
            session_id=str(updated_state.get("session_id") or ""),
        )
        return updated_state

    def _has_pending_user_input(self, state: Mapping[str, Any]) -> bool:
        return bool(
            str(state.get("response_status") or state.get("status") or "") == WAITING_USER_INPUT_STATUS
            and str(state.get("resume_kind") or "").strip()
            and str(state.get("resume_question") or "").strip()
        )

    def _waiting_event_payload(self, state: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "status": WAITING_USER_INPUT_STATUS,
            "resume_kind": str(state.get("resume_kind") or ""),
            "resume_question": str(state.get("resume_question") or ""),
            "thread_id": str(state.get("thread_id") or ""),
            "session_id": str(state.get("session_id") or ""),
            "route": str(state.get("next_route") or ""),
            "risk_level": str(state.get("risk_level") or ""),
            "sources": list(state.get("sources") or []),
            "review_note": str(state.get("review_note") or ""),
            "interrupt_payload": dict(state.get("interrupt_payload") or {}),
            "timing": self._build_timing_payload(state),
            "total_elapsed_ms": round(float(state.get("total_elapsed_ms") or 0.0), 3),
        }

    def _final_state_payload(self, state: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "status": str(state.get("response_status") or state.get("status") or "ok"),
            "thread_id": str(state.get("thread_id") or ""),
            "session_id": str(state.get("session_id") or ""),
            "resume_kind": str(state.get("resume_kind") or ""),
            "resume_question": str(state.get("resume_question") or ""),
            "route": str(state.get("next_route") or ""),
            "risk_level": str(state.get("risk_level") or ""),
            "timing": self._build_timing_payload(state),
            "total_elapsed_ms": round(float(state.get("total_elapsed_ms") or 0.0), 3),
        }

    def _prepare_full_pipeline_state(self, state: Mapping[str, Any], *, reason: str = "") -> AgentState:
        updates: dict[str, Any] = {
            "execution_profile": "full",
            "fast_path_enabled": False,
            "next_route": "legal-agent-path",
            "next_action": "",
            "retrieval_ok": False,
            "grounding_ok": False,
            "retrieval_failure_reason": "",
            "retrieved_docs": [],
            "reranked_docs": [],
            "context": "",
            "sources": [],
            "draft_answer": "",
            "final_answer": "",
            "draft_citations": [],
            "draft_confidence": 0.0,
            "unsupported_claims": [],
            "missing_evidence": [],
            "grounding_score": 0.0,
            "human_review_required": False,
            "review_note": "",
            "rewritten_queries": [],
        }
        if reason:
            updates["route_reason"] = reason
        return self._apply(state, updates)

    def _escalate_fast_path_to_full(
        self,
        state: AgentState,
        *,
        reason: str,
        emitter: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentState:
        escalated_reason = (
            f"{str(state.get('route_reason') or '').strip()} Fast-path chưa đủ chắc chắn nên chuyển sang legal-agent-path đầy đủ."
            if str(state.get("route_reason") or "").strip()
            else "Fast-path chưa đủ chắc chắn nên chuyển sang legal-agent-path đầy đủ."
        )
        prepared_state = self._prepare_full_pipeline_state(state, reason=reason or escalated_reason)
        prepared_state = self._apply(prepared_state, {"route_reason": escalated_reason})
        self._emit(
            emitter,
            "route",
            route="legal-agent-path",
            risk_level=prepared_state.get("risk_level"),
            intent=prepared_state.get("intent"),
            reason=escalated_reason,
            escalated_from="fast-path",
        )
        return prepared_state

    def _run_retrieval_cycle(
        self,
        state: AgentState,
        *,
        max_rounds: int | None = None,
        emitter: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentState:
        round_limit = max_rounds or self.app_config.max_retrieval_rounds
        retrieval_rounds = 0
        while retrieval_rounds < round_limit:
            retrieval_rounds += 1
            started_at = time.perf_counter()
            state = self._apply(state, self.dependencies.retrieve_node(state))
            state = self._log_step("retrieve_node", started_at, state, round=retrieval_rounds)
            started_at = time.perf_counter()
            state = self._apply(state, self.dependencies.rerank_node(state))
            state = self._log_step("rerank_node", started_at, state, round=retrieval_rounds)
            started_at = time.perf_counter()
            state = self._apply(state, self.dependencies.retrieval_check_node(state))
            state = self._log_step("retrieval_check_node", started_at, state, round=retrieval_rounds)
            self._emit(
                emitter,
                "retrieval_status",
                round=retrieval_rounds,
                retrieval_ok=state.get("retrieval_ok"),
                next_action=state.get("next_action"),
                retrieved_count=len(state.get("retrieved_docs") or []),
                reranked_count=len(state.get("reranked_docs") or []),
                elapsed_ms=round(
                    sum(
                        float(step.get("elapsed_ms") or 0.0)
                        for step in list(state.get("timing_debug", {}).get("steps") or [])[-3:]
                    ),
                    3,
                ),
            )
            if str(state.get("next_action") or "") != "retry":
                break
        return state

    def _run_reasoning_cycle(
        self,
        state: AgentState,
        *,
        emitter: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentState:
        started_at = time.perf_counter()
        state = self._apply(state, self.dependencies.generate_draft_node(state))
        state = self._log_step("generate_draft_node", started_at, state)
        if state.get("draft_answer"):
            self._emit(emitter, "partial_answer", draft_answer=state.get("draft_answer"))

        revision_loops = 0
        while revision_loops <= self.app_config.max_reasoning_loops:
            started_at = time.perf_counter()
            state = self._apply(state, self.dependencies.grounding_check_node(state))
            state = self._log_step("grounding_check_node", started_at, state, revision=revision_loops)
            self._emit(
                emitter,
                "grounding_status",
                grounding_ok=state.get("grounding_ok"),
                grounding_score=state.get("grounding_score"),
                next_action=state.get("next_action"),
                human_review_required=state.get("human_review_required"),
                elapsed_ms=float((state.get("timing_debug") or {}).get("steps", [{}])[-1].get("elapsed_ms") or 0.0),
            )
            next_action = str(state.get("next_action") or "proceed")

            if next_action == "proceed":
                if not str(state.get("final_answer") or "").strip():
                    state = self._apply(state, {"final_answer": str(state.get("draft_answer") or "").strip()})
                return state
            if next_action == "retrieve_again":
                return self._run_retrieval_then_reasoning_again(state, emitter=emitter)
            if next_action == "human_review":
                return state
            if next_action != "revise":
                return state

            revision_loops += 1
            started_at = time.perf_counter()
            state = self._apply(state, self.dependencies.revise_answer_node(state))
            state = self._log_step("revise_answer_node", started_at, state, revision=revision_loops)
            if state.get("final_answer"):
                self._emit(emitter, "partial_answer", draft_answer=state.get("final_answer"))
                state = self._apply(
                    state,
                    {
                        "draft_answer": str(state.get("final_answer") or "").strip(),
                        "loop_count": int(state.get("loop_count") or 0) + 1,
                    },
                )

            if revision_loops > self.app_config.max_reasoning_loops:
                state = self._apply(
                    state,
                    {
                        "human_review_required": True,
                        "review_note": "Đã vượt quá số vòng revise/grounding cho phép; cần human review trước khi phát hành.",
                        "next_action": "human_review",
                    },
                )
                return state
        return state

    def _run_retrieval_then_reasoning_again(
        self,
        state: AgentState,
        *,
        emitter: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentState:
        state = self._prepare_full_pipeline_state(state)
        state = self._run_retrieval_cycle(state, emitter=emitter)
        if not state.get("retrieval_ok"):
            return self._apply(state, retrieval_fallback_node(state))
        return self._run_reasoning_cycle(state, emitter=emitter)

    def _run_fast_path_subgraph(
        self,
        state: AgentState,
        *,
        emitter: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentState:
        state = self._apply(
            state,
            {
                "execution_profile": "fast",
                "fast_path_enabled": True,
                "next_route": "fast-path",
                "next_action": "",
            },
        )
        started_at = time.perf_counter()
        state = self._apply(state, self.dependencies.rewrite_query_node(state))
        state = self._log_step("rewrite_query_node_fast", started_at, state)
        self._emit(
            emitter,
            "retrieval_status",
            stage="rewrite_fast",
            rewritten_queries=list(state.get("rewritten_queries") or []),
            elapsed_ms=float((state.get("timing_debug") or {}).get("steps", [{}])[-1].get("elapsed_ms") or 0.0),
        )

        state = self._run_retrieval_cycle(state, max_rounds=1, emitter=emitter)
        next_action = str(state.get("next_action") or "")
        if not state.get("retrieval_ok"):
            if next_action == "escalate_to_full":
                reason = (
                    "Fast-path không tìm được evidence đủ mạnh ở vòng retrieval đầu tiên; chuyển sang legal-agent-path."
                )
                escalated_state = self._escalate_fast_path_to_full(state, reason=reason, emitter=emitter)
                return self._run_legal_agent_subgraph(escalated_state, emitter=emitter)
            return self._apply(state, retrieval_fallback_node(state))

        started_at = time.perf_counter()
        state = self._apply(state, self.dependencies.generate_draft_node(state))
        state = self._log_step("generate_draft_node_fast", started_at, state)
        if state.get("draft_answer"):
            self._emit(emitter, "partial_answer", draft_answer=state.get("draft_answer"))

        started_at = time.perf_counter()
        state = self._apply(state, self.dependencies.grounding_check_node(state))
        state = self._log_step("grounding_check_node_fast", started_at, state)
        self._emit(
            emitter,
            "grounding_status",
            grounding_ok=state.get("grounding_ok"),
            grounding_score=state.get("grounding_score"),
            next_action=state.get("next_action"),
            human_review_required=state.get("human_review_required"),
            elapsed_ms=float((state.get("timing_debug") or {}).get("steps", [{}])[-1].get("elapsed_ms") or 0.0),
        )

        next_action = str(state.get("next_action") or "proceed")
        if next_action == "proceed":
            if not str(state.get("final_answer") or "").strip():
                state = self._apply(state, {"final_answer": str(state.get("draft_answer") or "").strip()})
            return state
        if next_action == "escalate_to_full":
            reason = (
                "Grounding ở fast-path chưa đủ chắc chắn; chuyển sang legal-agent-path để chạy retrieval và reasoning đầy đủ."
            )
            escalated_state = self._escalate_fast_path_to_full(state, reason=reason, emitter=emitter)
            return self._run_legal_agent_subgraph(escalated_state, emitter=emitter)
        return state

    def _run_legal_agent_subgraph(
        self,
        state: AgentState,
        *,
        emitter: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentState:
        state = self._apply(
            state,
            {
                "execution_profile": "full",
                "fast_path_enabled": False,
                "next_route": "legal-agent-path",
                "next_action": "",
            },
        )
        started_at = time.perf_counter()
        state = self._apply(state, self.dependencies.rewrite_query_node(state))
        state = self._log_step("rewrite_query_node", started_at, state)
        self._emit(
            emitter,
            "retrieval_status",
            stage="rewrite",
            rewritten_queries=list(state.get("rewritten_queries") or []),
            elapsed_ms=float((state.get("timing_debug") or {}).get("steps", [{}])[-1].get("elapsed_ms") or 0.0),
        )

        state = self._run_retrieval_cycle(state, emitter=emitter)
        if not state.get("retrieval_ok"):
            return self._apply(state, retrieval_fallback_node(state))

        return self._run_reasoning_cycle(state, emitter=emitter)

    def _run_post_reasoning_human_review_if_needed(
        self,
        state: AgentState,
        *,
        emitter: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentState:
        if state.get("human_review_required") and str(state.get("next_action") or "") == "human_review":
            started_at = time.perf_counter()
            state = self._apply(state, self._manual_human_review_step(state))
            return self._log_step("human_review_node", started_at, state, stage="post_reasoning")
        return state

    def _continue_after_route(
        self,
        state: AgentState,
        *,
        emitter: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentState:
        route = str(state.get("next_route") or "legal-agent-path")
        if route == "clarify-path":
            started_at = time.perf_counter()
            state = self._apply(state, manual_clarify_node(state))
            state = self._log_step("clarify_node", started_at, state)
            return self._finish(state, emitter=emitter)
        if route == "unsupported-path":
            started_at = time.perf_counter()
            state = self._apply(state, unsupported_node(state))
            state = self._log_step("unsupported_node", started_at, state)
            return self._finish(state, emitter=emitter)
        if route == "human-review-path":
            started_at = time.perf_counter()
            state = self._apply(state, self._manual_human_review_step(state))
            state = self._log_step("human_review_node", started_at, state, stage="pre_retrieval")
            if self._has_pending_user_input(state) or str(state.get("next_action") or "") == "stop_after_review":
                return self._finish(state, emitter=emitter)
            if str(state.get("next_action") or "") == "resume_legal_agent":
                state = self._run_legal_agent_subgraph(state, emitter=emitter)
                state = self._run_post_reasoning_human_review_if_needed(state, emitter=emitter)
                return self._finish(state, emitter=emitter)

        if route == "fast-path":
            state = self._run_fast_path_subgraph(state, emitter=emitter)
        else:
            state = self._run_legal_agent_subgraph(state, emitter=emitter)
        state = self._run_post_reasoning_human_review_if_needed(state, emitter=emitter)
        return self._finish(state, emitter=emitter)

    def _finish(self, state: AgentState, *, emitter: Callable[[dict[str, Any]], None] | None = None) -> AgentState:
        started_at = time.perf_counter()
        state = self._apply(state, citation_format_node(state))
        state = self._log_step("citation_format_node", started_at, state)
        started_at = time.perf_counter()
        state = self._apply(state, final_answer_node(state))
        state = self._log_step("final_answer_node", started_at, state)
        state = self._finalize_timing(state, emitter=emitter)
        checkpoint_id = self.checkpoint_store.save_state(state)
        state = self._apply(state, {"app_checkpoint_id": checkpoint_id})

        if self._has_pending_user_input(state):
            event_name = "clarify_required" if str(state.get("resume_kind") or "") == "clarify" else "review_required"
            self._emit(emitter, event_name, **self._waiting_event_payload(state))
        elif state.get("final_answer"):
            self._emit(
                emitter,
                "final_answer",
                final_answer=state.get("final_answer"),
                sources=list(state.get("sources") or []),
                status=state.get("response_status"),
                thread_id=state.get("thread_id"),
                session_id=state.get("session_id"),
                timing=self._build_timing_payload(state),
                total_elapsed_ms=round(float(state.get("total_elapsed_ms") or 0.0), 3),
            )
        return state

    def invoke(
        self,
        state: Mapping[str, Any],
        *,
        emitter: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentState:
        """Invoke the top-level graph for one chat request."""

        working_state = self._start_timing(clone_state(state), phase="invoke")
        started_at = time.perf_counter()
        working_state = self._apply(working_state, analyze_node(working_state))
        working_state = self._log_step("analyze_node", started_at, working_state)

        started_at = time.perf_counter()
        working_state = self._apply(working_state, self.dependencies.route_node(working_state))
        working_state = self._log_step("route_node", started_at, working_state)
        self._emit(
            emitter,
            "route",
            route=working_state.get("next_route"),
            risk_level=working_state.get("risk_level"),
            intent=working_state.get("intent"),
            reason=working_state.get("route_reason"),
            elapsed_ms=float((working_state.get("timing_debug") or {}).get("steps", [{}])[-1].get("elapsed_ms") or 0.0),
        )

        return self._continue_after_route(working_state, emitter=emitter)

    def resume(
        self,
        *,
        thread_id: str,
        session_id: str,
        review_response: str = "",
        clarify_response: str = "",
        note: str = "",
        emitter: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentState:
        """Resume a paused/interrupt state from the local checkpoint store."""

        state = self._start_timing(
            self.checkpoint_store.load_state(thread_id=thread_id, session_id=session_id),
            phase="resume",
        )
        interrupt_payload = dict(state.get("interrupt_payload") or {})
        if not interrupt_payload:
            raise ValueError("Checkpoint exists but there is no pending interrupt to resume.")

        interrupt_kind = str(interrupt_payload.get("kind") or "")
        if interrupt_kind == "clarify":
            response_text = clarify_response or note
            if not response_text:
                raise ValueError("Clarify resume requires `clarify_response` or `note`.")
            state = self._apply(state, {"clarify_response": response_text})
            started_at = time.perf_counter()
            state = self._apply(state, manual_clarify_node(state))
            state = self._log_step("clarify_node_resume", started_at, state, resumed=True)
            started_at = time.perf_counter()
            state = self._apply(state, analyze_node(state))
            state = self._log_step("analyze_node_resume", started_at, state, resumed=True)
            started_at = time.perf_counter()
            state = self._apply(state, self.dependencies.route_node(state))
            state = self._log_step("route_node_resume", started_at, state, resumed=True)
            self._emit(
                emitter,
                "route",
                route=state.get("next_route"),
                risk_level=state.get("risk_level"),
                intent=state.get("intent"),
                reason=state.get("route_reason"),
                resumed=True,
                elapsed_ms=float((state.get("timing_debug") or {}).get("steps", [{}])[-1].get("elapsed_ms") or 0.0),
            )
            return self._continue_after_route(state, emitter=emitter)

        if interrupt_kind == "human_review":
            response_text = review_response or note
            if not response_text:
                raise ValueError("Human-review resume requires `review_response` or `note`.")
            state = self._apply(state, {"review_response": response_text})
            started_at = time.perf_counter()
            state = self._apply(state, self._manual_human_review_step(state))
            state = self._log_step("human_review_node_resume", started_at, state, resumed=True)
            if self._has_pending_user_input(state) or str(state.get("next_action") or "") == "stop_after_review":
                return self._finish(state, emitter=emitter)
            if str(state.get("next_action") or "") == "resume_legal_agent":
                state = self._run_legal_agent_subgraph(state, emitter=emitter)
                state = self._run_post_reasoning_human_review_if_needed(state, emitter=emitter)
            return self._finish(state, emitter=emitter)

        raise ValueError(f"Unsupported interrupt kind: {interrupt_kind}")

    def stream(self, state: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
        """Yield graph events incrementally for SSE streaming."""

        event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()

        def emit(event: dict[str, Any]) -> None:
            event_queue.put(event)

        def worker() -> None:
            try:
                final_state = self.invoke(state, emitter=emit)
                event_queue.put({"event": "final_state", "data": self._final_state_payload(final_state)})
            except Exception as exc:  # pragma: no cover - runtime path.
                self.logger.exception("Streaming graph invoke failed.")
                event_queue.put(
                    {
                        "event": "error",
                        "data": {
                            "status": "error",
                            "message": f"Graph execution failed: {exc}",
                        },
                    }
                )
            finally:
                event_queue.put(None)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = event_queue.get()
            if item is None:
                break
            yield item

    def stream_resume(
        self,
        *,
        thread_id: str,
        session_id: str,
        review_response: str = "",
        clarify_response: str = "",
        note: str = "",
    ) -> Iterator[dict[str, Any]]:
        """Yield ordered SSE events incrementally while resuming a paused thread."""

        event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()

        def emit(event: dict[str, Any]) -> None:
            event_queue.put(event)

        def worker() -> None:
            try:
                final_state = self.resume(
                    thread_id=thread_id,
                    session_id=session_id,
                    review_response=review_response,
                    clarify_response=clarify_response,
                    note=note,
                    emitter=emit,
                )
                event_queue.put({"event": "final_state", "data": self._final_state_payload(final_state)})
            except Exception as exc:  # pragma: no cover - runtime path.
                self.logger.exception("Streaming graph resume failed.")
                event_queue.put(
                    {
                        "event": "error",
                        "data": {
                            "status": "error",
                            "message": f"Resume failed: {exc}",
                        },
                    }
                )
            finally:
                event_queue.put(None)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = event_queue.get()
            if item is None:
                break
            yield item


def build_graph(
    *,
    app_config: AppConfig | None = None,
    app_config_path: str | Path | None = None,
    checkpoint_store: BaseCheckpointStore | None = None,
    dependencies: GraphDependencies | None = None,
    logger: logging.Logger | None = None,
) -> LegalQAGraphRuntime:
    """Build the top-level orchestration runtime and compile the LangGraph app."""

    resolved_logger = logger or LOGGER
    resolved_app_config = app_config or load_app_config(app_config_path)
    resolved_dependencies = dependencies or GraphDependencies()
    resolved_checkpoint_store = checkpoint_store or create_checkpoint_store(
        resolved_app_config.checkpoint_backend,
        base_dir=resolved_app_config.checkpoint_dir,
        logger=resolved_logger,
    )

    def prepare_full_from_fast_node(state: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "execution_profile": "full",
            "fast_path_enabled": False,
            "next_route": "legal-agent-path",
            "next_action": "",
            "retrieval_ok": False,
            "grounding_ok": False,
            "retrieved_docs": [],
            "reranked_docs": [],
            "context": "",
            "sources": [],
            "draft_answer": "",
            "final_answer": "",
            "unsupported_claims": [],
            "missing_evidence": [],
            "grounding_score": 0.0,
        }

    main_graph = StateGraph(AgentState)
    legal_agent_subgraph = build_legal_agent_subgraph(
        logger=resolved_logger,
        max_reasoning_loops=resolved_app_config.max_reasoning_loops,
    )
    review_subgraph = build_review_subgraph(logger=resolved_logger)
    fast_path_graph = StateGraph(AgentState)
    fast_path_graph.add_node("rewrite_query_node", resolved_dependencies.rewrite_query_node)
    fast_path_graph.add_node("retrieve_node", resolved_dependencies.retrieve_node)
    fast_path_graph.add_node("rerank_node", resolved_dependencies.rerank_node)
    fast_path_graph.add_node("retrieval_check_node", resolved_dependencies.retrieval_check_node)
    fast_path_graph.add_node("generate_draft_node", resolved_dependencies.generate_draft_node)
    fast_path_graph.add_node("grounding_check_node", resolved_dependencies.grounding_check_node)
    fast_path_graph.add_edge(START, "rewrite_query_node")
    fast_path_graph.add_edge("rewrite_query_node", "retrieve_node")
    fast_path_graph.add_edge("retrieve_node", "rerank_node")
    fast_path_graph.add_edge("rerank_node", "retrieval_check_node")
    fast_path_graph.add_conditional_edges(
        "retrieval_check_node",
        lambda state: str(state.get("next_action") or "proceed"),
        {
            "proceed": "generate_draft_node",
            "escalate_to_full": END,
            "fallback": END,
            "retry": END,
        },
    )
    fast_path_graph.add_edge("generate_draft_node", "grounding_check_node")
    fast_path_graph.add_conditional_edges(
        "grounding_check_node",
        lambda state: str(state.get("next_action") or "proceed"),
        {
            "proceed": END,
            "escalate_to_full": END,
            "retrieve_again": END,
            "revise": END,
            "human_review": END,
        },
    )
    fast_path_subgraph = fast_path_graph.compile()

    main_graph.add_node("analyze_node", analyze_node)
    main_graph.add_node("route_node", resolved_dependencies.route_node)
    main_graph.add_node("clarify_node", clarify_node)
    main_graph.add_node("unsupported_node", unsupported_node)
    main_graph.add_node("fast_path_subgraph", fast_path_subgraph)
    main_graph.add_node("prepare_full_from_fast_node", prepare_full_from_fast_node)
    main_graph.add_node("legal_agent_subgraph", legal_agent_subgraph)
    main_graph.add_node("human_review_node", review_subgraph)
    main_graph.add_node("citation_format_node", citation_format_node)
    main_graph.add_node("final_answer_node", final_answer_node)

    main_graph.add_edge(START, "analyze_node")
    main_graph.add_edge("analyze_node", "route_node")
    main_graph.add_conditional_edges(
        "route_node",
        lambda state: str(state.get("next_route") or "legal-agent-path"),
        {
            "clarify-path": "clarify_node",
            "unsupported-path": "unsupported_node",
            "human-review-path": "human_review_node",
            "legal-agent-path": "legal_agent_subgraph",
            "fast-path": "fast_path_subgraph",
        },
    )
    main_graph.add_edge("clarify_node", "analyze_node")
    main_graph.add_edge("unsupported_node", "citation_format_node")
    main_graph.add_conditional_edges(
        "fast_path_subgraph",
        lambda state: "legal-agent-path"
        if str(state.get("next_action") or "") == "escalate_to_full"
        or str(state.get("next_route") or "") == "legal-agent-path"
        else "done",
        {
            "legal-agent-path": "prepare_full_from_fast_node",
            "done": "citation_format_node",
        },
    )
    main_graph.add_edge("prepare_full_from_fast_node", "legal_agent_subgraph")
    main_graph.add_edge("legal_agent_subgraph", "citation_format_node")
    main_graph.add_edge("human_review_node", "citation_format_node")
    main_graph.add_edge("citation_format_node", "final_answer_node")
    main_graph.add_edge("final_answer_node", END)
    compiled_graph = main_graph.compile()

    return LegalQAGraphRuntime(
        app_config=resolved_app_config,
        checkpoint_store=resolved_checkpoint_store,
        dependencies=resolved_dependencies,
        compiled_graph=compiled_graph,
        logger=resolved_logger,
    )


__all__ = [
    "AppConfig",
    "GraphDependencies",
    "LegalQAGraphRuntime",
    "analyze_node",
    "build_graph",
    "citation_format_node",
    "clarify_node",
    "final_answer_node",
    "load_app_config",
    "retrieval_fallback_node",
    "unsupported_node",
]

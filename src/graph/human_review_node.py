from __future__ import annotations

import logging
import os
from functools import wraps
from typing import Any, Mapping

from langgraph.types import interrupt

from src.graph.state import append_history

LOGGER = logging.getLogger(__name__)

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

_POSITIVE_TOKENS = {"đồng ý", "duyệt", "ok", "tiếp tục", "approve", "yes", "được"}
_NEGATIVE_TOKENS = {"không", "từ chối", "stop", "dừng", "reject", "no"}


def _detect_review_stage(state: Mapping[str, Any]) -> str:
    if str(state.get("draft_answer") or "").strip() or str(state.get("final_answer") or "").strip() or str(
        state.get("context") or ""
    ).strip():
        return "post_reasoning"
    return "pre_retrieval"


def _build_interrupt_payload(state: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
    answer_preview = str(state.get("final_answer") or state.get("draft_answer") or "").strip()
    payload = {
        "kind": "human_review",
        "stage": stage,
        "thread_id": str(state.get("thread_id") or ""),
        "session_id": str(state.get("session_id") or ""),
        "question": str(state.get("normalized_question") or state.get("question") or ""),
        "review_note": str(state.get("review_note") or ""),
        "answer_preview": answer_preview,
        "sources": list(state.get("sources") or []),
    }
    return payload


def _classify_review_response(response_text: str) -> str:
    normalized = response_text.strip().lower()
    if any(token in normalized for token in _NEGATIVE_TOKENS):
        return "rejected"
    if any(token in normalized for token in _POSITIVE_TOKENS):
        return "approved"
    return "noted"


def _apply_review_response(state: Mapping[str, Any], response_text: str, *, stage: str) -> dict[str, Any]:
    decision = _classify_review_response(response_text)
    history = append_history(
        state,
        role="human_reviewer",
        content=response_text,
        kind="human_review",
        metadata={"stage": stage, "decision": decision},
    )

    if stage == "pre_retrieval":
        if decision == "rejected":
            return {
                "human_review_required": True,
                "interrupt_payload": None,
                "next_action": "stop_after_review",
                "response_status": "review_required",
                "status": "review_required",
                "review_note": "Human reviewer chưa chấp thuận cho hệ thống tiếp tục xử lý câu hỏi này.",
                "history": history,
            }
        return {
            "human_review_required": False,
            "interrupt_payload": None,
            "next_action": "resume_legal_agent",
            "response_status": "",
            "status": "",
            "resume_kind": "",
            "resume_question": "",
            "review_note": f"Đã nhận phản hồi human review: {response_text}",
            "history": history,
        }

    if decision == "rejected":
        return {
            "human_review_required": True,
            "interrupt_payload": None,
            "next_action": "stop_after_review",
            "response_status": "review_required",
            "status": "review_required",
            "review_note": "Human reviewer yêu cầu giữ trạng thái review trước khi phát hành câu trả lời cuối.",
            "history": history,
        }

    return {
        "human_review_required": False,
        "interrupt_payload": None,
        "next_action": "proceed",
        "response_status": "",
        "status": "",
        "resume_kind": "",
        "resume_question": "",
        "review_note": f"Đã nhận phản hồi human review: {response_text}",
        "history": history,
    }


@_optional_traceable(name="tv6.human_review_node", run_type="chain", tags=["tv6", "review"])
def human_review_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Pause for human review or apply a human-review resume payload."""

    if not bool(state.get("human_review_required")):
        return {}

    stage = _detect_review_stage(state)
    review_response = str(state.get("review_response") or "").strip()
    if review_response:
        return _apply_review_response(state, review_response, stage=stage)

    payload = _build_interrupt_payload(state, stage=stage)
    resume_value = interrupt(payload)
    if isinstance(resume_value, Mapping):
        review_response = str(resume_value.get("review_response") or resume_value.get("note") or "").strip()
    else:
        review_response = str(resume_value or "").strip()
    if not review_response:
        return {
            "interrupt_payload": payload,
            "next_action": "human_review",
            "response_status": "review_required",
            "status": "review_required",
        }
    return _apply_review_response(state, review_response, stage=stage)


@_optional_traceable(name="tv6.manual_human_review_node", run_type="chain", tags=["tv6", "review"])
def manual_human_review_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Manual-runtime equivalent of the human-review node without LangGraph interrupt."""

    if not bool(state.get("human_review_required")):
        return {}

    stage = _detect_review_stage(state)
    review_response = str(state.get("review_response") or "").strip()
    if review_response:
        return _apply_review_response(state, review_response, stage=stage)

    return {
        "interrupt_payload": _build_interrupt_payload(state, stage=stage),
        "next_action": "human_review",
        "response_status": "review_required",
        "status": "review_required",
    }


__all__ = ["human_review_node", "manual_human_review_node"]

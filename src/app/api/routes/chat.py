from __future__ import annotations

import logging
import os
from functools import wraps
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.graph.builder import LegalQAGraphRuntime
from src.graph.state import AgentState, create_initial_state

router = APIRouter(tags=["chat"])
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


def _safe_preview(text: str, *, max_chars: int = 240) -> str:
    normalized = " ".join((text or "").split())
    return normalized[:max_chars].strip()


def _request_trace_extra(*, endpoint: str, session_id: str | None, thread_id: str | None, question: str = "") -> dict[str, Any]:
    metadata = {
        "endpoint": endpoint,
        "session_id": str(session_id or ""),
        "thread_id": str(thread_id or ""),
    }
    if question:
        metadata["question_preview"] = _safe_preview(question)
    return {"metadata": metadata, "tags": ["api", "chat", "tv6"]}


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = None
    thread_id: str | None = None


class ChatResponse(BaseModel):
    status: str
    final_answer: str = ""
    resume_kind: str = ""
    resume_question: str = ""
    clarify_question: str = ""
    sources: list[str] = Field(default_factory=list)
    route: str = ""
    risk_level: str = ""
    thread_id: str = ""
    session_id: str = ""
    review_note: str = ""
    human_review_required: bool = False
    interrupt_payload: dict[str, Any] | None = None
    timing: dict[str, Any] = Field(default_factory=dict)
    total_elapsed_ms: float = 0.0


def _get_runtime(request: Request) -> LegalQAGraphRuntime:
    runtime = getattr(request.app.state, "graph_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=500, detail="Graph runtime is not initialized.")
    return runtime


def build_chat_response(state: AgentState) -> ChatResponse:
    """Convert AgentState into the stable /chat API response contract."""

    status = str(state.get("response_status") or state.get("status") or "ok")
    resume_question = str(state.get("resume_question") or state.get("clarify_question") or "")
    return ChatResponse(
        status=status,
        final_answer=str(state.get("final_answer") or ""),
        resume_kind=str(state.get("resume_kind") or ""),
        resume_question=resume_question,
        clarify_question=str(state.get("clarify_question") or ""),
        sources=list(state.get("sources") or []),
        route=str(state.get("next_route") or ""),
        risk_level=str(state.get("risk_level") or ""),
        thread_id=str(state.get("thread_id") or ""),
        session_id=str(state.get("session_id") or ""),
        review_note=str(state.get("review_note") or ""),
        human_review_required=bool(state.get("human_review_required") or False),
        interrupt_payload=state.get("interrupt_payload"),
        timing=dict(state.get("timing_debug") or {}),
        total_elapsed_ms=float(state.get("total_elapsed_ms") or 0.0),
    )


@_optional_traceable(name="api.chat.request", run_type="chain", tags=["api", "chat", "tv6"])
def _invoke_chat_request(runtime: LegalQAGraphRuntime, state: AgentState) -> AgentState:
    return runtime.invoke(state)


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Handle one synchronous chat request against the compiled legal QA graph."""

    runtime = _get_runtime(request)
    try:
        state = create_initial_state(
            question=payload.question,
            session_id=payload.session_id,
            thread_id=payload.thread_id,
        )
        final_state = _invoke_chat_request(
            runtime,
            state,
            langsmith_extra=_request_trace_extra(
                endpoint="/chat",
                session_id=payload.session_id,
                thread_id=payload.thread_id,
                question=payload.question,
            ),
        )
    except Exception as exc:  # pragma: no cover - runtime path.
        LOGGER.exception("Chat request failed.")
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {exc}") from exc
    return build_chat_response(final_state)


__all__ = ["ChatRequest", "ChatResponse", "build_chat_response", "router"]

from __future__ import annotations

import json
import logging
import os
from functools import wraps
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.app.api.routes.chat import ChatResponse, build_chat_response
from src.graph.builder import LegalQAGraphRuntime
from src.graph.checkpointing import CheckpointNotFoundError
from src.graph.state import create_initial_state

router = APIRouter(tags=["stream"])
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


def _request_trace_extra(
    *,
    endpoint: str,
    session_id: str | None,
    thread_id: str | None,
    question: str = "",
    note: str = "",
) -> dict[str, Any]:
    metadata = {
        "endpoint": endpoint,
        "session_id": str(session_id or ""),
        "thread_id": str(thread_id or ""),
    }
    if question:
        metadata["question_preview"] = _safe_preview(question)
    if note:
        metadata["note_preview"] = _safe_preview(note)
    return {"metadata": metadata, "tags": ["api", "stream", "tv6"]}


class ResumeRequest(BaseModel):
    thread_id: str
    session_id: str
    review_response: str | None = None
    clarify_response: str | None = None
    note: str | None = None


def _get_runtime(request: Request) -> LegalQAGraphRuntime:
    runtime = getattr(request.app.state, "graph_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=500, detail="Graph runtime is not initialized.")
    return runtime


def _sse_encode(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream_events(events: Iterator[dict[str, Any]]) -> Iterator[str]:
    for item in events:
        yield _sse_encode(str(item.get("event") or "message"), dict(item.get("data") or {}))


class StreamChatRequest(BaseModel):
    question: str
    session_id: str | None = None
    thread_id: str | None = None


@_optional_traceable(name="api.chat.stream_request", run_type="chain", tags=["api", "stream", "tv6"])
def _invoke_stream_request(runtime: LegalQAGraphRuntime, state: dict[str, Any]) -> Iterator[dict[str, Any]]:
    return runtime.stream(state)


@_optional_traceable(name="api.chat.resume_request", run_type="chain", tags=["api", "resume", "tv6"])
def _invoke_resume_request(
    runtime: LegalQAGraphRuntime,
    *,
    thread_id: str,
    session_id: str,
    review_response: str,
    clarify_response: str,
    note: str,
) -> Any:
    return runtime.resume(
        thread_id=thread_id,
        session_id=session_id,
        review_response=review_response,
        clarify_response=clarify_response,
        note=note,
    )


@router.post("/chat/stream")
def stream_chat(payload: StreamChatRequest, request: Request) -> StreamingResponse:
    """Stream graph progress as SSE events for a new chat request."""

    runtime = _get_runtime(request)
    state = create_initial_state(
        question=payload.question,
        session_id=payload.session_id,
        thread_id=payload.thread_id,
    )
    events = _invoke_stream_request(
        runtime,
        state,
        langsmith_extra=_request_trace_extra(
            endpoint="/chat/stream",
            session_id=payload.session_id,
            thread_id=payload.thread_id,
            question=payload.question,
        ),
    )
    return StreamingResponse(_stream_events(events), media_type="text/event-stream")


@router.get("/chat/stream")
def stream_chat_get(
    request: Request,
    question: str = Query(...),
    session_id: str | None = Query(default=None),
    thread_id: str | None = Query(default=None),
) -> StreamingResponse:
    """GET variant of the SSE endpoint for environments that prefer query params."""

    runtime = _get_runtime(request)
    state = create_initial_state(question=question, session_id=session_id, thread_id=thread_id)
    events = _invoke_stream_request(
        runtime,
        state,
        langsmith_extra=_request_trace_extra(
            endpoint="/chat/stream",
            session_id=session_id,
            thread_id=thread_id,
            question=question,
        ),
    )
    return StreamingResponse(_stream_events(events), media_type="text/event-stream")


@router.post("/chat/resume", response_model=ChatResponse)
def resume_chat(payload: ResumeRequest, request: Request) -> ChatResponse:
    """Resume a graph after clarify/human-review interrupt."""

    runtime = _get_runtime(request)
    try:
        final_state = _invoke_resume_request(
            runtime,
            thread_id=payload.thread_id,
            session_id=payload.session_id,
            review_response=str(payload.review_response or ""),
            clarify_response=str(payload.clarify_response or ""),
            note=str(payload.note or ""),
            langsmith_extra=_request_trace_extra(
                endpoint="/chat/resume",
                session_id=payload.session_id,
                thread_id=payload.thread_id,
                note=str(payload.note or payload.review_response or payload.clarify_response or ""),
            ),
        )
    except CheckpointNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime path.
        LOGGER.exception("Resume request failed.")
        raise HTTPException(status_code=500, detail=f"Resume failed: {exc}") from exc
    return build_chat_response(final_state)


__all__ = ["ResumeRequest", "StreamChatRequest", "router"]

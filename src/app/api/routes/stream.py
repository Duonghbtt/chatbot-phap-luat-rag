from __future__ import annotations

import json
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.app.api.routes.chat import ChatResponse, build_chat_response
from src.graph.builder import LegalQAGraphRuntime
from src.graph.checkpointing import CheckpointNotFoundError
from src.graph.state import create_initial_state

router = APIRouter(tags=["stream"])


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


@router.post("/chat/stream")
def stream_chat(payload: StreamChatRequest, request: Request) -> StreamingResponse:
    """Stream graph progress as SSE events for a new chat request."""

    runtime = _get_runtime(request)
    state = create_initial_state(
        question=payload.question,
        session_id=payload.session_id,
        thread_id=payload.thread_id,
    )
    return StreamingResponse(_stream_events(runtime.stream(state)), media_type="text/event-stream")


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
    return StreamingResponse(_stream_events(runtime.stream(state)), media_type="text/event-stream")


@router.post("/chat/resume", response_model=ChatResponse)
def resume_chat(payload: ResumeRequest, request: Request) -> ChatResponse:
    """Resume a graph after clarify/human-review interrupt."""

    runtime = _get_runtime(request)
    try:
        final_state = runtime.resume(
            thread_id=payload.thread_id,
            session_id=payload.session_id,
            review_response=str(payload.review_response or ""),
            clarify_response=str(payload.clarify_response or ""),
            note=str(payload.note or ""),
        )
    except CheckpointNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime path.
        raise HTTPException(status_code=500, detail=f"Resume failed: {exc}") from exc
    return build_chat_response(final_state)


__all__ = ["ResumeRequest", "StreamChatRequest", "router"]

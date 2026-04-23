from __future__ import annotations

import re
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping, TypedDict


class AgentState(TypedDict, total=False):
    """Shared LangGraph state for the legal QA multi-agent system."""

    question: str
    normalized_question: str
    intent: str
    intent_score: float
    risk_level: str
    rewritten_queries: list[str]
    retrieved_docs: list[dict[str, Any]]
    reranked_docs: list[dict[str, Any]]
    context: str
    sources: list[str]
    draft_answer: str
    final_answer: str
    retrieval_ok: bool
    grounding_ok: bool
    need_clarify: bool
    unsupported_query: bool
    human_review_required: bool
    review_note: str
    route_reason: str
    next_route: str
    next_action: str
    loop_count: int
    history: list[dict[str, Any]]
    thread_id: str
    session_id: str
    interrupt_payload: dict[str, Any] | None
    retrieval_debug: dict[str, Any]
    citation_findings: dict[str, Any]
    unsupported_claims: list[str]
    missing_evidence: list[str]
    grounding_score: float
    reasoning_notes: dict[str, Any]
    app_checkpoint_id: str
    clarify_question: str
    clarify_reason: str
    risk_reason: str
    top_intents: list[dict[str, Any]]
    retrieval_failure_reason: str
    response_status: str
    status: str
    draft_citations: list[str]
    draft_confidence: float
    review_response: str
    clarify_response: str
    missing_slots: list[str]
    resume_kind: str
    resume_question: str
    execution_profile: str
    fast_path_enabled: bool
    timing_debug: dict[str, Any]
    total_elapsed_ms: float
    timing_started_at: float


DEFAULT_STATE: AgentState = {
    "question": "",
    "normalized_question": "",
    "intent": "",
    "intent_score": 0.0,
    "risk_level": "",
    "rewritten_queries": [],
    "retrieved_docs": [],
    "reranked_docs": [],
    "context": "",
    "sources": [],
    "draft_answer": "",
    "final_answer": "",
    "retrieval_ok": False,
    "grounding_ok": False,
    "need_clarify": False,
    "unsupported_query": False,
    "human_review_required": False,
    "review_note": "",
    "route_reason": "",
    "next_route": "",
    "next_action": "",
    "loop_count": 0,
    "history": [],
    "thread_id": "",
    "session_id": "",
    "interrupt_payload": None,
    "retrieval_debug": {},
    "citation_findings": {},
    "unsupported_claims": [],
    "missing_evidence": [],
    "grounding_score": 0.0,
    "reasoning_notes": {},
    "app_checkpoint_id": "",
    "clarify_question": "",
    "clarify_reason": "",
    "risk_reason": "",
    "top_intents": [],
    "retrieval_failure_reason": "",
    "response_status": "",
    "status": "",
    "draft_citations": [],
    "draft_confidence": 0.0,
    "review_response": "",
    "clarify_response": "",
    "missing_slots": [],
    "resume_kind": "",
    "resume_question": "",
    "execution_profile": "",
    "fast_path_enabled": False,
    "timing_debug": {},
    "total_elapsed_ms": 0.0,
    "timing_started_at": 0.0,
}


def utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_user_text(text: str) -> str:
    """Normalize user-provided legal questions while preserving meaning."""

    return re.sub(r"\s+", " ", (text or "").strip())


def new_thread_id() -> str:
    """Generate a stable thread identifier for checkpointing."""

    return f"thread-{uuid.uuid4()}"


def new_session_id() -> str:
    """Generate a stable session identifier for checkpointing."""

    return f"session-{uuid.uuid4()}"


def clone_state(state: Mapping[str, Any] | None = None) -> AgentState:
    """Create a deep-copied state with all default fields populated."""

    merged: AgentState = deepcopy(DEFAULT_STATE)
    if state:
        for key, value in dict(state).items():
            merged[str(key)] = deepcopy(value)
    return merged


def merge_state(base_state: Mapping[str, Any], updates: Mapping[str, Any] | None) -> AgentState:
    """Merge state updates into an existing AgentState copy."""

    merged = clone_state(base_state)
    for key, value in dict(updates or {}).items():
        merged[str(key)] = value
    return merged


def append_history(
    state: Mapping[str, Any],
    *,
    role: str,
    content: str,
    kind: str = "message",
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Append one event to the state history and return the updated list."""

    history = [dict(item) for item in list(state.get("history") or [])]
    history.append(
        {
            "timestamp": utc_now_iso(),
            "role": role,
            "kind": kind,
            "content": content,
            "metadata": dict(metadata or {}),
        }
    )
    return history


def reset_for_new_question(state: Mapping[str, Any], *, question: str) -> AgentState:
    """Reset transient retrieval/reasoning fields for a fresh question in the same thread."""

    merged = clone_state(state)
    normalized_question = normalize_user_text(question)
    for field_name, default_value in DEFAULT_STATE.items():
        if field_name in {"history", "thread_id", "session_id", "app_checkpoint_id"}:
            continue
        merged[field_name] = deepcopy(default_value)
    merged["question"] = normalized_question
    merged["normalized_question"] = normalized_question
    merged["thread_id"] = str(state.get("thread_id") or new_thread_id())
    merged["session_id"] = str(state.get("session_id") or new_session_id())
    merged["history"] = append_history(
        state,
        role="user",
        content=normalized_question,
        kind="question",
    )
    return merged


def create_initial_state(
    *,
    question: str,
    session_id: str | None = None,
    thread_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> AgentState:
    """Create a new AgentState ready to enter the top-level graph."""

    state = clone_state(extra)
    normalized_question = normalize_user_text(question)
    state["question"] = normalized_question
    state["normalized_question"] = normalized_question
    state["thread_id"] = str(thread_id or extra.get("thread_id") if extra else thread_id or new_thread_id())
    if not state["thread_id"]:
        state["thread_id"] = new_thread_id()
    state["session_id"] = str(session_id or extra.get("session_id") if extra else session_id or new_session_id())
    if not state["session_id"]:
        state["session_id"] = new_session_id()
    state["history"] = append_history(
        state,
        role="user",
        content=normalized_question,
        kind="question",
    )
    return state


__all__ = [
    "AgentState",
    "DEFAULT_STATE",
    "append_history",
    "clone_state",
    "create_initial_state",
    "merge_state",
    "new_session_id",
    "new_thread_id",
    "normalize_user_text",
    "reset_for_new_question",
    "utc_now_iso",
]

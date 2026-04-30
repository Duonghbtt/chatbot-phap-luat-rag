from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping
from urllib import error, request

import streamlit as st

LOGGER = logging.getLogger(__name__)
DEFAULT_APP_CONFIG_PATH = Path("configs/app.yaml")
WAITING_USER_INPUT_STATUS = "waiting_user_input"
DEFAULT_CHECKPOINTS_DIR = Path(".checkpoints")
MAX_LOCAL_CONVERSATIONS = 30


@dataclass(slots=True, frozen=True)
class AppUIConfig:
    """Frontend configuration for the Streamlit legal QA client."""

    api_base_url: str = "http://localhost:8000"
    timeout_seconds: int = 120
    use_stream_endpoint: bool = True
    page_title: str = "Chatbot Pháp Luật Agentic"
    page_icon: str = "\u2696\ufe0f"
    app_title: str = "Hệ thống hỏi đáp pháp luật đa tác tử"
    app_subtitle: str = "LangGraph agentic mức 3 - Router, Retrieval, Grounding, Review"


@dataclass(slots=True, frozen=True)
class LocalConversationSummary:
    session_id: str
    thread_id: str
    updated_at: str
    title: str
    status: str
    resume_kind: str
    file_path: Path


def _load_yaml_module() -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("PyYAML is required to load configs/app.yaml.") from exc
    return yaml


def load_app_config(config_path: str | Path | None = None) -> AppUIConfig:
    """Load optional UI config from `configs/app.yaml`, falling back to safe defaults."""

    resolved_path = Path(config_path or DEFAULT_APP_CONFIG_PATH).resolve()
    if not resolved_path.exists() or resolved_path.stat().st_size == 0:
        return AppUIConfig()

    yaml = _load_yaml_module()
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        return AppUIConfig()

    return AppUIConfig(
        api_base_url=str(payload.get("api_base_url") or "http://localhost:8000"),
        timeout_seconds=int(payload.get("timeout_seconds") or 120),
        use_stream_endpoint=bool(payload.get("use_stream_endpoint", True)),
        page_title=str(payload.get("page_title") or "Chatbot Pháp Luật Agentic"),
        page_icon=str(payload.get("page_icon") or "\u2696\ufe0f"),
        app_title=str(payload.get("app_title") or "Hệ thống hỏi đáp pháp luật đa tác tử"),
        app_subtitle=str(
            payload.get("app_subtitle") or "LangGraph agentic mức 3 - Router, Retrieval, Grounding, Review"
        ),
    )


def configure_page(config: AppUIConfig) -> None:
    st.set_page_config(page_title=config.page_title, page_icon=config.page_icon, layout="wide")
    st.markdown(
        """
        <style>
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
            flex-direction: row-reverse;
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
            text-align: right;
            align-items: flex-end;
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {
            text-align: right;
        }
        .badge {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
        .badge-route { background: #d9edf7; color: #0b4f6c; }
        .badge-risk-low { background: #e6f6ec; color: #1e6f3d; }
        .badge-risk-medium { background: #fff4d6; color: #8a5d00; }
        .badge-risk-high { background: #fde7e9; color: #9f1c2a; }
        .badge-status { background: #ece9ff; color: #4a3cb2; }
        .source-box {
            border-left: 3px solid #d0d7de;
            padding-left: 0.75rem;
            margin: 0.4rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4()}"


def _truncate_text(text: str, *, max_chars: int = 72) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3].rstrip()}..."


def _parse_iso_datetime(raw_value: str) -> datetime:
    value = str(raw_value or "").strip()
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min


def _status_label(status: str, resume_kind: str = "") -> str:
    normalized_status = str(status or "").strip() or "unknown"
    if normalized_status == WAITING_USER_INPUT_STATUS and resume_kind:
        return f"{normalized_status} ({resume_kind})"
    return normalized_status


def _conversation_title_from_state(state: Mapping[str, Any], thread_id: str) -> str:
    question = str(state.get("question") or state.get("normalized_question") or "").strip()
    if question:
        return _truncate_text(question)

    history = list(state.get("history") or [])
    for item in history:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("role") or "") == "user":
            content = str(item.get("content") or "").strip()
            if content:
                return _truncate_text(content)
    return thread_id


def _load_local_conversations(
    *,
    base_dir: str | Path = DEFAULT_CHECKPOINTS_DIR,
    limit: int = MAX_LOCAL_CONVERSATIONS,
) -> list[LocalConversationSummary]:
    resolved_base_dir = Path(base_dir).resolve()
    if not resolved_base_dir.exists():
        return []

    summaries: list[LocalConversationSummary] = []
    for checkpoint_path in resolved_base_dir.glob("*/*.json"):
        try:
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("Skipping unreadable checkpoint file %s: %s", checkpoint_path, exc)
            continue
        if not isinstance(payload, Mapping):
            continue
        state = payload.get("state") or {}
        if not isinstance(state, Mapping):
            continue

        session_id = str(payload.get("session_id") or state.get("session_id") or checkpoint_path.parent.name).strip()
        thread_id = str(payload.get("thread_id") or state.get("thread_id") or checkpoint_path.stem).strip()
        updated_at = str(payload.get("updated_at") or "").strip()
        status = str(state.get("response_status") or state.get("status") or "ok").strip()
        resume_kind = str(state.get("resume_kind") or "").strip()
        summaries.append(
            LocalConversationSummary(
                session_id=session_id,
                thread_id=thread_id,
                updated_at=updated_at,
                title=_conversation_title_from_state(state, thread_id),
                status=status,
                resume_kind=resume_kind,
                file_path=checkpoint_path,
            )
        )

    summaries.sort(key=lambda item: (_parse_iso_datetime(item.updated_at), item.thread_id), reverse=True)
    return summaries[: max(1, limit)]


def _load_checkpoint_state_from_file(file_path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(file_path).resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint file has invalid structure: {file_path}")
    state = payload.get("state") or {}
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint file missing `state`: {file_path}")
    return dict(state)


def _history_entry_to_message(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    role = str(entry.get("role") or "").strip()
    content = str(entry.get("content") or "").strip()
    if not role or not content:
        return None

    if role == "assistant":
        metadata = dict(entry.get("metadata") or {})
        payload: dict[str, Any] = {}
        if metadata.get("status"):
            payload["status"] = metadata.get("status")
        return {"role": "assistant", "content": content, "payload": payload}

    if role in {"user", "human_reviewer"}:
        return {"role": "user", "content": content, "payload": {}}

    return None


def _build_messages_from_checkpoint_state(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    history = list(state.get("history") or [])
    for raw_entry in history:
        if not isinstance(raw_entry, Mapping):
            continue
        converted = _history_entry_to_message(raw_entry)
        if converted:
            messages.append(converted)

    normalized_payload = normalize_backend_response(state)["payload"]
    pending_messages = _build_assistant_messages_from_payload(normalized_payload)
    if str(normalized_payload.get("status") or "") == WAITING_USER_INPUT_STATUS:
        for pending_message in pending_messages:
            pending_content = str(pending_message.get("content") or "").strip()
            if not pending_content:
                continue
            if (
                not messages
                or messages[-1].get("role") != "assistant"
                or str(messages[-1].get("content") or "").strip() != pending_content
            ):
                messages.append(pending_message)
    else:
        for pending_message in pending_messages:
            assistant_text = str(pending_message.get("content") or "").strip()
            if not assistant_text:
                continue
            for message in reversed(messages):
                if message.get("role") == "assistant" and str(message.get("content") or "").strip() == assistant_text:
                    merged_payload = dict(message.get("payload") or {})
                    merged_payload.update(dict(pending_message.get("payload") or {}))
                    message["payload"] = merged_payload
                    break
            else:
                messages.append(pending_message)

    return messages

def _restore_conversation_from_checkpoint_state(state: Mapping[str, Any]) -> None:
    normalized_response = normalize_backend_response(state)
    payload = dict(normalized_response["payload"])
    st.session_state["messages"] = _build_messages_from_checkpoint_state(state)
    st.session_state["session_id_value"] = str(state.get("session_id") or st.session_state.get("session_id_value") or "")
    st.session_state["thread_id_value"] = str(state.get("thread_id") or st.session_state.get("thread_id_value") or "")
    st.session_state["last_response"] = payload
    st.session_state["pending_resume"] = _is_resume_required(payload)
    st.session_state["pending_resume_kind"] = str(payload.get("resume_kind") or "")
    st.session_state["pending_resume_question"] = str(payload.get("resume_question") or "")
    st.session_state["pending_resume_note_value"] = ""
    st.session_state["refresh_control_inputs"] = True


def _format_local_conversation_timestamp(updated_at: str) -> str:
    parsed = _parse_iso_datetime(updated_at)
    if parsed == datetime.min:
        return ""
    return parsed.strftime("%Y-%m-%d %H:%M")


def _format_local_conversation_option(summary: LocalConversationSummary) -> str:
    status_text = _status_label(summary.status, summary.resume_kind)
    timestamp = _format_local_conversation_timestamp(summary.updated_at)
    if timestamp:
        return f"[{status_text}] {summary.title} — {timestamp}"
    return f"[{status_text}] {summary.title}"


def _sync_control_values_from_inputs() -> None:
    """Copy sidebar widget inputs into stable non-widget session state."""

    st.session_state["session_id_value"] = str(st.session_state.get("session_id_input") or "").strip()
    st.session_state["thread_id_value"] = str(st.session_state.get("thread_id_input") or "").strip()
    st.session_state["api_base_url_value"] = str(st.session_state.get("api_base_url_input") or "").strip()
    st.session_state["use_stream_endpoint_value"] = bool(st.session_state.get("use_stream_endpoint_input"))
    st.session_state["pending_resume_note_value"] = str(st.session_state.get("pending_resume_note_input") or "")


def _sync_inputs_from_control_values() -> None:
    """Mirror stable non-widget session state into sidebar widget inputs."""

    st.session_state["session_id_input"] = str(st.session_state.get("session_id_value") or "")
    st.session_state["thread_id_input"] = str(st.session_state.get("thread_id_value") or "")
    st.session_state["api_base_url_input"] = str(st.session_state.get("api_base_url_value") or "")
    st.session_state["use_stream_endpoint_input"] = bool(st.session_state.get("use_stream_endpoint_value"))
    st.session_state["pending_resume_note_input"] = str(st.session_state.get("pending_resume_note_value") or "")


def init_session_state(config: AppUIConfig) -> None:
    """Initialize Streamlit session state for the legal QA frontend."""

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id_value", str(st.session_state.get("session_id") or _new_id("session")))
    st.session_state.setdefault("thread_id_value", str(st.session_state.get("thread_id") or _new_id("thread")))
    st.session_state.setdefault(
        "api_base_url_value",
        str(st.session_state.get("api_base_url") or config.api_base_url.rstrip("/")),
    )
    st.session_state.setdefault(
        "use_stream_endpoint_value",
        bool(st.session_state.get("use_stream_endpoint", config.use_stream_endpoint)),
    )
    st.session_state.setdefault("pending_resume", False)
    st.session_state.setdefault("pending_resume_kind", "")
    st.session_state.setdefault("pending_resume_question", "")
    st.session_state.setdefault("pending_resume_note_value", str(st.session_state.get("pending_resume_note") or ""))
    st.session_state.setdefault("last_response", {})
    st.session_state.setdefault("local_conversation_file", "")
    st.session_state.setdefault("refresh_control_inputs", False)
    st.session_state.setdefault("session_id_input", st.session_state["session_id_value"])
    st.session_state.setdefault("thread_id_input", st.session_state["thread_id_value"])
    st.session_state.setdefault("api_base_url_input", st.session_state["api_base_url_value"])
    st.session_state.setdefault("use_stream_endpoint_input", st.session_state["use_stream_endpoint_value"])
    st.session_state.setdefault("pending_resume_note_input", st.session_state["pending_resume_note_value"])
    if st.session_state.pop("refresh_control_inputs", False):
        _sync_inputs_from_control_values()


def reset_conversation() -> None:
    """Reset chat history and create a fresh thread id without touching widget keys directly."""

    st.session_state["messages"] = []
    st.session_state["thread_id_value"] = _new_id("thread")
    st.session_state["pending_resume"] = False
    st.session_state["pending_resume_kind"] = ""
    st.session_state["pending_resume_question"] = ""
    st.session_state["pending_resume_note_value"] = ""
    st.session_state["last_response"] = {}
    st.session_state["refresh_control_inputs"] = True


def append_message(role: str, content: str, *, payload: Mapping[str, Any] | None = None) -> None:
    st.session_state["messages"].append(
        {
            "role": role,
            "content": content,
            "payload": dict(payload or {}),
        }
    )


def _http_json_request(
    endpoint: str,
    *,
    payload: Mapping[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API returned HTTP {exc.code}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Không kết nối được backend tại {endpoint}") from exc

    payload_out = json.loads(body) if body else {}
    return dict(payload_out) if isinstance(payload_out, Mapping) else {"data": payload_out}


def _stream_json_events(
    endpoint: str,
    *,
    payload: Mapping[str, Any],
    timeout_seconds: int,
) -> Iterator[dict[str, Any]]:
    """Parse SSE responses into `{'event': ..., 'data': ...}` messages."""

    try:
        import requests
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("Cần `requests` để dùng /chat/stream trong Streamlit.") from exc

    with requests.post(
        endpoint,
        json=payload,
        timeout=(10, timeout_seconds),
        stream=True,
        headers={"Accept": "text/event-stream"},
    ) as response:
        response.raise_for_status()
        current_event = "message"
        data_lines: list[str] = []

        def flush_event() -> Iterator[dict[str, Any]]:
            nonlocal current_event, data_lines
            if not data_lines:
                current_event = "message"
                return
            raw_data = "\n".join(data_lines).strip()
            data_lines = []
            if not raw_data:
                current_event = "message"
                return
            try:
                parsed_data = json.loads(raw_data)
            except json.JSONDecodeError:
                parsed_data = {"message": raw_data}
            if not isinstance(parsed_data, Mapping):
                parsed_data = {"value": parsed_data}
            yield {"event": current_event or "message", "data": dict(parsed_data)}
            current_event = "message"

        for raw_line in response.iter_lines(decode_unicode=True):
            line = str(raw_line or "").rstrip("\r")
            if not line:
                yield from flush_event()
                continue
            if line.startswith(":"):
                continue

            field, separator, value = line.partition(":")
            if not separator:
                continue
            value = value.lstrip(" ")
            if field == "event":
                current_event = value or "message"
            elif field == "data":
                data_lines.append(value)

        yield from flush_event()


def call_chat_api(
    question: str,
    *,
    session_id: str,
    thread_id: str,
    api_base_url: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Call the backend `/chat` endpoint."""

    endpoint = f"{api_base_url.rstrip('/')}/chat"
    payload = {
        "question": question,
        "session_id": session_id,
        "thread_id": thread_id,
    }
    return _http_json_request(endpoint, payload=payload, timeout_seconds=timeout_seconds)


def call_resume_api(
    note: str,
    *,
    resume_kind: str,
    session_id: str,
    thread_id: str,
    api_base_url: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Call the backend `/chat/resume` endpoint."""

    endpoint = f"{api_base_url.rstrip('/')}/chat/resume"
    payload: dict[str, Any] = {
        "session_id": session_id,
        "thread_id": thread_id,
    }
    if resume_kind == "clarify":
        payload["clarify_response"] = note
    elif resume_kind == "human_review":
        payload["review_response"] = note
    else:
        payload["note"] = note
    return _http_json_request(endpoint, payload=payload, timeout_seconds=timeout_seconds)


def stream_chat_api(
    question: str,
    *,
    session_id: str,
    thread_id: str,
    api_base_url: str,
    timeout_seconds: int,
) -> Iterator[dict[str, Any]]:
    """Stream events from `/chat/stream` when the backend supports SSE."""

    endpoint = f"{api_base_url.rstrip('/')}/chat/stream"
    payload = {
        "question": question,
        "session_id": session_id,
        "thread_id": thread_id,
    }
    yield from _stream_json_events(endpoint, payload=payload, timeout_seconds=timeout_seconds)


def _extract_response_body_text(payload: Mapping[str, Any]) -> str:
    return str(
        payload.get("final_answer")
        or payload.get("answer")
        or payload.get("draft_answer")
        or payload.get("clarify_question")
        or payload.get("review_note")
        or payload.get("message")
        or "Hệ thống chưa trả về nội dung trả lời."
    )


def _extract_waiting_prompt_text(payload: Mapping[str, Any]) -> str:
    status = str(payload.get("status") or payload.get("response_status") or "").strip()
    if status != WAITING_USER_INPUT_STATUS:
        return ""
    return str(payload.get("resume_question") or "").strip()


def _extract_answer_text(payload: Mapping[str, Any]) -> str:
    """Choose the best answer field while keeping `/chat` compatibility."""

    waiting_prompt = _extract_waiting_prompt_text(payload)
    if waiting_prompt:
        return waiting_prompt
    return _extract_response_body_text(payload)


def _build_draft_assistant_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    draft_payload = dict(payload)
    draft_payload.pop("resume_kind", None)
    draft_payload.pop("resume_question", None)
    if str(draft_payload.get("status") or "").strip() == WAITING_USER_INPUT_STATUS:
        draft_payload["status"] = "draft"
        draft_payload["response_status"] = "draft"
    return draft_payload


def _build_assistant_messages_from_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    normalized_payload = dict(payload)
    body_text = _extract_response_body_text(normalized_payload).strip()
    waiting_prompt = _extract_waiting_prompt_text(normalized_payload)
    default_text = "Hệ thống chưa trả về nội dung trả lời."
    messages: list[dict[str, Any]] = []

    if body_text and body_text != default_text and body_text != waiting_prompt:
        body_payload = _build_draft_assistant_payload(normalized_payload) if waiting_prompt else dict(normalized_payload)
        messages.append({"role": "assistant", "content": body_text, "payload": body_payload})

    if waiting_prompt:
        prompt_payload = dict(normalized_payload)
        prompt_payload["assistant_prompt_kind"] = str(normalized_payload.get("resume_kind") or "waiting_user_input")
        messages.append({"role": "assistant", "content": waiting_prompt, "payload": prompt_payload})

    if not messages:
        fallback_text = waiting_prompt or body_text or default_text
        messages.append({"role": "assistant", "content": fallback_text, "payload": dict(normalized_payload)})

    return messages

def _build_stream_fallback_payload(
    question: str,
    *,
    response_payload: Mapping[str, Any],
    fallback_reason: str,
) -> dict[str, Any]:
    payload = dict(response_payload)
    event_trace = list(payload.get("event_trace") or [])
    event_trace.insert(
        0,
        {
            "event": "stream_fallback",
            "message": f"/chat/stream lỗi hoặc timeout, đã fallback sang /chat. Lý do: {fallback_reason}",
        },
    )
    payload["event_trace"] = event_trace
    payload["ui_notice"] = "/chat/stream lỗi hoặc timeout, UI đã fallback sang /chat."
    payload.setdefault("question", question)
    return payload


def _compact_trace_entry(event_name: str, event_data: Mapping[str, Any]) -> dict[str, Any]:
    trace_entry: dict[str, Any] = {"event": event_name}
    if event_name == "route":
        trace_entry.update(
            {
                "route": event_data.get("route"),
                "risk_level": event_data.get("risk_level"),
                "intent": event_data.get("intent"),
                "route_reason": event_data.get("reason") or event_data.get("route_reason"),
            }
        )
    elif event_name == "retrieval_status":
        trace_entry.update(
            {
                "stage": event_data.get("stage"),
                "rewritten_queries": list(event_data.get("rewritten_queries") or []),
                "round": event_data.get("round"),
                "retrieval_ok": event_data.get("retrieval_ok"),
                "retrieved_count": event_data.get("retrieved_count"),
                "reranked_count": event_data.get("reranked_count"),
                "next_action": event_data.get("next_action"),
            }
        )
    elif event_name == "partial_answer":
        trace_entry["draft_answer"] = event_data.get("draft_answer") or event_data.get("answer") or ""
    elif event_name == "grounding_status":
        trace_entry.update(
            {
                "grounding_ok": event_data.get("grounding_ok"),
                "grounding_score": event_data.get("grounding_score"),
                "next_action": event_data.get("next_action"),
            }
        )
    elif event_name in {"clarify_required", "review_required"}:
        trace_entry.update(
            {
                "status": event_data.get("status"),
                "resume_kind": event_data.get("resume_kind"),
                "resume_question": event_data.get("resume_question"),
                "review_note": event_data.get("review_note"),
            }
        )
    elif event_name == "final_answer":
        trace_entry.update(
            {
                "status": event_data.get("status"),
                "sources": list(event_data.get("sources") or []),
            }
        )
    elif event_name == "final_state":
        trace_entry.update(
            {
                "status": event_data.get("status"),
                "thread_id": event_data.get("thread_id"),
                "session_id": event_data.get("session_id"),
                "resume_kind": event_data.get("resume_kind"),
            }
        )
    elif event_name == "stream_fallback":
        trace_entry["message"] = event_data.get("message")
    elif event_name == "error":
        trace_entry["message"] = event_data.get("message") or event_data.get("detail")
    else:
        trace_entry.update(dict(event_data))
    return {key: value for key, value in trace_entry.items() if value not in (None, "", [], {})}


def update_stream_state_from_event(
    stream_payload: dict[str, Any],
    event_message: Mapping[str, Any],
    *,
    event_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge one SSE event into a UI-friendly accumulated payload."""

    event_name = str(event_message.get("event") or "message").strip() or "message"
    event_data = dict(event_message.get("data") or {})
    updated_payload = dict(stream_payload)

    if event_name == "route":
        updated_payload["route"] = event_data.get("route") or updated_payload.get("route")
        updated_payload["next_route"] = updated_payload["route"]
        updated_payload["risk_level"] = event_data.get("risk_level") or updated_payload.get("risk_level")
        updated_payload["intent"] = event_data.get("intent") or updated_payload.get("intent")
        updated_payload["route_reason"] = (
            event_data.get("reason") or event_data.get("route_reason") or updated_payload.get("route_reason")
        )
    elif event_name == "retrieval_status":
        if event_data.get("stage"):
            updated_payload["retrieval_stage"] = event_data.get("stage")
        if event_data.get("rewritten_queries"):
            updated_payload["rewritten_queries"] = list(event_data.get("rewritten_queries") or [])
        if "retrieval_ok" in event_data:
            updated_payload["retrieval_ok"] = bool(event_data.get("retrieval_ok"))
        if "retrieved_count" in event_data:
            updated_payload["retrieved_count"] = int(event_data.get("retrieved_count") or 0)
        if "reranked_count" in event_data:
            updated_payload["reranked_count"] = int(event_data.get("reranked_count") or 0)
        if event_data.get("next_action"):
            updated_payload["next_action"] = event_data.get("next_action")
    elif event_name == "partial_answer":
        partial_text = str(
            event_data.get("draft_answer")
            or event_data.get("answer")
            or event_data.get("final_answer")
            or event_data.get("message")
            or ""
        )
        if partial_text:
            updated_payload["draft_answer"] = partial_text
    elif event_name == "grounding_status":
        if "grounding_ok" in event_data:
            updated_payload["grounding_ok"] = bool(event_data.get("grounding_ok"))
        if "grounding_score" in event_data:
            updated_payload["grounding_score"] = event_data.get("grounding_score")
        if event_data.get("next_action"):
            updated_payload["next_action"] = event_data.get("next_action")
    elif event_name in {"clarify_required", "review_required"}:
        updated_payload["status"] = event_data.get("status") or WAITING_USER_INPUT_STATUS
        updated_payload["resume_kind"] = event_data.get("resume_kind") or updated_payload.get("resume_kind")
        updated_payload["resume_question"] = (
            event_data.get("resume_question") or updated_payload.get("resume_question") or ""
        )
        updated_payload["thread_id"] = event_data.get("thread_id") or updated_payload.get("thread_id")
        updated_payload["session_id"] = event_data.get("session_id") or updated_payload.get("session_id")
        updated_payload["review_note"] = event_data.get("review_note") or updated_payload.get("review_note") or ""
        if event_data.get("sources"):
            updated_payload["sources"] = list(event_data.get("sources") or [])
        if updated_payload.get("resume_kind") == "human_review":
            updated_payload["human_review_required"] = True
    elif event_name == "final_answer":
        if event_data.get("final_answer"):
            updated_payload["final_answer"] = event_data.get("final_answer")
        if event_data.get("answer"):
            updated_payload["answer"] = event_data.get("answer")
        if event_data.get("sources"):
            updated_payload["sources"] = list(event_data.get("sources") or [])
        if event_data.get("status"):
            updated_payload["status"] = event_data.get("status")
        if event_data.get("thread_id"):
            updated_payload["thread_id"] = event_data.get("thread_id")
        if event_data.get("session_id"):
            updated_payload["session_id"] = event_data.get("session_id")
    elif event_name == "final_state":
        updated_payload["status"] = event_data.get("status") or updated_payload.get("status")
        updated_payload["thread_id"] = event_data.get("thread_id") or updated_payload.get("thread_id")
        updated_payload["session_id"] = event_data.get("session_id") or updated_payload.get("session_id")
        updated_payload["resume_kind"] = event_data.get("resume_kind") or updated_payload.get("resume_kind")
        updated_payload["resume_question"] = event_data.get("resume_question") or updated_payload.get(
            "resume_question"
        )
    elif event_name == "error":
        updated_payload["status"] = event_data.get("status") or "error"
        updated_payload["message"] = event_data.get("message") or event_data.get("detail") or "Backend gặp lỗi."
    elif event_name == "stream_fallback":
        updated_payload["ui_notice"] = event_data.get("message")
    else:
        updated_payload.update(event_data)

    trace_entry = _compact_trace_entry(event_name, event_data)
    if trace_entry:
        event_trace.append(trace_entry)
    updated_payload["event_trace"] = list(event_trace)
    return updated_payload


def render_badges(payload: Mapping[str, Any]) -> None:
    route = str(payload.get("next_route") or payload.get("route") or "").strip()
    risk_level = str(payload.get("risk_level") or "").strip().lower()
    status = str(payload.get("status") or payload.get("response_status") or "").strip()

    badge_html: list[str] = []
    if route:
        badge_html.append(f'<span class="badge badge-route">Route: {route}</span>')
    if risk_level:
        badge_html.append(f'<span class="badge badge-risk-{risk_level}">Risk: {risk_level}</span>')
    if status:
        badge_html.append(f'<span class="badge badge-status">Status: {status}</span>')
    if badge_html:
        st.markdown("".join(badge_html), unsafe_allow_html=True)


def render_sources(sources: Iterable[str]) -> None:
    normalized_sources = [str(source).strip() for source in sources if str(source).strip()]
    if not normalized_sources:
        return
    st.markdown("**Nguồn trích dẫn**")
    for source in normalized_sources:
        st.markdown(f'<div class="source-box">{source}</div>', unsafe_allow_html=True)


def render_event_trace(event_trace: Iterable[Mapping[str, Any]]) -> None:
    """Render a compact trace for route/retrieval/grounding/review events."""

    for event in event_trace:
        event_name = str(event.get("event") or "").strip()
        if event_name == "route":
            details = []
            if event.get("route"):
                details.append(f"Route: {event.get('route')}")
            if event.get("risk_level"):
                details.append(f"Risk: {event.get('risk_level')}")
            if event.get("intent"):
                details.append(f"Intent: {event.get('intent')}")
            if details:
                st.caption(" | ".join(details))
            if event.get("route_reason"):
                st.caption(f"Lý do route: {event.get('route_reason')}")
        elif event_name == "retrieval_status":
            details = []
            if event.get("stage"):
                details.append(f"Stage: {event.get('stage')}")
            if event.get("round") is not None:
                details.append(f"Round: {event.get('round')}")
            if event.get("retrieved_count") is not None:
                details.append(f"Retrieved: {event.get('retrieved_count')}")
            if event.get("reranked_count") is not None:
                details.append(f"Reranked: {event.get('reranked_count')}")
            if event.get("next_action"):
                details.append(f"Next: {event.get('next_action')}")
            if details:
                st.caption(" | ".join(details))
            if event.get("rewritten_queries"):
                rewritten = ", ".join(str(item) for item in event.get("rewritten_queries") or [])
                st.caption(f"Rewrite truy vấn: {rewritten}")
        elif event_name == "grounding_status":
            details = []
            if event.get("grounding_ok") is not None:
                details.append(f"Grounding OK: {event.get('grounding_ok')}")
            if event.get("grounding_score") is not None:
                details.append(f"Score: {event.get('grounding_score')}")
            if event.get("next_action"):
                details.append(f"Next: {event.get('next_action')}")
            if details:
                st.caption(" | ".join(details))
        elif event_name in {"clarify_required", "review_required"}:
            st.caption(f"Chờ user input ({event.get('resume_kind', '')})")
            if event.get("resume_question"):
                st.caption(str(event.get("resume_question")))
        elif event_name == "stream_fallback":
            st.info(str(event.get("message") or "Đã fallback sang /chat."))
        elif event_name == "error":
            st.error(str(event.get("message") or "Backend gặp lỗi khi xử lý yêu cầu."))


def render_message(message: Mapping[str, Any]) -> None:
    role = str(message.get("role") or "assistant")
    content = str(message.get("content") or "")
    payload = dict(message.get("payload") or {})
    with st.chat_message("assistant" if role == "assistant" else "user"):
        st.write(content)
        if role == "assistant":
            render_badges(payload)
            if payload.get("intent"):
                st.caption(f"Intent: {payload.get('intent')} | Score: {payload.get('intent_score', '')}")
            if payload.get("route_reason"):
                st.caption(f"Lý do route: {payload.get('route_reason')}")
            waiting_prompt = _extract_waiting_prompt_text(payload)
            if waiting_prompt:
                resume_kind = str(payload.get("resume_kind") or "user_input").strip()
                display_kind = "human review" if resume_kind == "human_review" else resume_kind
                st.info(f"Đang chờ {display_kind}. Hãy trả lời ngay trong ô chat để hệ thống tiếp tục.")
            if payload.get("review_note"):
                review_note = str(payload.get("review_note"))
                if not waiting_prompt or review_note.strip() != content.strip():
                    st.warning(review_note)
            if payload.get("ui_notice"):
                st.info(str(payload.get("ui_notice")))
            if payload.get("event_trace"):
                with st.expander("Trace agentic / luồng xử lý", expanded=False):
                    render_event_trace(payload.get("event_trace") or [])
            elif payload.get("grounding_score") is not None and payload.get("grounding_score") != "":
                st.caption(
                    f"Grounding: {payload.get('grounding_score')} | "
                    f"OK: {payload.get('grounding_ok')} | Next: {payload.get('next_action')}"
                )
            render_sources(payload.get("sources") or [])

def render_chat_history() -> None:
    for message in st.session_state["messages"]:
        render_message(message)


def normalize_backend_response(response: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize both `/chat` and streamed `/chat/stream` payloads for the UI."""

    payload = dict(response)
    if payload.get("reason") and not payload.get("route_reason"):
        payload["route_reason"] = payload.get("reason")
    if payload.get("route") and not payload.get("next_route"):
        payload["next_route"] = payload.get("route")
    if payload.get("response_status") and not payload.get("status"):
        payload["status"] = payload.get("response_status")
    payload.setdefault("sources", response.get("sources") or [])
    payload.setdefault("event_trace", response.get("event_trace") or [])
    payload["human_review_required"] = bool(payload.get("human_review_required") or False)
    payload["resume_kind"] = str(payload.get("resume_kind") or "")
    payload["resume_question"] = str(payload.get("resume_question") or "")
    if payload.get("review_note"):
        payload["review_note"] = str(payload.get("review_note"))
    return {"answer": _extract_answer_text(payload), "payload": payload}


def _render_live_stream_state(
    trace_placeholder: Any,
    answer_placeholder: Any,
    stream_payload: Mapping[str, Any],
) -> None:
    with trace_placeholder.container():
        st.caption("Luồng xử lý live")
        render_badges(stream_payload)
        if stream_payload.get("ui_notice"):
            st.info(str(stream_payload.get("ui_notice")))
        render_event_trace(stream_payload.get("event_trace") or [])

    answer_text = _extract_answer_text(stream_payload)
    if answer_text and answer_text != "Hệ thống chưa trả về nội dung trả lời.":
        if str(stream_payload.get("status") or "").strip().lower() == "error":
            answer_placeholder.error(answer_text)
        else:
            answer_placeholder.markdown(answer_text)


def consume_stream_events(
    question: str,
    *,
    session_id: str,
    thread_id: str,
    api_base_url: str,
    timeout_seconds: int,
    trace_placeholder: Any,
    answer_placeholder: Any,
) -> dict[str, Any]:
    """Consume TV6 SSE events and accumulate a renderable payload."""

    stream_payload: dict[str, Any] = {
        "question": question,
        "thread_id": thread_id,
        "session_id": session_id,
        "sources": [],
        "event_trace": [],
    }
    event_trace: list[dict[str, Any]] = []

    for event_message in stream_chat_api(
        question,
        session_id=session_id,
        thread_id=thread_id,
        api_base_url=api_base_url,
        timeout_seconds=timeout_seconds,
    ):
        stream_payload = update_stream_state_from_event(stream_payload, event_message, event_trace=event_trace)
        _render_live_stream_state(trace_placeholder, answer_placeholder, stream_payload)

    if not stream_payload.get("event_trace") and not _extract_answer_text(stream_payload).strip():
        raise RuntimeError("Không nhận được SSE event nào từ backend.")

    return stream_payload


def _is_resume_required(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload.get("status") == WAITING_USER_INPUT_STATUS
        or payload.get("resume_kind")
        or payload.get("interrupt_payload")
    )


def _store_response_state(payload: Mapping[str, Any]) -> None:
    st.session_state["last_response"] = dict(payload)
    st.session_state["thread_id_value"] = str(payload.get("thread_id") or st.session_state["thread_id_value"])
    if payload.get("session_id"):
        st.session_state["session_id_value"] = str(payload.get("session_id"))
    st.session_state["refresh_control_inputs"] = True
    st.session_state["pending_resume"] = _is_resume_required(payload)
    st.session_state["pending_resume_kind"] = str(payload.get("resume_kind") or "")
    st.session_state["pending_resume_question"] = str(payload.get("resume_question") or "")
    if not st.session_state["pending_resume"]:
        st.session_state["pending_resume_kind"] = ""
        st.session_state["pending_resume_question"] = ""
        st.session_state["pending_resume_note_value"] = ""


def submit_question(config: AppUIConfig, question: str) -> None:
    _sync_control_values_from_inputs()

    try:
        if st.session_state["use_stream_endpoint_value"]:
            with st.chat_message("assistant"):
                trace_placeholder = st.empty()
                answer_placeholder = st.empty()
                try:
                    response_payload = consume_stream_events(
                        question,
                        session_id=st.session_state["session_id_value"],
                        thread_id=st.session_state["thread_id_value"],
                        api_base_url=st.session_state["api_base_url_value"],
                        timeout_seconds=config.timeout_seconds,
                        trace_placeholder=trace_placeholder,
                        answer_placeholder=answer_placeholder,
                    )
                except Exception as exc:
                    LOGGER.warning("Streaming /chat/stream failed, falling back to /chat: %s", exc)
                    trace_placeholder.empty()
                    answer_placeholder.empty()
                    response_payload = _build_stream_fallback_payload(
                        question,
                        response_payload=call_chat_api(
                            question,
                            session_id=st.session_state["session_id_value"],
                            thread_id=st.session_state["thread_id_value"],
                            api_base_url=st.session_state["api_base_url_value"],
                            timeout_seconds=config.timeout_seconds,
                        ),
                        fallback_reason=str(exc),
                    )
        else:
            response_payload = call_chat_api(
                question,
                session_id=st.session_state["session_id_value"],
                thread_id=st.session_state["thread_id_value"],
                api_base_url=st.session_state["api_base_url_value"],
                timeout_seconds=config.timeout_seconds,
            )
    except Exception as exc:
        append_message(
            "assistant",
            f"Không thể gọi backend: {exc}",
            payload={"status": "error"},
        )
        return

    normalized_response = normalize_backend_response(response_payload)
    for assistant_message in _build_assistant_messages_from_payload(normalized_response["payload"]):
        append_message(
            "assistant",
            str(assistant_message.get("content") or ""),
            payload=assistant_message.get("payload"),
        )
    _store_response_state(normalized_response["payload"])

def submit_resume(config: AppUIConfig, note: str) -> None:
    _sync_control_values_from_inputs()
    resume_kind = str(
        st.session_state.get("pending_resume_kind")
        or st.session_state.get("last_response", {}).get("resume_kind")
        or ""
    )
    try:
        response = call_resume_api(
            note,
            resume_kind=resume_kind,
            session_id=st.session_state["session_id_value"],
            thread_id=st.session_state["thread_id_value"],
            api_base_url=st.session_state["api_base_url_value"],
            timeout_seconds=config.timeout_seconds,
        )
    except Exception as exc:
        append_message("assistant", f"Không thể resume backend: {exc}", payload={"status": "error"})
        return

    normalized_response = normalize_backend_response(response)
    for assistant_message in _build_assistant_messages_from_payload(normalized_response["payload"]):
        append_message(
            "assistant",
            str(assistant_message.get("content") or ""),
            payload=assistant_message.get("payload"),
        )
    _store_response_state(normalized_response["payload"])

def render_sidebar(config: AppUIConfig) -> None:
    with st.sidebar:
        st.header("Điều khiển phiên")
        st.text_input("Session ID", key="session_id_input")
        st.text_input("Thread ID", key="thread_id_input")
        st.text_input("Backend URL", key="api_base_url_input")
        st.checkbox("Ưu tiên /chat/stream", key="use_stream_endpoint_input")
        _sync_control_values_from_inputs()
        st.caption("UI này kết nối FastAPI backend của TV6 qua /chat, /chat/stream, /chat/resume.")

        if st.button("Tạo phiên mới"):
            reset_conversation()
            st.rerun()

        st.subheader("Hội thoại cũ (local)")
        local_conversations = _load_local_conversations()
        if not local_conversations:
            st.caption("Chưa tìm thấy checkpoint nào trong `.checkpoints`.")
        else:
            option_ids = [str(item.file_path) for item in local_conversations]
            option_labels = {str(item.file_path): _format_local_conversation_option(item) for item in local_conversations}
            if st.session_state.get("local_conversation_file") not in option_ids:
                st.session_state["local_conversation_file"] = option_ids[0]

            selected_file = st.selectbox(
                "Chọn cuộc trò chuyện đã lưu",
                options=option_ids,
                format_func=lambda option_id: option_labels.get(option_id, option_id),
                key="local_conversation_file",
            )
            selected_summary = next(
                (item for item in local_conversations if str(item.file_path) == str(selected_file)),
                None,
            )
            if selected_summary is not None:
                st.caption(f"session_id: {selected_summary.session_id}")
                st.caption(f"thread_id: {selected_summary.thread_id}")
                if st.button("Mở hội thoại đã chọn"):
                    try:
                        checkpoint_state = _load_checkpoint_state_from_file(selected_summary.file_path)
                    except Exception as exc:
                        st.error(f"Không thể đọc checkpoint: {exc}")
                    else:
                        _restore_conversation_from_checkpoint_state(checkpoint_state)
                        st.rerun()

        last_response = st.session_state.get("last_response") or {}
        if last_response:
            st.subheader("Trạng thái gần nhất")
            render_badges(last_response)
            if last_response.get("resume_question"):
                st.info(str(last_response.get("resume_question")))
            if last_response.get("ui_notice"):
                st.info(str(last_response.get("ui_notice")))

        if st.session_state.get("pending_resume"):
            st.subheader("Debug resume")
            if st.session_state.get("pending_resume_kind"):
                st.caption(f"resume_kind: {st.session_state.get('pending_resume_kind')}")
            if st.session_state.get("pending_resume_question"):
                st.caption(str(st.session_state.get("pending_resume_question")))
            note = st.text_area(
                "Nội dung bổ sung / xác nhận",
                key="pending_resume_note_input",
                placeholder="Chỉ dùng khi cần debug thủ công luồng /chat/resume...",
            )
            st.session_state["pending_resume_note_value"] = str(note or "")
            if st.button("Gửi resume (debug)"):
                submit_resume(config, note)
                st.rerun()


def render_header(config: AppUIConfig) -> None:
    st.title(config.app_title)
    st.caption(config.app_subtitle)


def main() -> None:
    config = load_app_config()
    configure_page(config)
    init_session_state(config)
    render_sidebar(config)
    render_header(config)
    render_chat_history()

    user_prompt = st.chat_input("Nhập câu hỏi pháp luật tiếng Việt...")
    if user_prompt:
        append_message("user", user_prompt)
        render_message(st.session_state["messages"][-1])
        if st.session_state.get("pending_resume"):
            submit_resume(config, user_prompt)
        else:
            submit_question(config, user_prompt)
        st.rerun()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    main()

from __future__ import annotations

# Terminal 1:
# python -m uvicorn src.app.api.main:app --host 127.0.0.1 --port 8000
#
# Terminal 2:
# PowerShell:
# $env:LANGSMITH_TRACING="true"
# $env:LANGSMITH_API_KEY="..."
# $env:LANGSMITH_PROJECT="legal-rag-tv6"
# $env:EVAL_LIMIT="5"
# python evaluation\run_langsmith_eval.py

import json
import os
import uuid
from importlib.metadata import version
from typing import Any, Iterable

import requests
from langsmith import Client, evaluate

LANGSMITH_VERSION = version("langsmith")
DATASET_NAME = os.getenv("LANGSMITH_DATASET_NAME", "vietnamese-legal-qa-routing-eval-v1")
CHATBOT_API_URL = os.getenv("CHATBOT_API_URL", "http://127.0.0.1:8000/chat")
EXPERIMENT_PREFIX = os.getenv("LANGSMITH_EXPERIMENT_PREFIX", "legal-qa-routing-eval")
MAX_CONCURRENCY = int(os.getenv("EVAL_MAX_CONCURRENCY", "1"))
TIMEOUT_SECONDS = int(os.getenv("EVAL_TIMEOUT_SECONDS", "180"))
EVAL_LIMIT_RAW = os.getenv("EVAL_LIMIT", "").strip()
EVAL_DEBUG = os.getenv("EVAL_DEBUG", "true")

client = Client()


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)


DEBUG_ENABLED = _to_bool(EVAL_DEBUG)


def _debug_print(*parts: Any) -> None:
    if DEBUG_ENABLED:
        print(*parts)


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        if isinstance(value, list) and len(value) == 0:
            continue
        return value
    return None


def _unwrap_payload(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    for key in ("payload", "data", "result", "state", "response"):
        candidate = raw.get(key)
        if isinstance(candidate, dict):
            return candidate
    return raw


def _normalize_slots(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        normalized = [str(item).strip() for item in value if str(item).strip()]
        return sorted(normalized)
    text = str(value).strip()
    if not text:
        return []
    separators = [",", ";"]
    for separator in separators:
        if separator in text:
            normalized = [part.strip() for part in text.split(separator) if part.strip()]
            return sorted(normalized)
    return [text]


def _normalize_sources(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return [value]


def _preview_text(value: Any, limit: int = 500) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    return text[:limit]


def _infer_route(
    *,
    pred_route: str,
    data: dict[str, Any],
    raw: dict[str, Any],
    outputs_so_far: dict[str, Any],
) -> str:
    normalized_route = (pred_route or "").strip()
    if normalized_route:
        return normalized_route

    execution_profile = str(data.get("execution_profile") or raw.get("execution_profile") or "").strip().lower()
    status = str(
        _first_non_empty(
            data.get("response_status"),
            data.get("status"),
            raw.get("response_status"),
            raw.get("status"),
        )
        or ""
    ).strip()
    resume_kind = str(_first_non_empty(data.get("resume_kind"), raw.get("resume_kind")) or "").strip().lower()
    unsupported_query = _to_bool(
        _first_non_empty(data.get("unsupported_query"), raw.get("unsupported_query"))
    )
    need_clarify = _to_bool(outputs_so_far.get("pred_need_clarify"))
    human_review_required = _to_bool(
        _first_non_empty(
            data.get("human_review_required"),
            raw.get("human_review_required"),
            data.get("review_required"),
            raw.get("review_required"),
        )
    )

    if execution_profile == "fast":
        return "fast-path"
    if status == "waiting_user_input":
        if resume_kind == "clarify" or need_clarify:
            return "clarify-path"
        if resume_kind == "human_review" or human_review_required:
            return "human-review-path"
    if unsupported_query or status == "unsupported":
        return "unsupported-path"
    if outputs_so_far.get("answer") and outputs_so_far.get("ok"):
        return "legal-agent-path"
    return ""


def _base_error_output(error_type: str, error: str, raw_response_preview: str = "") -> dict[str, Any]:
    return {
        "ok": False,
        "error_type": error_type,
        "error": error,
        "answer": "",
        "pred_intent": "",
        "pred_route": "",
        "pred_risk_level": "",
        "pred_need_clarify": False,
        "pred_missing_slots": [],
        "sources": [],
        "raw_response_preview": raw_response_preview,
        "raw_keys": [],
        "data_keys": [],
    }


def run_chatbot(inputs: dict[str, Any]) -> dict[str, Any]:
    question = str(inputs.get("question") or "").strip()
    if not question:
        return _base_error_output("missing_question", "Input does not contain a non-empty `question` field.")

    session_id = f"langsmith-eval-{uuid.uuid4()}"
    thread_id = f"thread-{uuid.uuid4()}"

    _debug_print("\n=== QUESTION ===")
    _debug_print(question)
    _debug_print("Session ID:", session_id)
    _debug_print("Thread ID:", thread_id)

    payload = {
        "question": question,
        "message": question,
        "session_id": session_id,
        "thread_id": thread_id,
    }

    try:
        response = requests.post(CHATBOT_API_URL, json=payload, timeout=TIMEOUT_SECONDS)
    except requests.exceptions.Timeout as exc:
        return _base_error_output("timeout", str(exc))
    except requests.exceptions.ConnectionError as exc:
        return _base_error_output("connection_error", str(exc))
    except requests.exceptions.RequestException as exc:
        return _base_error_output("request_error", str(exc))

    _debug_print("HTTP status:", response.status_code)

    if not response.ok:
        preview = response.text[:500]
        _debug_print("Raw response preview:", preview)
        return _base_error_output(
            f"http_{response.status_code}",
            f"HTTP {response.status_code}: {response.reason}",
            preview,
        )

    try:
        raw = response.json()
    except ValueError as exc:
        preview = response.text[:500]
        _debug_print("Raw response preview:", preview)
        return _base_error_output("json_decode_error", str(exc), preview)

    data = _unwrap_payload(raw)
    raw_preview = _preview_text(raw)
    _debug_print("Raw response preview:", raw_preview)

    answer = _first_non_empty(
        data.get("answer"),
        data.get("final_answer"),
        data.get("draft_answer"),
        data.get("message"),
        data.get("content"),
        raw.get("answer") if isinstance(raw, dict) else None,
        raw.get("final_answer") if isinstance(raw, dict) else None,
    )
    pred_intent = _first_non_empty(
        data.get("intent"),
        data.get("pred_intent"),
        raw.get("intent") if isinstance(raw, dict) else None,
    )
    pred_route = _first_non_empty(
        data.get("next_route"),
        data.get("route"),
        data.get("pred_route"),
        raw.get("next_route") if isinstance(raw, dict) else None,
        raw.get("route") if isinstance(raw, dict) else None,
    )
    pred_risk_level = _first_non_empty(
        data.get("risk_level"),
        data.get("pred_risk_level"),
        raw.get("risk_level") if isinstance(raw, dict) else None,
    )
    pred_need_clarify = _first_non_empty(
        data.get("need_clarify"),
        data.get("pred_need_clarify"),
        raw.get("need_clarify") if isinstance(raw, dict) else None,
    )
    pred_missing_slots = _first_non_empty(
        data.get("missing_slots"),
        data.get("pred_missing_slots"),
        raw.get("missing_slots") if isinstance(raw, dict) else None,
    )
    sources = _first_non_empty(
        data.get("sources"),
        data.get("citations"),
        data.get("retrieved_docs"),
        raw.get("sources") if isinstance(raw, dict) else None,
    )

    output: dict[str, Any] = {
        "ok": True,
        "error_type": "",
        "error": "",
        "answer": str(answer or ""),
        "pred_intent": str(pred_intent or ""),
        "pred_route": "",
        "pred_risk_level": str(pred_risk_level or ""),
        "pred_need_clarify": _to_bool(pred_need_clarify),
        "pred_missing_slots": _normalize_slots(pred_missing_slots),
        "sources": _normalize_sources(sources),
        "raw_response_preview": raw_preview,
        "raw_keys": sorted(list(raw.keys())) if isinstance(raw, dict) else [],
        "data_keys": sorted(list(data.keys())) if isinstance(data, dict) else [],
    }

    if not output["pred_need_clarify"] and str(
        _first_non_empty(data.get("resume_kind"), raw.get("resume_kind")) or ""
    ).strip().lower() == "clarify":
        output["pred_need_clarify"] = True

    output["pred_route"] = _infer_route(
        pred_route=str(pred_route or ""),
        data=data,
        raw=raw if isinstance(raw, dict) else {},
        outputs_so_far=output,
    )
    if output["pred_route"] == "clarify-path":
        output["pred_need_clarify"] = True

    _debug_print(
        "Extracted:",
        {
            "pred_intent": output["pred_intent"],
            "pred_route": output["pred_route"],
            "pred_risk_level": output["pred_risk_level"],
            "pred_need_clarify": output["pred_need_clarify"],
            "pred_missing_slots": output["pred_missing_slots"],
        },
    )
    return output


def api_success(inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    score = int(outputs.get("ok") is True)
    comment = "ok" if score == 1 else f"error={outputs.get('error_type')}: {outputs.get('error')}"
    return {"key": "api_success", "score": score, "comment": comment}


def intent_accuracy(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    pred = str(outputs.get("pred_intent") or "")
    gold = str(reference_outputs.get("gold_intent") or "")
    score = int(outputs.get("ok") is True and pred == gold)
    comment = f"pred={pred}, gold={gold}"
    if not outputs.get("ok"):
        comment = f"{comment}, error={outputs.get('error_type')}: {outputs.get('error')}"
    return {"key": "intent_accuracy", "score": score, "comment": comment}


def route_accuracy(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    pred = str(outputs.get("pred_route") or "")
    gold = str(reference_outputs.get("gold_route") or "")
    score = int(outputs.get("ok") is True and pred == gold)
    comment = f"pred={pred}, gold={gold}"
    if not outputs.get("ok"):
        comment = f"{comment}, error={outputs.get('error_type')}: {outputs.get('error')}"
    return {"key": "route_accuracy", "score": score, "comment": comment}


def risk_accuracy(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    pred = str(outputs.get("pred_risk_level") or "")
    gold = str(reference_outputs.get("gold_risk_level") or "")
    score = int(outputs.get("ok") is True and pred == gold)
    comment = f"pred={pred}, gold={gold}"
    if not outputs.get("ok"):
        comment = f"{comment}, error={outputs.get('error_type')}: {outputs.get('error')}"
    return {"key": "risk_accuracy", "score": score, "comment": comment}


def clarify_accuracy(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    pred = _to_bool(outputs.get("pred_need_clarify"))
    gold = _to_bool(reference_outputs.get("gold_need_clarify"))
    score = int(outputs.get("ok") is True and pred == gold)
    comment = f"pred={pred}, gold={gold}"
    if not outputs.get("ok"):
        comment = f"{comment}, error={outputs.get('error_type')}: {outputs.get('error')}"
    return {"key": "clarify_accuracy", "score": score, "comment": comment}


def missing_slot_accuracy(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    pred_slots = _normalize_slots(outputs.get("pred_missing_slots"))
    gold_slots = _normalize_slots(reference_outputs.get("gold_missing_slots"))
    score = int(outputs.get("ok") is True and pred_slots == gold_slots)
    comment = f"pred={pred_slots}, gold={gold_slots}"
    if not outputs.get("ok"):
        comment = f"{comment}, error={outputs.get('error_type')}: {outputs.get('error')}"
    return {"key": "missing_slot_accuracy", "score": score, "comment": comment}


def retrieval_hint_present(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    gold_hint = str(reference_outputs.get("gold_source_hint") or "").strip()
    if not gold_hint:
        return {
            "key": "retrieval_hint_present",
            "score": 1,
            "comment": "no gold source hint",
        }

    sources_text = json.dumps(outputs.get("sources") or [], ensure_ascii=False).lower()
    hint_tokens = [
        token.strip().lower()
        for chunk in gold_hint.replace(";", ",").split(",")
        for token in [chunk]
        if token.strip()
    ]
    matched = any(len(token) > 3 and token in sources_text for token in hint_tokens)
    score = int(matched)
    comment = f"gold_hint={hint_tokens}, matched={matched}"
    if not outputs.get("ok"):
        comment = f"{comment}, error={outputs.get('error_type')}: {outputs.get('error')}"
    return {"key": "retrieval_hint_present", "score": score, "comment": comment}


def _get_api_key_prefix() -> str:
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "LANGSMITH_API_KEY is not set. PowerShell example: $env:LANGSMITH_API_KEY=\"...\""
        )
    return api_key[:8]


def _resolve_data_for_eval() -> tuple[Any, int]:
    dataset = client.read_dataset(dataset_name=DATASET_NAME)
    examples = list(client.list_examples(dataset_id=dataset.id))
    example_count = len(examples)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Dataset ID: {dataset.id}")
    print(f"Example count: {example_count}")

    if not EVAL_LIMIT_RAW:
        print(f"Examples to run: {example_count}")
        return DATASET_NAME, example_count

    limit = int(EVAL_LIMIT_RAW)
    selected_examples = examples[:limit]
    print(f"EVAL_LIMIT={limit}")
    print(f"Examples to run: {len(selected_examples)}")
    return selected_examples, len(selected_examples)


def main() -> None:
    print(f"langsmith version: {LANGSMITH_VERSION}")
    api_key_prefix = _get_api_key_prefix()
    print(f"LANGSMITH_API_KEY prefix: {api_key_prefix}...")
    print(f"CHATBOT_API_URL: {CHATBOT_API_URL}")
    print(f"EXPERIMENT_PREFIX: {EXPERIMENT_PREFIX}")
    print(f"MAX_CONCURRENCY: {MAX_CONCURRENCY}")
    print(f"TIMEOUT_SECONDS: {TIMEOUT_SECONDS}")
    print(f"EVAL_DEBUG: {DEBUG_ENABLED}")

    data_for_eval, run_count = _resolve_data_for_eval()

    experiment = evaluate(
        run_chatbot,
        data=data_for_eval,
        evaluators=[
            api_success,
            intent_accuracy,
            route_accuracy,
            risk_accuracy,
            clarify_accuracy,
            missing_slot_accuracy,
            retrieval_hint_present,
        ],
        experiment_prefix=EXPERIMENT_PREFIX,
        max_concurrency=MAX_CONCURRENCY,
        client=client,
        metadata={
            "dataset": DATASET_NAME,
            "api_url": CHATBOT_API_URL,
            "purpose": "Vietnamese Legal QA routing/risk/clarify evaluation",
            "run_count": run_count,
        },
    )

    print("\nExperiment:")
    print(experiment)


if __name__ == "__main__":
    main()

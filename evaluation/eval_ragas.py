from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from openai import AsyncOpenAI
import requests

LOGGER = logging.getLogger(__name__)
_RAGAS_SYMBOLS: dict[str, Any] | None = None
DEFAULT_QA_DATASET = os.getenv("RAGAS_HF_DATASET", "thangvip/vietnamese-legal-qa")


@dataclass(slots=True)
class EvalConfig:
    llm_model: str = os.getenv("RAGAS_LLM_MODEL", "qwen2.5:7b")
    embedding_model: str = os.getenv("RAGAS_EMBED_MODEL", "bge-m3")
    base_url: str = os.getenv("RAGAS_BASE_URL", "http://127.0.0.1:11434")
    api_key: str = os.getenv("RAGAS_API_KEY", "ollama")
    mode: str = os.getenv("RAGAS_MODE", "auto")
    output_dir: str = "evaluation/results"
    api_url: str = os.getenv("RAGAS_CHATBOT_API_URL", "http://127.0.0.1:8000/chat")
    timeout_seconds: int = int(os.getenv("RAGAS_TIMEOUT_SECONDS", "180"))
    call_backend: bool = True


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def _load_ragas_symbols() -> dict[str, Any]:
    global _RAGAS_SYMBOLS
    if _RAGAS_SYMBOLS is not None:
        return _RAGAS_SYMBOLS

    try:
        try:
            from ragas import SingleTurnSample  # type: ignore
        except ImportError:  # pragma: no cover - compatibility fallback
            from ragas.dataset_schema import SingleTurnSample  # type: ignore

        from ragas.embeddings.base import embedding_factory  # type: ignore
        from ragas.llms import llm_factory  # type: ignore
        from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall  # type: ignore
        from ragas.metrics.collections import (  # type: ignore
            AnswerCorrectness,
            AnswerRelevancy,
            ContextRecall,
            Faithfulness,
            SemanticSimilarity,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Failed to import the ragas runtime. Please install compatible dependencies "
            "(see requirements.txt) and verify the Python environment can import `ragas`."
        ) from exc

    _RAGAS_SYMBOLS = {
        "SingleTurnSample": SingleTurnSample,
        "embedding_factory": embedding_factory,
        "llm_factory": llm_factory,
        "NonLLMContextPrecisionWithReference": NonLLMContextPrecisionWithReference,
        "NonLLMContextRecall": NonLLMContextRecall,
        "AnswerCorrectness": AnswerCorrectness,
        "AnswerRelevancy": AnswerRelevancy,
        "ContextRecall": ContextRecall,
        "Faithfulness": Faithfulness,
        "SemanticSimilarity": SemanticSimilarity,
    }
    return _RAGAS_SYMBOLS


def _normalize_base_url(base_url: str) -> str:
    text = (base_url or "").strip().rstrip("/")
    if not text:
        return "http://127.0.0.1:11434/v1"
    if text.endswith("/v1"):
        return text
    return f"{text}/v1"


def _unwrap_payload(raw: Any) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    for key in ("payload", "data", "result", "state", "response"):
        value = raw.get(key)
        if isinstance(value, Mapping):
            return value
    return raw


def _parse_json_like(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text.startswith("[") or text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None or value == "" or value == []:
            continue
        return value
    return None


def _normalize_text_list(value: Any) -> list[str]:
    parsed = _parse_json_like(value)
    if parsed in (None, "", []):
        return []
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if isinstance(parsed, str):
        return [item.strip() for item in parsed.replace(";", ",").split(",") if item.strip()]
    return [str(parsed).strip()]


def _extract_contexts(value: Any) -> list[str]:
    parsed = _parse_json_like(value)
    if parsed in (None, "", []):
        return []
    if isinstance(parsed, list):
        contexts: list[str] = []
        for item in parsed:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    contexts.append(text)
                continue
            if isinstance(item, Mapping):
                metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
                parts = [
                    item.get("text"),
                    item.get("content"),
                    item.get("page_content"),
                    metadata.get("article_name") if metadata else None,
                    metadata.get("title") if metadata else None,
                    metadata.get("law_id") if metadata else None,
                ]
                text = " ".join(str(part).strip() for part in parts if part)
                if text:
                    contexts.append(text)
        return contexts
    if isinstance(parsed, str):
        return [parsed.strip()] if parsed.strip() else []
    return [str(parsed).strip()]


def _resolve_text(value: Any) -> str:
    return str(value or "").strip()


def _extract_generated_pairs(value: Any) -> list[dict[str, Any]]:
    parsed = _parse_json_like(value)
    if parsed in (None, "", []):
        return []
    if not isinstance(parsed, list):
        return []
    pairs: list[dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, Mapping):
            pairs.append(dict(item))
    return pairs


def _extract_question_answer(pair: Mapping[str, Any]) -> tuple[str, str]:
    question = _resolve_text(
        _first_non_empty(
            pair.get("question"),
            pair.get("query"),
            pair.get("prompt"),
            pair.get("q"),
            pair.get("cau_hoi"),
        )
    )
    answer = _resolve_text(
        _first_non_empty(
            pair.get("answer"),
            pair.get("response"),
            pair.get("a"),
            pair.get("tra_loi"),
        )
    )
    return question, answer


def _select_dataset_split(dataset: Any, requested_split: str) -> Sequence[Mapping[str, Any]]:
    if not isinstance(dataset, Mapping):
        return dataset
    if requested_split and requested_split.lower() != "auto":
        if requested_split not in dataset:
            available = ", ".join(str(key) for key in dataset.keys())
            raise KeyError(f"Split `{requested_split}` not found. Available splits: {available}")
        return dataset[requested_split]
    for candidate in ("test", "validation", "train"):
        if candidate in dataset:
            return dataset[candidate]
    first_key = next(iter(dataset.keys()))
    return dataset[first_key]


def load_hf_rows(dataset_name: str, *, split: str = "auto", limit: int | None = None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Failed to import Hugging Face `datasets`. Please install compatible `datasets`/`pyarrow` packages."
        ) from exc

    dataset = load_dataset(dataset_name)
    split_rows = _select_dataset_split(dataset, split)
    flattened_rows: list[dict[str, Any]] = []
    qa_count = 0
    for raw_row in split_rows:
        row = dict(raw_row)
        article_content = _resolve_text(
            _first_non_empty(
                row.get("article_content"),
                row.get("context"),
                row.get("content"),
                row.get("document"),
            )
        )
        qa_pairs = _extract_generated_pairs(row.get("generated_qa_pairs") or row.get("qa_pairs"))
        for pair_index, pair in enumerate(qa_pairs, start=1):
            question, answer = _extract_question_answer(pair)
            if not question or not answer:
                continue
            flattened_rows.append(
                {
                    "id": f"hf-{qa_count + 1}",
                    "source_row_id": row.get("id") or row.get("row_id") or qa_count + 1,
                    "pair_index": pair_index,
                    "question": question,
                    "reference_answer": answer,
                    "contexts": [article_content] if article_content else [],
                    "reference_contexts": [],
                    "raw_dataset_row": row,
                }
            )
            qa_count += 1
            if limit is not None and qa_count >= limit:
                return flattened_rows
    return flattened_rows


def _unwrap_response(raw: Any) -> Mapping[str, Any]:
    return _unwrap_payload(raw) if isinstance(raw, Mapping) else {}


def run_chatbot(question: str, config: EvalConfig) -> dict[str, Any]:
    session_id = f"ragas-eval-{uuid.uuid4()}"
    thread_id = f"ragas-eval-{uuid.uuid4()}"
    payload = {
        "question": question,
        "message": question,
        "session_id": session_id,
        "thread_id": thread_id,
    }
    try:
        response = requests.post(config.api_url, json=payload, timeout=config.timeout_seconds)
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        return {"ok": False, "error": f"timeout: {exc}", "answer": "", "raw": {}}
    except requests.exceptions.ConnectionError as exc:
        return {"ok": False, "error": f"connection_error: {exc}", "answer": "", "raw": {}}
    except requests.exceptions.RequestException as exc:
        return {"ok": False, "error": f"http_error: {exc}", "answer": "", "raw": {}}

    try:
        raw = response.json()
    except ValueError as exc:
        return {"ok": False, "error": f"json_decode_error: {exc}", "answer": "", "raw": {}}

    data = _unwrap_response(raw)
    answer = _resolve_text(
        _first_non_empty(
            data.get("answer"),
            data.get("final_answer"),
            data.get("draft_answer"),
            data.get("message"),
            data.get("content"),
            raw.get("answer") if isinstance(raw, Mapping) else None,
            raw.get("final_answer") if isinstance(raw, Mapping) else None,
        )
    )
    return {
        "ok": True,
        "answer": answer,
        "route": _resolve_text(_first_non_empty(data.get("route"), data.get("next_route"), raw.get("route") if isinstance(raw, Mapping) else None)),
        "status": _resolve_text(_first_non_empty(data.get("status"), data.get("response_status"), raw.get("status") if isinstance(raw, Mapping) else None)),
        "sources": _extract_contexts(_first_non_empty(data.get("sources"), raw.get("sources") if isinstance(raw, Mapping) else None)),
        "raw": raw,
    }


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    resolved_path = Path(path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Evaluation input file not found: {resolved_path}")

    suffix = resolved_path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with resolved_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, Mapping):
                    rows.append(dict(payload))
        return rows

    if suffix == ".json":
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, Mapping)]
        if isinstance(payload, Mapping):
            rows = payload.get("rows") or payload.get("predictions") or payload.get("examples") or payload
            if isinstance(rows, list):
                return [dict(item) for item in rows if isinstance(item, Mapping)]
        raise ValueError(f"Unsupported JSON structure in {resolved_path}")

    if suffix == ".csv":
        with resolved_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]

    raise ValueError(f"Unsupported evaluation input extension: {resolved_path.suffix}")


def normalize_row(raw_row: Mapping[str, Any], index: int) -> dict[str, Any]:
    row = dict(raw_row)
    payload = _unwrap_payload(row)

    question = str(_first_non_empty(row.get("question"), payload.get("question"), row.get("query")) or "").strip()
    answer = str(
        _first_non_empty(
            row.get("answer"),
            row.get("final_answer"),
            payload.get("answer"),
            payload.get("final_answer"),
            payload.get("draft_answer"),
            payload.get("message"),
            payload.get("content"),
        )
        or ""
    ).strip()
    reference = str(
        _first_non_empty(
            row.get("reference_answer"),
            row.get("gold_answer"),
            row.get("expected_answer"),
            row.get("ground_truth_answer"),
            payload.get("reference_answer"),
        )
        or ""
    ).strip()
    contexts = _extract_contexts(
        _first_non_empty(
            row.get("contexts"),
            row.get("retrieved_contexts"),
            payload.get("contexts"),
            payload.get("retrieved_contexts"),
            row.get("sources"),
            payload.get("sources"),
            payload.get("reranked_docs"),
            payload.get("retrieved_docs"),
        )
    )
    reference_contexts = _normalize_text_list(
        _first_non_empty(
            row.get("reference_contexts"),
            row.get("gold_contexts"),
            row.get("ground_truth_contexts"),
            row.get("gold_source_hint"),
        )
    )
    row_id = str(_first_non_empty(row.get("id"), row.get("row_id"), row.get("example_id")) or f"row-{index}")
    return {
        "id": row_id,
        "question": question,
        "answer": answer,
        "reference": reference,
        "contexts": contexts,
        "reference_contexts": reference_contexts,
        "backend_status": _resolve_text(_first_non_empty(row.get("status"), payload.get("status"))),
        "backend_route": _resolve_text(_first_non_empty(row.get("route"), payload.get("route"), payload.get("next_route"))),
        "backend_sources": _extract_contexts(_first_non_empty(row.get("sources"), payload.get("sources"))),
        "raw": dict(row),
    }


def build_sample(row: Mapping[str, Any]) -> Any:
    SingleTurnSample = _load_ragas_symbols()["SingleTurnSample"]
    payload: dict[str, Any] = {
        "user_input": row.get("question") or "",
        "response": row.get("answer") or "",
    }
    if row.get("reference"):
        payload["reference"] = row.get("reference")
    if row.get("contexts"):
        payload["retrieved_contexts"] = list(row.get("contexts") or [])
    if row.get("reference_contexts"):
        payload["reference_contexts"] = list(row.get("reference_contexts") or [])
    return SingleTurnSample(**payload)


def build_clients(config: EvalConfig) -> tuple[Any, Any]:
    ragas_symbols = _load_ragas_symbols()
    client = AsyncOpenAI(api_key=config.api_key, base_url=_normalize_base_url(config.base_url))
    llm = ragas_symbols["llm_factory"](config.llm_model, client=client)
    embeddings = ragas_symbols["embedding_factory"]("openai", model=config.embedding_model, client=client)
    return llm, embeddings


def build_metrics(config: EvalConfig, has_reference_contexts: bool, has_reference_answers: bool) -> list[tuple[str, Any]]:
    ragas_symbols = _load_ragas_symbols()
    metrics: list[tuple[str, Any]] = []

    if has_reference_contexts:
        metrics.append(
            ("non_llm_context_precision_with_reference", ragas_symbols["NonLLMContextPrecisionWithReference"]())
        )
        metrics.append(("non_llm_context_recall", ragas_symbols["NonLLMContextRecall"]()))

    if config.mode == "non_llm":
        return metrics

    llm, embeddings = build_clients(config)
    metrics.extend(
        [
            ("answer_relevancy", ragas_symbols["AnswerRelevancy"](llm=llm, embeddings=embeddings)),
            ("faithfulness", ragas_symbols["Faithfulness"](llm=llm)),
        ]
    )
    if has_reference_answers:
        metrics.extend(
            [
                ("answer_correctness", ragas_symbols["AnswerCorrectness"](llm=llm, embeddings=embeddings)),
                ("semantic_similarity", ragas_symbols["SemanticSimilarity"](embeddings=embeddings)),
                ("context_recall", ragas_symbols["ContextRecall"](llm=llm)),
            ]
        )
    return metrics


def _sample_kwargs(sample: Any) -> dict[str, Any]:
    if hasattr(sample, "model_dump"):
        return dict(sample.model_dump(exclude_none=True))
    if hasattr(sample, "dict"):
        return dict(sample.dict(exclude_none=True))
    return dict(sample.__dict__)


def _coerce_score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "value"):
        inner = getattr(value, "value")
        if isinstance(inner, (int, float)):
            return float(inner)
    if isinstance(value, Mapping):
        candidate = value.get("score") or value.get("value")
        if isinstance(candidate, (int, float)):
            return float(candidate)
    return None


def _metric_result_to_payload(value: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"score": _coerce_score(value)}
    if isinstance(value, Mapping):
        for key, inner in value.items():
            if isinstance(inner, (str, int, float, bool)) or inner is None:
                payload[str(key)] = inner
    elif hasattr(value, "reason"):
        payload["reason"] = getattr(value, "reason")
    return payload


def score_metric(metric_name: str, metric: Any, sample: Any) -> dict[str, Any]:
    kwargs = _sample_kwargs(sample)
    last_error: Exception | None = None

    for method_name in ("score", "single_turn_score"):
        method = getattr(metric, method_name, None)
        if method is None:
            continue
        try:
            signature = inspect.signature(method)
            if len(signature.parameters) == 1:
                result = method(sample)
            else:
                filtered_kwargs = {name: kwargs.get(name) for name in signature.parameters if name in kwargs}
                result = method(**filtered_kwargs)
            payload = _metric_result_to_payload(result)
            payload["metric"] = metric_name
            return payload
        except Exception as exc:  # pragma: no cover - runtime dependent
            last_error = exc

    error_message = str(last_error) if last_error else "Metric does not expose a compatible score method."
    LOGGER.warning("Metric %s failed: %s", metric_name, error_message)
    return {"metric": metric_name, "score": None, "error": error_message}


def _average(values: Sequence[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return round(mean(filtered), 4)


def write_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(dict(summary))


def run_evaluation(*, input_path: str | Path, output_dir: str | Path, config: EvalConfig) -> dict[str, Any]:
    rows = [normalize_row(row, idx) for idx, row in enumerate(load_rows(input_path), start=1)]
    backend_success_count = 0
    for row in rows:
        if config.call_backend:
            run_result = run_chatbot(str(row.get("question") or ""), config)
            row["backend_ok"] = bool(run_result.get("ok"))
            row["backend_error"] = run_result.get("error", "")
            row["backend_status"] = str(run_result.get("status") or "")
            row["backend_route"] = str(run_result.get("route") or "")
            row["backend_sources"] = list(run_result.get("sources") or [])
            if run_result.get("ok"):
                backend_success_count += 1
                row["answer"] = str(run_result.get("answer") or "")
        else:
            row["backend_ok"] = True
            row["backend_error"] = ""
            backend_success_count += 1
    has_reference_answers = any(bool(row["reference"]) for row in rows)
    has_reference_contexts = any(bool(row["reference_contexts"]) for row in rows)
    metrics = build_metrics(config, has_reference_contexts=has_reference_contexts, has_reference_answers=has_reference_answers)

    details: list[dict[str, Any]] = []
    metric_scores: dict[str, list[float | None]] = {name: [] for name, _ in metrics}

    for row in rows:
        sample = build_sample(row)
        row_detail: dict[str, Any] = {
            "id": row["id"],
            "question": row["question"],
            "answer_preview": row["answer"][:300],
            "reference_present": bool(row["reference"]),
            "context_count": len(row["contexts"]),
            "reference_context_count": len(row["reference_contexts"]),
            "backend_ok": bool(row.get("backend_ok")),
            "backend_status": row.get("backend_status"),
            "backend_route": row.get("backend_route"),
            "backend_error": row.get("backend_error"),
            "metrics": {},
        }
        for metric_name, metric in metrics:
            payload = score_metric(metric_name, metric, sample)
            row_detail["metrics"][metric_name] = payload
            metric_scores[metric_name].append(payload.get("score"))
        details.append(row_detail)

    summary: dict[str, Any] = {
        "sample_count": len(rows),
        "mode": config.mode,
        "llm_model": config.llm_model if config.mode != "non_llm" else None,
        "embedding_model": config.embedding_model if config.mode != "non_llm" else None,
        "api_url": config.api_url if config.call_backend else None,
        "backend_call_enabled": config.call_backend,
        "backend_success_rate": round(backend_success_count / len(rows), 4) if rows else None,
        "rows_with_reference_answer": sum(1 for row in rows if row["reference"]),
        "rows_with_reference_contexts": sum(1 for row in rows if row["reference_contexts"]),
    }
    for metric_name, scores in metric_scores.items():
        summary[f"avg_{metric_name}"] = _average(scores)
        summary[f"scored_{metric_name}_count"] = sum(1 for score in scores if score is not None)

    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_json = output_root / "eval_ragas_summary.json"
    summary_csv = output_root / "eval_ragas_summary.csv"
    details_jsonl = output_root / "eval_ragas_details.jsonl"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(summary_csv, summary)
    with details_jsonl.open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    return {
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "details_jsonl": str(details_jsonl),
        "metrics": summary,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate answer quality using the actual ragas library.")
    parser.add_argument(
        "--input",
        help=(
            "Path to JSON/JSONL/CSV rows containing question, answer/final_answer, optional "
            "reference_answer, contexts/sources, and optional reference_contexts. If omitted, the script "
            "loads the Hugging Face QA dataset."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_QA_DATASET,
        help="Hugging Face QA dataset to use when --input is omitted.",
    )
    parser.add_argument(
        "--split",
        default="auto",
        help="Dataset split to use for Hugging Face loading (default: auto -> test/validation/train).",
    )
    parser.add_argument("--limit", type=int, help="Optional max number of QA pairs to evaluate.")
    parser.add_argument("--output-dir", default="evaluation/results", help="Directory for evaluation artifacts.")
    parser.add_argument(
        "--mode",
        default=os.getenv("RAGAS_MODE", "auto"),
        choices=["auto", "llm", "non_llm"],
        help="auto: use non-LLM metrics plus LLM metrics when possible; non_llm: only reference-context metrics; llm: require LLM-backed metrics.",
    )
    parser.add_argument("--llm-model", default=os.getenv("RAGAS_LLM_MODEL", "qwen2.5:7b"))
    parser.add_argument("--embedding-model", default=os.getenv("RAGAS_EMBED_MODEL", "bge-m3"))
    parser.add_argument("--base-url", default=os.getenv("RAGAS_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--api-key", default=os.getenv("RAGAS_API_KEY", "ollama"))
    parser.add_argument("--api-url", default=os.getenv("RAGAS_CHATBOT_API_URL", "http://127.0.0.1:8000/chat"))
    parser.add_argument("--timeout-seconds", type=int, default=int(os.getenv("RAGAS_TIMEOUT_SECONDS", "180")))
    parser.add_argument(
        "--no-backend",
        action="store_true",
        help="Do not call the local chatbot API; use answers already present in the input rows.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)
    config = EvalConfig(
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        base_url=args.base_url,
        api_key=args.api_key,
        mode=args.mode,
        output_dir=args.output_dir,
        api_url=args.api_url,
        timeout_seconds=args.timeout_seconds,
        call_backend=not args.no_backend,
    )
    input_path = args.input
    temp_dataset_path: Path | None = None
    if not input_path:
        rows = load_hf_rows(args.dataset, split=args.split, limit=args.limit)
        temp_dataset_path = Path(config.output_dir).resolve() / "_ragas_hf_rows.jsonl"
        temp_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with temp_dataset_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
        input_path = str(temp_dataset_path)
        LOGGER.info("Loaded %s QA rows from Hugging Face dataset %s", len(rows), args.dataset)

    summary = run_evaluation(input_path=input_path, output_dir=args.output_dir, config=config)
    if temp_dataset_path is not None:
        summary["hf_dataset_cache"] = str(temp_dataset_path)
    LOGGER.info(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

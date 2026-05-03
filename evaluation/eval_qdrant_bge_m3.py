from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tv2_index.embedding_registry import DEFAULT_CONFIG_PATH
from src.tv2_index.search_with_filters import QdrantSearchService, normalize_filters, result_matches_filters

LOGGER = logging.getLogger(__name__)
DEFAULT_RETRIEVAL_DATASET = os.getenv(
    "RETRIEVAL_HF_DATASET",
    "YuITC/Vietnamese-Legal-Doc-Retrieval-Data",
)
DATASET_ALIASES = {
    "YuITC/Vietnamese-Legal-Doc-Retrieval-Data": "YuITC/Vietnamese-legal-documents",
}
_STOPWORDS = {
    "các",
    "cho",
    "có",
    "của",
    "đã",
    "đang",
    "để",
    "được",
    "hoặc",
    "khi",
    "không",
    "là",
    "một",
    "như",
    "này",
    "theo",
    "thì",
    "từ",
    "và",
    "về",
    "vì",
    "với",
}


@dataclass(slots=True)
class QueryCase:
    """One evaluation query paired with optional filters and ground truth."""

    query: str
    filters: dict[str, Any] = field(default_factory=dict)
    ground_truth_ids: list[str] = field(default_factory=list)
    ground_truth_mapc: list[str] = field(default_factory=list)
    ground_truth_article_codes: list[str] = field(default_factory=list)
    ground_truth_law_ids: list[str] = field(default_factory=list)
    ground_truth_titles: list[str] = field(default_factory=list)
    reference_contexts: list[str] = field(default_factory=list)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def _parse_list_field(value: Any) -> list[str]:
    if value in (None, "", []):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    return [item.strip() for item in text.split(",") if item.strip()]


def _normalize_text(value: str) -> str:
    lowered = (value or "").lower()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return re.sub(r"\s+", " ", lowered).strip()


def _informative_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", _normalize_text(value), flags=re.UNICODE)
        if len(token) > 2 and token not in _STOPWORDS and not token.isdigit()
    }


def _result_to_text(result: Mapping[str, Any]) -> str:
    metadata = dict(result.get("metadata") or {})
    parts = [
        result.get("content"),
        metadata.get("article_name"),
        metadata.get("title"),
        metadata.get("law_id"),
        metadata.get("source_note"),
    ]
    return " ".join(str(part).strip() for part in parts if part)


def _matches_reference_context(result: Mapping[str, Any], reference_context: str) -> bool:
    result_text = _normalize_text(_result_to_text(result))
    context_text = _normalize_text(reference_context)
    if not result_text or not context_text:
        return False

    shorter, longer = sorted((result_text, context_text), key=len)
    if len(shorter) >= 48 and shorter in longer:
        return True

    result_tokens = _informative_tokens(result_text)
    context_tokens = _informative_tokens(context_text)
    if not result_tokens or not context_tokens:
        return False

    overlap = len(result_tokens & context_tokens)
    min_size = min(len(result_tokens), len(context_tokens))
    return overlap >= 12 or (min_size > 0 and (overlap / min_size) >= 0.55)


def _resolve_dataset_name(dataset_name: str) -> str:
    text = (dataset_name or "").strip()
    return DATASET_ALIASES.get(text, text)


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


def load_hf_query_cases(dataset_name: str, *, split: str = "auto", limit: int | None = None) -> list[QueryCase]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Failed to import Hugging Face `datasets`. Please install compatible `datasets`/`pyarrow` packages."
        ) from exc

    resolved_name = _resolve_dataset_name(dataset_name)
    dataset = load_dataset(resolved_name)
    split_rows = _select_dataset_split(dataset, split)
    cases: list[QueryCase] = []
    for index, raw_row in enumerate(split_rows):
        if limit is not None and index >= limit:
            break
        row = dict(raw_row)
        query = str(row.get("question") or row.get("query") or "").strip()
        if not query:
            continue
        reference_contexts = _parse_list_field(
            row.get("context_list") or row.get("reference_contexts") or row.get("ground_truth_contexts")
        )
        cases.append(
            QueryCase(
                query=query,
                filters={},
                ground_truth_ids=_parse_list_field(row.get("cid") or row.get("ground_truth_ids")),
                ground_truth_titles=_parse_list_field(row.get("title") or row.get("ground_truth_titles")),
                reference_contexts=reference_contexts,
            )
        )
    return cases


def parse_query_case(raw: Mapping[str, Any]) -> QueryCase:
    query = str(raw.get("query") or raw.get("question") or "").strip()
    if not query:
        raise ValueError("Each evaluation row must provide `query` or `question`.")

    filters = raw.get("filters") or {}
    if isinstance(filters, str) and filters.strip():
        filters = json.loads(filters)
    filters = normalize_filters(filters if isinstance(filters, Mapping) else {})

    return QueryCase(
        query=query,
        filters=dict(filters),
        ground_truth_ids=_parse_list_field(raw.get("ground_truth_ids") or raw.get("point_ids")),
        ground_truth_mapc=_parse_list_field(raw.get("ground_truth_mapc") or raw.get("mapc")),
        ground_truth_article_codes=_parse_list_field(
            raw.get("ground_truth_article_codes") or raw.get("article_code")
        ),
        ground_truth_law_ids=_parse_list_field(raw.get("ground_truth_law_ids") or raw.get("law_id")),
        ground_truth_titles=_parse_list_field(raw.get("ground_truth_titles") or raw.get("title")),
        reference_contexts=_parse_list_field(
            raw.get("reference_contexts") or raw.get("ground_truth_contexts") or raw.get("context_list")
        ),
    )


def load_query_cases(path: str | Path) -> list[QueryCase]:
    """Load evaluation data from JSON, JSONL, or CSV."""

    resolved_path = Path(path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Evaluation query file not found: {resolved_path}")

    if resolved_path.suffix.lower() == ".json":
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        rows = payload.get("queries", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON list in {resolved_path}")
        return [parse_query_case(row) for row in rows]

    if resolved_path.suffix.lower() == ".jsonl":
        cases: list[QueryCase] = []
        with resolved_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                cases.append(parse_query_case(json.loads(line)))
        return cases

    if resolved_path.suffix.lower() == ".csv":
        with resolved_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [parse_query_case(row) for row in reader]

    raise ValueError(f"Unsupported evaluation input extension: {resolved_path.suffix}")


def result_matches_ground_truth(result: Mapping[str, Any], query_case: QueryCase) -> bool:
    """Flexible GT matching by id/mapc/article_code/law_id/title."""

    metadata = dict(result.get("metadata") or {})
    point_id = str(result.get("point_id") or "")
    candidates = {
        "id": point_id,
        "mapc": str(metadata.get("mapc") or ""),
        "article_code": str(metadata.get("article_code") or ""),
        "law_id": str(metadata.get("law_id") or ""),
        "title": str(metadata.get("title") or ""),
    }

    if query_case.ground_truth_ids and candidates["id"] in set(query_case.ground_truth_ids):
        return True
    if query_case.ground_truth_mapc and candidates["mapc"] in set(query_case.ground_truth_mapc):
        return True
    if query_case.ground_truth_article_codes and candidates["article_code"] in set(
        query_case.ground_truth_article_codes
    ):
        return True
    if query_case.ground_truth_law_ids and candidates["law_id"] in set(query_case.ground_truth_law_ids):
        return True
    if query_case.ground_truth_titles and candidates["title"] in set(query_case.ground_truth_titles):
        return True
    if query_case.reference_contexts and any(
        _matches_reference_context(result, reference_context)
        for reference_context in query_case.reference_contexts
    ):
        return True
    return False


def _safe_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(statistics.mean(values), 4)


def _safe_quantile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(float(values[0]), 3)
    sorted_values = sorted(float(value) for value in values)
    index = (len(sorted_values) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return round(sorted_values[lower], 3)
    fraction = index - lower
    blended = sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    return round(blended, 3)


def _find_first_relevant_rank(results: Sequence[Mapping[str, Any]], query_case: QueryCase) -> int | None:
    for rank, result in enumerate(results, start=1):
        if result_matches_ground_truth(result, query_case):
            return rank
    return None


def compute_filter_hit_rate(results: Sequence[Mapping[str, Any]], filters: Mapping[str, Any]) -> float | None:
    if not filters:
        return None
    if not results:
        return 0.0
    hits = sum(1 for result in results if result_matches_filters(result, filters))
    return round(hits / len(results), 4)


def benchmark_scenario(
    *,
    service: QdrantSearchService,
    query_cases: Sequence[QueryCase],
    level: str,
    use_filters: bool,
    top_k: int,
    collection_name: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one benchmark scenario and return aggregate + per-query details."""

    latencies_ms: list[float] = []
    filter_hit_rates: list[float] = []
    result_counts: list[float] = []
    details: list[dict[str, Any]] = []
    hit_counts = {1: 0, 3: 0, 5: 0, 10: 0}
    reciprocal_ranks: list[float] = []
    gt_case_count = 0

    for case in query_cases:
        applied_filters = case.filters if use_filters else {}
        start = time.perf_counter()
        if level == "chunk":
            results = service.search_chunk_level(
                case.query,
                filters=applied_filters,
                top_k=top_k,
                collection_name=collection_name,
            )
        else:
            results = service.search_article_level(
                case.query,
                filters=applied_filters,
                top_k=top_k,
                collection_name=collection_name,
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(latency_ms)
        result_counts.append(float(len(results)))

        has_ground_truth = any(
            [
                case.ground_truth_ids,
                case.ground_truth_mapc,
                case.ground_truth_article_codes,
                case.ground_truth_law_ids,
                case.ground_truth_titles,
            ]
        )
        relevant_rank = None
        hit_at_1 = None
        hit_at_3 = None
        hit_at_5 = None
        hit_at_10 = None
        if has_ground_truth:
            gt_case_count += 1
            relevant_rank = _find_first_relevant_rank(results[:top_k], case)
            if relevant_rank is not None:
                reciprocal_ranks.append(1.0 / relevant_rank)
            hit_at_1 = relevant_rank is not None and relevant_rank <= 1
            hit_at_3 = relevant_rank is not None and relevant_rank <= 3
            hit_at_5 = relevant_rank is not None and relevant_rank <= 5
            hit_at_10 = relevant_rank is not None and relevant_rank <= 10
            for cutoff in hit_counts:
                if relevant_rank is not None and relevant_rank <= cutoff:
                    hit_counts[cutoff] += 1

        scenario_filter_hit_rate = compute_filter_hit_rate(results, applied_filters)
        if scenario_filter_hit_rate is not None:
            filter_hit_rates.append(scenario_filter_hit_rate)

        details.append(
            {
                "level": level,
                "use_filters": use_filters,
                "top_k": top_k,
                "query": case.query,
                "filters": applied_filters,
                "latency_ms": round(latency_ms, 3),
                "result_count": len(results),
                "filter_hit_rate": scenario_filter_hit_rate,
                "first_relevant_rank": relevant_rank,
                "reciprocal_rank": round(1.0 / relevant_rank, 4) if relevant_rank else None,
                "hit_at_1": hit_at_1,
                "hit_at_3": hit_at_3,
                "hit_at_5": hit_at_5,
                "hit_at_10": hit_at_10,
            }
        )

    summary = {
        "level": level,
        "use_filters": use_filters,
        "top_k": top_k,
        "query_count": len(query_cases),
        "ground_truth_case_count": gt_case_count,
        "latency_mean_ms": _safe_mean(latencies_ms),
        "latency_p50_ms": round(statistics.median(latencies_ms), 3) if latencies_ms else None,
        "latency_p95_ms": _safe_quantile(latencies_ms, 0.95),
        "avg_result_count": _safe_mean(result_counts),
        "hit_rate_at_1": round(hit_counts[1] / gt_case_count, 4) if gt_case_count and top_k >= 1 else None,
        "hit_rate_at_3": round(hit_counts[3] / gt_case_count, 4) if gt_case_count and top_k >= 3 else None,
        "hit_rate_at_5": round(hit_counts[5] / gt_case_count, 4) if gt_case_count and top_k >= 5 else None,
        "hit_rate_at_10": round(hit_counts[10] / gt_case_count, 4) if gt_case_count and top_k >= 10 else None,
        "mrr_at_k": _safe_mean(reciprocal_ranks),
        "filter_hit_rate": _safe_mean(filter_hit_rates),
    }
    return summary, details


def write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "level",
        "use_filters",
        "top_k",
        "query_count",
        "ground_truth_case_count",
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "avg_result_count",
        "hit_rate_at_1",
        "hit_rate_at_3",
        "hit_rate_at_5",
        "hit_rate_at_10",
        "mrr_at_k",
        "filter_hit_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def run_benchmark(
    *,
    queries_path: str | Path | None = None,
    dataset_name: str | None = None,
    dataset_split: str = "auto",
    dataset_limit: int | None = None,
    level: str,
    config_path: str | Path | None = None,
    collection_name: str | None = None,
    top_k_values: Sequence[int] = (3, 5, 10),
    output_dir: str | Path = "evaluation/results",
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Benchmark Qdrant retrieval for chunk/article and filter/no-filter scenarios."""

    resolved_logger = logger or LOGGER
    if queries_path:
        query_cases = load_query_cases(queries_path)
        data_source = str(Path(queries_path).resolve())
    else:
        query_cases = load_hf_query_cases(
            dataset_name or DEFAULT_RETRIEVAL_DATASET,
            split=dataset_split,
            limit=dataset_limit,
        )
        data_source = f"hf://{_resolve_dataset_name(dataset_name or DEFAULT_RETRIEVAL_DATASET)}[{dataset_split}]"
    service = QdrantSearchService(config_path=config_path or DEFAULT_CONFIG_PATH, logger=resolved_logger)
    levels = [level] if level != "both" else ["chunk", "article"]

    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    has_any_filters = any(case.filters for case in query_cases)
    filter_modes = [False, True] if has_any_filters else [False]

    for current_level in levels:
        for use_filters in filter_modes:
            for top_k in top_k_values:
                summary, scenario_details = benchmark_scenario(
                    service=service,
                    query_cases=query_cases,
                    level=current_level,
                    use_filters=use_filters,
                    top_k=int(top_k),
                    collection_name=collection_name,
                )
                summaries.append(summary)
                details.extend(scenario_details)
                resolved_logger.info(
                    "Benchmarked level=%s filters=%s top_k=%s => latency_p50=%s",
                    current_level,
                    use_filters,
                    top_k,
                    summary["latency_p50_ms"],
                )

    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_json = output_root / "eval_qdrant_bge_m3_summary.json"
    summary_csv = output_root / "eval_qdrant_bge_m3_summary.csv"
    detail_jsonl = output_root / "eval_qdrant_bge_m3_details.jsonl"

    summary_json.write_text(
        json.dumps(
            {
                "data_source": data_source,
                "query_count": len(query_cases),
                "summaries": summaries,
                "details_count": len(details),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary_csv(summary_csv, summaries)
    with detail_jsonl.open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    return {
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "details_jsonl": str(detail_jsonl),
        "scenario_count": len(summaries),
        "data_source": data_source,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Qdrant + bge-m3 retrieval quality and latency.")
    parser.add_argument(
        "--queries",
        help="Path to query+ground-truth JSON/JSONL/CSV. If omitted, the script loads the Hugging Face retrieval dataset.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_RETRIEVAL_DATASET,
        help="Hugging Face dataset name for retrieval evaluation when --queries is omitted.",
    )
    parser.add_argument(
        "--split",
        default="auto",
        help="Dataset split to use for Hugging Face loading (default: auto -> test/validation/train).",
    )
    parser.add_argument("--limit", type=int, help="Optional max number of dataset rows to evaluate.")
    parser.add_argument(
        "--level",
        default="both",
        choices=["chunk", "article", "both"],
        help="Which retrieval granularity to benchmark.",
    )
    parser.add_argument("--collection", help="Optional explicit collection or alias name.")
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=[3, 5, 10],
        help="Top-k values to benchmark.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results",
        help="Output directory for CSV/JSON benchmark artifacts.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to configs/indexing.yaml")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)
    summary = run_benchmark(
        queries_path=args.queries,
        dataset_name=args.dataset,
        dataset_split=args.split,
        dataset_limit=args.limit,
        level=args.level,
        config_path=args.config,
        collection_name=args.collection,
        top_k_values=args.top_k,
        output_dir=args.output,
        logger=LOGGER,
    )
    LOGGER.info(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

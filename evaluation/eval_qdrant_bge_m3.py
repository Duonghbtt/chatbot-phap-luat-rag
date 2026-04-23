from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.tv2_index.embedding_registry import DEFAULT_CONFIG_PATH
from src.tv2_index.search_with_filters import QdrantSearchService, normalize_filters, result_matches_filters

LOGGER = logging.getLogger(__name__)


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
    return False


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
    details: list[dict[str, Any]] = []
    recall_hits_at_5 = 0
    recall_hits_at_10 = 0
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

        has_ground_truth = any(
            [
                case.ground_truth_ids,
                case.ground_truth_mapc,
                case.ground_truth_article_codes,
                case.ground_truth_law_ids,
                case.ground_truth_titles,
            ]
        )
        hit_at_5 = None
        hit_at_10 = None
        if has_ground_truth:
            gt_case_count += 1
            if top_k >= 5:
                hit_at_5 = any(result_matches_ground_truth(result, case) for result in results[:5])
                if hit_at_5:
                    recall_hits_at_5 += 1
            if top_k >= 10:
                hit_at_10 = any(result_matches_ground_truth(result, case) for result in results[:10])
                if hit_at_10:
                    recall_hits_at_10 += 1

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
                "hit_at_5": hit_at_5,
                "hit_at_10": hit_at_10,
            }
        )

    summary = {
        "level": level,
        "use_filters": use_filters,
        "top_k": top_k,
        "query_count": len(query_cases),
        "latency_p50_ms": round(statistics.median(latencies_ms), 3) if latencies_ms else None,
        "recall_at_5": round(recall_hits_at_5 / gt_case_count, 4) if gt_case_count and top_k >= 5 else None,
        "recall_at_10": round(recall_hits_at_10 / gt_case_count, 4) if gt_case_count and top_k >= 10 else None,
        "filter_hit_rate": round(statistics.mean(filter_hit_rates), 4) if filter_hit_rates else None,
    }
    return summary, details


def write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "level",
        "use_filters",
        "top_k",
        "query_count",
        "latency_p50_ms",
        "recall_at_5",
        "recall_at_10",
        "filter_hit_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def run_benchmark(
    *,
    queries_path: str | Path,
    level: str,
    config_path: str | Path | None = None,
    collection_name: str | None = None,
    top_k_values: Sequence[int] = (3, 5, 10),
    output_dir: str | Path = "evaluation/results",
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Benchmark Qdrant retrieval for chunk/article and filter/no-filter scenarios."""

    resolved_logger = logger or LOGGER
    query_cases = load_query_cases(queries_path)
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
        json.dumps({"summaries": summaries, "details_count": len(details)}, ensure_ascii=False, indent=2),
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
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Qdrant + bge-m3 retrieval quality and latency.")
    parser.add_argument("--queries", required=True, help="Path to query+ground-truth JSON/JSONL/CSV.")
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

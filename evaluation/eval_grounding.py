from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

LOGGER = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    """Load grounding predictions from JSONL or JSON."""

    resolved_path = Path(path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {resolved_path}")

    if resolved_path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with resolved_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, Mapping):
                    rows.append(dict(payload))
        return rows

    if resolved_path.suffix.lower() == ".json":
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, Mapping)]
        if isinstance(payload, Mapping):
            rows = payload.get("predictions", payload.get("rows", []))
            if isinstance(rows, list):
                return [dict(item) for item in rows if isinstance(item, Mapping)]
        raise ValueError(f"Unsupported JSON structure in {resolved_path}")

    raise ValueError(f"Unsupported predictions format: {resolved_path.suffix}")


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _count_items(value: Any) -> int:
    if value in (None, "", []):
        return 0
    if isinstance(value, list):
        return len(value)
    return 1


def compute_grounding_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute TV5 grounding metrics from prediction records."""

    total = len(rows)
    if total == 0:
        return {
            "sample_count": 0,
            "grounded_answer_rate": 0.0,
            "unsupported_claim_rate": 0.0,
            "citation_correctness_rate": 0.0,
            "avg_grounding_score": None,
            "avg_faithfulness": None,
            "avg_answer_relevancy": None,
        }

    grounded_count = sum(1 for row in rows if _to_bool(row.get("grounding_ok")))
    citation_ok_count = 0
    total_unsupported_claims = 0
    total_claims = 0
    grounding_scores: list[float] = []
    faithfulness_scores: list[float] = []
    answer_relevancy_scores: list[float] = []

    for row in rows:
        citation_findings = row.get("citation_findings") or {}
        if isinstance(citation_findings, Mapping):
            citation_ok = _to_bool(citation_findings.get("citation_ok"))
        else:
            citation_ok = _to_bool(row.get("citation_ok"))
        if citation_ok:
            citation_ok_count += 1

        unsupported_count = _count_items(row.get("unsupported_claims"))
        claim_count = int(row.get("claim_count") or 0)
        if claim_count <= 0:
            claim_count = max(1, unsupported_count if unsupported_count else 1)
        total_unsupported_claims += unsupported_count
        total_claims += claim_count

        grounding_score = _to_float(row.get("grounding_score"))
        if grounding_score is not None:
            grounding_scores.append(grounding_score)
        faithfulness = _to_float(row.get("faithfulness"))
        if faithfulness is not None:
            faithfulness_scores.append(faithfulness)
        answer_relevancy = _to_float(row.get("answer_relevancy"))
        if answer_relevancy is not None:
            answer_relevancy_scores.append(answer_relevancy)

    return {
        "sample_count": total,
        "grounded_answer_rate": round(grounded_count / total, 4),
        "unsupported_claim_rate": round(total_unsupported_claims / max(total_claims, 1), 4),
        "citation_correctness_rate": round(citation_ok_count / total, 4),
        "avg_grounding_score": round(sum(grounding_scores) / len(grounding_scores), 4)
        if grounding_scores
        else None,
        "avg_faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 4)
        if faithfulness_scores
        else None,
        "avg_answer_relevancy": round(sum(answer_relevancy_scores) / len(answer_relevancy_scores), 4)
        if answer_relevancy_scores
        else None,
    }


def write_summary_csv(path: Path, metrics: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(dict(metrics))


def run_evaluation(
    *,
    predictions_path: str | Path,
    output_dir: str | Path = "evaluation/results",
) -> dict[str, Any]:
    """Evaluate TV5 grounding results and write summary artifacts."""

    rows = load_predictions(predictions_path)
    metrics = compute_grounding_metrics(rows)
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "eval_grounding_summary.json"
    csv_path = output_root / "eval_grounding_summary.csv"

    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(csv_path, metrics)
    return {
        "summary_json": str(json_path),
        "summary_csv": str(csv_path),
        "metrics": metrics,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate TV5 grounding outputs from JSONL/JSON predictions.")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to JSONL/JSON predictions produced by the reasoning subgraph.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/results",
        help="Directory where JSON/CSV evaluation summaries will be written.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)
    summary = run_evaluation(predictions_path=args.predictions, output_dir=args.output_dir)
    LOGGER.info(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

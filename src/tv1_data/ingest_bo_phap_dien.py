from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.tv1_data.chunk_legal_docs import ChunkConfig, chunk_document
from src.tv1_data.parse_clean import CorpusLookup, ParsedDocument, load_corpus_lookup, parse_html_file

LOGGER = logging.getLogger(__name__)
DEFAULT_MANIFEST = Path("data/manifests/legal_corpus_manifest.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/processed")
SKIP_HTML_FILENAMES = {"bophapdien.html"}
SKIP_HTML_DIRS = {"lib"}


@dataclass(slots=True)
class IngestArtifacts:
    """Full TV1 artifact bundle ready for export or incremental sync."""

    chunks: list[dict[str, Any]]
    manifest_entries: list[dict[str, Any]]
    stats: dict[str, Any]


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def canonicalize_path(path: str | Path) -> str:
    return str(Path(path).resolve()).lower()


def _should_skip_html(path: Path) -> bool:
    if path.name.lower() in SKIP_HTML_FILENAMES:
        return True
    return any(part.lower() in SKIP_HTML_DIRS for part in path.parts)


def discover_html_files(input_path: Path) -> list[Path]:
    """Discover local Bộ pháp điển HTML files from either one file or a directory."""

    resolved_input = input_path.resolve()
    if not resolved_input.exists():
        raise FileNotFoundError(f"Không tìm thấy đầu vào: {resolved_input}")

    if resolved_input.is_file():
        if resolved_input.suffix.lower() != ".html":
            raise ValueError(f"Đầu vào phải là file .html: {resolved_input}")
        if _should_skip_html(resolved_input):
            return []
        return [resolved_input]

    html_files = [
        path.resolve()
        for path in resolved_input.rglob("*.html")
        if not _should_skip_html(path)
    ]
    return sorted(html_files)


def infer_snapshot_root(input_path: Path) -> Path | None:
    """Infer the official snapshot root that contains `jsonData.js`."""

    resolved_input = input_path.resolve()
    candidates: list[Path] = []
    if resolved_input.is_dir():
        candidates.extend([resolved_input, resolved_input.parent])
    else:
        candidates.extend([resolved_input.parent, resolved_input.parent.parent])

    for candidate in candidates:
        if candidate.exists() and (candidate / "jsonData.js").exists():
            return candidate
    return None


def compute_file_signature(source_path: Path) -> dict[str, Any]:
    """Compute a stable checksum and lightweight version info for one source file."""

    data = source_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    stat = source_path.stat()
    return {
        "content_sha256": digest,
        "snapshot_version": digest[:16],
        "file_size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _sort_chunks(chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        chunks,
        key=lambda chunk: (
            str(chunk.get("metadata", {}).get("file_id", "")),
            str(chunk.get("metadata", {}).get("article_code", "")),
            str(chunk.get("metadata", {}).get("law_id", "")),
            str(chunk.get("content", ""))[:80],
        ),
    )


def _collect_source_laws(document: ParsedDocument) -> list[dict[str, str]]:
    law_index: dict[str, dict[str, str]] = {}

    for article in document.articles:
        if not article.law_id and not article.title:
            continue

        key = article.law_id or article.title
        current = law_index.setdefault(
            key,
            {
                "law_id": article.law_id,
                "title": article.title,
                "issuer": article.issuer,
                "effective_date": article.effective_date,
            },
        )
        title_candidate = article.title
        if title_candidate and len(title_candidate) > len(current.get("title", "")):
            current["title"] = title_candidate

        issuer_candidate = article.issuer
        current_issuer = current.get("issuer", "")
        if issuer_candidate and (not current_issuer or ("ngày" in current_issuer.lower() and "ngày" not in issuer_candidate.lower())):
            current["issuer"] = issuer_candidate

        effective_candidate = article.effective_date
        if effective_candidate and len(effective_candidate) > len(current.get("effective_date", "")):
            current["effective_date"] = effective_candidate

    return sorted(
        law_index.values(),
        key=lambda item: (item.get("law_id", ""), item.get("title", "")),
    )


def build_manifest_entry(
    document: ParsedDocument,
    chunk_records: Sequence[dict[str, Any]],
    snapshot_root: Path | None,
    synced_at: str,
) -> dict[str, Any]:
    """Build one manifest entry per source HTML file."""

    signature = compute_file_signature(document.source_path)
    relative_path = ""
    if snapshot_root is not None:
        try:
            relative_path = str(document.source_path.relative_to(snapshot_root))
        except ValueError:
            relative_path = str(document.source_path)
    else:
        relative_path = str(document.source_path)

    source_laws = _collect_source_laws(document)
    return {
        "file_id": document.file_id,
        "source_path": str(document.source_path),
        "relative_path": relative_path,
        "de_muc": document.de_muc,
        "topic_id": document.topic_id,
        "topic_name": document.topic_name,
        "source_laws": source_laws,
        "law_ids": [law["law_id"] for law in source_laws if law.get("law_id")],
        "article_count": len(document.articles),
        "chunk_count": len(chunk_records),
        "synced_at": synced_at,
        **signature,
    }


def build_stats(chunks: Sequence[dict[str, Any]], total_files: int) -> dict[str, Any]:
    """Aggregate corpus-level statistics required by the TV1 report."""

    normalized_chunks = list(chunks)
    content_lengths = [len(str(chunk.get("content", "")).strip()) for chunk in normalized_chunks]
    chunk_count_by_de_muc = Counter()
    chunk_count_by_law_id = Counter()
    total_empty_content = 0
    total_missing_issuer = 0
    total_missing_effective_date = 0

    for chunk in normalized_chunks:
        content = str(chunk.get("content", "")).strip()
        metadata = chunk.get("metadata", {}) or {}
        de_muc = str(metadata.get("de_muc", "") or "(missing)")
        law_id = str(metadata.get("law_id", "") or "(missing)")
        issuer = str(metadata.get("issuer", "") or "").strip()
        effective_date = str(metadata.get("effective_date", "") or "").strip()

        chunk_count_by_de_muc[de_muc] += 1
        chunk_count_by_law_id[law_id] += 1
        if not content:
            total_empty_content += 1
        if not issuer:
            total_missing_issuer += 1
        if not effective_date:
            total_missing_effective_date += 1

    average_length = round(sum(content_lengths) / len(content_lengths), 2) if content_lengths else 0.0
    max_length = max(content_lengths) if content_lengths else 0

    return {
        "total_files": total_files,
        "total_chunks": len(normalized_chunks),
        "total_empty_content": total_empty_content,
        "total_missing_issuer": total_missing_issuer,
        "total_missing_effective_date": total_missing_effective_date,
        "avg_content_length": average_length,
        "max_content_length": max_length,
        "chunk_count_by_de_muc": dict(sorted(chunk_count_by_de_muc.items())),
        "chunk_count_by_law_id": dict(sorted(chunk_count_by_law_id.items())),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _write_preview_csv(path: Path, chunks: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_id",
        "de_muc",
        "topic_id",
        "law_id",
        "title",
        "article",
        "article_code",
        "article_name",
        "issuer",
        "effective_date",
        "content_length",
        "content_preview",
        "related_articles_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for chunk in chunks:
            metadata = chunk.get("metadata", {}) or {}
            content = str(chunk.get("content", "") or "")
            writer.writerow(
                {
                    "file_id": metadata.get("file_id", ""),
                    "de_muc": metadata.get("de_muc", ""),
                    "topic_id": metadata.get("topic_id", ""),
                    "law_id": metadata.get("law_id", ""),
                    "title": metadata.get("title", ""),
                    "article": metadata.get("article", ""),
                    "article_code": metadata.get("article_code", ""),
                    "article_name": metadata.get("article_name", ""),
                    "issuer": metadata.get("issuer", ""),
                    "effective_date": metadata.get("effective_date", ""),
                    "content_length": len(content.strip()),
                    "content_preview": content.replace("\n", " ")[:240],
                    "related_articles_count": len(metadata.get("related_articles", []) or []),
                }
            )


def load_chunk_records(chunk_json_path: Path) -> list[dict[str, Any]]:
    if not chunk_json_path.exists():
        return []

    payload = json.loads(chunk_json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"File chunks không hợp lệ: {chunk_json_path}")
    return payload


def export_corpus_artifacts(
    chunks: Sequence[dict[str, Any]],
    output_dir: Path,
    manifest_entries: Sequence[dict[str, Any]],
    manifest_path: Path,
) -> dict[str, Any]:
    """Write TV1 outputs to disk for downstream Qdrant indexing."""

    normalized_chunks = _sort_chunks(chunks)
    normalized_manifest = sorted(
        manifest_entries,
        key=lambda entry: (str(entry.get("file_id", "")), str(entry.get("source_path", ""))),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    _write_json(output_dir / "all_chunks.json", normalized_chunks)
    _write_jsonl(output_dir / "all_chunks.jsonl", normalized_chunks)
    _write_preview_csv(output_dir / "chunks_preview.csv", normalized_chunks)
    _write_jsonl(manifest_path, normalized_manifest)

    stats = build_stats(normalized_chunks, total_files=len(normalized_manifest))
    _write_json(output_dir / "stats.json", stats)
    return stats


def collect_corpus_artifacts(
    source_files: Sequence[Path],
    snapshot_root: Path | None,
    chunk_config: ChunkConfig | None = None,
    logger: logging.Logger | None = None,
) -> IngestArtifacts:
    """Parse, clean, chunk, and manifest one batch of local HTML files."""

    log = logger or LOGGER
    config = chunk_config or ChunkConfig()
    lookup: CorpusLookup = load_corpus_lookup(snapshot_root, logger=log)
    synced_at = datetime.now().astimezone().isoformat()

    all_chunks: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []

    for source_file in source_files:
        log.info("Đang ingest %s", source_file)
        document = parse_html_file(source_file, lookup=lookup, logger=log)
        chunk_records = chunk_document(document, config=config)
        all_chunks.extend(chunk_records)
        manifest_entries.append(build_manifest_entry(document, chunk_records, snapshot_root, synced_at))
        log.info(
            "Hoàn thành %s: %s Điều -> %s chunk",
            source_file.name,
            len(document.articles),
            len(chunk_records),
        )

    stats = build_stats(all_chunks, total_files=len(source_files))
    return IngestArtifacts(chunks=all_chunks, manifest_entries=manifest_entries, stats=stats)


def run_ingestion(
    input_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
    chunk_config: ChunkConfig | None = None,
    logger: logging.Logger | None = None,
) -> IngestArtifacts:
    """High-level TV1 ingestion entrypoint used by the CLI and sync module."""

    log = logger or LOGGER
    source_files = discover_html_files(input_path)
    if not source_files:
        raise FileNotFoundError(f"Không tìm thấy file HTML hợp lệ trong {input_path.resolve()}")

    snapshot_root = infer_snapshot_root(input_path)
    artifacts = collect_corpus_artifacts(
        source_files=source_files,
        snapshot_root=snapshot_root,
        chunk_config=chunk_config,
        logger=log,
    )
    artifacts.stats = export_corpus_artifacts(
        chunks=artifacts.chunks,
        output_dir=output_dir.resolve(),
        manifest_entries=artifacts.manifest_entries,
        manifest_path=manifest_path.resolve(),
    )
    return artifacts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TV1 local-only ingest pipeline for Bộ pháp điển điện tử HTML.",
    )
    parser.add_argument("--input", required=True, help="Đường dẫn tới file HTML hoặc thư mục HTML chính thức.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Thư mục sinh đầu ra `all_chunks.*`, `chunks_preview.csv`, `stats.json`.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Đường dẫn manifest `legal_corpus_manifest.jsonl`.",
    )
    parser.add_argument("--chunk-size", type=int, default=800, help="Kích thước chunk mặc định cho Recursive splitter.")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Độ chồng chunk mặc định.")
    parser.add_argument("--log-level", default="INFO", help="Mức log, ví dụ INFO hoặc DEBUG.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)
    artifacts = run_ingestion(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        manifest_path=Path(args.manifest),
        chunk_config=ChunkConfig(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap),
        logger=LOGGER,
    )

    LOGGER.info("TV1 ingest hoàn tất")
    LOGGER.info("  Tổng file: %s", artifacts.stats["total_files"])
    LOGGER.info("  Tổng chunk: %s", artifacts.stats["total_chunks"])
    LOGGER.info("  Manifest: %s", Path(args.manifest).resolve())
    LOGGER.info("  Output dir: %s", Path(args.output).resolve())


if __name__ == "__main__":
    main()

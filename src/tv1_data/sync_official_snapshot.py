from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.tv1_data.chunk_legal_docs import ChunkConfig
from src.tv1_data.ingest_bo_phap_dien import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    canonicalize_path,
    collect_corpus_artifacts,
    compute_file_signature,
    discover_html_files,
    export_corpus_artifacts,
    infer_snapshot_root,
    load_chunk_records,
    setup_logging,
)

LOGGER = logging.getLogger(__name__)


def load_manifest_map(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Load existing manifest entries indexed by canonical source path."""

    if not manifest_path.exists():
        return {}

    manifest_entries: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            source_path = entry.get("source_path")
            if not source_path:
                continue
            manifest_entries[canonicalize_path(source_path)] = entry
    return manifest_entries


def detect_snapshot_changes(
    source_files: list[Path],
    manifest_map: dict[str, dict[str, Any]],
) -> tuple[list[Path], list[Path], list[dict[str, Any]]]:
    """Detect new or modified HTML files using SHA-256 signatures."""

    changed_or_new: list[Path] = []
    unchanged: list[Path] = []
    current_keys = {canonicalize_path(path) for path in source_files}

    for source_file in source_files:
        key = canonicalize_path(source_file)
        signature = compute_file_signature(source_file)
        manifest_entry = manifest_map.get(key)
        if manifest_entry and manifest_entry.get("content_sha256") == signature["content_sha256"]:
            unchanged.append(source_file)
        else:
            changed_or_new.append(source_file)

    removed_entries = [entry for key, entry in manifest_map.items() if key not in current_keys]
    return changed_or_new, unchanged, removed_entries


def merge_chunk_records(
    existing_chunks: list[dict[str, Any]],
    updated_chunks: list[dict[str, Any]],
    changed_file_ids: set[str],
    removed_file_ids: set[str],
) -> list[dict[str, Any]]:
    """Replace chunk records for changed sources while preserving unchanged files."""

    affected_ids = changed_file_ids | removed_file_ids
    preserved = [
        chunk
        for chunk in existing_chunks
        if str(chunk.get("metadata", {}).get("file_id", "")) not in affected_ids
    ]
    return preserved + updated_chunks


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Incremental sync for the official Bộ pháp điển offline HTML snapshot.",
    )
    parser.add_argument("--input", required=True, help="Thư mục hoặc file HTML nguồn chính thức.")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Manifest hiện tại để so sánh checksum và cập nhật incremental.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Thư mục đầu ra chứa `all_chunks.*`, `chunks_preview.csv`, `stats.json`.",
    )
    parser.add_argument("--chunk-size", type=int, default=800, help="Kích thước chunk khi cần chia nhỏ Điều dài.")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Độ chồng chunk mặc định.")
    parser.add_argument("--log-level", default="INFO", help="Mức log, ví dụ INFO hoặc DEBUG.")
    return parser


def run_incremental_sync(
    input_path: Path,
    manifest_path: Path,
    output_dir: Path,
    chunk_config: ChunkConfig | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Sync only new or changed HTML files and update aggregate outputs."""

    log = logger or LOGGER
    source_files = discover_html_files(input_path)
    snapshot_root = infer_snapshot_root(input_path)
    manifest_map = load_manifest_map(manifest_path)

    changed_files, unchanged_files, removed_entries = detect_snapshot_changes(source_files, manifest_map)

    all_chunks_path = output_dir / "all_chunks.json"
    if not all_chunks_path.exists() and source_files:
        log.info("Không tìm thấy %s; sẽ build lại aggregate output từ snapshot hiện tại.", all_chunks_path)
        changed_files = source_files
        unchanged_files = []
        removed_entries = []

    if not changed_files and not removed_entries:
        log.info("Snapshot không có thay đổi. Bỏ qua rebuild.")
        return {
            "changed_files": 0,
            "unchanged_files": len(unchanged_files),
            "removed_files": 0,
            "total_chunks": len(load_chunk_records(all_chunks_path)),
            "manifest": str(manifest_path.resolve()),
            "output_dir": str(output_dir.resolve()),
        }

    updated_artifacts = collect_corpus_artifacts(
        source_files=changed_files,
        snapshot_root=snapshot_root,
        chunk_config=chunk_config,
        logger=log,
    )

    existing_chunks = load_chunk_records(all_chunks_path)
    changed_file_ids = {path.stem for path in changed_files}
    removed_file_ids = {
        str(entry.get("file_id") or Path(str(entry.get("source_path", ""))).stem)
        for entry in removed_entries
    }
    merged_chunks = merge_chunk_records(
        existing_chunks=existing_chunks,
        updated_chunks=updated_artifacts.chunks,
        changed_file_ids=changed_file_ids,
        removed_file_ids=removed_file_ids,
    )

    new_manifest_map = dict(manifest_map)
    for removed_entry in removed_entries:
        source_path = removed_entry.get("source_path")
        if source_path:
            new_manifest_map.pop(canonicalize_path(source_path), None)
    for entry in updated_artifacts.manifest_entries:
        new_manifest_map[canonicalize_path(entry["source_path"])] = entry

    stats = export_corpus_artifacts(
        chunks=merged_chunks,
        output_dir=output_dir.resolve(),
        manifest_entries=list(new_manifest_map.values()),
        manifest_path=manifest_path.resolve(),
    )

    return {
        "changed_files": len(changed_files),
        "unchanged_files": len(unchanged_files),
        "removed_files": len(removed_entries),
        "total_chunks": stats["total_chunks"],
        "manifest": str(manifest_path.resolve()),
        "output_dir": str(output_dir.resolve()),
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)
    summary = run_incremental_sync(
        input_path=Path(args.input),
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output),
        chunk_config=ChunkConfig(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap),
        logger=LOGGER,
    )

    LOGGER.info("Sync snapshot hoàn tất")
    LOGGER.info("  File mới/thay đổi: %s", summary["changed_files"])
    LOGGER.info("  File không đổi: %s", summary["unchanged_files"])
    LOGGER.info("  File bị xóa: %s", summary["removed_files"])
    LOGGER.info("  Tổng chunk hiện tại: %s", summary["total_chunks"])
    LOGGER.info("  Manifest: %s", summary["manifest"])
    LOGGER.info("  Output dir: %s", summary["output_dir"])


if __name__ == "__main__":
    main()

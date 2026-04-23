from __future__ import annotations

import argparse
import hashlib
import json
import logging
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence, Mapping

from src.tv2_index.embedding_registry import (
    AppConfig,
    DEFAULT_CONFIG_PATH,
    get_embedder,
    load_indexing_config,
)
from src.tv2_index.qdrant_manager import QdrantManager

LOGGER = logging.getLogger(__name__)
UUID_NAMESPACE = uuid.UUID("9ad3d3f9-e77e-4b1f-bd48-c552df8e6d42")


@dataclass(slots=True)
class IndexDocument:
    """Prepared document payload ready for embedding and Qdrant upsert."""

    point_id: str
    content: str
    payload: dict[str, Any]


def _dedupe_text_parts(parts: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = " ".join(str(part or "").split()).strip(" .")
        if not cleaned:
            continue
        normalized = cleaned.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(cleaned)
    return deduped


def build_article_retrieval_text(metadata: Mapping[str, Any], content: str) -> str:
    """Construct retrieval text that blends article content with strong legal identifiers."""

    text_parts = _dedupe_text_parts(
        [
            str(metadata.get("title") or ""),
            str(metadata.get("law_id") or ""),
            str(metadata.get("article") or ""),
            str(metadata.get("article_code") or ""),
            str(metadata.get("article_name") or ""),
            str(metadata.get("de_muc") or ""),
            str(content or ""),
        ]
    )
    return ". ".join(text_parts).strip()


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def load_tv1_records(input_path: str | Path) -> list[dict[str, Any]]:
    """Load TV1 chunks from JSON or JSONL."""

    resolved_path = Path(input_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Input file not found: {resolved_path}")

    if resolved_path.suffix.lower() == ".json":
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list in {resolved_path}")
        return [normalize_tv1_record(item) for item in payload]

    if resolved_path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with resolved_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                records.append(normalize_tv1_record(json.loads(line)))
        return records

    raise ValueError(f"Unsupported input extension: {resolved_path.suffix}")


def normalize_tv1_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize one TV1 chunk record into a stable dict shape."""

    metadata = dict(record.get("metadata") or {})
    related_articles = metadata.get("related_articles") or []
    if not isinstance(related_articles, list):
        related_articles = [str(related_articles)]
    metadata["related_articles"] = [str(item) for item in related_articles if str(item).strip()]

    return {
        "content": str(record.get("content") or "").strip(),
        "metadata": {key: "" if value is None else value for key, value in metadata.items()},
    }


def parse_effective_date_to_iso(date_text: str) -> str:
    """Convert `dd/mm/yyyy` into RFC3339 date format for Qdrant datetime filtering."""

    normalized = (date_text or "").strip()
    if not normalized:
        return ""
    for pattern in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(normalized, pattern)
            return parsed.strftime("%Y-%m-%dT00:00:00Z")
        except ValueError:
            continue
    return ""


def stable_point_id(level: str, key: str) -> str:
    """Build a deterministic point id for idempotent upserts."""

    return str(uuid.uuid5(UUID_NAMESPACE, f"tv2:{level}:{key}"))


def generate_versioned_collection_name(prefix: str, version_tag: str | None = None) -> str:
    """Generate a versioned collection name like `legal_chunks_v20260422`."""

    tag = version_tag or datetime.now().strftime("%Y%m%d")
    return f"{prefix}_v{tag}"


def derive_article_group_key(metadata: dict[str, Any]) -> str:
    """Build the grouping key for article-level aggregation."""

    key_parts = [
        str(metadata.get("file_id") or ""),
        str(metadata.get("law_id") or ""),
        str(metadata.get("article_code") or metadata.get("article") or ""),
        str(metadata.get("article_name") or ""),
    ]
    return "|".join(key_parts)


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def prepare_chunk_documents(records: Sequence[dict[str, Any]]) -> list[IndexDocument]:
    """Prepare chunk-level Qdrant documents from TV1 records."""

    documents: list[IndexDocument] = []
    for record in records:
        content = str(record.get("content") or "").strip()
        metadata = dict(record.get("metadata") or {})
        metadata.setdefault("related_articles", [])
        point_key = "|".join(
            [
                str(metadata.get("file_id") or ""),
                str(metadata.get("law_id") or ""),
                str(metadata.get("article_code") or ""),
                str(metadata.get("mapc") or ""),
                _hash_text(content),
            ]
        )
        payload = {
            **metadata,
            "metadata": metadata,
            "content": content,
            "retrieval_text": content,
            "doc_type": "chunk",
            "content_length": len(content),
            "effective_date_iso": parse_effective_date_to_iso(str(metadata.get("effective_date") or "")),
        }
        documents.append(
            IndexDocument(
                point_id=stable_point_id("chunk", point_key),
                content=content,
                payload=payload,
            )
        )
    return documents


def prepare_article_documents(records: Sequence[dict[str, Any]]) -> list[IndexDocument]:
    """Aggregate TV1 chunk records into article-level documents."""

    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for record in records:
        content = str(record.get("content") or "").strip()
        metadata = dict(record.get("metadata") or {})
        metadata.setdefault("related_articles", [])
        group_key = derive_article_group_key(metadata)

        if group_key not in grouped:
            grouped[group_key] = {
                "metadata": metadata,
                "contents": [],
                "related_articles": OrderedDict(),
                "chunk_count": 0,
            }

        bucket = grouped[group_key]
        if content and content not in bucket["contents"]:
            bucket["contents"].append(content)
        bucket["chunk_count"] += 1
        for article_ref in metadata.get("related_articles") or []:
            bucket["related_articles"][str(article_ref)] = True

    documents: list[IndexDocument] = []
    for group_key, bucket in grouped.items():
        metadata = dict(bucket["metadata"])
        metadata["related_articles"] = list(bucket["related_articles"].keys())
        content = "\n\n".join(bucket["contents"]).strip()
        retrieval_text = build_article_retrieval_text(metadata, content)
        payload = {
            **metadata,
            "metadata": metadata,
            "content": content,
            "retrieval_text": retrieval_text,
            "doc_type": "article",
            "content_length": len(content),
            "chunk_count": int(bucket["chunk_count"]),
            "effective_date_iso": parse_effective_date_to_iso(str(metadata.get("effective_date") or "")),
        }
        documents.append(
            IndexDocument(
                point_id=stable_point_id("article", group_key),
                content=retrieval_text or content,
                payload=payload,
            )
        )
    return documents


def _batched(items: Sequence[IndexDocument], batch_size: int) -> Iterable[Sequence[IndexDocument]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _infer_vector_dim(documents: Sequence[IndexDocument], config: AppConfig) -> int:
    vector_dim = int(config.embedding.vector_dim or 0)
    if vector_dim > 0:
        return vector_dim
    if not documents:
        raise ValueError("Cannot infer embedding dimension from an empty document set.")
    embedder = get_embedder(config, logger=LOGGER)
    sample_vector = embedder.embed_query(documents[0].content or "BGE-M3 dimension probe")
    return len(sample_vector)


def resolve_collection_name(
    level: str,
    config: AppConfig,
    *,
    explicit_collection_name: str | None = None,
    version_tag: str | None = None,
) -> str:
    if explicit_collection_name:
        return explicit_collection_name
    if level == "chunk":
        return generate_versioned_collection_name(config.collections.chunk_collection_prefix, version_tag)
    if level == "article":
        return generate_versioned_collection_name(config.collections.article_collection_prefix, version_tag)
    raise ValueError(f"Unsupported index level: {level}")


def build_documents_for_level(level: str, records: Sequence[dict[str, Any]]) -> list[IndexDocument]:
    if level == "chunk":
        return prepare_chunk_documents(records)
    if level == "article":
        return prepare_article_documents(records)
    raise ValueError(f"Unsupported index level: {level}")


def activate_level_alias(level: str, manager: QdrantManager, collection_name: str) -> str:
    alias_name = (
        manager.config.collections.active_chunk_alias
        if level == "chunk"
        else manager.config.collections.active_article_alias
    )
    manager.switch_alias(alias_name, collection_name)
    return alias_name


def index_documents(
    documents: Sequence[IndexDocument],
    *,
    level: str,
    collection_name: str,
    manager: QdrantManager,
) -> dict[str, Any]:
    """Embed and upsert prepared documents into Qdrant."""

    if not documents:
        return {
            "level": level,
            "collection_name": collection_name,
            "document_count": 0,
            "upserted_points": 0,
        }

    config = manager.config
    vector_dim = _infer_vector_dim(documents, config)
    manager.create_collection_if_not_exists(
        collection_name,
        vector_dim=vector_dim,
        recreate_if_exists=config.indexing.recreate_if_exists,
    )
    manager.ensure_payload_indexes(collection_name)

    embedder = get_embedder(config, logger=LOGGER)
    point_class = manager.models.PointStruct
    batch_size = max(1, config.embedding.batch_size)
    upsert_size = max(1, config.indexing.batch_upsert_size)

    total_upserted = 0
    for batch_number, document_batch in enumerate(_batched(list(documents), batch_size), start=1):
        texts = [document.content for document in document_batch]
        vectors = embedder.embed_texts(texts)
        if len(vectors) != len(document_batch):
            raise RuntimeError(
                f"Embedding count mismatch for {collection_name}: {len(vectors)} != {len(document_batch)}"
            )
        points = [
            point_class(id=document.point_id, vector=vector, payload=document.payload)
            for document, vector in zip(document_batch, vectors)
        ]
        total_upserted += manager.upsert_points(
            collection_name,
            points,
            batch_size=upsert_size,
            max_retries=config.indexing.insert_max_retries,
            retry_backoff_seconds=config.indexing.insert_retry_backoff_seconds,
        )
        LOGGER.info(
            "Indexed batch %s into %s (%s documents)",
            batch_number,
            collection_name,
            len(document_batch),
        )

    return {
        "level": level,
        "collection_name": collection_name,
        "document_count": len(documents),
        "upserted_points": total_upserted,
    }


def run_build(
    input_path: str | Path,
    *,
    level: str,
    config_path: str | Path | None = None,
    collection_name: str | None = None,
    version_tag: str | None = None,
    activate_alias: bool = False,
    incremental: bool = False,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    """Build one or both TV2 indexes from TV1 chunk data."""

    resolved_logger = logger or LOGGER
    config = load_indexing_config(config_path or DEFAULT_CONFIG_PATH)
    manager = QdrantManager(config=config, logger=resolved_logger)
    records = load_tv1_records(input_path)
    if level == "both" and collection_name:
        raise ValueError("`--collection-name` cannot be used with `--level both`.")
    levels = [level] if level != "both" else ["chunk", "article"]
    summaries: list[dict[str, Any]] = []

    file_ids = sorted(
        {
            str(record.get("metadata", {}).get("file_id") or "")
            for record in records
            if str(record.get("metadata", {}).get("file_id") or "")
        }
    )

    for current_level in levels:
        target_collection = resolve_collection_name(
            current_level,
            config,
            explicit_collection_name=collection_name,
            version_tag=version_tag,
        )
        alias_target = manager.get_alias_target(target_collection)
        if alias_target:
            resolved_logger.info("Resolved alias %s -> %s for indexing", target_collection, alias_target)
            target_collection = alias_target
        documents = build_documents_for_level(current_level, records)
        resolved_logger.info(
            "Preparing %s documents for level=%s into collection=%s",
            len(documents),
            current_level,
            target_collection,
        )

        if (
            incremental
            and manager.collection_exists(target_collection)
            and config.indexing.delete_stale_points_on_incremental
        ):
            manager.delete_points_by_field_values(
                target_collection,
                field_name="file_id",
                values=file_ids,
            )

        summary = index_documents(
            documents,
            level=current_level,
            collection_name=target_collection,
            manager=manager,
        )
        summary["incremental"] = incremental

        if activate_alias:
            alias_name = activate_level_alias(current_level, manager, target_collection)
            summary["activated_alias"] = alias_name
            protected = {target_collection, manager.get_alias_target(alias_name) or ""}
            prefix = (
                config.collections.chunk_collection_prefix
                if current_level == "chunk"
                else config.collections.article_collection_prefix
            )
            deleted_collections = manager.delete_old_collections(
                prefix,
                keep_last_n=config.indexing.keep_last_n_versions,
                exclude_names=protected,
            )
            summary["deleted_old_collections"] = deleted_collections

        summaries.append(summary)

    return summaries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Qdrant indexes from TV1 chunk JSON/JSONL using bge-m3.",
    )
    parser.add_argument("--input", required=True, help="Path to TV1 all_chunks.json or all_chunks.jsonl")
    parser.add_argument(
        "--level",
        default="chunk",
        choices=["chunk", "article", "both"],
        help="Index granularity to build.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to configs/indexing.yaml",
    )
    parser.add_argument(
        "--collection-name",
        help="Explicit Qdrant collection name. If omitted, a versioned name is generated.",
    )
    parser.add_argument(
        "--version-tag",
        help="Override the generated collection version tag, e.g. 20260422.",
    )
    parser.add_argument(
        "--activate-alias",
        action="store_true",
        help="Switch the active alias to the newly built collection after indexing succeeds.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Delete stale points by file_id in the target collection before upserting the new subset.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)
    summaries = run_build(
        input_path=args.input,
        level=args.level,
        config_path=args.config,
        collection_name=args.collection_name,
        version_tag=args.version_tag,
        activate_alias=args.activate_alias,
        incremental=args.incremental,
        logger=LOGGER,
    )
    LOGGER.info("TV2 indexing finished")
    for summary in summaries:
        LOGGER.info(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

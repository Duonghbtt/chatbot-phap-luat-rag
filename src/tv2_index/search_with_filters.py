from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from src.tv2_index.embedding_registry import (
    AppConfig,
    DEFAULT_CONFIG_PATH,
    get_embedder,
    load_indexing_config,
)
from src.tv2_index.qdrant_manager import QdrantManager
from src.tv3_retrieval.fallback_policy import RetrievalConfig, load_retrieval_config

LOGGER = logging.getLogger(__name__)
SPACE_PATTERN = re.compile(r"\s+")


@dataclass(slots=True)
class SearchFilters:
    """Supported metadata filters for legal search."""

    law_id: str = ""
    topic_id: str = ""
    title: str = ""
    issuer: str = ""
    article: str = ""
    effective_date: str = ""
    effective_date_from: str = ""
    effective_date_to: str = ""
    de_muc: str = ""
    article_code: str = ""
    article_name: str = ""


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def parse_effective_date_to_iso(date_text: str) -> str:
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


def normalize_filters(filters: SearchFilters | Mapping[str, Any] | None) -> dict[str, Any]:
    if filters is None:
        return {}
    if isinstance(filters, SearchFilters):
        payload = asdict(filters)
    else:
        payload = dict(filters)
    return {key: value for key, value in payload.items() if value not in (None, "", [], {})}


def _normalize_match_text(text: str) -> str:
    return SPACE_PATTERN.sub(" ", (text or "").strip().lower())


def _matches_exact(metadata: Mapping[str, Any], field: str, expected: str) -> bool:
    actual = _normalize_match_text(str(metadata.get(field) or ""))
    target = _normalize_match_text(expected)
    if not actual or not target:
        return False
    return actual == target


def _compute_exact_legal_signals(
    metadata: Mapping[str, Any],
    filters: Mapping[str, Any],
    config: RetrievalConfig,
) -> dict[str, Any]:
    exact_hit_fields: list[str] = []
    matched_filters: dict[str, str] = {}
    ranking_components: dict[str, float] = {}
    bonus = 0.0

    field_bonus_pairs = (
        ("article_code", float(config.article_code_match_bonus)),
        ("article", float(config.article_match_bonus)),
        ("law_id", float(config.law_id_match_bonus)),
        ("title", float(config.title_match_bonus)),
        ("article_name", float(config.article_name_match_bonus)),
    )
    for field, field_bonus in field_bonus_pairs:
        expected = str(filters.get(field) or "").strip()
        if not expected or not _matches_exact(metadata, field, expected):
            continue
        exact_hit_fields.append(field)
        matched_filters[field] = expected
        ranking_components[f"{field}_match_bonus"] = field_bonus
        bonus += field_bonus

    structured_pair_bonus = 0.0
    if "article_code" in exact_hit_fields:
        structured_pair_bonus = float(config.structured_match_bonus)
    elif "article" in exact_hit_fields and ("title" in exact_hit_fields or "law_id" in exact_hit_fields):
        structured_pair_bonus = float(config.structured_match_bonus)

    if structured_pair_bonus > 0:
        ranking_components["structured_match_bonus"] = structured_pair_bonus
        bonus += structured_pair_bonus

    return {
        "bonus": round(bonus, 6),
        "matched_filters": matched_filters,
        "exact_hit_fields": exact_hit_fields,
        "ranking_components": ranking_components,
    }


def _metadata_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("metadata"), dict):
        metadata = dict(payload["metadata"])
    else:
        metadata_fields = [
            "de_muc",
            "file_id",
            "article_code",
            "article_name",
            "mapc",
            "law_id",
            "topic_id",
            "title",
            "issuer",
            "article",
            "effective_date",
            "source_note",
            "related_articles",
        ]
        metadata = {field: payload.get(field, "") for field in metadata_fields if field in payload}
    related_articles = metadata.get("related_articles") or []
    if not isinstance(related_articles, list):
        related_articles = [str(related_articles)]
    metadata["related_articles"] = related_articles
    return metadata


def result_matches_filters(result: Mapping[str, Any], filters: SearchFilters | Mapping[str, Any] | None) -> bool:
    """Check whether a retrieved result satisfies the requested metadata filters."""

    normalized_filters = normalize_filters(filters)
    if not normalized_filters:
        return True

    metadata = dict(result.get("metadata") or {})
    for field in (
        "law_id",
        "topic_id",
        "title",
        "issuer",
        "article",
        "effective_date",
        "de_muc",
        "article_code",
        "article_name",
    ):
        expected = normalized_filters.get(field)
        if expected and _normalize_match_text(str(metadata.get(field) or "")) != _normalize_match_text(str(expected)):
            return False

    date_from = normalized_filters.get("effective_date_from")
    date_to = normalized_filters.get("effective_date_to")
    if date_from or date_to:
        result_iso = parse_effective_date_to_iso(str(metadata.get("effective_date") or ""))
        if date_from:
            from_iso = parse_effective_date_to_iso(str(date_from))
            if result_iso and from_iso and result_iso < from_iso:
                return False
        if date_to:
            to_iso = parse_effective_date_to_iso(str(date_to))
            if result_iso and to_iso and result_iso > to_iso:
                return False
    return True


class QdrantSearchService:
    """Semantic search service used by TV3 retrieve nodes."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        config_path: str | Path | None = None,
        manager: QdrantManager | None = None,
        retrieval_config: RetrievalConfig | None = None,
        retrieval_config_path: str | Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or load_indexing_config(config_path or DEFAULT_CONFIG_PATH)
        self.logger = logger or LOGGER
        self.manager = manager or QdrantManager(config=self.config, logger=self.logger)
        self.embedder = get_embedder(self.config, logger=self.logger)
        self.retrieval_config = retrieval_config or load_retrieval_config(retrieval_config_path)

    def _resolve_collection_name(self, level: str, explicit_collection_name: str | None = None) -> str:
        if explicit_collection_name:
            return explicit_collection_name
        if level == "chunk":
            return self.config.collections.active_chunk_alias
        if level == "article":
            return self.config.collections.active_article_alias
        raise ValueError(f"Unsupported level: {level}")

    def build_qdrant_filter(self, filters: SearchFilters | Mapping[str, Any] | None) -> Any | None:
        """Convert application filters to a Qdrant filter object."""

        normalized = normalize_filters(filters)
        if not normalized:
            return None

        models = self.manager.models
        must_conditions: list[Any] = []
        for field in (
            "law_id",
            "topic_id",
            "title",
            "issuer",
            "article",
            "effective_date",
            "de_muc",
            "article_code",
            "article_name",
        ):
            value = normalized.get(field)
            if value:
                must_conditions.append(
                    models.FieldCondition(
                        key=field,
                        match=models.MatchValue(value=value),
                    )
                )

        date_from = normalized.get("effective_date_from")
        date_to = normalized.get("effective_date_to")
        if date_from or date_to:
            must_conditions.append(
                models.FieldCondition(
                    key="effective_date_iso",
                    range=models.DatetimeRange(
                        gte=parse_effective_date_to_iso(str(date_from)) or None,
                        lte=parse_effective_date_to_iso(str(date_to)) or None,
                    ),
                )
            )

        if not must_conditions:
            return None
        return models.Filter(must=must_conditions)

    def _search(
        self,
        *,
        level: str,
        query_text: str,
        filters: SearchFilters | Mapping[str, Any] | None = None,
        top_k: int | None = None,
        collection_name: str | None = None,
    ) -> list[dict[str, Any]]:
        resolved_collection = self._resolve_collection_name(level, collection_name)
        resolved_top_k = int(top_k or self.config.indexing.top_k_default)
        fetch_limit = max(resolved_top_k * 4, resolved_top_k + 6)
        vector = self.embedder.embed_query(query_text)
        normalized_filters = normalize_filters(filters)
        qdrant_filter = self.build_qdrant_filter(normalized_filters)
        response = self.manager.client.query_points(
            collection_name=resolved_collection,
            query=vector,
            query_filter=qdrant_filter,
            limit=fetch_limit,
            with_payload=True,
        )
        raw_points = getattr(response, "points", response)
        results: list[dict[str, Any]] = []
        for point in raw_points or []:
            payload = getattr(point, "payload", {}) or {}
            metadata = _metadata_from_payload(payload)
            semantic_score = float(getattr(point, "score", 0.0) or 0.0)
            exact_signals = _compute_exact_legal_signals(metadata, normalized_filters, self.retrieval_config)
            results.append(
                {
                    "point_id": str(getattr(point, "id", "") or ""),
                    "content": str(payload.get("content") or ""),
                    "score": semantic_score + exact_signals["bonus"],
                    "semantic_score": semantic_score,
                    "exact_match_bonus": exact_signals["bonus"],
                    "metadata": metadata,
                    "matched_filters": exact_signals["matched_filters"],
                    "exact_hit_fields": exact_signals["exact_hit_fields"],
                    "ranking_components": {
                        "semantic_score": semantic_score,
                        **exact_signals["ranking_components"],
                    },
                }
            )
        results.sort(
            key=lambda item: (
                float(item.get("score") or 0.0),
                len(item.get("exact_hit_fields") or []),
                float(item.get("semantic_score") or 0.0),
            ),
            reverse=True,
        )
        return results[:resolved_top_k]

    def search_chunk_level(
        self,
        query_text: str,
        *,
        filters: SearchFilters | Mapping[str, Any] | None = None,
        top_k: int | None = None,
        collection_name: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._search(
            level="chunk",
            query_text=query_text,
            filters=filters,
            top_k=top_k,
            collection_name=collection_name,
        )

    def search_article_level(
        self,
        query_text: str,
        *,
        filters: SearchFilters | Mapping[str, Any] | None = None,
        top_k: int | None = None,
        collection_name: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._search(
            level="article",
            query_text=query_text,
            filters=filters,
            top_k=top_k,
            collection_name=collection_name,
        )

    def search_with_filter(
        self,
        query_text: str,
        *,
        filters: SearchFilters | Mapping[str, Any] | None = None,
        top_k: int | None = None,
        collection_name: str,
    ) -> list[dict[str, Any]]:
        return self._search(
            level="chunk",
            query_text=query_text,
            filters=filters,
            top_k=top_k,
            collection_name=collection_name,
        )


def search_chunk_level(
    query_text: str,
    *,
    filters: SearchFilters | Mapping[str, Any] | None = None,
    top_k: int | None = None,
    collection_name: str | None = None,
    config_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    service = QdrantSearchService(config_path=config_path)
    return service.search_chunk_level(
        query_text,
        filters=filters,
        top_k=top_k,
        collection_name=collection_name,
    )


def search_article_level(
    query_text: str,
    *,
    filters: SearchFilters | Mapping[str, Any] | None = None,
    top_k: int | None = None,
    collection_name: str | None = None,
    config_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    service = QdrantSearchService(config_path=config_path)
    return service.search_article_level(
        query_text,
        filters=filters,
        top_k=top_k,
        collection_name=collection_name,
    )


def search_with_filter(
    query_text: str,
    *,
    filters: SearchFilters | Mapping[str, Any] | None = None,
    top_k: int | None = None,
    collection_name: str,
    config_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    service = QdrantSearchService(config_path=config_path)
    return service.search_with_filter(
        query_text,
        filters=filters,
        top_k=top_k,
        collection_name=collection_name,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search Qdrant indexes with metadata filters.")
    parser.add_argument("--query", required=True, help="User query for semantic retrieval.")
    parser.add_argument(
        "--level",
        default="chunk",
        choices=["chunk", "article"],
        help="Collection granularity to search.",
    )
    parser.add_argument("--collection", help="Explicit collection name or alias.")
    parser.add_argument("--law-id", help="Exact filter for law_id.")
    parser.add_argument("--topic-id", help="Exact filter for topic_id.")
    parser.add_argument("--issuer", help="Exact filter for issuer.")
    parser.add_argument("--de-muc", help="Exact filter for de_muc.")
    parser.add_argument("--title", help="Exact filter for title.")
    parser.add_argument("--article", help="Exact filter for article.")
    parser.add_argument("--article-code", help="Exact filter for article_code.")
    parser.add_argument("--article-name", help="Exact filter for article_name.")
    parser.add_argument("--effective-date", help="Exact filter for effective_date (dd/mm/yyyy).")
    parser.add_argument("--effective-date-from", help="Lower bound for effective_date.")
    parser.add_argument("--effective-date-to", help="Upper bound for effective_date.")
    parser.add_argument("--top-k", type=int, help="Number of results to return.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to configs/indexing.yaml")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)
    service = QdrantSearchService(config_path=args.config, logger=LOGGER)
    filters = SearchFilters(
        law_id=args.law_id or "",
        topic_id=args.topic_id or "",
        title=args.title or "",
        issuer=args.issuer or "",
        article=args.article or "",
        effective_date=args.effective_date or "",
        effective_date_from=args.effective_date_from or "",
        effective_date_to=args.effective_date_to or "",
        de_muc=args.de_muc or "",
        article_code=args.article_code or "",
        article_name=args.article_name or "",
    )
    if args.level == "chunk":
        results = service.search_chunk_level(
            args.query,
            filters=filters,
            top_k=args.top_k,
            collection_name=args.collection,
        )
    else:
        results = service.search_article_level(
            args.query,
            filters=filters,
            top_k=args.top_k,
            collection_name=args.collection,
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

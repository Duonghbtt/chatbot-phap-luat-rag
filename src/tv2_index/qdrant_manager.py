from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.tv2_index.embedding_registry import (
    AppConfig,
    DEFAULT_CONFIG_PATH,
    MissingDependencyError,
    load_indexing_config,
)

LOGGER = logging.getLogger(__name__)


def _load_qdrant_modules() -> tuple[Any, Any]:
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise MissingDependencyError(
            "qdrant-client is required for TV2 indexing/search operations. Install `qdrant-client`."
        ) from exc
    return QdrantClient, models


class QdrantManager:
    """Manage versioned Qdrant collections, payload indexes, and aliases."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        config_path: str | Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or load_indexing_config(config_path or DEFAULT_CONFIG_PATH)
        self.logger = logger or LOGGER
        qdrant_client_cls, models = _load_qdrant_modules()
        self.models = models
        self.client = qdrant_client_cls(
            url=self.config.qdrant.url,
            api_key=self.config.qdrant.api_key or None,
            prefer_grpc=self.config.qdrant.prefer_grpc,
        )

    def _distance_enum(self, distance_name: str) -> Any:
        normalized = distance_name.strip().upper()
        mapping = {
            "COSINE": self.models.Distance.COSINE,
            "DOT": self.models.Distance.DOT,
            "EUCLID": self.models.Distance.EUCLID,
            "MANHATTAN": self.models.Distance.MANHATTAN,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported Qdrant distance metric: {distance_name}")
        return mapping[normalized]

    def _payload_schema(self, schema_name: str) -> Any:
        normalized = schema_name.strip().upper()
        mapping = {
            "KEYWORD": self.models.PayloadSchemaType.KEYWORD,
            "INTEGER": self.models.PayloadSchemaType.INTEGER,
            "FLOAT": self.models.PayloadSchemaType.FLOAT,
            "BOOL": self.models.PayloadSchemaType.BOOL,
            "DATETIME": self.models.PayloadSchemaType.DATETIME,
            "TEXT": self.models.PayloadSchemaType.TEXT,
            "UUID": self.models.PayloadSchemaType.UUID,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported payload schema: {schema_name}")
        return mapping[normalized]

    def list_collection_names(self) -> list[str]:
        response = self.client.get_collections()
        collections = getattr(response, "collections", None)
        if collections is None and isinstance(response, dict):
            collections = response.get("collections", [])
        names: list[str] = []
        for item in collections or []:
            if isinstance(item, dict):
                name = item.get("name")
            else:
                name = getattr(item, "name", None)
            if name:
                names.append(str(name))
        return sorted(names)

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in set(self.list_collection_names())

    def create_collection_if_not_exists(
        self,
        collection_name: str,
        *,
        vector_dim: int,
        recreate_if_exists: bool = False,
    ) -> None:
        """Create a collection with the configured HNSW and distance settings."""

        if self.collection_exists(collection_name):
            if recreate_if_exists:
                self.logger.info("Recreating existing collection %s", collection_name)
                self.recreate_collection(collection_name, vector_dim=vector_dim)
            else:
                self.logger.info("Collection %s already exists; reusing it.", collection_name)
            return

        vectors_config = self.models.VectorParams(
            size=vector_dim,
            distance=self._distance_enum(self.config.qdrant.distance),
        )
        kwargs: dict[str, Any] = {
            "collection_name": collection_name,
            "vectors_config": vectors_config,
        }
        if self.config.qdrant.hnsw_config:
            kwargs["hnsw_config"] = self.models.HnswConfigDiff(**self.config.qdrant.hnsw_config)

        self.client.create_collection(**kwargs)
        self.logger.info("Created collection %s (dim=%s)", collection_name, vector_dim)

    def recreate_collection(self, collection_name: str, *, vector_dim: int) -> None:
        """Delete and recreate a collection with the configured schema."""

        if self.collection_exists(collection_name):
            self.client.delete_collection(collection_name=collection_name)
        self.create_collection_if_not_exists(
            collection_name,
            vector_dim=vector_dim,
            recreate_if_exists=False,
        )

    def ensure_payload_indexes(self, collection_name: str) -> None:
        """Create payload indexes for TV2 metadata filters."""

        for field_name, schema_name in self.config.payload_fields.items():
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=self._payload_schema(schema_name),
                    wait=True,
                )
                self.logger.debug(
                    "Ensured payload index collection=%s field=%s schema=%s",
                    collection_name,
                    field_name,
                    schema_name,
                )
            except Exception as exc:  # pragma: no cover - depends on Qdrant runtime.
                self.logger.warning(
                    "Skipping payload index for %s.%s (%s)",
                    collection_name,
                    field_name,
                    exc,
                )

    def create_alias(self, alias_name: str, collection_name: str) -> None:
        """Create a Qdrant alias that points at a collection."""

        actions = [
            self.models.CreateAliasOperation(
                create_alias=self.models.CreateAlias(
                    collection_name=collection_name,
                    alias_name=alias_name,
                )
            )
        ]
        self.client.update_collection_aliases(change_aliases_operations=actions)
        self.logger.info("Created alias %s -> %s", alias_name, collection_name)

    def get_alias_target(self, alias_name: str) -> str | None:
        """Resolve the current collection behind an alias, if any."""

        response = self.client.get_aliases()
        aliases = getattr(response, "aliases", None)
        if aliases is None and isinstance(response, dict):
            aliases = response.get("aliases", [])

        for item in aliases or []:
            if isinstance(item, dict):
                candidate_alias = item.get("alias_name")
                collection_name = item.get("collection_name")
            else:
                candidate_alias = getattr(item, "alias_name", None)
                collection_name = getattr(item, "collection_name", None)
            if candidate_alias == alias_name:
                return str(collection_name)
        return None

    def switch_alias(self, alias_name: str, target_collection: str) -> str | None:
        """Atomically switch an alias to a new collection."""

        current_collection = self.get_alias_target(alias_name)
        if current_collection == target_collection:
            self.logger.info("Alias %s already points to %s", alias_name, target_collection)
            return current_collection

        actions: list[Any] = []
        if current_collection:
            actions.append(
                self.models.DeleteAliasOperation(
                    delete_alias=self.models.DeleteAlias(alias_name=alias_name)
                )
            )
        actions.append(
            self.models.CreateAliasOperation(
                create_alias=self.models.CreateAlias(
                    collection_name=target_collection,
                    alias_name=alias_name,
                )
            )
        )
        self.client.update_collection_aliases(change_aliases_operations=actions)
        self.logger.info(
            "Switched alias %s from %s to %s",
            alias_name,
            current_collection or "<none>",
            target_collection,
        )
        return current_collection

    def delete_old_collections(
        self,
        prefix: str,
        *,
        keep_last_n: int,
        exclude_names: Iterable[str] | None = None,
    ) -> list[str]:
        """Delete older versioned collections while keeping the newest N."""

        exclude = {name for name in (exclude_names or []) if name}
        candidates = [name for name in self.list_collection_names() if name.startswith(prefix)]
        candidates.sort(reverse=True)

        deleted: list[str] = []
        for collection_name in candidates[keep_last_n:]:
            if collection_name in exclude:
                continue
            self.client.delete_collection(collection_name=collection_name)
            deleted.append(collection_name)
            self.logger.info("Deleted old collection %s", collection_name)
        return deleted

    def get_collection_info(self, collection_name: str) -> Any:
        """Fetch raw Qdrant collection information."""

        return self.client.get_collection(collection_name=collection_name)

    def upsert_points(
        self,
        collection_name: str,
        points: Sequence[Any],
        *,
        batch_size: int,
        max_retries: int,
        retry_backoff_seconds: float,
    ) -> int:
        """Upsert points in batches with basic retry handling."""

        total = 0
        for start in range(0, len(points), batch_size):
            batch = list(points[start : start + batch_size])
            if not batch:
                continue
            for attempt in range(1, max_retries + 1):
                try:
                    self.client.upsert(
                        collection_name=collection_name,
                        points=batch,
                        wait=True,
                    )
                    total += len(batch)
                    break
                except Exception as exc:  # pragma: no cover - depends on Qdrant runtime.
                    if attempt >= max_retries:
                        raise RuntimeError(
                            f"Failed to upsert batch into {collection_name} after {max_retries} attempts"
                        ) from exc
                    sleep_for = retry_backoff_seconds * attempt
                    self.logger.warning(
                        "Upsert attempt %s/%s failed for %s (%s). Retrying in %.1fs.",
                        attempt,
                        max_retries,
                        collection_name,
                        exc,
                        sleep_for,
                    )
                    time.sleep(sleep_for)
        return total

    def delete_points_by_field_values(
        self,
        collection_name: str,
        *,
        field_name: str,
        values: Sequence[str],
    ) -> int:
        """Delete existing points for a set of payload values, used for incremental refresh."""

        deleted = 0
        for value in values:
            if not value:
                continue
            filter_selector = self.models.FilterSelector(
                filter=self.models.Filter(
                    must=[
                        self.models.FieldCondition(
                            key=field_name,
                            match=self.models.MatchValue(value=value),
                        )
                    ]
                )
            )
            self.client.delete(
                collection_name=collection_name,
                points_selector=filter_selector,
                wait=True,
            )
            deleted += 1
        if deleted:
            self.logger.info(
                "Deleted stale points in %s for %s distinct %s values",
                collection_name,
                deleted,
                field_name,
            )
        return deleted


__all__ = ["QdrantManager"]

from __future__ import annotations

import json
import logging
import math
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
from urllib import error, request

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path("configs/indexing.yaml")
ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::([^}]*))?\}")


class MissingDependencyError(RuntimeError):
    """Raised when an optional runtime dependency is required but missing."""


@dataclass(slots=True, frozen=True)
class EmbeddingConfig:
    """Embedding-specific settings for TV2."""

    provider: str = "ollama"
    fallback_provider: str = "sentence_transformers"
    model_name: str = "bge-m3"
    sentence_transformers_model_name: str = "BAAI/bge-m3"
    vector_dim: int = 1024
    batch_size: int = 16
    normalize_embeddings: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_seconds: int = 120


@dataclass(slots=True, frozen=True)
class QdrantConfig:
    """Qdrant connection and indexing settings."""

    url: str = "http://localhost:6333"
    api_key: str = ""
    prefer_grpc: bool = False
    distance: str = "cosine"
    hnsw_config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class CollectionsConfig:
    """Collection naming conventions for TV2."""

    chunk_collection_prefix: str = "legal_chunks"
    article_collection_prefix: str = "legal_articles"
    active_chunk_alias: str = "legal_chunks_active"
    active_article_alias: str = "legal_articles_active"


@dataclass(slots=True, frozen=True)
class IndexingRuntimeConfig:
    """Runtime indexing knobs."""

    level: str = "chunk"
    top_k_default: int = 5
    batch_upsert_size: int = 64
    recreate_if_exists: bool = False
    keep_last_n_versions: int = 3
    insert_max_retries: int = 3
    insert_retry_backoff_seconds: float = 2.0
    delete_stale_points_on_incremental: bool = True


@dataclass(slots=True, frozen=True)
class AppConfig:
    """Combined config loaded from `configs/indexing.yaml`."""

    qdrant: QdrantConfig
    embedding: EmbeddingConfig
    collections: CollectionsConfig
    indexing: IndexingRuntimeConfig
    payload_fields: dict[str, str]
    config_path: Path


class BaseEmbedder(ABC):
    """Unified embedding interface for TV2."""

    def __init__(self, config: EmbeddingConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or LOGGER
        self._vector_dim = max(int(config.vector_dim or 0), 0)

    @property
    def vector_dim(self) -> int:
        return self._vector_dim

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of document texts."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single retrieval query."""

    def _set_vector_dim_from_embeddings(self, embeddings: Sequence[Sequence[float]]) -> None:
        if self._vector_dim or not embeddings:
            return
        first = embeddings[0]
        if first:
            self._vector_dim = len(first)


class OllamaBgeM3Embedder(BaseEmbedder):
    """Embedder that calls a local Ollama `/api/embed` endpoint."""

    def __init__(self, config: EmbeddingConfig, logger: logging.Logger | None = None) -> None:
        super().__init__(config=config, logger=logger)
        self.base_url = config.ollama_base_url.rstrip("/")
        self.timeout_seconds = int(config.ollama_timeout_seconds)
        self.logger.info(
            "Using Ollama embedder model=%s base_url=%s",
            self.config.model_name,
            self.base_url,
        )

    def _post_embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = json.dumps(
            {
                "model": self.config.model_name,
                "input": list(texts),
            }
        ).encode("utf-8")
        endpoint = f"{self.base_url}/api/embed"
        http_request = request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama embed request failed with status {exc.code}: {details}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Unable to reach Ollama at {endpoint}. Check that the local server is running."
            ) from exc

        data = json.loads(body)
        embeddings = data.get("embeddings") or []
        if not isinstance(embeddings, list):
            raise RuntimeError("Ollama response did not contain a valid `embeddings` array.")

        normalized = _normalize_embeddings(embeddings, self.config.normalize_embeddings)
        self._set_vector_dim_from_embeddings(normalized)
        return normalized

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return self._post_embed(texts)

    def embed_query(self, text: str) -> list[float]:
        embeddings = self._post_embed([text])
        if not embeddings:
            raise RuntimeError("Ollama returned no embedding for the query.")
        return embeddings[0]


class SentenceTransformersBgeM3Embedder(BaseEmbedder):
    """Embedder that runs `BAAI/bge-m3` locally via sentence-transformers."""

    def __init__(self, config: EmbeddingConfig, logger: logging.Logger | None = None) -> None:
        super().__init__(config=config, logger=logger)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency.
            raise MissingDependencyError(
                "sentence-transformers is required for the local fallback embedder. "
                "Install `sentence-transformers` to use provider=sentence_transformers."
            ) from exc

        model_name = config.sentence_transformers_model_name or config.model_name
        self.logger.info("Using sentence-transformers embedder model=%s", model_name)
        self.model = SentenceTransformer(model_name)
        dimension = getattr(self.model, "get_sentence_embedding_dimension", lambda: None)()
        if dimension and not self._vector_dim:
            self._vector_dim = int(dimension)

    def _encode(self, texts: Sequence[str], *, is_query: bool) -> list[list[float]]:
        if not texts:
            return []

        encode_kwargs = {
            "batch_size": self.config.batch_size,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": self.config.normalize_embeddings,
        }

        if is_query and hasattr(self.model, "encode_query"):
            embeddings = self.model.encode_query(list(texts), **encode_kwargs)
        elif not is_query and hasattr(self.model, "encode_document"):
            embeddings = self.model.encode_document(list(texts), **encode_kwargs)
        else:
            embeddings = self.model.encode(list(texts), **encode_kwargs)

        if hasattr(embeddings, "tolist"):
            vectors = embeddings.tolist()
        else:
            vectors = [list(vector) for vector in embeddings]

        normalized = _normalize_embeddings(vectors, self.config.normalize_embeddings)
        self._set_vector_dim_from_embeddings(normalized)
        return normalized

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts, is_query=False)

    def embed_query(self, text: str) -> list[float]:
        embeddings = self._encode([text], is_query=True)
        if not embeddings:
            raise RuntimeError("sentence-transformers returned no embedding for the query.")
        return embeddings[0]


def _load_yaml_module() -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise MissingDependencyError(
            "PyYAML is required to load configs/indexing.yaml. Install `PyYAML`."
        ) from exc
    return yaml


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _substitute_env_placeholders(raw_text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(key, default)

    return ENV_PATTERN.sub(replace, raw_text)


def _normalize_embeddings(vectors: Sequence[Sequence[float]], should_normalize: bool) -> list[list[float]]:
    normalized_vectors: list[list[float]] = []
    for vector in vectors:
        cast_vector = [float(value) for value in vector]
        if should_normalize:
            norm = math.sqrt(sum(value * value for value in cast_vector))
            if norm > 0:
                cast_vector = [value / norm for value in cast_vector]
        normalized_vectors.append(cast_vector)
    return normalized_vectors


def load_indexing_config(config_path: str | Path | None = None) -> AppConfig:
    """Load TV2 indexing config from YAML with optional env substitution."""

    yaml = _load_yaml_module()
    resolved_path = Path(config_path or DEFAULT_CONFIG_PATH).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    raw_text = resolved_path.read_text(encoding="utf-8")
    config_data = yaml.safe_load(_substitute_env_placeholders(raw_text)) or {}
    if not isinstance(config_data, dict):
        raise ValueError(f"Invalid config structure in {resolved_path}")

    qdrant_data = dict(config_data.get("qdrant") or {})
    embedding_data = dict(config_data.get("embedding") or {})
    collections_data = dict(config_data.get("collections") or {})
    indexing_data = dict(config_data.get("indexing") or {})
    payload_fields = dict(config_data.get("payload_fields") or {})

    return AppConfig(
        qdrant=QdrantConfig(
            url=str(qdrant_data.get("url") or "http://localhost:6333"),
            api_key=str(qdrant_data.get("api_key") or ""),
            prefer_grpc=_coerce_bool(qdrant_data.get("prefer_grpc", False)),
            distance=str(qdrant_data.get("distance") or "cosine"),
            hnsw_config=dict(qdrant_data.get("hnsw_config") or {}),
        ),
        embedding=EmbeddingConfig(
            provider=str(embedding_data.get("provider") or "ollama"),
            fallback_provider=str(embedding_data.get("fallback_provider") or "sentence_transformers"),
            model_name=str(embedding_data.get("model_name") or "bge-m3"),
            sentence_transformers_model_name=str(
                embedding_data.get("sentence_transformers_model_name") or "BAAI/bge-m3"
            ),
            vector_dim=int(embedding_data.get("vector_dim") or 1024),
            batch_size=int(embedding_data.get("batch_size") or 16),
            normalize_embeddings=_coerce_bool(embedding_data.get("normalize_embeddings", True)),
            ollama_base_url=str(embedding_data.get("ollama_base_url") or "http://localhost:11434"),
            ollama_timeout_seconds=int(embedding_data.get("ollama_timeout_seconds") or 120),
        ),
        collections=CollectionsConfig(
            chunk_collection_prefix=str(collections_data.get("chunk_collection_prefix") or "legal_chunks"),
            article_collection_prefix=str(
                collections_data.get("article_collection_prefix") or "legal_articles"
            ),
            active_chunk_alias=str(collections_data.get("active_chunk_alias") or "legal_chunks_active"),
            active_article_alias=str(
                collections_data.get("active_article_alias") or "legal_articles_active"
            ),
        ),
        indexing=IndexingRuntimeConfig(
            level=str(indexing_data.get("level") or "chunk"),
            top_k_default=int(indexing_data.get("top_k_default") or 5),
            batch_upsert_size=int(indexing_data.get("batch_upsert_size") or 64),
            recreate_if_exists=_coerce_bool(indexing_data.get("recreate_if_exists", False)),
            keep_last_n_versions=int(indexing_data.get("keep_last_n_versions") or 3),
            insert_max_retries=int(indexing_data.get("insert_max_retries") or 3),
            insert_retry_backoff_seconds=float(
                indexing_data.get("insert_retry_backoff_seconds") or 2.0
            ),
            delete_stale_points_on_incremental=_coerce_bool(
                indexing_data.get("delete_stale_points_on_incremental", True)
            ),
        ),
        payload_fields={str(key): str(value) for key, value in payload_fields.items()},
        config_path=resolved_path,
    )


def get_embedder(
    config: AppConfig | EmbeddingConfig | None = None,
    *,
    config_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> BaseEmbedder:
    """Instantiate the configured TV2 embedder."""

    resolved_logger = logger or LOGGER
    if config is None:
        resolved_config = load_indexing_config(config_path)
        embedding_config = resolved_config.embedding
    elif isinstance(config, AppConfig):
        embedding_config = config.embedding
    else:
        embedding_config = config

    provider = embedding_config.provider.strip().lower().replace("-", "_")
    if provider == "ollama":
        return OllamaBgeM3Embedder(config=embedding_config, logger=resolved_logger)
    if provider in {"sentence_transformers", "sentence_transformer"}:
        return SentenceTransformersBgeM3Embedder(config=embedding_config, logger=resolved_logger)

    fallback_provider = embedding_config.fallback_provider.strip().lower().replace("-", "_")
    if fallback_provider == provider:
        raise ValueError(f"Unsupported embedding provider: {embedding_config.provider}")
    resolved_logger.warning(
        "Unknown embedding provider '%s'. Falling back to '%s'.",
        embedding_config.provider,
        embedding_config.fallback_provider,
    )
    fallback_config = EmbeddingConfig(
        provider=fallback_provider,
        fallback_provider=embedding_config.fallback_provider,
        model_name=embedding_config.model_name,
        sentence_transformers_model_name=embedding_config.sentence_transformers_model_name,
        vector_dim=embedding_config.vector_dim,
        batch_size=embedding_config.batch_size,
        normalize_embeddings=embedding_config.normalize_embeddings,
        ollama_base_url=embedding_config.ollama_base_url,
        ollama_timeout_seconds=embedding_config.ollama_timeout_seconds,
    )
    return get_embedder(fallback_config, logger=resolved_logger)


def embed_texts(
    texts: Sequence[str],
    config: AppConfig | EmbeddingConfig | None = None,
    *,
    config_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> list[list[float]]:
    """Convenience wrapper for embedding batches."""

    return get_embedder(config, config_path=config_path, logger=logger).embed_texts(texts)


def embed_query(
    text: str,
    config: AppConfig | EmbeddingConfig | None = None,
    *,
    config_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> list[float]:
    """Convenience wrapper for embedding a single query."""

    return get_embedder(config, config_path=config_path, logger=logger).embed_query(text)


__all__ = [
    "AppConfig",
    "BaseEmbedder",
    "CollectionsConfig",
    "DEFAULT_CONFIG_PATH",
    "EmbeddingConfig",
    "IndexingRuntimeConfig",
    "MissingDependencyError",
    "OllamaBgeM3Embedder",
    "QdrantConfig",
    "SentenceTransformersBgeM3Embedder",
    "embed_query",
    "embed_texts",
    "get_embedder",
    "load_indexing_config",
]

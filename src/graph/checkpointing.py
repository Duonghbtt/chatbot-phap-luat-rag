from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from src.graph.state import AgentState, clone_state, utc_now_iso

LOGGER = logging.getLogger(__name__)


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when a stored graph checkpoint cannot be found."""


@dataclass(slots=True, frozen=True)
class CheckpointRecord:
    """Serializable representation of one persisted graph checkpoint."""

    checkpoint_id: str
    session_id: str
    thread_id: str
    updated_at: str
    state: dict[str, Any]


class BaseCheckpointStore(ABC):
    """Abstraction for graph checkpoint persistence backends."""

    @abstractmethod
    def save_state(self, state: Mapping[str, Any]) -> str:
        """Persist the latest state and return the checkpoint id."""

    @abstractmethod
    def load_state(self, *, thread_id: str, session_id: str | None = None) -> AgentState:
        """Load a previously persisted state for a given thread."""

    @abstractmethod
    def delete_state(self, *, thread_id: str, session_id: str | None = None) -> None:
        """Delete a stored checkpoint."""

    @abstractmethod
    def exists(self, *, thread_id: str, session_id: str | None = None) -> bool:
        """Return whether a checkpoint exists for the given thread."""


class LocalJSONCheckpointStore(BaseCheckpointStore):
    """Local filesystem JSON checkpoint backend for Windows/dev mode."""

    def __init__(self, base_dir: str | Path = ".checkpoints", logger: logging.Logger | None = None) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.logger = logger or LOGGER
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, *, thread_id: str, session_id: str | None = None) -> Path:
        safe_session = (session_id or "default-session").strip() or "default-session"
        safe_thread = thread_id.strip()
        return self.base_dir / safe_session / f"{safe_thread}.json"

    def save_state(self, state: Mapping[str, Any]) -> str:
        resolved_state = clone_state(state)
        checkpoint_id = str(resolved_state.get("app_checkpoint_id") or f"ckpt-{uuid.uuid4()}")
        resolved_state["app_checkpoint_id"] = checkpoint_id
        record = CheckpointRecord(
            checkpoint_id=checkpoint_id,
            session_id=str(resolved_state.get("session_id") or "default-session"),
            thread_id=str(resolved_state.get("thread_id") or ""),
            updated_at=utc_now_iso(),
            state=dict(resolved_state),
        )
        target_path = self._file_path(thread_id=record.thread_id, session_id=record.session_id)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            json.dumps(asdict(record), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.logger.info("Saved checkpoint session=%s thread=%s file=%s", record.session_id, record.thread_id, target_path)
        return checkpoint_id

    def load_state(self, *, thread_id: str, session_id: str | None = None) -> AgentState:
        target_path = self._file_path(thread_id=thread_id, session_id=session_id)
        if not target_path.exists():
            raise CheckpointNotFoundError(f"Checkpoint not found for thread_id={thread_id} session_id={session_id or ''}")
        payload = json.loads(target_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Invalid checkpoint structure: {target_path}")
        state = payload.get("state") or {}
        self.logger.info("Loaded checkpoint session=%s thread=%s file=%s", session_id or "", thread_id, target_path)
        return clone_state(state)

    def delete_state(self, *, thread_id: str, session_id: str | None = None) -> None:
        target_path = self._file_path(thread_id=thread_id, session_id=session_id)
        if target_path.exists():
            target_path.unlink()
            self.logger.info("Deleted checkpoint session=%s thread=%s", session_id or "", thread_id)

    def exists(self, *, thread_id: str, session_id: str | None = None) -> bool:
        return self._file_path(thread_id=thread_id, session_id=session_id).exists()


def create_checkpoint_store(
    backend: str = "local_json",
    *,
    base_dir: str | Path = ".checkpoints",
    logger: logging.Logger | None = None,
) -> BaseCheckpointStore:
    """Factory for the configured checkpoint backend."""

    normalized = backend.strip().lower()
    if normalized in {"local", "local_json", "json"}:
        return LocalJSONCheckpointStore(base_dir=base_dir, logger=logger)
    raise ValueError(f"Unsupported checkpoint backend: {backend}")


__all__ = [
    "BaseCheckpointStore",
    "CheckpointNotFoundError",
    "CheckpointRecord",
    "LocalJSONCheckpointStore",
    "create_checkpoint_store",
]

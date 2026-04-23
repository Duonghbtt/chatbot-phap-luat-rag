from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.tv2_index.embedding_registry import DEFAULT_CONFIG_PATH, load_indexing_config
from src.tv2_index.qdrant_manager import QdrantManager

LOGGER = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def swap_active_collection(
    *,
    alias_name: str,
    target_collection: str,
    config_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Atomically move an active alias to a new versioned collection."""

    resolved_logger = logger or LOGGER
    config = load_indexing_config(config_path or DEFAULT_CONFIG_PATH)
    manager = QdrantManager(config=config, logger=resolved_logger)
    previous_target = manager.switch_alias(alias_name, target_collection)
    return {
        "alias": alias_name,
        "previous_target": previous_target,
        "current_target": target_collection,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Switch a Qdrant alias to a new collection version for safe rollout.",
    )
    parser.add_argument("--alias", required=True, help="Alias name to switch, e.g. legal_chunks_active")
    parser.add_argument("--target", required=True, help="Target collection name, e.g. legal_chunks_v20260422")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to configs/indexing.yaml")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)
    summary = swap_active_collection(
        alias_name=args.alias,
        target_collection=args.target,
        config_path=args.config,
        logger=LOGGER,
    )
    LOGGER.info("Alias switch completed")
    LOGGER.info(json.dumps(summary, ensure_ascii=False))
    if summary.get("previous_target"):
        LOGGER.info(
            "Rollback tip: rerun with --alias %s --target %s if you need to revert.",
            summary["alias"],
            summary["previous_target"],
        )


if __name__ == "__main__":
    main()

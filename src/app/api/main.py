from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI

from src.app.api.routes.chat import router as chat_router
from src.app.api.routes.stream import router as stream_router
from src.graph.builder import build_graph, load_app_config

LOGGER = logging.getLogger(__name__)


def _langsmith_tracing_enabled() -> bool:
    tracing_flag = str(os.getenv("LANGSMITH_TRACING") or "").strip().lower() in {"1", "true", "yes", "on"}
    api_key = str(os.getenv("LANGSMITH_API_KEY") or "").strip()
    return tracing_flag and bool(api_key)


def _configure_langsmith_tracing() -> dict[str, str | bool]:
    project = str(os.getenv("LANGSMITH_PROJECT") or "").strip()
    workspace_id = str(os.getenv("LANGSMITH_WORKSPACE_ID") or "").strip()
    tracing_flag = str(os.getenv("LANGSMITH_TRACING") or "").strip().lower() in {"1", "true", "yes", "on"}
    api_key = str(os.getenv("LANGSMITH_API_KEY") or "").strip()
    enabled = _langsmith_tracing_enabled()

    if tracing_flag and not api_key:
        LOGGER.warning("LangSmith tracing disabled because LANGSMITH_API_KEY is not set.")
        return {"enabled": False, "project": project, "workspace_id": workspace_id}

    if not enabled:
        LOGGER.info("LangSmith tracing disabled.")
        return {"enabled": False, "project": project, "workspace_id": workspace_id}

    try:
        import langsmith  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency.
        LOGGER.warning("LangSmith tracing disabled because langsmith is unavailable: %s", exc)
        return {"enabled": False, "project": project, "workspace_id": workspace_id}

    LOGGER.info(
        "LangSmith tracing enabled. project=%s workspace_id=%s",
        project or "<default>",
        workspace_id or "<default>",
    )
    return {"enabled": True, "project": project, "workspace_id": workspace_id}


def create_app(config_path: str | Path | None = None) -> FastAPI:
    """Create and configure the FastAPI backend for the legal QA graph."""

    app_config = load_app_config(config_path)
    runtime = build_graph(app_config=app_config, app_config_path=config_path, logger=LOGGER)
    tracing_config = _configure_langsmith_tracing()

    app = FastAPI(title=app_config.app_name)
    app.state.app_config = app_config
    app.state.graph_runtime = runtime
    app.state.langsmith = tracing_config

    app.include_router(chat_router)
    app.include_router(stream_router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "app_name": app_config.app_name}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    config = load_app_config()
    uvicorn.run("src.app.api.main:app", host=config.host, port=config.port, reload=False)

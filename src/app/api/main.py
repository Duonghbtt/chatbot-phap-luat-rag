from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI

from src.app.api.routes.chat import router as chat_router
from src.app.api.routes.stream import router as stream_router
from src.graph.builder import build_graph, load_app_config

LOGGER = logging.getLogger(__name__)


def create_app(config_path: str | Path | None = None) -> FastAPI:
    """Create and configure the FastAPI backend for the legal QA graph."""

    app_config = load_app_config(config_path)
    runtime = build_graph(app_config=app_config, app_config_path=config_path, logger=LOGGER)

    app = FastAPI(title=app_config.app_name)
    app.state.app_config = app_config
    app.state.graph_runtime = runtime

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

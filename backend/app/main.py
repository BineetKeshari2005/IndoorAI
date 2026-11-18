"""IndoorAI FastAPI Application Entry Point.

Provides multimodal dataset ingestion, quality validation, device session tracking,
and machine learning inference endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.datasets import router as datasets_router
from backend.app.api.routes.realtime import router as realtime_router
from src.core.config import get_settings
from src.core.logging import setup_logger

logger = setup_logger("FastAPIBackend")


def create_app() -> FastAPI:
    """Factory creating and configuring the primary IndoorAI FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="IndoorAI Backend API",
        version="0.1.0",
        description="Multimodal indoor localization, survey dataset ingestion, and spatial mapping backend.",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Cross-Origin Resource Sharing (CORS) Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API v1 Router Mount
    api_v1_router = APIRouter(prefix="/api")
    api_v1_router.include_router(datasets_router)
    api_v1_router.include_router(realtime_router)
    app.include_router(api_v1_router)

    @app.get("/", tags=["health"])
    def root():
        """Root API metadata endpoint."""
        return {
            "name": "IndoorAI Backend API",
            "version": "0.1.0",
            "environment": settings.environment,
            "status": "running",
            "documentation": "/docs"
        }

    @app.get("/health", tags=["health"])
    def health():
        """Health check endpoint for container orchestrators and monitoring probes."""
        return {
            "status": "healthy",
            "service": "indoorai-backend"
        }

    logger.info("IndoorAI FastAPI application initialized successfully.")
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)

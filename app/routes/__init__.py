from fastapi import APIRouter
from app.routes.benchmarks import router as benchmarks_router
from app.routes.providers import router as providers_router
from app.routes.history import router as history_router


def create_api_router() -> APIRouter:
    api = APIRouter(prefix="/api/v1")
    api.include_router(benchmarks_router)
    api.include_router(providers_router)
    api.include_router(history_router)
    return api

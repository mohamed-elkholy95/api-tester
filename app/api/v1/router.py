from fastapi import APIRouter

from app.api.v1.endpoints import benchmarks, history, providers, settings

api_router = APIRouter()
api_router.include_router(benchmarks.router)
api_router.include_router(history.router)
api_router.include_router(providers.router)
api_router.include_router(settings.router)

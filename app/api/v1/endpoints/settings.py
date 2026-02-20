"""CRUD routes for user-managed LLM provider settings."""

from __future__ import annotations

import json
import httpx

from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db_session
from app.models.api_settings import ProviderSetting
from app.schemas.api_settings import (
    ProviderSettingCreate,
    ProviderSettingResponse,
)

router = APIRouter(prefix="/settings", tags=["settings"])


def _row_to_response(row: ProviderSetting) -> ProviderSettingResponse:
    try:
        models = json.loads(row.models_json or "[]")
    except Exception:
        models = []
    try:
        extra_headers = json.loads(row.extra_headers_json or "{}")
    except Exception:
        extra_headers = {}

    key = row.api_key or ""
    key_hint = ("…" + key[-4:]) if len(key) >= 4 else ("*" * len(key))

    return ProviderSettingResponse(
        id=row.id,
        name=row.name,
        base_url=row.base_url,
        api_key_hint=key_hint,
        models=models,
        default_model=row.default_model or "",
        supports_stream_usage=row.supports_stream_usage,
        min_temperature=row.min_temperature,
        extra_headers=extra_headers,
        source="db",
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.get("/providers", response_model=list[ProviderSettingResponse])
async def list_db_providers(session: AsyncSession = Depends(get_db_session)):
    result = await session.execute(select(ProviderSetting).order_by(ProviderSetting.name))
    rows = result.scalars().all()
    return [_row_to_response(r) for r in rows]


@router.post(
    "/providers",
    response_model=ProviderSettingResponse,
    status_code=status.HTTP_200_OK,
)
async def upsert_provider(body: ProviderSettingCreate, session: AsyncSession = Depends(get_db_session)):
    result = await session.execute(
        select(ProviderSetting).where(ProviderSetting.id == body.id)
    )
    row = result.scalar_one_or_none()

    models_json = json.dumps(body.models)
    extra_headers_json = json.dumps(body.extra_headers)

    if row is None:
        row = ProviderSetting(
            id=body.id,
            name=body.name,
            base_url=body.base_url,
            api_key=body.api_key,
            models_json=models_json,
            default_model=body.default_model,
            supports_stream_usage=body.supports_stream_usage,
            min_temperature=body.min_temperature,
            extra_headers_json=extra_headers_json,
        )
        session.add(row)
    else:
        row.name = body.name
        row.base_url = body.base_url
        row.api_key = body.api_key
        row.models_json = models_json
        row.default_model = body.default_model
        row.supports_stream_usage = body.supports_stream_usage
        row.min_temperature = body.min_temperature
        row.extra_headers_json = extra_headers_json

    await session.commit()
    # Re-fetch after commit
    fresh = await session.get(ProviderSetting, body.id)
    return _row_to_response(fresh)


@router.delete("/providers/{provider_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_provider(provider_id: str, session: AsyncSession = Depends(get_db_session)):
    result = await session.execute(
        delete(ProviderSetting).where(ProviderSetting.id == provider_id)
    )
    if result.rowcount == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider '{provider_id}' not found in database",
        )


@router.get("/providers/{provider_id}/models")
async def fetch_remote_models(provider_id: str, session: AsyncSession = Depends(get_db_session)):
    row = await session.get(ProviderSetting, provider_id)

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider '{provider_id}' not found in database",
        )

    headers: dict[str, str] = {}
    if row.api_key:
        headers["Authorization"] = f"Bearer {row.api_key}"
    try:
        extra = json.loads(row.extra_headers_json or "{}")
        headers.update(extra)
    except Exception:
        pass

    base = row.base_url.rstrip("/")
    models_url = f"{base}/models"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(models_url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Provider returned {exc.response.status_code}: {exc.response.text[:300]}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not reach provider: {exc}",
        )

    if isinstance(data, list):
        raw_models = data
    elif isinstance(data, dict):
        raw_models = data.get("data") or data.get("models") or []
    else:
        raw_models = []

    model_ids: list[str] = []
    for m in raw_models:
        if isinstance(m, str):
            model_ids.append(m)
        elif isinstance(m, dict):
            mid = m.get("id") or m.get("name") or ""
            if mid:
                model_ids.append(mid)

    return {"provider_id": provider_id, "models": sorted(model_ids)}

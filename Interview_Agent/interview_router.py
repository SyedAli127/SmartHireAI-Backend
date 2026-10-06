"""
FastAPI router — browser interview endpoints.
Mounted into campaign_api.py so everything runs on one server (port 8001).
GET /interview redirects to the Next.js frontend interview page.
"""
from __future__ import annotations

import base64
import logging
import os
import tempfile
import uuid as _uuid
from typing import Any, Dict, List

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse

from Interview_Agent.browser_interview_app import (
    _build_context,
    _check_token,
    _complete_pipeline_interview,
    _generate_question,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_ERROR_MESSAGES = {
    "completed":  "This interview has already been submitted and cannot be reopened.",
    "expired":    "This interview link has expired.",
    "cancelled":  "This interview has been cancelled.",
    "invalid":    "Invalid or unknown interview token.",
}


def _token_error_response(error: str) -> JSONResponse:
    return JSONResponse(
        status_code=410 if error != "invalid" else 404,
        content={"error": error, "message": _ERROR_MESSAGES.get(error, "Unknown error.")},
    )


# ── GET /interview — redirect to Next.js frontend ─────────────────────────────

@router.get("/interview", include_in_schema=False)
async def interview_redirect(token: str = ""):
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
    target = f"{frontend_url}/interview"
    if token:
        target += f"?token={token}"
    return RedirectResponse(url=target, status_code=302)


# ── GET /api/session ──────────────────────────────────────────────────────────

@router.get("/api/session")
async def interview_session(token: str = ""):
    if not token:
        return JSONResponse(status_code=400, content={"error": "missing_token", "message": "No interview token provided."})

    _, error = _check_token(token)
    if error:
        return _token_error_response(error)

    context = _build_context(token)
    if not context:
        return _token_error_response("invalid")

    return context.__dict__


# ── POST /api/question ────────────────────────────────────────────────────────

@router.post("/api/question")
async def interview_question(payload: Dict[str, Any]):
    token = str(payload.get("token") or "").strip()
    history: List[Dict[str, Any]] = payload.get("history") or []

    if not token:
        return JSONResponse(status_code=400, content={"error": "missing_token"})
    if not isinstance(history, list):
        return JSONResponse(status_code=400, content={"error": "history must be a list"})

    _, error = _check_token(token)
    if error:
        return _token_error_response(error)

    context = _build_context(token)
    if not context:
        return _token_error_response("invalid")

    try:
        return _generate_question(context, history)
    except Exception as exc:
        logger.exception("Question generation failed")
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ── POST /api/transcribe ──────────────────────────────────────────────────────

@router.post("/api/transcribe")
async def interview_transcribe(request: Request):
    import os
    from openai import OpenAI

    token = str(request.headers.get("X-Interview-Token") or "").strip()
    if not token:
        return JSONResponse(status_code=400, content={"error": "missing_token"})

    _, error = _check_token(token)
    if error:
        return _token_error_response(error)

    context = _build_context(token)
    if not context:
        return _token_error_response("invalid")

    audio_bytes = await request.body()
    if not audio_bytes:
        return JSONResponse(status_code=400, content={"error": "No audio received"})

    content_type = (request.headers.get("content-type") or "audio/webm").split(";")[0].strip().lower()
    suffix = ".webm"
    if "wav" in content_type:
        suffix = ".wav"
    elif "mp4" in content_type:
        suffix = ".mp4"
    elif "mpeg" in content_type or "mp3" in content_type:
        suffix = ".mp3"

    stt_model = os.getenv("INTERVIEW_STT_MODEL", "gpt-4o-mini-transcribe")
    client = OpenAI()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        try:
            with open(temp_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(model=stt_model, file=audio_file)
            text = transcription.text.strip()
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

        return {"text": text, "candidate_name": context.candidate_name}
    except Exception as exc:
        logger.exception("Transcription failed")
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ── POST /api/complete ────────────────────────────────────────────────────────

@router.post("/api/complete")
async def interview_complete(payload: Dict[str, Any]):
    token = str(payload.get("token") or "").strip()
    history: List[Dict[str, Any]] = payload.get("history") or []

    if not token:
        return JSONResponse(status_code=400, content={"error": "missing_token"})
    if not isinstance(history, list):
        return JSONResponse(status_code=400, content={"error": "history must be a list"})

    _, error = _check_token(token)
    if error:
        return _token_error_response(error)

    context = _build_context(token)
    if not context:
        return _token_error_response("invalid")

    try:
        return _complete_pipeline_interview(token, context, history)
    except Exception as exc:
        logger.exception("Interview completion failed")
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ── POST /api/snapshot ────────────────────────────────────────────────────────

@router.post("/api/snapshot")
async def interview_snapshot(payload: Dict[str, Any]):
    """Receive a base64 JPEG snapshot taken during an interview, upload to S3,
    and store the URL in the interview result so HR can see it."""
    token = str(payload.get("token") or "").strip()
    image_b64 = str(payload.get("image") or "").strip()

    if not token or not image_b64:
        return JSONResponse(status_code=400, content={"error": "missing_fields"})

    _, error = _check_token(token)
    if error:
        return _token_error_response(error)

    context = _build_context(token)
    if not context:
        return _token_error_response("invalid")

    try:
        # Strip optional data-URL prefix (data:image/jpeg;base64,...)
        raw = image_b64.split(",", 1)[-1]
        image_bytes = base64.b64decode(raw)

        from services import s3_service
        interview_id = str(context.interview_id)
        s3_key = f"interviews/snapshots/{interview_id}.jpg"
        snapshot_url = s3_service.upload_bytes(image_bytes, s3_key, "image/jpeg")

        import main_v2 as pipeline_runtime
        from database.repo import session_scope
        from database.models import Interview

        pipeline_runtime._ensure_backend()
        sf = pipeline_runtime.get_session_factory()
        with session_scope(sf) as session:
            inv = session.get(Interview, _uuid.UUID(interview_id))
            if inv and inv.result:
                scores = dict(inv.result.scores or {})
                scores["snapshot_url"] = snapshot_url
                inv.result.scores = scores

        return {"ok": True, "snapshot_url": snapshot_url}
    except Exception as exc:
        logger.exception("Snapshot upload failed")
        return JSONResponse(status_code=500, content={"error": str(exc)})

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from Pipeline.interview_tokens import hash_interview_token  # noqa: E402
from database.repo import (  # noqa: E402
    add_audit_finding,
    complete_interview,
    get_interview_by_token_hash,
    log_event,
    log_hr_conversation,
    mark_interview_no_show,
    set_application_status,
)
from database.session import get_session_factory, init_db, session_scope  # noqa: E402
from database import models as m  # noqa: E402
from Pipeline.audit_service import audit_interview  # noqa: E402

logger = logging.getLogger(__name__)

MODEL      = os.getenv("INTERVIEW_MODEL", "gpt-4o-mini")
EVAL_MODEL = os.getenv("INTERVIEW_EVAL_MODEL", "gpt-4o")
STT_MODEL  = os.getenv("INTERVIEW_STT_MODEL", "gpt-4o-mini-transcribe")
DEFAULT_QUESTION_COUNT = max(1, int(os.getenv("INTERVIEW_QUESTION_COUNT", "5")))
MIN_ANSWER_WORDS = int(os.getenv("INTERVIEW_MIN_ANSWER_WORDS", "10"))

client = OpenAI()

# ── Question type templates ───────────────────────────────────────────────────

QUESTION_TYPES: Dict[int, str] = {
    1: "INTRO — ask a warm, open-ended question about the candidate's background, career journey, or motivation for applying to this specific role",
    2: "TECHNICAL — ask a specific technical question directly tied to required skills in the job description and the candidate's stated experience in their resume",
    3: "BEHAVIOURAL — ask a behavioural question using the STAR framework about a key competency required for this role",
    4: "SITUATIONAL — present a realistic challenge they would face in this role and ask how they would approach it",
    5: "CLOSING — ask about their expectations, career goals, or what excites them about this role to assess culture fit and seriousness",
}


@dataclass
class InterviewContext:
    interview_id: str
    application_id: str
    job_id: str
    candidate_name: str
    candidate_email: str
    job_title: str
    job_department: Optional[str]
    expires_at: str
    status: str
    question_count: int
    jd_preview: str
    resume_storage_uri: Optional[str]
    resume_text: str = ""


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(raw)


def _html_response(handler: BaseHTTPRequestHandler, status: int, payload: str) -> None:
    raw = payload.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(raw)


def _read_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length else b""
    if not raw:
        return {}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    return payload


def _read_raw_body(handler: BaseHTTPRequestHandler) -> bytes:
    length = int(handler.headers.get("Content-Length", "0"))
    return handler.rfile.read(length) if length else b""


def _truncate(text: str, limit: int = 3500) -> str:
    clean = (text or "").strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."


def _safe_json_loads(raw: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(raw[start: end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _normalize_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    if score > 1.5:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def _format_history(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "No previous answers yet."
    lines: List[str] = []
    for index, item in enumerate(history, start=1):
        question = str(item.get("question") or item.get("q") or "")
        answer = str(item.get("answer") or item.get("a") or "")
        lines.append(f"Q{index}: {question}\nA{index}: {answer}")
    return "\n\n".join(lines)


# ── Resume fetching ───────────────────────────────────────────────────────────

def _fetch_resume_text(uri: Optional[str]) -> str:
    if not uri:
        return ""
    try:
        local_path = Path(uri)
        if local_path.exists():
            from Parser.resume_parser import read_resume
            return _truncate(read_resume(str(local_path)), 3000)
    except Exception:
        pass
    try:
        import boto3
        bucket = os.getenv("AWS_S3_BUCKET", "smarthireai-files")
        region = os.getenv("AWS_REGION", "ap-south-1")
        if "amazonaws.com/" in uri:
            key = uri.split(".amazonaws.com/", 1)[1]
        elif uri.startswith("s3://"):
            key = uri.split("/", 3)[-1]
        else:
            key = uri
        s3 = boto3.client("s3", region_name=region)
        ext = Path(key).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            s3.download_fileobj(bucket, key, tmp)
            temp_path = tmp.name
        try:
            from Parser.resume_parser import read_resume
            return _truncate(read_resume(temp_path), 3000)
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
    except Exception:
        return ""


# ── Token validation & context ────────────────────────────────────────────────

def _check_token(raw_token: str) -> tuple[Optional[Any], Optional[str]]:
    """
    Returns (interview, error_code).
    error_code is None (valid), "invalid", "completed", "expired", "cancelled".
    """
    with session_scope(get_session_factory()) as session:
        interview = get_interview_by_token_hash(session, hash_interview_token(raw_token))
        if not interview:
            return None, "invalid"
        if interview.status == m.InterviewStatus.completed:
            return None, "completed"
        if interview.status in (m.InterviewStatus.cancelled,):
            return None, "cancelled"
        now = datetime.now(timezone.utc)
        exp = interview.expires_at
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if now > exp:
            if interview.status not in (m.InterviewStatus.no_show, m.InterviewStatus.completed):
                mark_interview_no_show(session, interview)
            return None, "expired"
        return interview, None


def _build_context(raw_token: str) -> Optional[InterviewContext]:
    with session_scope(get_session_factory()) as session:
        interview = get_interview_by_token_hash(session, hash_interview_token(raw_token))
        if not interview:
            return None

        application = interview.application
        candidate = application.candidate if application else None
        job = application.job if application else None
        jd_version = application.jd_version if application else None

        resume_uri = application.resume_storage_uri if application else None
        resume_text = _fetch_resume_text(resume_uri)

        return InterviewContext(
            interview_id=str(interview.id),
            application_id=str(application.id) if application else "",
            job_id=str(job.id) if job else "",
            candidate_name=(candidate.name if candidate and candidate.name else None) or "Candidate",
            candidate_email=(candidate.email if candidate and candidate.email else None) or "",
            job_title=(job.title if job and job.title else None) or "Interview",
            job_department=job.department if job else None,
            expires_at=interview.expires_at.isoformat(),
            status=interview.status.value,
            question_count=DEFAULT_QUESTION_COUNT,
            jd_preview=_truncate(jd_version.jd_text if jd_version and jd_version.jd_text else "", 2500),
            resume_storage_uri=resume_uri,
            resume_text=resume_text,
        )


# ── Question generation ───────────────────────────────────────────────────────

def _question_prompt(
    context: InterviewContext,
    history: List[Dict[str, Any]],
    question_number: int,
    is_follow_up: bool = False,
) -> str:
    q_type = QUESTION_TYPES.get(question_number, "GENERAL — ask a relevant question about the candidate's experience and fit for the role")

    resume_section = (
        f"\nCandidate resume:\n{context.resume_text}"
        if context.resume_text
        else ""
    )

    follow_up_note = ""
    if is_follow_up and history:
        last_answer = history[-1].get("answer", "")
        follow_up_note = (
            f"\nThe candidate gave a very brief answer: \"{last_answer}\". "
            "Ask a concise follow-up to probe deeper on the same topic before moving on."
        )

    return (
        f"You are conducting a professional browser-based interview.\n"
        f"Question type for this turn: {q_type}\n\n"
        f"Candidate: {context.candidate_name}\n"
        f"Role: {context.job_title}\n"
        f"Department: {context.job_department or 'N/A'}\n"
        f"JD excerpt:\n{context.jd_preview or 'N/A'}"
        f"{resume_section}\n\n"
        f"This is question {question_number} of {context.question_count}.\n"
        f"Previous transcript:\n{_format_history(history)}"
        f"{follow_up_note}\n\n"
        "Ask ONE concise, focused question. Do not mention you are an AI. "
        "Do not ask multiple questions at once. Return only the question text."
    )


def _should_follow_up(history: List[Dict[str, Any]]) -> bool:
    """Returns True if the last answer was too brief and warrants a follow-up."""
    if not history:
        return False
    last = history[-1]
    if str(last.get("question", "")).startswith("(Follow-up)"):
        return False
    answer = str(last.get("answer", ""))
    return len(answer.split()) < MIN_ANSWER_WORDS


def _answered_count(history: List[Dict[str, Any]]) -> int:
    """Count non-follow-up turns to determine real question progression."""
    return sum(
        1 for h in history
        if not str(h.get("question", "")).startswith("(Follow-up)")
    )


def _generate_question(context: InterviewContext, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    is_follow_up = _should_follow_up(history)
    question_number = _answered_count(history) + 1

    if question_number > context.question_count and not is_follow_up:
        return {
            "complete": True,
            "question_number": question_number,
            "message": "All questions answered. Please finish your interview.",
        }

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a precise, professional interviewer running a structured browser interview.",
            },
            {"role": "user", "content": _question_prompt(context, history, question_number, is_follow_up)},
        ],
        temperature=0.55,
    )
    question_text = response.choices[0].message.content.strip()
    if is_follow_up:
        question_text = f"(Follow-up) {question_text}"

    return {
        "complete": False,
        "question_number": question_number,
        "question": question_text,
        "is_follow_up": is_follow_up,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def _extract_jd_skills(jd_text: str) -> List[str]:
    if not jd_text or not jd_text.strip():
        return []
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract 3-5 key skills or competencies from this job description. "
                        'Return JSON: {"skills": ["skill1", "skill2", ...]}'
                    ),
                },
                {"role": "user", "content": jd_text[:2000]},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)
        skills = parsed.get("skills") or []
        return [str(s) for s in skills[:5]] if isinstance(skills, list) else []
    except Exception:
        return []


def _score_single_answer(question: str, answer: str, role: str) -> Dict[str, Any]:
    try:
        resp = client.chat.completions.create(
            model=EVAL_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a hiring evaluator. Score this single interview answer. "
                        'Return JSON: {"score": 0-100, "feedback": "one sentence"}'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Role: {role}\nQuestion: {question}\nAnswer: {answer}",
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"score": 50, "feedback": "Unable to evaluate this answer."}


def _evaluate_interview(context: InterviewContext, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    transcript = _format_history(history)

    # Score each answer in parallel
    with concurrent.futures.ThreadPoolExecutor() as pool:
        answer_futures = [
            pool.submit(
                _score_single_answer,
                item.get("question", ""),
                item.get("answer", ""),
                context.job_title,
            )
            for item in history
        ]
        # Also extract JD skills in parallel
        skills_future = pool.submit(_extract_jd_skills, context.jd_preview)

    answer_scores = [f.result() for f in answer_futures]
    jd_skills = skills_future.result()

    # Build dynamic skill score keys
    skill_keys = [s.lower().replace(" ", "_").replace("/", "_")[:30] for s in jd_skills]
    skill_score_spec = "\n".join(f'- {k}: number 0-100 (score for "{s}")' for k, s in zip(skill_keys, jd_skills))

    prompt = f"""You are a senior hiring manager evaluating a candidate after a structured interview.

Candidate: {context.candidate_name}
Role: {context.job_title}
Department: {context.job_department or "N/A"}

Full transcript:
{transcript}

Per-answer scores (reference): {json.dumps(answer_scores)}

Return valid JSON with exactly these keys:
- overall: number 0-100
- communication: number 0-100
- technical: number 0-100
{skill_score_spec if skill_score_spec else ""}
- recommendation: one of "STRONG_HIRE" | "HIRE" | "MAYBE" | "NO_HIRE"
- strengths: array of short strings
- weaknesses: array of short strings
- summary: concise evaluation paragraph (3-5 sentences)""".strip()

    response = client.chat.completions.create(
        model=EVAL_MODEL,
        messages=[
            {"role": "system", "content": "You return strict JSON only. No markdown, no code fences."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    parsed = _safe_json_loads(response.choices[0].message.content.strip())
    if not parsed:
        parsed = {
            "overall": 60,
            "communication": 60,
            "technical": 60,
            "recommendation": "MAYBE",
            "strengths": [],
            "weaknesses": [],
            "summary": "Evaluation parsing failed. Manual review required.",
        }

    parsed["answer_scores"] = answer_scores
    parsed["jd_skills_assessed"] = jd_skills
    return parsed


# ── Pipeline completion ───────────────────────────────────────────────────────

def _complete_pipeline_interview(
    raw_token: str,
    context: InterviewContext,
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    evaluation = _evaluate_interview(context, history)
    payload = {
        "overall": _normalize_score(evaluation.get("overall", 0.0)),
        "communication": _normalize_score(evaluation.get("communication", 0.0)),
        "technical": _normalize_score(evaluation.get("technical", 0.0)),
        "notes": str(evaluation.get("summary") or evaluation.get("recommendation") or "Completed."),
        "evaluation": evaluation,
        "history": history,
        "candidate_context": {
            "candidate_name": context.candidate_name,
            "candidate_email": context.candidate_email,
            "job_title": context.job_title,
            "job_department": context.job_department,
            "application_id": context.application_id,
            "job_id": context.job_id,
        },
    }

    with session_scope(get_session_factory()) as session:
        interview = get_interview_by_token_hash(session, hash_interview_token(raw_token))
        if not interview:
            return {"completed": False, "evaluation": evaluation, "scores": payload}

        application = interview.application
        job = application.job if application else None
        correlation_id = job.correlation_id if job else uuid.uuid4()

        for finding in audit_interview(payload):
            add_audit_finding(
                session,
                correlation_id=correlation_id,
                job_id=application.job_id,
                application_id=application.id,
                stage="interview",
                severity=finding["severity"],
                finding_type=finding["finding_type"],
                summary=finding["summary"],
                details=finding.get("details"),
            )

        _, completed = complete_interview(
            session,
            interview,
            scores=payload,
            model_version=os.getenv("INTERVIEW_MODEL_VERSION", "browser-interview-v2"),
            media_uri=None,
            idempotency_key=f"browser-interview-complete-{interview.id}",
        )
        if completed and application:
            set_application_status(session, application, m.ApplicationStatus.interview_completed)
            log_event(
                session,
                correlation_id=correlation_id,
                agent_name="interview_agent",
                entity_type="interview",
                entity_id=str(interview.id),
                from_state=m.InterviewStatus.link_sent.value,
                to_state=m.InterviewStatus.completed.value,
                payload={"source": "browser_interview_v2"},
                idempotency_key=f"browser-interview-complete-{interview.id}-evt",
            )
            if application.candidate:
                log_hr_conversation(
                    session,
                    correlation_id=correlation_id,
                    job_id=application.job_id,
                    application_id=application.id,
                    speaker="agent",
                    stage="interview_completion",
                    message=f"Interview completed for {application.candidate.email}",
                    meta_json={"interview_id": str(interview.id)},
                )

    # Send thank-you email
    if completed and context.candidate_email:
        try:
            from Pipeline.communication_adapter import send_interview_complete_email
            send_interview_complete_email(
                candidate_name=context.candidate_name,
                candidate_email=context.candidate_email,
                job_title=context.job_title,
            )
        except Exception as e:
            logger.warning("Thank-you email failed: %s", e)

    return {
        "completed": completed,
        "evaluation": evaluation,
        "scores": payload,
    }


# ── HTML page (legacy — now frontend handles UI, this is fallback only) ───────

def _render_page() -> str:
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta http-equiv="refresh" content="0; url={frontend_url}/interview"/>
  <title>Redirecting...</title>
</head>
<body>
  <p>Redirecting to interview room... <a href="{frontend_url}/interview">click here</a> if not redirected.</p>
</body>
</html>"""


# ── HTTP handler (legacy — kept for standalone mode) ─────────────────────────

class BrowserInterviewHandler(BaseHTTPRequestHandler):
    server_version = "SmartHireAIInterview/2.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        logger.info("%s - - [%s] %s", self.address_string(), self.log_date_time_string(), format % args)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/interview"}:
            _html_response(self, 200, _render_page())
            return
        if parsed.path == "/api/session":
            token = parse_qs(parsed.query).get("token", [""])[0].strip()
            if not token:
                _json_response(self, 400, {"error": "missing_token"})
                return
            _, error = _check_token(token)
            if error:
                _json_response(self, 410 if error != "invalid" else 404, {"error": error})
                return
            context = _build_context(token)
            if not context:
                _json_response(self, 404, {"error": "invalid"})
                return
            _json_response(self, 200, context.__dict__)
            return
        if parsed.path == "/health":
            _json_response(self, 200, {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()})
            return
        _json_response(self, 404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/question":
            self._handle_question()
            return
        if parsed.path == "/api/transcribe":
            self._handle_transcribe()
            return
        if parsed.path == "/api/complete":
            self._handle_complete()
            return
        _json_response(self, 404, {"error": "Not found"})

    def _handle_question(self) -> None:
        try:
            payload = _read_json_body(self)
            token = str(payload.get("token") or "").strip()
            history = payload.get("history") or []
            if not token:
                _json_response(self, 400, {"error": "Missing token"})
                return
            _, error = _check_token(token)
            if error:
                _json_response(self, 410 if error != "invalid" else 404, {"error": error})
                return
            context = _build_context(token)
            if not context:
                _json_response(self, 404, {"error": "Invalid token"})
                return
            result = _generate_question(context, history)
            _json_response(self, 200, result)
        except Exception as exc:
            logger.exception("Question generation failed")
            _json_response(self, 500, {"error": str(exc)})

    def _handle_transcribe(self) -> None:
        try:
            token = str(self.headers.get("X-Interview-Token") or "").strip()
            if not token:
                _json_response(self, 400, {"error": "Missing interview token"})
                return
            _, error = _check_token(token)
            if error:
                _json_response(self, 410 if error != "invalid" else 404, {"error": error})
                return
            context = _build_context(token)
            if not context:
                _json_response(self, 404, {"error": "Invalid token"})
                return
            audio_bytes = _read_raw_body(self)
            if not audio_bytes:
                _json_response(self, 400, {"error": "No audio received"})
                return
            content_type = (self.headers.get("Content-Type") or "audio/webm").split(";")[0].strip().lower()
            suffix = ".webm"
            if "wav" in content_type:
                suffix = ".wav"
            elif "mp4" in content_type:
                suffix = ".mp4"
            elif "mpeg" in content_type or "mp3" in content_type:
                suffix = ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                temp_path = tmp.name
            try:
                with open(temp_path, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(model=STT_MODEL, file=audio_file)
                text = transcription.text.strip()
            finally:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            _json_response(self, 200, {"text": text, "candidate_name": context.candidate_name})
        except Exception as exc:
            logger.exception("Transcription failed")
            _json_response(self, 500, {"error": str(exc)})

    def _handle_complete(self) -> None:
        try:
            payload = _read_json_body(self)
            token = str(payload.get("token") or "").strip()
            history = payload.get("history") or []
            if not token:
                _json_response(self, 400, {"error": "Missing token"})
                return
            _, error = _check_token(token)
            if error:
                _json_response(self, 410 if error != "invalid" else 404, {"error": error})
                return
            context = _build_context(token)
            if not context:
                _json_response(self, 404, {"error": "Invalid token"})
                return
            result = _complete_pipeline_interview(token, context, history)
            _json_response(self, 200, result)
        except Exception as exc:
            logger.exception("Interview completion failed")
            _json_response(self, 500, {"error": str(exc)})


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    init_db()
    host = os.getenv("INTERVIEW_HOST", "0.0.0.0")
    port = int(os.getenv("INTERVIEW_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), BrowserInterviewHandler)
    logger.info("Interview server on http://%s:%s/interview", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

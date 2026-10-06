"""
SmartHireAI — Unified Pipeline Entry Point
==========================================
HR passes a single intake dict and this script drives the full pipeline:

    JD creation → HR approval → Ad generation → HR approval → Ad posting
    → Collect applications (threshold gate) → Resume screening → Shortlist 1
    → Send interview links → Wait for interviews → Shortlist 2 → Offer letters

HR interaction is limited to:
  - Approving / rejecting the generated JD (with optional feedback)
  - Approving / rejecting the generated ads

Everything else is automated by the orchestrator and sub-agents.
All HR conversations are persisted to the HR_Conversations table.
"""

import time
import uuid
import os
import argparse
import sys
import subprocess
# import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from sqlalchemy import func, select

_BACKEND_READY = False
_BACKEND_IMPORT_ERROR: Optional[Exception] = None

HiringPipeline = None
HiringPipelineConfig = None
run_learning_pass = None
audit_batch_prescreen = None
audit_batch_interview = None
generate_audit_summary = None
GoogleFormsService = None
session_scope = None
get_session_factory = None
init_db = None
m = None


def _ensure_backend() -> None:
    global _BACKEND_READY, _BACKEND_IMPORT_ERROR
    global HiringPipeline, HiringPipelineConfig, run_learning_pass
    global audit_batch_prescreen, audit_batch_interview, generate_audit_summary
    global GoogleFormsService, session_scope, get_session_factory, init_db, m

    if _BACKEND_READY:
        return

    try:
        from Pipeline.hiring_pipeline import HiringPipeline as _HiringPipeline, HiringPipelineConfig as _HiringPipelineConfig
        from Pipeline.learning_service import run_learning_pass as _run_learning_pass
        from Pipeline.audit_service import (
            audit_batch_prescreen as _audit_batch_prescreen,
            audit_batch_interview as _audit_batch_interview,
            generate_audit_summary as _generate_audit_summary,
        )
        from services.google_forms_service import GoogleFormsService as _GoogleFormsService
        from database.repo import session_scope as _session_scope
        from database.session import get_session_factory as _get_session_factory, init_db as _init_db
        from database import models as _m
    except ModuleNotFoundError as exc:
        _BACKEND_IMPORT_ERROR = exc
        raise

    HiringPipeline = _HiringPipeline
    HiringPipelineConfig = _HiringPipelineConfig
    run_learning_pass = _run_learning_pass
    audit_batch_prescreen = _audit_batch_prescreen
    audit_batch_interview = _audit_batch_interview
    generate_audit_summary = _generate_audit_summary
    GoogleFormsService = _GoogleFormsService
    session_scope = _session_scope
    get_session_factory = _get_session_factory
    init_db = _init_db
    m = _m
    _BACKEND_READY = True


# ---------------------------------------------------------------------------
# Communication preflight
# ---------------------------------------------------------------------------

def _is_truthy_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _validate_communication_config() -> None:
    """
    Fail fast when real communication is enabled but provider configuration is invalid.
    This prevents silent interview dispatch with no actual email delivery.
    """
    if not _is_truthy_env("COMMUNICATION_REAL", default=False):
        print("[Preflight] COMMUNICATION_REAL is disabled; email delivery will run in dry mode.")
        return

    provider = os.getenv("COMMUNICATION_PROVIDER", "gmail").strip().lower()
    print(f"[Preflight] Real communication enabled. Provider: {provider}")

    if provider == "smtp":
        missing = [
            name for name in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS")
            if not os.getenv(name)
        ]
        if missing:
            raise RuntimeError(
                "SMTP configuration is incomplete. Missing: " + ", ".join(missing)
            )

        use_tls = _is_truthy_env("SMTP_USE_TLS", default=True)
        use_ssl = _is_truthy_env("SMTP_USE_SSL", default=False)
        if use_tls and use_ssl:
            raise RuntimeError("SMTP config invalid: enable only one of SMTP_USE_TLS or SMTP_USE_SSL.")

        try:
            int(os.getenv("SMTP_PORT", ""))
        except ValueError as exc:
            raise RuntimeError("SMTP_PORT must be a valid integer.") from exc

        from_email = os.getenv("RECRUITER_FROM_EMAIL")
        if not from_email:
            print("[Preflight] RECRUITER_FROM_EMAIL not set; SMTP_USER will be used as sender.")
        return

    if provider == "gmail":
        from_email = os.getenv("RECRUITER_FROM_EMAIL")
        if not from_email:
            print("[Preflight] RECRUITER_FROM_EMAIL not set; Gmail API default sender will be used.")
        return

    raise RuntimeError(
        "COMMUNICATION_PROVIDER must be either 'smtp' or 'gmail'. "
        f"Received: {provider!r}"
    )


# ---------------------------------------------------------------------------
# HR approval callbacks (interactive prompts)
# ---------------------------------------------------------------------------

def _hr_jd_approval(state: Dict[str, Any]) -> Tuple[bool, str]:
    """Prompt HR to review and approve or reject the generated JD."""
    print("\n" + "=" * 70)
    print("DRAFT JOB DESCRIPTION (awaiting your approval)")
    print("=" * 70)
    jd_text = (state.get("winner") or {}).get("jd_text") or state.get("final_jd") or ""
    print(jd_text or "<No JD text available>")
    print("=" * 70)

    while True:
        choice = input("\nApprove this JD? [y = approve / n = reject]: ").strip().lower()
        if choice in ("y", "n"):
            break
        print("  Please enter 'y' or 'n'.")

    feedback = ""
    if choice == "n":
        feedback = input("Feedback / revision notes: ").strip()

    return choice == "y", feedback


def _hr_ad_approval(state: Dict[str, Any]) -> Tuple[bool, str, Optional[List[dict]]]:
    """Prompt HR to review and approve or reject the generated ad pack."""
    print("\n" + "=" * 70)
    print("GENERATED AD PACK (awaiting your approval)")
    print("=" * 70)
    ads = state.get("combined_ads") or state.get("ad_variants") or []
    if not ads:
        print("<No ads generated yet>")
    else:
        for ad in ads:
            platform = getattr(ad, "platform", None) or (ad.get("platform") if isinstance(ad, dict) else "?")
            variant  = getattr(ad, "variant",  None) or (ad.get("variant")  if isinstance(ad, dict) else "?")
            headline = getattr(ad, "headline", None) or (ad.get("headline") if isinstance(ad, dict) else "")
            text     = getattr(ad, "text",     None) or (ad.get("text")     if isinstance(ad, dict) else "")
            cta      = getattr(ad, "cta",      None) or (ad.get("cta")      if isinstance(ad, dict) else "")
            hashtags = getattr(ad, "hashtags", None) or (ad.get("hashtags") if isinstance(ad, dict) else [])
            print(f"\n[{platform}] Variant {variant}")
            print(f"  Headline : {headline}")
            print(f"  Text     : {text}")
            print(f"  CTA      : {cta}")
            print(f"  Hashtags : {' '.join(hashtags or [])}")
    print("=" * 70)

    while True:
        choice = input("\nApprove these ads? [y = approve / n = reject]: ").strip().lower()
        if choice in ("y", "n"):
            break
        print("  Please enter 'y' or 'n'.")

    feedback = ""
    if choice == "n":
        feedback = input("Feedback on ads: ").strip()
        return False, feedback, None

    # Auto-select one variant per platform from combined_ads
    approved: Dict[str, dict] = {}
    for ad in (state.get("combined_ads") or []):
        platform = getattr(ad, "platform", None) or (ad.get("platform") if isinstance(ad, dict) else None)
        if platform and platform not in approved:
            approved[platform] = ad.model_dump() if hasattr(ad, "model_dump") else dict(ad)

    return True, "Approved by HR.", list(approved.values()) if approved else None


# ---------------------------------------------------------------------------
# Post-interview polling helpers
# ---------------------------------------------------------------------------

def _count_completed_interviews(session_factory, job_id: uuid.UUID) -> int:
    with session_scope(session_factory) as session:
        return session.scalar(
            select(func.count(m.Interview.id)).where(
                m.Interview.application_id.in_(
                    select(m.Application.id).where(
                        m.Application.job_id == job_id,
                        m.Application.status.in_([
                            m.ApplicationStatus.interview_link_sent,
                            m.ApplicationStatus.interview_completed,
                        ]),
                    )
                ),
                m.Interview.status == m.InterviewStatus.completed,
            )
        ) or 0


def _count_dispatched_interviews(session_factory, job_id: uuid.UUID) -> int:
    with session_scope(session_factory) as session:
        return session.scalar(
            select(func.count(m.Application.id)).where(
                m.Application.job_id == job_id,
                m.Application.status.in_([
                    m.ApplicationStatus.interview_link_sent,
                    m.ApplicationStatus.interview_scheduled,
                    m.ApplicationStatus.interview_completed,
                ]),
            )
        ) or 0


def _get_job_counts(session_factory, job_id: uuid.UUID) -> Dict[str, int]:
    with session_scope(session_factory) as session:
        job = session.get(m.Job, job_id)
        if not job:
            return {
                "current": 0,
                "threshold": 1,
                "submitted": 0,
                "dispatched": 0,
                "completed": 0,
            }

        submitted = session.scalar(
            select(func.count(m.Application.id)).where(
                m.Application.job_id == job_id,
                m.Application.status == m.ApplicationStatus.submitted,
            )
        ) or 0

        dispatched = session.scalar(
            select(func.count(m.Application.id)).where(
                m.Application.job_id == job_id,
                m.Application.status.in_([
                    m.ApplicationStatus.interview_link_sent,
                    m.ApplicationStatus.interview_scheduled,
                    m.ApplicationStatus.interview_completed,
                ]),
            )
        ) or 0

        completed = session.scalar(
            select(func.count(m.Interview.id)).where(
                m.Interview.application_id.in_(
                    select(m.Application.id).where(
                        m.Application.job_id == job_id,
                        m.Application.status.in_([
                            m.ApplicationStatus.interview_link_sent,
                            m.ApplicationStatus.interview_completed,
                        ]),
                    )
                ),
                m.Interview.status == m.InterviewStatus.completed,
            )
        ) or 0

        return {
            "current": int(job.current_applications_count or 0),
            "threshold": int(job.min_applications_threshold or 1),
            "submitted": int(submitted),
            "dispatched": int(dispatched),
            "completed": int(completed),
        }


def _collect_status_snapshot(session_factory, job_id: uuid.UUID) -> Dict[str, Any]:
    with session_scope(session_factory) as session:
        job = session.get(m.Job, job_id)
        if not job:
            return {
                "job_found": False,
                "job_id": str(job_id),
            }

        app_rows = session.execute(
            select(m.Application.status, func.count(m.Application.id))
            .where(m.Application.job_id == job_id)
            .group_by(m.Application.status)
        ).all()
        app_counts = {
            (status.value if hasattr(status, "value") else str(status)): int(count)
            for status, count in app_rows
        }

        interview_rows = session.execute(
            select(m.Interview.status, func.count(m.Interview.id))
            .join(m.Application, m.Application.id == m.Interview.application_id)
            .where(m.Application.job_id == job_id)
            .group_by(m.Interview.status)
        ).all()
        interview_counts = {
            (status.value if hasattr(status, "value") else str(status)): int(count)
            for status, count in interview_rows
        }

        offer_rows = session.execute(
            select(m.Offer.status, func.count(m.Offer.id))
            .join(m.Application, m.Application.id == m.Offer.application_id)
            .where(m.Application.job_id == job_id)
            .group_by(m.Offer.status)
        ).all()
        offer_counts = {
            (status.value if hasattr(status, "value") else str(status)): int(count)
            for status, count in offer_rows
        }

        pending_interviews = max(
            0,
            int(app_counts.get(m.ApplicationStatus.interview_link_sent.value, 0))
            + int(app_counts.get(m.ApplicationStatus.interview_scheduled.value, 0))
            - int(interview_counts.get(m.InterviewStatus.completed.value, 0)),
        )

        return {
            "job_found": True,
            "job_id": str(job.id),
            "job_status": job.status.value if job.status else None,
            "title": job.title,
            "department": job.department,
            "current_applications": int(job.current_applications_count or 0),
            "threshold": int(job.min_applications_threshold or 1),
            "app_counts": app_counts,
            "interview_counts": interview_counts,
            "offer_counts": offer_counts,
            "pending_interviews": int(pending_interviews),
        }


def _print_status_snapshot(session_factory, job_id: uuid.UUID, label: str) -> None:
    snap = _collect_status_snapshot(session_factory, job_id)
    print("\n" + "-" * 70)
    print(f"[STATUS] {label}")
    print("-" * 70)
    if not snap.get("job_found"):
        print(f"  Job not found: {job_id}")
        return

    print(f"  Job ID            : {snap['job_id']}")
    print(f"  Title             : {snap.get('title')}")
    print(f"  Job status        : {snap.get('job_status')}")
    print(f"  Applications      : {snap.get('current_applications')}/{snap.get('threshold')} (received/threshold)")

    app_counts = snap.get("app_counts") or {}
    if app_counts:
        print("  Application states:")
        for state_name in sorted(app_counts.keys()):
            print(f"    - {state_name}: {app_counts[state_name]}")
    else:
        print("  Application states: none")

    interview_counts = snap.get("interview_counts") or {}
    if interview_counts:
        print("  Interview states  :")
        for state_name in sorted(interview_counts.keys()):
            print(f"    - {state_name}: {interview_counts[state_name]}")
    else:
        print("  Interview states  : none")

    print(f"  Pending interviews: {snap.get('pending_interviews', 0)}")

    offer_counts = snap.get("offer_counts") or {}
    if offer_counts:
        print("  Offer states      :")
        for state_name in sorted(offer_counts.keys()):
            print(f"    - {state_name}: {offer_counts[state_name]}")
    else:
        print("  Offer states      : none")


def _resolve_application_briefs(session_factory, application_ids: List[str]) -> List[Dict[str, Any]]:
    """Resolve application IDs into lightweight candidate metadata for runtime logs."""
    if not application_ids:
        return []

    out: List[Dict[str, Any]] = []
    with session_scope(session_factory) as session:
        for idx, app_id in enumerate(application_ids, start=1):
            try:
                app_uuid = uuid.UUID(str(app_id))
            except Exception:
                out.append(
                    {
                        "rank": idx,
                        "application_id": str(app_id),
                        "candidate_email": None,
                        "candidate_name": None,
                        "status": None,
                    }
                )
                continue

            app = session.get(m.Application, app_uuid)
            out.append(
                {
                    "rank": idx,
                    "application_id": str(app_id),
                    "candidate_email": app.candidate.email if app and app.candidate else None,
                    "candidate_name": app.candidate.name if app and app.candidate else None,
                    "status": app.status.value if app and app.status else None,
                }
            )
    return out


def _status_mode(session_factory, job_id: uuid.UUID, *, watch: bool, poll_seconds: int, max_polls: int) -> None:
    polls = 0
    while True:
        _print_status_snapshot(session_factory, job_id, label="Manual status command")
        polls += 1
        if not watch:
            return
        if max_polls > 0 and polls >= max_polls:
            return
        time.sleep(max(1, int(poll_seconds)))


def _launch_interview_server() -> None:
    interview_path = Path(__file__).resolve().parent / "Interview_Agent" / "browser_interview_app.py"
    if not interview_path.exists():
        raise FileNotFoundError(f"Interview server entrypoint not found: {interview_path}")

    try:
        import urllib.request

        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1) as response:
            if response.status == 200:
                print("[Interview] Browser interview server already running on port 8000.")
                return
    except Exception:
        pass

    print("[Interview] Starting browser interview server on port 8000...")
    subprocess.Popen(
        [sys.executable, str(interview_path)],
        cwd=str(Path(__file__).resolve().parent),
        stdout=None,
        stderr=None,
        start_new_session=True,
    )

    try:
        import urllib.request

        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1) as response:
                    if response.status == 200:
                        print("[Interview] Browser interview server is ready.")
                        return
            except Exception:
                time.sleep(0.5)
    except Exception:
        pass

    print("[Interview] Browser interview server is starting in the background.")


def _wait_for_application_threshold(
    pipeline: HiringPipeline,
    session_factory,
    job_id: uuid.UUID,
    form_link: Optional[str],
    shortlist_top_n: int,
    poll_seconds: int = 10,
    max_wait_minutes: int = 60,
) -> bool:
    """
    Poll job/application counters until min_applications_threshold is met.
    Returns True if threshold met, False if timed out.
    """
    deadline = time.time() + (max_wait_minutes * 60)
    print(
        f"\n[Phase 3/5] Waiting for applications to reach threshold "
        f"(polling every {poll_seconds}s, timeout {max_wait_minutes}m)..."
    )
    seen_response_ids: set = set()
    forms_client_cache: Dict[str, Any] = {}

    while time.time() < deadline:
        _ingest_new_form_responses(
            pipeline,
            job_id,
            form_link=form_link,
            shortlist_top_n=shortlist_top_n,
            seen_response_ids=seen_response_ids,
            forms_client_cache=forms_client_cache,
        )

        counts = _get_job_counts(session_factory, job_id)
        print(
            "  Applications received: "
            f"{counts['current']}/{counts['threshold']} | "
            f"queued for screening: {counts['submitted']}"
        )
        _print_status_snapshot(session_factory, job_id, label="Phase 3 heartbeat")
        if counts["current"] >= counts["threshold"]:
            return True
        time.sleep(poll_seconds)

    print("[Phase 3/5] Timeout waiting for application threshold.")
    return False


def _extract_form_id(form_link: Optional[str]) -> Optional[str]:
    if not form_link:
        return None
    try:
        parts = [p for p in urlparse(form_link).path.split("/") if p]
        if len(parts) >= 3 and parts[0] == "forms" and parts[1] == "d":
            return parts[2]
    except Exception:
        return None
    return None


def _get_answer_values(answer_payload: dict) -> List[str]:
    if not isinstance(answer_payload, dict):
        return []

    values: List[str] = []

    text_answers = answer_payload.get("textAnswers", {}).get("answers", [])
    if isinstance(text_answers, list):
        for answer in text_answers:
            if not isinstance(answer, dict):
                continue
            value = answer.get("value")
            if value:
                values.append(str(value).strip())

    # Google Forms file-upload answers return Drive file ids.
    file_answers = answer_payload.get("fileUploadAnswers", {}).get("answers", [])
    if isinstance(file_answers, list):
        for answer in file_answers:
            if not isinstance(answer, dict):
                continue
            file_id = answer.get("fileId")
            if file_id:
                values.append(f"https://drive.google.com/file/d/{file_id}/view")

    return [v for v in values if v]


def _extract_application_from_form_response(response: dict, question_map: Dict[str, str]) -> Optional[Dict[str, str]]:
    answers = response.get("answers") or {}
    if not isinstance(answers, dict):
        return None

    email = ""
    name = ""
    phone = ""
    resume_link = ""

    for qid, payload in answers.items():
        title = (question_map.get(qid) or "").lower()
        values = _get_answer_values(payload)
        if not values:
            continue
        value = values[0]
        if "email" in title:
            email = value
        elif "name" in title:
            name = value
        elif "phone" in title:
            phone = value
        elif "resume" in title or "cv" in title:
            resume_link = value

    if not email:
        return None

    return {
        "email": email,
        "name": name or "Candidate",
        "phone": phone or None,
        "resume_path": resume_link or "",
    }


def _ingest_new_form_responses(
    pipeline: HiringPipeline,
    job_id: uuid.UUID,
    *,
    form_link: Optional[str],
    shortlist_top_n: int,
    seen_response_ids: set,
    forms_client_cache: Dict[str, Any],
) -> int:
    form_id = _extract_form_id(form_link)
    if not form_id:
        return 0

    forms = forms_client_cache.get("client")
    if forms is None:
        try:
            forms = GoogleFormsService()
            forms_client_cache["client"] = forms
        except Exception as e:
            print(f"  [INFO] Google Forms polling unavailable: {e}")
            return 0

    try:
        form_data = forms.forms_service.forms().get(formId=form_id).execute()
        question_map: Dict[str, str] = {}
        for item in form_data.get("items", []) or []:
            q = ((item.get("questionItem") or {}).get("question") or {})
            qid = q.get("questionId")
            title = item.get("title") or ""
            if qid:
                question_map[str(qid)] = str(title)

        responses = forms.forms_service.forms().responses().list(formId=form_id).execute().get("responses", [])
    except Exception as e:
        print(f"  [WARNING] Failed to poll Google Form responses: {e}")
        return 0

    ingested = 0
    for response in responses:
        response_id = response.get("responseId")
        if not response_id or response_id in seen_response_ids:
            continue
        seen_response_ids.add(response_id)

        app = _extract_application_from_form_response(response, question_map)
        if not app:
            continue

        # Form captures resume URL, which we store as resume_path for downstream processing.
        # If a local file path is required by parser, screening marks record for review.
        submit = pipeline.submit_application(
            job_id=job_id,
            email=app["email"],
            name=app.get("name"),
            phone=app.get("phone"),
            resume_path=app.get("resume_path") or "",
            shortlist_top_n=max(1, int(shortlist_top_n)),
            auto_process_when_threshold_met=False,
        )
        ingested += 1
        print(f"  [FORMS] Ingested application from Google Form: {app['email']} ({submit.get('status')})")

    return ingested


def _poll_interviews_complete(
    session_factory,
    job_id: uuid.UUID,
    poll_seconds: int = 10,
    max_wait_minutes: int = 60,
) -> bool:
    """
    Poll until all dispatched interviews are completed (or timeout).
    Returns True when all done, False on timeout.
    """
    deadline = time.time() + (max_wait_minutes * 60)
    print(f"\n[Pipeline] Waiting for candidates to complete their interviews "
          f"(polling every {poll_seconds}s, timeout {max_wait_minutes}m)...")

    while time.time() < deadline:
        dispatched = _count_dispatched_interviews(session_factory, job_id)
        completed  = _count_completed_interviews(session_factory, job_id)
        print(f"  Interviews dispatched: {dispatched}  |  completed: {completed}")
        _print_status_snapshot(session_factory, job_id, label="Phase 4 heartbeat")
        if dispatched > 0 and completed >= dispatched:
            return True
        time.sleep(poll_seconds)

    print("[Pipeline] Timeout waiting for interview completions.")
    return False


# ---------------------------------------------------------------------------
# Audit helper
# ---------------------------------------------------------------------------

def _run_and_print_audit(session_factory, job_id: uuid.UUID, correlation_id: uuid.UUID) -> None:
    """Fetch latest prescreen + interview scores and run the audit agents."""
    from database.repo import add_audit_finding

    with session_scope(session_factory) as session:
        apps = list(session.scalars(
            select(m.Application).where(m.Application.job_id == job_id)
        ).all())

        prescreen_scores = []
        for app in apps:
            if app.prescreens:
                latest = max(app.prescreens, key=lambda r: r.created_at or 0)
                prescreen_scores.append(latest.scores or {})

        interview_scores = []
        for app in apps:
            for inv in (app.interviews or []):
                if inv.result and inv.result.scores:
                    interview_scores.append(inv.result.scores)

        if prescreen_scores:
            batch_findings = audit_batch_prescreen(prescreen_scores)
            for finding in batch_findings:
                add_audit_finding(
                    session,
                    correlation_id=correlation_id,
                    job_id=job_id,
                    application_id=None,
                    stage="batch_prescreen_audit",
                    severity=finding["severity"],
                    finding_type=finding["finding_type"],
                    summary=finding["summary"],
                    details=finding.get("details"),
                )

        if interview_scores:
            batch_findings = audit_batch_interview(interview_scores)
            for finding in batch_findings:
                add_audit_finding(
                    session,
                    correlation_id=correlation_id,
                    job_id=job_id,
                    application_id=None,
                    stage="batch_interview_audit",
                    severity=finding["severity"],
                    finding_type=finding["finding_type"],
                    summary=finding["summary"],
                    details=finding.get("details"),
                )

        all_findings = list(session.scalars(
            select(m.AuditFinding).where(m.AuditFinding.job_id == job_id)
        ).all())

        summary = generate_audit_summary([
            {"severity": f.severity, "finding_type": f.finding_type, "summary": f.summary}
            for f in all_findings
        ])

    print("\n" + "=" * 70)
    print("SELF-AUDIT REPORT")
    print("=" * 70)
    print(summary)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pipeline(intake: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the full SmartHireAI hiring pipeline for a single role.

    intake keys:
        title                    (str, required)
        department               (str)
        seniority                (str)
        location                 (str)
        employment_type          (str)
        hr_notes                 (str)
        pass_threshold           (float, default 6.0)
        platforms                (list[str], default ["Facebook","Instagram","X"])
        shortlist_top_n          (int, default 3)
        min_applications_threshold (int, default 5)
    """
    _ensure_backend()

    print("\n" + "=" * 70)
    print(f"SmartHireAI Pipeline — Starting for role: {intake.get('title', '?')}")
    print("=" * 70)

    _validate_communication_config()

    init_db()
    session_factory = get_session_factory()

    config = HiringPipelineConfig(
        hr_auto_approve_jd=False,
        hr_auto_approve_ads=False,
    )
    pipeline = HiringPipeline(session_factory=session_factory, config=config)

    # Phase 1-3: JD → Ads → Posting (with HR approvals)
    print("\n[Phase 1/5] Generating Job Description and Advertisements...")
    result = pipeline.run_full_pipeline(
        intake,
        candidates=[],  # No candidates yet — ads must go live first
        hr_decision_provider=_hr_jd_approval,
        ad_decision_provider=_hr_ad_approval,
    )

    job_id_str = result.get("job_id")
    correlation_id_str = result.get("correlation_id")
    if not job_id_str:
        print("[ERROR] Pipeline did not return a job_id. Aborting.")
        return result

    job_id = uuid.UUID(job_id_str)
    correlation_id = uuid.UUID(correlation_id_str) if correlation_id_str else uuid.uuid4()
    _print_status_snapshot(session_factory, job_id, label="After phase 1/2 setup")

    print(f"\n[Phase 2/5] Ads are live! Job ID: {job_id}")
    print(f"  Posting platforms : {', '.join(intake.get('platforms', []))}")
    print(f"  Min applications  : {intake.get('min_applications_threshold', 5)}")
    print("  Candidates are now applying via the job form link...")
    job_form_link = result.get("job_form_link")
    if job_form_link:
        print(f"  Job form link     : {job_form_link}")

    # Phase 3: Wait for threshold and process queued applications
    poll_seconds = int(intake.get("poll_seconds", 10))
    max_wait_app_minutes = int(intake.get("max_wait_application_minutes", 60))
    max_wait_interview_minutes = int(intake.get("max_wait_interview_minutes", 60))

    counts = _get_job_counts(session_factory, job_id)
    threshold_met = counts["current"] >= counts["threshold"]
    if not threshold_met:
        threshold_met = _wait_for_application_threshold(
            pipeline,
            session_factory,
            job_id,
            form_link=job_form_link,
            shortlist_top_n=int(intake.get("shortlist_top_n", 3)),
            poll_seconds=poll_seconds,
            max_wait_minutes=max_wait_app_minutes,
        )

    if threshold_met:
        print("\n[Phase 3/5] Threshold met. Running screening + shortlist-1 + interview dispatch...")
        phase3 = pipeline.process_waiting_applications(
            job_id,
            shortlist_top_n=int(intake.get("shortlist_top_n", 3)),
        )
        result["phase3"] = phase3

        processed_count = len(phase3.get("processed", []))
        stage1_count = len(phase3.get("shortlist_stage1_application_ids", []))
        mcq_count = len(phase3.get("prescreen_mcq_dispatch", []))
        interview_dispatch_count = len(phase3.get("interview_dispatch", []))
        print(f"  Screened applications : {processed_count}")
        print(f"  Stage-1 shortlist     : {stage1_count}")
        stage1_ids = phase3.get("shortlist_stage1_application_ids", [])
        if stage1_ids:
            print("  Stage-1 shortlist details:")
            for row in _resolve_application_briefs(session_factory, stage1_ids):
                print(
                    f"    - rank {row['rank']}: {row.get('candidate_email') or 'unknown'}"
                    f" | app={row.get('application_id')} | status={row.get('status') or '-'}"
                )
        if mcq_count:
            print(f"  MCQ links dispatched  : {mcq_count}")
        if interview_dispatch_count:
            print(f"  Interview links sent  : {interview_dispatch_count}")

        details = phase3.get("prescreen_results") or []
        if details:
            print("  Prescreen details:")
            for row in details:
                email = row.get("candidate_email") or "unknown"
                app_id = row.get("application_id") or "-"
                status = row.get("status") or "-"
                hybrid = row.get("hybrid_score")
                hybrid_str = "n/a" if hybrid is None else f"{float(hybrid):.3f}"
                explanation = (row.get("explanation") or "")
                if len(explanation) > 100:
                    explanation = explanation[:97] + "..."
                print(
                    f"    - {email} | app={app_id} | status={status} | hybrid={hybrid_str}"
                    + (f" | note={explanation}" if explanation else "")
                )
        _print_status_snapshot(session_factory, job_id, label="After phase 3 processing")
    else:
        print("\n[Phase 3/5] Threshold not met yet. Leaving job in collection mode.")
        _print_status_snapshot(session_factory, job_id, label="Phase 3 threshold pending")

    # Phase 4: Wait for interview completion when interviews have been dispatched
    dispatched = _count_dispatched_interviews(session_factory, job_id)
    if dispatched > 0:
        print(f"\n[Phase 4/5] {dispatched} interview(s) dispatched. Waiting for completions...")
        interviews_done = _poll_interviews_complete(
            session_factory,
            job_id,
            poll_seconds,
            max_wait_interview_minutes,
        )

        # Phase 5: finalize only after interviews are complete
        if interviews_done:
            print("\n[Phase 5/5] Running final shortlist and sending offer letters...")
            final = pipeline.finalize_after_interviews(
                job_id,
                top_n=int(intake.get("shortlist_top_n", 3)),
            )
            result.update(final)
            stage2_ids = final.get("shortlist_stage2_application_ids", [])
            if stage2_ids:
                print("  Stage-2 shortlist details:")
                for row in _resolve_application_briefs(session_factory, stage2_ids):
                    print(
                        f"    - rank {row['rank']}: {row.get('candidate_email') or 'unknown'}"
                        f" | app={row.get('application_id')} | status={row.get('status') or '-'}"
                    )
            _print_status_snapshot(session_factory, job_id, label="After phase 5 finalize")
        else:
            print("\n[Phase 5/5] Skipped because interviews are still pending.")
            _print_status_snapshot(session_factory, job_id, label="Finalize skipped (pending interviews)")
    else:
        print("\n[Phase 4/5] No interviews dispatched yet.")
        print("[Phase 5/5] Skipped because interview stage has not started.")
        _print_status_snapshot(session_factory, job_id, label="Interview stage not started")

    # Self-audit report
    print("\n[Audit] Running self-audit agent...")
    _run_and_print_audit(session_factory, job_id, correlation_id)

    # Self-learning pass
    print("\n[Learning] Running self-learning agent...")
    learning_report = run_learning_pass(session_factory)
    if learning_report.get("adjustments"):
        print("  Learning agent recommendations:")
        for adj in learning_report["adjustments"]:
            print(f"    {adj['agent']}: {adj['recommendation']}")
    else:
        print("  Not enough feedback data yet for weight adjustments.")

    print("\n" + "=" * 70)
    print("Pipeline run complete.")
    print(f"  Job ID          : {job_id}")
    print(f"  Correlation ID  : {correlation_id}")
    shortlist = result.get("shortlist_stage2_application_ids", [])
    if shortlist:
        print(f"  Final shortlist : {shortlist}")
    print("=" * 70)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SmartHireAI pipeline runner with live status commands")
    parser.add_argument("--status-job-id", help="Print live status for an existing job UUID instead of running a new pipeline")
    parser.add_argument("--watch-status", action="store_true", help="Continuously print status snapshot for --status-job-id")
    parser.add_argument("--status-poll-seconds", type=int, default=10, help="Poll interval for --watch-status")
    parser.add_argument("--status-max-polls", type=int, default=0, help="Optional max polls for watch mode (0 = infinite)")
    args = parser.parse_args()

    if args.status_job_id:
        _ensure_backend()
        init_db()
        session_factory = get_session_factory()
        _status_mode(
            session_factory,
            uuid.UUID(args.status_job_id),
            watch=bool(args.watch_status),
            poll_seconds=max(1, int(args.status_poll_seconds)),
            max_polls=max(0, int(args.status_max_polls)),
        )
        return


    _ensure_backend()
    init_db()
    session_factory = get_session_factory()

    intake = {
        "title": "AI Engineer",
        "department": "R&D",
        "seniority": "Mid-level",
        "location": "Remote",
        "employment_type": "Full-time",
        "hr_notes": "Python, ML, APIs.",
        "pass_threshold": 6.0,
        "platforms": ["Facebook", "Instagram", "X"],
        "shortlist_top_n": 3,
        "min_applications_threshold": 5,
    }

    print("\nStatus commands while pipeline is running:")
    print("  python main_v2.py --status-job-id <JOB_ID>")
    print("  python main_v2.py --status-job-id <JOB_ID> --watch-status --status-poll-seconds 10")

    run_pipeline(intake)


if __name__ == "__main__":
    main()

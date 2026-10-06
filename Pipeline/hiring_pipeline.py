from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

from sqlalchemy import select
from sqlalchemy.orm import Session

from Orchestrator.orchestrator import Orchestrator
from database import models as m
from database.repo import (
    add_jd_version,
    add_audit_finding,
    add_learning_feedback,
    complete_interview,
    create_application,
    create_interview_slot,
    create_job,
    create_offer,
    get_interview_by_token_hash,
    get_or_create_candidate,
    log_hr_conversation,
    log_event,
    mark_interview_link_sent,
    mark_interview_reminder_sent,
    mark_job_needs_review,
    record_ad_post,
    record_prescreen,
    replace_shortlist,
    session_scope,
    set_application_status,
    list_pending_screening_applications,
    mark_interview_no_show,
    withdraw_application,
)
from database.session import get_session_factory, init_db
from Pipeline.communication_adapter import (
    compute_expiry,
    send_interview_link_email,
    send_prescreen_mcq_email,
    send_offer_email,
    send_rejection_email,
)
from Pipeline.audit_service import audit_interview, audit_prescreen
from Pipeline.interview_tokens import hash_interview_token, mint_interview_token
from Offer_Letter.offer_letter import generate_offer_letter
from Pipeline.learning_service import summarize_learning_feedback
from Pipeline.prescreen_service import (
    PRESCREEN_MODEL_ID,
    PRESCREEN_PROMPT_VERSION,
    run_prescreen,
)

logger = logging.getLogger(__name__)


def _normalize_interview_base_url(base_url: str) -> str:
    """
    Ensure interview links always target the browser interview route.
    If the configured URL is host-only (or '/'), default to '/interview'.
    """
    raw = (base_url or "").strip()
    if not raw:
        raw = "http://localhost:8000/interview"

    parsed = urlparse(raw)
    path = (parsed.path or "").rstrip("/")
    if not path or path == "/":
        path = "/interview"

    return urlunparse((parsed.scheme, parsed.netloc, path, "", parsed.query, ""))


@dataclass
class HiringPipelineConfig:
    prescreen_min_hybrid: float = 0.45
    interview_model_version: str = "synthetic-interview-v1"
    base_public_url: str = field(
        default_factory=lambda: os.getenv("INTERVIEW_PUBLIC_BASE_URL", "http://localhost:8000/interview")
    )
    send_reminder: bool = field(
        default_factory=lambda: os.getenv("INTERVIEW_SEND_REMINDER", "").lower() in ("1", "true", "yes")
    )
    hr_auto_approve_jd: bool = field(
        default_factory=lambda: os.getenv("HR_AUTO_APPROVE_JD", "1").lower() in ("1", "true", "yes")
    )
    hr_auto_approve_ads: bool = field(
        default_factory=lambda: os.getenv("HR_AUTO_APPROVE_ADS", "1").lower() in ("1", "true", "yes")
    )
    prescreen_mcq_enabled: bool = field(
        default_factory=lambda: os.getenv("PRESCREEN_MCQ_ENABLED", "0").lower() in ("1", "true", "yes")
    )
    prescreen_mcq_base_url: str = field(
        default_factory=lambda: os.getenv("PRESCREEN_MCQ_BASE_URL", "http://localhost:8000/prescreen-mcq")
    )


class HiringPipeline:
    """
    End-to-end flow: JD → ads → post → threshold gate → prescreen → shortlist-1 → schedule + link
    → interview result → shortlist-2 → offer.
    HR can approve/reject JD, application-volume threshold controls screening start,
    and all transitions are persisted in PostgreSQL with idempotency and audit events.
    """

    def __init__(self, session_factory: Optional[Callable[[], Session]] = None, config: Optional[HiringPipelineConfig] = None):
        self.session_factory = session_factory or get_session_factory()
        self.config = config or HiringPipelineConfig()

    def ensure_schema(self) -> None:
        init_db()

    def run_jd_through_post(
        self,
        intake: Dict[str, Any],
        *,
        correlation_id: uuid.UUID,
        job_id: uuid.UUID,
        hr_decision_provider: Optional[Callable[[Dict[str, Any]], Tuple[bool, str]]] = None,
        ad_decision_provider: Optional[Callable[[Dict[str, Any]], Tuple[bool, str, Optional[List[dict]]]]] = None,
        ad_text_decision_provider: Optional[Callable[[Dict[str, Any]], Tuple[bool, str, Optional[Dict[str, Any]]]]] = None,
        ad_image_script_decision_provider: Optional[Callable[[Dict[str, Any]], Tuple[bool, str, Optional[str]]]] = None,
        ad_image_decision_provider: Optional[Callable[[Dict[str, Any]], Tuple[bool, str]]] = None,
    ) -> Dict[str, Any]:
        thread_id = str(uuid.uuid4())
        orchestrator = Orchestrator()
        initials: Dict[str, Any] = {
            "title": intake["title"],
            "department": intake.get("department"),
            "seniority": intake.get("seniority"),
            "location": intake.get("location"),
            "employment_type": intake.get("employment_type", "Full-time"),
            "hr_notes": intake.get("hr_notes", ""),
            "pass_threshold": float(intake.get("pass_threshold", 6.0)),
            "candidates": [],
            "judgements": {},
            "status": "running",
            "thread_id": thread_id,
            "job_id": str(job_id),
            "iteration": 0,
            "hr_approved": None,
            "orchestration_status": None,
            "platforms": intake.get("platforms", ["Facebook", "Instagram", "X"]),
            "skip_hr_wait": False,
            "require_ad_approval": True,
            "ad_approved": False,
        }
        try:
            result = orchestrator.run(initials, thread_id)
            while True:
                if result.get("status") == "await_hr":
                    jd_preview = (result.get("winner") or {}).get("jd_text") or result.get("final_jd") or ""
                    decision_fn = hr_decision_provider or self._default_hr_decision_provider
                    approved, feedback = decision_fn(result)
                    with session_scope(self.session_factory) as session:
                        log_hr_conversation(
                            session,
                            correlation_id=correlation_id,
                            job_id=job_id,
                            speaker="agent",
                            stage="jd_review",
                            message="JD generated. Awaiting HR approval.",
                            meta_json={"preview_chars": len(jd_preview)},
                        )
                        log_hr_conversation(
                            session,
                            correlation_id=correlation_id,
                            job_id=job_id,
                            speaker="hr",
                            stage="jd_review",
                            message=feedback or ("approved" if approved else "rejected"),
                            meta_json={"approved": approved},
                        )
                    if approved:
                        result["hr_approved"] = True
                    else:
                        # Reset to None so decision_router pauses at await_hr after
                        # regenerating the JD, rather than immediately re-triggering revise
                        result["hr_approved"] = None
                    if feedback:
                        result["review_feedback"] = feedback
                    result = orchestrator.run(result, thread_id)
                    continue

                if result.get("status") == "await_ad_hr":
                    ad_fn = ad_decision_provider or self._default_ad_decision_provider
                    approved, feedback, approved_ads = ad_fn(result)
                    with session_scope(self.session_factory) as session:
                        log_hr_conversation(
                            session,
                            correlation_id=correlation_id,
                            job_id=job_id,
                            speaker="agent",
                            stage="ad_review",
                            message="Ads generated. Awaiting HR approval.",
                            meta_json={"ad_count": len(result.get("combined_ads") or [])},
                        )
                        log_hr_conversation(
                            session,
                            correlation_id=correlation_id,
                            job_id=job_id,
                            speaker="hr",
                            stage="ad_review",
                            message=feedback or ("approved" if approved else "rejected"),
                            meta_json={"approved": approved},
                        )
                    if not approved:
                        raise RuntimeError("Ad pack rejected by HR.")
                    result["ad_approved"] = True
                    result["orchestration_status"] = "AD_Generation_Agent"
                    if approved_ads:
                        result["approved_ads"] = approved_ads
                    result = orchestrator.run(result, thread_id)
                    continue

                # ── Phase-based ad review pauses ─────────────────────────────
                if result.get("orchestration_status") == "AD_TEXT_AWAIT_HR":
                    text_fn = ad_text_decision_provider or self._default_ad_text_decision_provider
                    approved, feedback, approved_texts = text_fn(result)
                    with session_scope(self.session_factory) as session:
                        log_hr_conversation(
                            session,
                            correlation_id=correlation_id,
                            job_id=job_id,
                            speaker="hr",
                            stage="ad_text_review",
                            message=feedback or ("approved" if approved else "rejected"),
                            meta_json={"approved": approved},
                        )
                    if approved and approved_texts:
                        result["approved_texts"] = approved_texts
                        result["ad_text_feedback"] = None
                    else:
                        result["approved_texts"] = None
                        result["ad_text_feedback"] = feedback or "Rejected — please revise."
                        result["text_variants"] = None  # force full regeneration
                    result = orchestrator.run(result, thread_id)
                    continue

                if result.get("orchestration_status") == "AD_IMAGE_SCRIPT_AWAIT_HR":
                    script_fn = ad_image_script_decision_provider or self._default_ad_image_script_decision_provider
                    approved, feedback, approved_script = script_fn(result)
                    with session_scope(self.session_factory) as session:
                        log_hr_conversation(
                            session,
                            correlation_id=correlation_id,
                            job_id=job_id,
                            speaker="hr",
                            stage="ad_image_script_review",
                            message=feedback or ("approved" if approved else "rejected"),
                            meta_json={"approved": approved},
                        )
                    if approved and approved_script:
                        result["approved_image_script"] = approved_script
                        result["image_script_feedback"] = None
                    else:
                        result["approved_image_script"] = None
                        result["image_script_feedback"] = feedback or "Rejected — please revise."
                        result["image_script_variants"] = None  # force regeneration
                    result = orchestrator.run(result, thread_id)
                    continue

                if result.get("orchestration_status") == "AD_IMAGE_AWAIT_HR":
                    image_fn = ad_image_decision_provider or self._default_ad_image_decision_provider
                    approved, feedback = image_fn(result)
                    with session_scope(self.session_factory) as session:
                        log_hr_conversation(
                            session,
                            correlation_id=correlation_id,
                            job_id=job_id,
                            speaker="hr",
                            stage="ad_image_review",
                            message=feedback or ("approved" if approved else "rejected"),
                            meta_json={"approved": approved},
                        )
                    if approved:
                        result["ad_image_approved"] = True
                        result["image_feedback"] = None
                    else:
                        result["ad_image_approved"] = None
                        result["image_feedback"] = feedback or "Rejected — please regenerate."
                        result["generated_image_url"] = None  # force regeneration
                    result = orchestrator.run(result, thread_id)
                    continue

                break
        except Exception as e:
            logger.exception("Orchestrator failed")
            with session_scope(self.session_factory) as session:
                job = session.get(m.Job, job_id)
                if job:
                    mark_job_needs_review(session, job, str(e))
            raise
        return result

    def _default_hr_decision_provider(self, _: Dict[str, Any]) -> Tuple[bool, str]:
        if self.config.hr_auto_approve_jd:
            return True, "Auto-approved by pipeline policy."
        return False, "Auto-rejected by pipeline policy."

    def _default_ad_decision_provider(self, state: Dict[str, Any]) -> Tuple[bool, str, Optional[List[dict]]]:
        if not self.config.hr_auto_approve_ads:
            return False, "Auto-rejected ads by pipeline policy.", None
        # Prefer approved_ads already in state — hr_approval_node sets these with generated_image_path.
        # Rebuilding from combined_ads would strip the image paths.
        existing = state.get("approved_ads")
        if existing:
            ads = [ad.model_dump() if hasattr(ad, "model_dump") else dict(ad) for ad in existing]
            return True, "Ads auto-approved by pipeline policy.", ads
        combined_ads = state.get("combined_ads") or []
        approved: Dict[str, dict] = {}
        for ad in combined_ads:
            platform = ad.platform if hasattr(ad, "platform") else ad.get("platform")
            if platform in approved:
                continue
            approved[platform] = ad.model_dump() if hasattr(ad, "model_dump") else dict(ad)
        return True, "Ads auto-approved by pipeline policy.", list(approved.values())

    def _default_ad_text_decision_provider(
        self, state: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        if not self.config.hr_auto_approve_ads:
            return False, "Auto-rejected ad texts by pipeline policy.", None
        text_variants = state.get("text_variants") or {}
        # Auto-select variant A for every platform
        approved_texts = {
            platform: {**variants["A"], "_selected_variant": "A"}
            for platform, variants in text_variants.items()
            if "A" in variants
        }
        return True, "Ad texts auto-approved by pipeline policy.", approved_texts

    def _default_ad_image_script_decision_provider(
        self, state: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[str]]:
        if not self.config.hr_auto_approve_ads:
            return False, "Auto-rejected image script by pipeline policy.", None
        variants = state.get("image_script_variants") or {}
        script = variants.get("professional") or next(iter(variants.values()), None)
        return True, "Image script auto-approved by pipeline policy.", script

    def _default_ad_image_decision_provider(
        self, state: Dict[str, Any]
    ) -> Tuple[bool, str]:
        if not self.config.hr_auto_approve_ads:
            return False, "Auto-rejected ad image by pipeline policy."
        return True, "Ad image auto-approved by pipeline policy."

    def persist_posting_phase(
        self,
        session: Session,
        job: m.Job,
        jd_version: m.JDVersion,
        result: Dict[str, Any],
        correlation_id: uuid.UUID,
    ) -> None:
        log_event(
            session,
            correlation_id=correlation_id,
            agent_name="persist",
            entity_type="job",
            entity_id=str(job.id),
            from_state=job.status.value,
            to_state=m.JobStatus.posted.value,
            payload={"jd_version_id": str(jd_version.id)},
            idempotency_key=f"job-{job.id}-persist-post",
        )
        job.status = m.JobStatus.posted
        posting_urls = result.get("posting_urls") or {}
        if not isinstance(posting_urls, dict):
            posting_urls = {}
        for platform, url in posting_urls.items():
            record_ad_post(
                session,
                job,
                str(platform),
                url=str(url) if url else None,
                external_post_id=None,
                payload={"source": "ad_posting_v2"},
                idempotency_key=f"post-{job.id}-{platform}",
            )
        job.status = m.JobStatus.collecting

    def _run_prescreen(
        self,
        session: Session,
        *,
        app: m.Application,
        jd_text: str,
        correlation_id: uuid.UUID,
        resume_path: str,
    ) -> m.Application:
        idem_ps = f"prescreen-{app.id}-{PRESCREEN_PROMPT_VERSION}"
        try:
            out = run_prescreen(jd_text, resume_path)
        except Exception as e:
            logger.exception("Prescreen failed for %s", app.id)
            set_application_status(session, app, m.ApplicationStatus.needs_review)
            log_event(
                session,
                correlation_id=correlation_id,
                agent_name="prescreen",
                entity_type="application",
                entity_id=str(app.id),
                from_state=m.ApplicationStatus.submitted.value,
                to_state=m.ApplicationStatus.needs_review.value,
                payload={"error": str(e)},
                idempotency_key=f"{idem_ps}-error",
            )
            return app

        scores = out["scores"]
        for finding in audit_prescreen(scores):
            add_audit_finding(
                session,
                correlation_id=correlation_id,
                job_id=app.job_id,
                application_id=app.id,
                stage="prescreen",
                severity=finding["severity"],
                finding_type=finding["finding_type"],
                summary=finding["summary"],
                details=finding.get("details"),
            )
        hybrid = float(scores.get("hybrid_score", 0.0))
        passed = hybrid >= self.config.prescreen_min_hybrid
        record_prescreen(
            session,
            app,
            model_id=PRESCREEN_MODEL_ID,
            prompt_version=PRESCREEN_PROMPT_VERSION,
            scores=scores,
            passed=passed,
            idempotency_key=idem_ps,
        )
        new_status = m.ApplicationStatus.prescreen_passed if passed else m.ApplicationStatus.prescreen_failed
        set_application_status(session, app, new_status)
        log_event(
            session,
            correlation_id=correlation_id,
            agent_name="prescreen",
            entity_type="application",
            entity_id=str(app.id),
            from_state=m.ApplicationStatus.submitted.value,
            to_state=new_status.value,
            payload={"hybrid_score": hybrid},
            idempotency_key=f"{idem_ps}-status",
        )
        return app

    def _schedule_and_send_interview(
        self,
        session: Session,
        *,
        app: m.Application,
        correlation_id: uuid.UUID,
    ) -> Dict[str, Any]:
        candidate = app.candidate
        candidate_name = (candidate.name if candidate else None) or "Candidate"
        candidate_email = (candidate.email if candidate else None) or "candidate@example.com"

        raw_token, token_hash = mint_interview_token()
        exp = compute_expiry()
        sched_key = f"sched-{app.id}"
        interview, created = create_interview_slot(
            session,
            app,
            token_hash=token_hash,
            expires_at=exp,
            scheduled_at=datetime.now(timezone.utc),
            timezone_name=intake_tz(),
            idempotency_schedule_key=sched_key,
        )

        if not created:
            return {
                "application_id": str(app.id),
                "interview_token": None,
                "interview_url": None,
                "status": app.status.value,
                "note": "interview_already_scheduled",
            }

        set_application_status(session, app, m.ApplicationStatus.interview_scheduled)
        base_url = _normalize_interview_base_url(self.config.base_public_url)
        separator = "&" if "?" in base_url else "?"
        interview_url = f"{base_url}{separator}token={raw_token}"
        _MAX_EMAIL_ATTEMPTS = 3
        sent = False
        for attempt in range(1, _MAX_EMAIL_ATTEMPTS + 1):
            sent = send_interview_link_email(
                candidate_name=candidate_name,
                candidate_email=candidate_email,
                interview_url=interview_url,
                expires_at=exp,
            )
            if sent:
                break
            if attempt < _MAX_EMAIL_ATTEMPTS:
                logger.warning(
                    "Interview link email attempt %d/%d failed for %s — retrying in 5s",
                    attempt, _MAX_EMAIL_ATTEMPTS, candidate_email,
                )
                time.sleep(5)
            else:
                logger.error(
                    "Interview link email failed after %d attempts for %s — candidate marked as email_failed",
                    _MAX_EMAIL_ATTEMPTS, candidate_email,
                )

        if sent:
            mark_interview_link_sent(session, interview)
            set_application_status(session, app, m.ApplicationStatus.interview_link_sent)
            log_event(
                session,
                correlation_id=correlation_id,
                agent_name="communication_scheduling",
                entity_type="interview",
                entity_id=str(interview.id),
                from_state=m.InterviewStatus.scheduled.value,
                to_state=m.InterviewStatus.link_sent.value,
                payload={"expires_at": exp.isoformat()},
                idempotency_key=f"link-{interview.id}",
            )
            log_hr_conversation(
                session,
                correlation_id=correlation_id,
                job_id=app.job_id,
                application_id=app.id,
                speaker="agent",
                stage="candidate_communication",
                message=f"Interview link sent to {candidate_email}",
                meta_json={"type": "interview_link", "expires_at": exp.isoformat()},
            )
        else:
            set_application_status(session, app, m.ApplicationStatus.not_selected)
            log_hr_conversation(
                session,
                correlation_id=correlation_id,
                job_id=app.job_id,
                application_id=app.id,
                speaker="agent",
                stage="candidate_communication",
                message=f"Interview link send failed after {_MAX_EMAIL_ATTEMPTS} attempts for {candidate_email} — candidate automatically rejected",
                meta_json={
                    "type": "interview_link_failed",
                    "expires_at": exp.isoformat(),
                    "attempts": _MAX_EMAIL_ATTEMPTS,
                    "reason": "Email delivery failed after 3 attempts",
                },
            )

        if self.config.send_reminder and sent:
            send_interview_link_email(
                candidate_name=candidate_name,
                candidate_email=candidate_email,
                interview_url=interview_url,
                expires_at=exp,
                reminder_note=True,
            )
            mark_interview_reminder_sent(session, interview)

        result: Dict[str, Any] = {
            "application_id": str(app.id),
            "interview_token": raw_token if sent else None,
            "interview_url": interview_url if sent else None,
            "status": app.status.value,
            "email_sent": bool(sent),
        }
        if not sent:
            result["email_failure_reason"] = "Email delivery failed after 3 attempts — candidate automatically rejected"
        return result

    def _build_prescreen_mcq_url(self, app: m.Application) -> str:
        base_url = (self.config.prescreen_mcq_base_url or "").strip() or "http://localhost:8000/prescreen-mcq"
        sep = "&" if "?" in base_url else "?"
        return f"{base_url}{sep}application_id={app.id}"

    def _send_prescreen_mcq_and_log(
        self,
        session: Session,
        *,
        app: m.Application,
        correlation_id: uuid.UUID,
    ) -> Dict[str, Any]:
        candidate = app.candidate
        candidate_name = (candidate.name if candidate else None) or "Candidate"
        candidate_email = (candidate.email if candidate else None) or "candidate@example.com"
        mcq_url = self._build_prescreen_mcq_url(app)
        sent = send_prescreen_mcq_email(
            candidate_name=candidate_name,
            candidate_email=candidate_email,
            mcq_url=mcq_url,
        )

        log_event(
            session,
            correlation_id=correlation_id,
            agent_name="mcq_prescreen",
            entity_type="application",
            entity_id=str(app.id),
            from_state=app.status.value if app.status else None,
            to_state="mcq_link_sent",
            payload={"mcq_url": mcq_url, "email_sent": bool(sent)},
            idempotency_key=f"mcq-link-{app.id}",
        )
        log_hr_conversation(
            session,
            correlation_id=correlation_id,
            job_id=app.job_id,
            application_id=app.id,
            speaker="agent",
            stage="candidate_communication",
            message=(
                f"MCQ prescreen link sent to {candidate_email}"
                if sent
                else f"MCQ prescreen link send failed for {candidate_email}"
            ),
            meta_json={"type": "mcq_link", "mcq_url": mcq_url, "email_sent": bool(sent)},
        )

        return {
            "application_id": str(app.id),
            "prescreen_mcq_url": mcq_url,
            "email_sent": bool(sent),
        }

    def _prescreen_shortlist_and_dispatch(
        self,
        session: Session,
        *,
        job: m.Job,
        correlation_id: uuid.UUID,
        shortlist_top_n: int,
    ) -> Dict[str, Any]:
        apps = list(
            session.scalars(
                select(m.Application).where(
                    m.Application.job_id == job.id,
                    m.Application.status == m.ApplicationStatus.prescreen_passed,
                )
            ).all()
        )

        def prescreen_score(a: m.Application) -> float:
            if not a.prescreens:
                return 0.0
            latest = max(a.prescreens, key=lambda r: r.created_at or datetime.min.replace(tzinfo=timezone.utc))
            return float((latest.scores or {}).get("hybrid_score", 0.0))

        ranked = sorted(apps, key=prescreen_score, reverse=True)
        top_ids = [a.id for a in ranked[:max(1, int(shortlist_top_n))]]
        replace_shortlist(session, job.id, top_ids)
        log_event(
            session,
            correlation_id=correlation_id,
            agent_name="shortlist_stage1",
            entity_type="job",
            entity_id=str(job.id),
            from_state=job.status.value,
            to_state="prescreen_shortlisted",
            payload={"application_ids": [str(x) for x in top_ids]},
            idempotency_key=f"shortlist-stage1-{job.id}",
        )

        dispatches: List[Dict[str, Any]] = []
        mcq_dispatches: List[Dict[str, Any]] = []
        for app in ranked[:max(1, int(shortlist_top_n))]:
            if self.config.prescreen_mcq_enabled:
                mcq_dispatches.append(
                    self._send_prescreen_mcq_and_log(
                        session,
                        app=app,
                        correlation_id=correlation_id,
                    )
                )
            else:
                dispatches.append(
                    self._schedule_and_send_interview(
                        session,
                        app=app,
                        correlation_id=correlation_id,
                    )
                )

        return {
            "shortlist_stage1_application_ids": [str(x) for x in top_ids],
            "interview_dispatch": dispatches,
            "prescreen_mcq_dispatch": mcq_dispatches,
        }

    def complete_prescreen_mcq(
        self,
        application_id: uuid.UUID,
        *,
        passed: bool,
        score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        with session_scope(self.session_factory) as session:
            app = session.get(m.Application, application_id)
            if not app:
                return {"status": "application_not_found", "application_id": str(application_id)}

            corr_id = app.job.correlation_id if app.job else uuid.uuid4()
            payload: Dict[str, Any] = {"passed": bool(passed)}
            if score is not None:
                payload["score"] = float(score)
            if metadata:
                payload["metadata"] = metadata

            log_event(
                session,
                correlation_id=corr_id,
                agent_name="mcq_prescreen",
                entity_type="application",
                entity_id=str(app.id),
                from_state=app.status.value if app.status else None,
                to_state="mcq_passed" if passed else m.ApplicationStatus.not_selected.value,
                payload=payload,
                idempotency_key=f"mcq-complete-{app.id}",
            )

            if not passed:
                set_application_status(session, app, m.ApplicationStatus.not_selected)
                log_hr_conversation(
                    session,
                    correlation_id=corr_id,
                    job_id=app.job_id,
                    application_id=app.id,
                    speaker="agent",
                    stage="mcq_prescreen",
                    message=f"MCQ failed for {app.candidate.email if app.candidate else app.id}",
                    meta_json=payload,
                )
                return {
                    "status": "mcq_failed",
                    "application_id": str(app.id),
                    "interview_dispatch": None,
                }

            dispatch = self._schedule_and_send_interview(
                session,
                app=app,
                correlation_id=corr_id,
            )
            log_hr_conversation(
                session,
                correlation_id=corr_id,
                job_id=app.job_id,
                application_id=app.id,
                speaker="agent",
                stage="mcq_prescreen",
                message=f"MCQ passed. Interview dispatched for {app.candidate.email if app.candidate else app.id}",
                meta_json=payload,
            )
            return {
                "status": "mcq_passed_interview_dispatched",
                "application_id": str(app.id),
                "interview_dispatch": dispatch,
            }

    def ingest_candidate(
        self,
        session: Session,
        job: m.Job,
        jd_version: m.JDVersion,
        correlation_id: uuid.UUID,
        *,
        email: str,
        name: Optional[str],
        resume_path: str,
        phone: Optional[str] = None,
        resume_storage_uri: Optional[str] = None,
    ) -> m.Application:
        effective_resume_uri = resume_storage_uri or resume_path
        cand, _ = get_or_create_candidate(session, email, name, phone, None, None)
        app = create_application(session, job, cand, jd_version, effective_resume_uri)
        log_event(
            session,
            correlation_id=correlation_id,
            agent_name="application",
            entity_type="application",
            entity_id=str(app.id),
            from_state=None,
            to_state=m.ApplicationStatus.submitted.value,
            payload={"candidate_email": email},
            idempotency_key=f"app-submit-{app.id}",
        )
        return self._run_prescreen(
            session,
            app=app,
            jd_text=jd_version.jd_text,
            correlation_id=correlation_id,
            resume_path=resume_path,
        )

    def process_submitted_application(
        self,
        session: Session,
        app: m.Application,
        jd_version: m.JDVersion,
        correlation_id: uuid.UUID,
        *,
        resume_path: str,
    ) -> m.Application:
        return self._run_prescreen(
            session,
            app=app,
            jd_text=jd_version.jd_text,
            correlation_id=correlation_id,
            resume_path=resume_path,
        )

    def register_application_only(
        self,
        session: Session,
        job: m.Job,
        jd_version: m.JDVersion,
        correlation_id: uuid.UUID,
        *,
        email: str,
        name: Optional[str],
        phone: Optional[str] = None,
        resume_storage_uri: Optional[str] = None,
    ) -> m.Application:
        effective_resume_uri = resume_storage_uri
        cand, _ = get_or_create_candidate(session, email, name, phone, None, None)
        app = create_application(session, job, cand, jd_version, effective_resume_uri)
        log_event(
            session,
            correlation_id=correlation_id,
            agent_name="application",
            entity_type="application",
            entity_id=str(app.id),
            from_state=None,
            to_state=m.ApplicationStatus.submitted.value,
            payload={"candidate_email": email, "screening": "deferred"},
            idempotency_key=f"app-submit-{app.id}",
        )
        return app

    def submit_application(
        self,
        *,
        job_id: uuid.UUID,
        email: str,
        resume_path: str,
        name: Optional[str] = None,
        phone: Optional[str] = None,
        shortlist_top_n: int = 3,
        auto_process_when_threshold_met: bool = True,
    ) -> Dict[str, Any]:
        """
        Register one candidate application after ads are live.
        Screening + interview dispatch will auto-start only when
        min_applications_threshold is reached for the job.

        Set auto_process_when_threshold_met=False to only enqueue applications
        and let a separate explicit processing step handle screening/dispatch.
        """
        with session_scope(self.session_factory) as session:
            job = session.get(m.Job, job_id)
            if not job:
                return {"status": "job_not_found", "job_id": str(job_id)}
            jd_version = session.scalars(
                select(m.JDVersion).where(m.JDVersion.job_id == job_id).order_by(m.JDVersion.version.desc())
            ).first()
            if not jd_version:
                return {"status": "jd_not_found", "job_id": str(job_id)}

            app = self.register_application_only(
                session,
                job,
                jd_version,
                job.correlation_id,
                email=email,
                name=name,
                phone=phone,
                resume_storage_uri=resume_path,
            )
            current_count = int(job.current_applications_count or 0)
            min_threshold = int(job.min_applications_threshold or 1)

        if current_count < min_threshold:
            return {
                "status": "waiting_for_min_applications",
                "job_id": str(job_id),
                "application_id": str(app.id),
                "current_applications_count": current_count,
                "min_applications_threshold": min_threshold,
            }

        if not auto_process_when_threshold_met:
            return {
                "status": "threshold_reached_pending_processing",
                "job_id": str(job_id),
                "application_id": str(app.id),
                "current_applications_count": current_count,
                "min_applications_threshold": min_threshold,
            }

        processed = self.process_waiting_applications(
            job_id,
            shortlist_top_n=max(1, int(shortlist_top_n)),
        )
        return {
            "status": "application_submitted_and_processed",
            "job_id": str(job_id),
            "application_id": str(app.id),
            "processing": processed,
        }

    def complete_interview_by_token(
        self,
        raw_token: str,
        *,
        scores: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[uuid.UUID] = None,
        media_uri: Optional[str] = None,
    ) -> bool:
        th = hash_interview_token(raw_token)
        cid = correlation_id or uuid.uuid4()
        with session_scope(self.session_factory) as session:
            inv = get_interview_by_token_hash(session, th)
            if not inv:
                return False
            if inv.expires_at < datetime.now(timezone.utc):
                logger.warning("Interview token expired")
                return False
            app = inv.application
            job = app.job
            if job:
                cid = job.correlation_id
            default_scores = scores or {
                "overall": 0.75,
                "communication": 0.7,
                "technical": 0.8,
                "interview_score": 0.75,
                "notes": "Synthetic completion — replace with real Interview_Agent output",
            }
            default_scores.setdefault("interview_score", float(default_scores.get("overall", 0.0)))
            for finding in audit_interview(default_scores):
                add_audit_finding(
                    session,
                    correlation_id=cid,
                    job_id=app.job_id,
                    application_id=app.id,
                    stage="interview",
                    severity=finding["severity"],
                    finding_type=finding["finding_type"],
                    summary=finding["summary"],
                    details=finding.get("details"),
                )
            res, new_row = complete_interview(
                session,
                inv,
                scores=default_scores,
                model_version=self.config.interview_model_version,
                media_uri=media_uri,
                idempotency_key=f"interview-complete-{inv.id}",
            )
            if new_row:
                set_application_status(session, app, m.ApplicationStatus.interview_completed)
                log_event(
                    session,
                    correlation_id=cid,
                    agent_name="interview_agent",
                    entity_type="interview",
                    entity_id=str(inv.id),
                    from_state=m.InterviewStatus.link_sent.value,
                    to_state=m.InterviewStatus.completed.value,
                    payload={"result_id": str(res.id) if res else None},
                    idempotency_key=f"interview-complete-{inv.id}-evt",
                )
                if app and app.candidate:
                    log_hr_conversation(
                        session,
                        correlation_id=cid,
                        job_id=app.job_id,
                        application_id=app.id,
                        speaker="agent",
                        stage="interview_completion",
                        message=f"Interview completed for {app.candidate.email}",
                        meta_json={"interview_id": str(inv.id)},
                    )
            return True

    def mark_no_show(self, application_id: uuid.UUID) -> None:
        with session_scope(self.session_factory) as session:
            app = session.get(m.Application, application_id)
            if not app:
                return
            invs = list(session.scalars(select(m.Interview).where(m.Interview.application_id == app.id)).all())
            for inv in invs:
                if inv.status != m.InterviewStatus.completed:
                    mark_interview_no_show(session, inv)
            set_application_status(session, app, m.ApplicationStatus.interview_no_show)

    def withdraw(self, application_id: uuid.UUID) -> None:
        with session_scope(self.session_factory) as session:
            app = session.get(m.Application, application_id)
            if app:
                withdraw_application(session, app)

    def process_waiting_applications(self, job_id: uuid.UUID, *, shortlist_top_n: int = 3) -> Dict[str, Any]:
        """
        Starts prescreening/scheduling for applications that were queued while job was below threshold.
        """
        processed: List[str] = []
        prescreen_results: List[Dict[str, Any]] = []
        with session_scope(self.session_factory) as session:
            job = session.get(m.Job, job_id)
            if not job:
                return {"processed": processed, "status": "job_not_found"}
            min_threshold = int(job.min_applications_threshold or 1)
            current_count = int(job.current_applications_count or 0)
            if current_count < min_threshold:
                return {
                    "processed": processed,
                    "status": "waiting_for_min_applications",
                    "current_applications_count": current_count,
                    "min_applications_threshold": min_threshold,
                }
            jd_version = session.scalars(
                select(m.JDVersion).where(m.JDVersion.job_id == job_id).order_by(m.JDVersion.version.desc())
            ).first()
            pending = list_pending_screening_applications(session, job_id)
            for app in pending:
                resume_path = app.resume_storage_uri
                if not resume_path:
                    set_application_status(session, app, m.ApplicationStatus.needs_review)
                    add_audit_finding(
                        session,
                        correlation_id=job.correlation_id,
                        job_id=job.id,
                        application_id=app.id,
                        stage="prescreen",
                        severity="high",
                        finding_type="missing_resume_uri",
                        summary="Application cannot be screened because resume URI/path is missing.",
                        details=None,
                    )
                    prescreen_results.append(
                        {
                            "application_id": str(app.id),
                            "candidate_email": app.candidate.email if app.candidate else None,
                            "status": m.ApplicationStatus.needs_review.value,
                            "hybrid_score": None,
                            "explanation": "missing_resume_uri",
                        }
                    )
                    continue
                self.process_submitted_application(
                    session,
                    app,
                    jd_version,
                    job.correlation_id,
                    resume_path=resume_path,
                )
                processed.append(str(app.id))
                latest_ps = max(app.prescreens, key=lambda r: r.created_at or datetime.min.replace(tzinfo=timezone.utc)) if app.prescreens else None
                latest_scores = (latest_ps.scores or {}) if latest_ps else {}
                explanation = latest_scores.get("explanation")
                if not explanation and app.status == m.ApplicationStatus.needs_review:
                    latest_event = session.execute(
                        select(m.PipelineEvent)
                        .where(
                            m.PipelineEvent.entity_type == "application",
                            m.PipelineEvent.entity_id == str(app.id),
                            m.PipelineEvent.agent_name == "prescreen",
                        )
                        .order_by(m.PipelineEvent.created_at.desc())
                    ).scalars().first()
                    if latest_event and isinstance(latest_event.payload, dict):
                        explanation = latest_event.payload.get("error")
                prescreen_results.append(
                    {
                        "application_id": str(app.id),
                        "candidate_email": app.candidate.email if app.candidate else None,
                        "status": app.status.value if app.status else None,
                        "hybrid_score": latest_scores.get("hybrid_score"),
                        "embedding_score": latest_scores.get("embedding_score"),
                        "explanation": explanation,
                    }
                )
            stage1 = self._prescreen_shortlist_and_dispatch(
                session,
                job=job,
                correlation_id=job.correlation_id,
                shortlist_top_n=shortlist_top_n,
            )
        return {"processed": processed, "status": "processed", "prescreen_results": prescreen_results, **stage1}

    def record_learning_feedback(
        self,
        *,
        correlation_id: str,
        agent_name: str,
        feedback_type: str,
        created_by: str = "hr",
        label: Optional[str] = None,
        notes: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
        application_id: Optional[str] = None,
    ) -> str:
        with session_scope(self.session_factory) as session:
            row = add_learning_feedback(
                session,
                correlation_id=uuid.UUID(correlation_id),
                agent_name=agent_name,
                feedback_type=feedback_type,
                created_by=created_by,
                label=label,
                notes=notes,
                payload=payload,
                job_id=uuid.UUID(job_id) if job_id else None,
                application_id=uuid.UUID(application_id) if application_id else None,
            )
            return str(row.id)

    def learning_summary(self) -> Dict[str, Any]:
        with session_scope(self.session_factory) as session:
            return summarize_learning_feedback(session)

    def shortlist_and_offer_top(
        self,
        job_id: uuid.UUID,
        *,
        top_n: int = 3,
        offer_top: bool = True,
    ) -> List[uuid.UUID]:
        with session_scope(self.session_factory) as session:
            job = session.get(m.Job, job_id)
            if not job:
                return []
            apps = list(
                session.scalars(
                    select(m.Application).where(
                        m.Application.job_id == job_id,
                        m.Application.status == m.ApplicationStatus.interview_completed,
                    )
                ).all()
            )

            def sort_key(a: m.Application) -> float:
                best = 0.0
                for inv in a.interviews:
                    if inv.result and inv.result.scores:
                        best = max(best, float(inv.result.scores.get("overall", 0.0)))
                return best

            ranked = sorted(apps, key=sort_key, reverse=True)
            top_ids = [a.id for a in ranked[:top_n]]
            replace_shortlist(session, job_id, top_ids)
            log_event(
                session,
                correlation_id=job.correlation_id,
                agent_name="shortlist",
                entity_type="job",
                entity_id=str(job_id),
                from_state=None,
                to_state="shortlisted",
                payload={"application_ids": [str(x) for x in top_ids]},
                idempotency_key=f"shortlist-{job_id}",
            )
            if offer_top and top_ids:
                winner = session.get(m.Application, top_ids[0])
                if winner and winner.candidate:
                    payload = generate_offer_letter(
                        job_title=job.title,
                        candidate_name=winner.candidate.name or winner.candidate.email,
                        role=job.title,
                        department=job.department,
                        start_date=os.getenv("OFFER_DEFAULT_START", "TBD"),
                        salary_band=os.getenv("OFFER_DEFAULT_SALARY", "Per discussion"),
                    )
                    create_offer(
                        session,
                        winner,
                        payload,
                        idempotency_key=f"offer-{winner.id}",
                    )
                    offer_evt = log_event(
                        session,
                        correlation_id=job.correlation_id,
                        agent_name="offer_letter",
                        entity_type="application",
                        entity_id=str(winner.id),
                        from_state=m.ApplicationStatus.shortlisted.value,
                        to_state=m.ApplicationStatus.offer_sent.value,
                        payload={"offer_keys": list(payload.keys())},
                        idempotency_key=f"offer-{winner.id}-evt",
                    )
                    if offer_evt:
                        send_offer_email(
                            candidate_name=winner.candidate.name or winner.candidate.email,
                            candidate_email=winner.candidate.email,
                            job_title=job.title,
                        )
                        log_hr_conversation(
                            session,
                            correlation_id=job.correlation_id,
                            job_id=job_id,
                            application_id=winner.id,
                            speaker="agent",
                            stage="candidate_communication",
                            message=f"Offer email sent to {winner.candidate.email}",
                            meta_json={"type": "offer_sent"},
                        )

            for rejected in ranked[max(0, int(top_n)):]:
                if not rejected.candidate:
                    continue
                rej_evt = log_event(
                    session,
                    correlation_id=job.correlation_id,
                    agent_name="candidate_updates",
                    entity_type="application",
                    entity_id=str(rejected.id),
                    from_state=rejected.status.value if rejected.status else None,
                    to_state=m.ApplicationStatus.not_selected.value,
                    payload={"reason": "not_in_top_shortlist"},
                    idempotency_key=f"rejection-email-{rejected.id}",
                )
                if not rej_evt:
                    continue
                send_rejection_email(
                    candidate_name=rejected.candidate.name or rejected.candidate.email,
                    candidate_email=rejected.candidate.email,
                    job_title=job.title,
                )
                log_hr_conversation(
                    session,
                    correlation_id=job.correlation_id,
                    job_id=job_id,
                    application_id=rejected.id,
                    speaker="agent",
                    stage="candidate_communication",
                    message=f"Rejection email sent to {rejected.candidate.email}",
                    meta_json={"type": "rejection_sent"},
                )
            return top_ids

    def finalize_after_interviews(
        self,
        job_id: uuid.UUID,
        *,
        top_n: int = 3,
    ) -> Dict[str, Any]:
        winners = self.shortlist_and_offer_top(job_id, top_n=top_n, offer_top=True)
        return {
            "job_id": str(job_id),
            "shortlist_stage2_application_ids": [str(x) for x in winners],
            "status": "finalized_post_interviews",
        }

    def get_offer_candidates(
        self,
        job_id: uuid.UUID,
        *,
        top_n: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Return ranked shortlisted candidates with interview scores for HR review.
        Shortlists the top N but does NOT send offers or rejections.
        """
        with session_scope(self.session_factory) as session:
            apps = list(
                session.scalars(
                    select(m.Application).where(
                        m.Application.job_id == job_id,
                        m.Application.status == m.ApplicationStatus.interview_completed,
                    )
                ).all()
            )

            def sort_key(a: m.Application) -> float:
                best = 0.0
                for inv in a.interviews:
                    if inv.result and inv.result.scores:
                        best = max(best, float(inv.result.scores.get("overall", 0.0)))
                return best

            ranked = sorted(apps, key=sort_key, reverse=True)
            top_apps = ranked[:max(1, top_n)]
            replace_shortlist(session, job_id, [a.id for a in top_apps])

            candidates = []
            for app in top_apps:
                candidate = app.candidate
                best_score = 0.0
                recommendation = ""
                summary = ""
                for inv in app.interviews:
                    if inv.result and inv.result.scores:
                        score = float(inv.result.scores.get("overall", 0.0))
                        if score > best_score:
                            best_score = score
                            recommendation = (inv.result.scores.get("evaluation") or {}).get("recommendation", "")
                            summary = inv.result.scores.get("notes") or (inv.result.scores.get("evaluation") or {}).get("summary", "")
                display_score = round(best_score * 100) if best_score <= 1.0 else round(best_score)
                candidates.append({
                    "application_id": str(app.id),
                    "candidate_name": (candidate.name if candidate else None) or "Unknown",
                    "candidate_email": (candidate.email if candidate else None) or "",
                    "overall_score": display_score,
                    "recommendation": recommendation,
                    "summary": summary,
                })
            return candidates

    def send_offer_to_candidate(
        self,
        job_id: uuid.UUID,
        application_id: uuid.UUID,
    ) -> Dict[str, Any]:
        """
        Send offer letter to the HR-chosen candidate and rejection emails to others.
        """
        with session_scope(self.session_factory) as session:
            job = session.get(m.Job, job_id)
            if not job:
                return {"status": "job_not_found"}

            winner = session.get(m.Application, application_id)
            if not winner or not winner.candidate:
                return {"status": "application_not_found"}

            # All other completed/shortlisted apps for this job get rejected
            other_apps = list(
                session.scalars(
                    select(m.Application).where(
                        m.Application.job_id == job_id,
                        m.Application.id != application_id,
                        m.Application.status.in_([
                            m.ApplicationStatus.interview_completed,
                            m.ApplicationStatus.shortlisted,
                        ]),
                    )
                ).all()
            )

            payload = generate_offer_letter(
                job_title=job.title,
                candidate_name=winner.candidate.name or winner.candidate.email,
                role=job.title,
                department=job.department,
                start_date=os.getenv("OFFER_DEFAULT_START", "TBD"),
                salary_band=os.getenv("OFFER_DEFAULT_SALARY", "Per discussion"),
            )
            create_offer(
                session,
                winner,
                payload,
                idempotency_key=f"offer-{winner.id}",
            )
            offer_evt = log_event(
                session,
                correlation_id=job.correlation_id,
                agent_name="offer_letter",
                entity_type="application",
                entity_id=str(winner.id),
                from_state=m.ApplicationStatus.shortlisted.value,
                to_state=m.ApplicationStatus.offer_sent.value,
                payload={"offer_keys": list(payload.keys()), "hr_approved": True},
                idempotency_key=f"offer-{winner.id}-evt",
            )
            if offer_evt:
                send_offer_email(
                    candidate_name=winner.candidate.name or winner.candidate.email,
                    candidate_email=winner.candidate.email,
                    job_title=job.title,
                )
                log_hr_conversation(
                    session,
                    correlation_id=job.correlation_id,
                    job_id=job_id,
                    application_id=winner.id,
                    speaker="agent",
                    stage="candidate_communication",
                    message=f"Offer email sent to {winner.candidate.email} (HR-approved)",
                    meta_json={"type": "offer_sent", "hr_approved": True},
                )

            for rejected in other_apps:
                if not rejected.candidate:
                    continue
                rej_evt = log_event(
                    session,
                    correlation_id=job.correlation_id,
                    agent_name="candidate_updates",
                    entity_type="application",
                    entity_id=str(rejected.id),
                    from_state=rejected.status.value if rejected.status else None,
                    to_state=m.ApplicationStatus.not_selected.value,
                    payload={"reason": "not_selected_by_hr"},
                    idempotency_key=f"rejection-email-{rejected.id}",
                )
                if not rej_evt:
                    continue
                send_rejection_email(
                    candidate_name=rejected.candidate.name or rejected.candidate.email,
                    candidate_email=rejected.candidate.email,
                    job_title=job.title,
                )

            return {
                "job_id": str(job_id),
                "winner_application_id": str(winner.id),
                "winner_email": winner.candidate.email,
                "status": "offer_sent",
            }

    def run_full_pipeline(
        self,
        intake: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        *,
        hr_decision_provider: Optional[Callable[[Dict[str, Any]], Tuple[bool, str]]] = None,
        ad_decision_provider: Optional[Callable[[Dict[str, Any]], Tuple[bool, str, Optional[List[dict]]]]] = None,
        ad_text_decision_provider: Optional[Callable[[Dict[str, Any]], Tuple[bool, str, Optional[Dict[str, Any]]]]] = None,
        ad_image_script_decision_provider: Optional[Callable[[Dict[str, Any]], Tuple[bool, str, Optional[str]]]] = None,
        ad_image_decision_provider: Optional[Callable[[Dict[str, Any]], Tuple[bool, str]]] = None,
    ) -> Dict[str, Any]:
        """
        intake: title, department, seniority, location, platforms, pass_threshold, hr_notes, ...
        candidates: list of {email, name?, resume_path, phone?}
        """
        self.ensure_schema()
        correlation_id = uuid.uuid4()
        thread_id = str(uuid.uuid4())

        with session_scope(self.session_factory) as session:
            job = create_job(
                session,
                title=intake["title"],
                correlation_id=correlation_id,
                thread_id=thread_id,
                department=intake.get("department"),
                pass_threshold=float(intake.get("pass_threshold", 6.0)),
                min_applications_threshold=int(intake.get("min_applications_threshold", 1)),
            )
            job_id = job.id
            log_event(
                session,
                correlation_id=correlation_id,
                agent_name="pipeline",
                entity_type="job",
                entity_id=str(job_id),
                from_state=None,
                to_state=m.JobStatus.draft.value,
                payload={"title": intake["title"]},
                idempotency_key=f"job-create-{job_id}",
            )
            log_hr_conversation(
                session,
                correlation_id=correlation_id,
                job_id=job_id,
                speaker="hr",
                stage="intake",
                message="HR intake submitted",
                meta_json={
                    "title": intake.get("title"),
                    "department": intake.get("department"),
                    "location": intake.get("location"),
                    "pass_threshold": intake.get("pass_threshold"),
                    "min_applications_threshold": intake.get("min_applications_threshold"),
                },
            )

        result = self.run_jd_through_post(
            intake,
            correlation_id=correlation_id,
            job_id=job_id,
            hr_decision_provider=hr_decision_provider,
            ad_decision_provider=ad_decision_provider,
            ad_text_decision_provider=ad_text_decision_provider,
            ad_image_script_decision_provider=ad_image_script_decision_provider,
            ad_image_decision_provider=ad_image_decision_provider,
        )

        final_jd = result.get("final_jd") or (result.get("winner") or {}).get("jd_text")
        if not final_jd:
            with session_scope(self.session_factory) as session:
                j = session.get(m.Job, job_id)
                if j:
                    mark_job_needs_review(session, j, "Missing final_jd from orchestrator")
            raise RuntimeError("Pipeline stopped: no final JD produced")

        with session_scope(self.session_factory) as session:
            job = session.get(m.Job, job_id)
            jd_version = add_jd_version(
                session,
                job,
                final_jd,
                requirements_snapshot={"essentials": _essentials_dump(result)},
            )
            job.status = m.JobStatus.jd_ready
            self.persist_posting_phase(session, job, jd_version, result, correlation_id)

        applications_out: List[Dict[str, Any]] = []
        with session_scope(self.session_factory) as session:
            job = session.get(m.Job, job_id)
            jd_version = session.scalars(
                select(m.JDVersion).where(m.JDVersion.job_id == job_id).order_by(m.JDVersion.version.desc())
            ).first()
            min_threshold = int(job.min_applications_threshold or 1)
            current_count_before = int(job.current_applications_count or 0)
            projected_count = current_count_before + len(candidates)
            defer_screening = projected_count < min_threshold
            for c in candidates:
                if defer_screening:
                    app = self.register_application_only(
                        session,
                        job,
                        jd_version,
                        correlation_id,
                        email=c["email"],
                        name=c.get("name"),
                        phone=c.get("phone"),
                        resume_storage_uri=c.get("resume_storage_uri") or c.get("resume_path"),
                    )
                else:
                    app = self.ingest_candidate(
                        session,
                        job,
                        jd_version,
                        correlation_id,
                        email=c["email"],
                        name=c.get("name"),
                        resume_path=c["resume_path"],
                        phone=c.get("phone"),
                        resume_storage_uri=c.get("resume_storage_uri") or c.get("resume_path"),
                    )
                applications_out.append(
                    {
                        "application_id": str(app.id),
                        "status": app.status.value,
                    }
                )
            current_count = int(job.current_applications_count or 0)
            if defer_screening:
                log_event(
                    session,
                    correlation_id=correlation_id,
                    agent_name="pipeline",
                    entity_type="job",
                    entity_id=str(job_id),
                    from_state=job.status.value,
                    to_state="waiting_for_min_applications",
                    payload={"current_count": current_count, "min_threshold": min_threshold},
                    idempotency_key=f"waiting-threshold-{job_id}-{current_count}",
                )
                return {
                    "job_id": str(job_id),
                    "correlation_id": str(correlation_id),
                    "jd_version_id": str(jd_version.id) if jd_version else None,
                    "job_form_link": result.get("job_form_link"),
                    "posting_urls": result.get("posting_urls") or {},
                    "candidates": applications_out,
                    "status": "waiting_for_min_applications",
                    "current_applications_count": current_count,
                    "min_applications_threshold": min_threshold,
                    "shortlist_application_ids": [],
                }

            stage1 = self._prescreen_shortlist_and_dispatch(
                session,
                job=job,
                correlation_id=correlation_id,
                shortlist_top_n=int(intake.get("shortlist_top_n", 3)),
            )

        return {
            "job_id": str(job_id),
            "correlation_id": str(correlation_id),
            "jd_version_id": str(jd_version.id) if jd_version else None,
            "job_form_link": result.get("job_form_link"),
            "posting_urls": result.get("posting_urls") or {},
            "candidates": applications_out,
            "shortlist_stage1_application_ids": stage1["shortlist_stage1_application_ids"],
            "interview_dispatch": stage1["interview_dispatch"],
            "status": "awaiting_interviews",
        }


def intake_tz() -> str:
    return os.getenv("DEFAULT_TZ", "UTC")


def _essentials_dump(result: Dict[str, Any]) -> Any:
    ess = result.get("essentials")
    if hasattr(ess, "model_dump"):
        return ess.model_dump()
    return ess

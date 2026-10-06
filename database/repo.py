from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from database.models import (
    AdPost,
    Application,
    ApplicationStatus,
    Candidate,
    Interview,
    InterviewResult,
    InterviewStatus,
    JDVersion,
    Job,
    JobStatus,
    HRConversation,
    AuditFinding,
    LearningFeedback,
    Offer,
    OfferStatus,
    OrgSettings,
    PipelineEvent,
    PrescreeningRun,
    ShortlistEntry,
    _ORG_SETTINGS_ID,
)

logger = logging.getLogger(__name__)


@contextmanager
def session_scope(session_factory: Callable[[], Session]):
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def log_event(
    session: Session,
    *,
    correlation_id: uuid.UUID,
    agent_name: str,
    entity_type: str,
    entity_id: Optional[str],
    from_state: Optional[str],
    to_state: Optional[str],
    payload: Optional[dict],
    idempotency_key: str,
) -> bool:
    dup = session.execute(
        select(PipelineEvent.id).where(
            PipelineEvent.correlation_id == correlation_id,
            PipelineEvent.idempotency_key == idempotency_key,
        )
    ).first()
    if dup:
        return False
    session.add(
        PipelineEvent(
            correlation_id=correlation_id,
            entity_type=entity_type,
            entity_id=entity_id,
            from_state=from_state,
            to_state=to_state,
            agent_name=agent_name,
            payload=payload,
            idempotency_key=idempotency_key,
        )
    )
    session.flush()
    return True


def create_job(
    session: Session,
    *,
    title: str,
    correlation_id: uuid.UUID,
    thread_id: Optional[str],
    department: Optional[str],
    pass_threshold: float,
    min_applications_threshold: int = 1,
) -> Job:
    job = Job(
        title=title,
        department=department,
        correlation_id=correlation_id,
        thread_id=thread_id,
        pass_threshold=pass_threshold,
        min_applications_threshold=max(1, int(min_applications_threshold)),
        current_applications_count=0,
        status=JobStatus.draft,
    )
    session.add(job)
    session.flush()
    return job


def add_jd_version(
    session: Session,
    job: Job,
    jd_text: str,
    requirements_snapshot: Optional[dict],
) -> JDVersion:
    q = select(JDVersion).where(JDVersion.job_id == job.id)
    max_v = session.execute(q).scalars().all()
    next_v = max((v.version for v in max_v), default=-1) + 1
    row = JDVersion(
        job_id=job.id,
        version=next_v,
        jd_text=jd_text,
        requirements_snapshot=requirements_snapshot or {},
    )
    session.add(row)
    session.flush()
    return row


def record_ad_post(
    session: Session,
    job: Job,
    platform: str,
    *,
    url: Optional[str],
    external_post_id: Optional[str],
    payload: Optional[dict],
    idempotency_key: str,
) -> Optional[AdPost]:
    existing = session.execute(
        select(AdPost).where(
            AdPost.job_id == job.id,
            AdPost.platform == platform,
            AdPost.idempotency_key == idempotency_key,
        )
    ).scalar_one_or_none()
    if existing:
        return existing
    row = AdPost(
        job_id=job.id,
        platform=platform,
        url=url,
        external_post_id=external_post_id,
        payload=payload,
        idempotency_key=idempotency_key,
    )
    session.add(row)
    session.flush()
    return row


def get_or_create_candidate(
    session: Session,
    email: str,
    name: Optional[str],
    phone: Optional[str],
    pii: Optional[dict],
    retention_until: Optional[datetime],
) -> Tuple[Candidate, bool]:
    existing = session.execute(select(Candidate).where(Candidate.email == email)).scalar_one_or_none()
    if existing:
        if name and not existing.name:
            existing.name = name
        if phone and not existing.phone:
            existing.phone = phone
        if pii:
            existing.pii = {**(existing.pii or {}), **pii}
        if retention_until:
            existing.retention_until = retention_until
        return existing, False
    c = Candidate(email=email, name=name, phone=phone, pii=pii or {}, retention_until=retention_until)
    session.add(c)
    session.flush()
    return c, True


def next_reapplication_seq(session: Session, job_id: uuid.UUID, candidate_id: uuid.UUID) -> int:
    q = select(Application.reapplication_seq).where(
        Application.job_id == job_id,
        Application.candidate_id == candidate_id,
    )
    rows = session.execute(q).scalars().all()
    return max(rows, default=-1) + 1


def create_application(
    session: Session,
    job: Job,
    candidate: Candidate,
    jd_version: Optional[JDVersion],
    resume_storage_uri: Optional[str],
    reapplication_seq: Optional[int] = None,
) -> Application:
    seq = reapplication_seq if reapplication_seq is not None else next_reapplication_seq(session, job.id, candidate.id)
    app = Application(
        job_id=job.id,
        candidate_id=candidate.id,
        jd_version_id=jd_version.id if jd_version else None,
        reapplication_seq=seq,
        status=ApplicationStatus.submitted,
        resume_storage_uri=resume_storage_uri,
    )
    session.add(app)
    session.flush()
    if job.current_applications_count is None:
        job.current_applications_count = 0
    job.current_applications_count += 1
    return app


def log_hr_conversation(
    session: Session,
    *,
    correlation_id: uuid.UUID,
    job_id: uuid.UUID,
    speaker: str,
    stage: str,
    message: str,
    application_id: Optional[uuid.UUID] = None,
    meta_json: Optional[dict] = None,
) -> HRConversation:
    row = HRConversation(
        correlation_id=correlation_id,
        job_id=job_id,
        application_id=application_id,
        speaker=speaker,
        stage=stage,
        message=message,
        meta_json=meta_json,
    )
    session.add(row)
    session.flush()
    return row


def add_audit_finding(
    session: Session,
    *,
    correlation_id: uuid.UUID,
    job_id: uuid.UUID,
    stage: str,
    severity: str,
    finding_type: str,
    summary: str,
    application_id: Optional[uuid.UUID] = None,
    details: Optional[dict] = None,
) -> AuditFinding:
    row = AuditFinding(
        correlation_id=correlation_id,
        job_id=job_id,
        application_id=application_id,
        stage=stage,
        severity=severity,
        finding_type=finding_type,
        summary=summary,
        details=details,
    )
    session.add(row)
    session.flush()
    return row


def add_learning_feedback(
    session: Session,
    *,
    correlation_id: uuid.UUID,
    agent_name: str,
    feedback_type: str,
    created_by: str = "hr",
    label: Optional[str] = None,
    notes: Optional[str] = None,
    payload: Optional[dict] = None,
    job_id: Optional[uuid.UUID] = None,
    application_id: Optional[uuid.UUID] = None,
) -> LearningFeedback:
    row = LearningFeedback(
        correlation_id=correlation_id,
        job_id=job_id,
        application_id=application_id,
        agent_name=agent_name,
        feedback_type=feedback_type,
        label=label,
        notes=notes,
        payload=payload,
        created_by=created_by,
    )
    session.add(row)
    session.flush()
    return row


def record_prescreen(
    session: Session,
    application: Application,
    *,
    model_id: str,
    prompt_version: str,
    scores: dict,
    passed: bool,
    idempotency_key: str,
) -> Optional[PrescreeningRun]:
    existing = session.execute(
        select(PrescreeningRun).where(
            PrescreeningRun.application_id == application.id,
            PrescreeningRun.idempotency_key == idempotency_key,
        )
    ).scalar_one_or_none()
    if existing:
        return existing
    row = PrescreeningRun(
        application_id=application.id,
        model_id=model_id,
        prompt_version=prompt_version,
        scores=scores,
        passed=passed,
        idempotency_key=idempotency_key,
    )
    session.add(row)
    session.flush()
    return row


def set_application_status(session: Session, app: Application, status: ApplicationStatus) -> None:
    app.status = status


def create_interview_slot(
    session: Session,
    application: Application,
    *,
    token_hash: str,
    expires_at: datetime,
    scheduled_at: Optional[datetime],
    timezone_name: Optional[str],
    idempotency_schedule_key: str,
) -> Tuple[Interview, bool]:
    existing = session.execute(
        select(Interview).where(Interview.idempotency_schedule_key == idempotency_schedule_key)
    ).scalar_one_or_none()
    if existing:
        return existing, False
    inv = Interview(
        application_id=application.id,
        token_hash=token_hash,
        expires_at=expires_at,
        scheduled_at=scheduled_at,
        timezone_name=timezone_name,
        status=InterviewStatus.scheduled,
        idempotency_schedule_key=idempotency_schedule_key,
    )
    session.add(inv)
    session.flush()
    return inv, True


def mark_interview_link_sent(session: Session, interview: Interview) -> None:
    interview.status = InterviewStatus.link_sent
    interview.link_sent_at = datetime.now(timezone.utc)


def mark_interview_reminder_sent(session: Session, interview: Interview) -> None:
    interview.reminder_sent_at = datetime.now(timezone.utc)


def get_interview_by_token_hash(session: Session, token_hash: str) -> Optional[Interview]:
    return session.execute(select(Interview).where(Interview.token_hash == token_hash)).scalar_one_or_none()


def complete_interview(
    session: Session,
    interview: Interview,
    *,
    scores: dict,
    model_version: str,
    media_uri: Optional[str],
    idempotency_key: str,
) -> Tuple[Optional[InterviewResult], bool]:
    existing = session.execute(
        select(InterviewResult).where(InterviewResult.idempotency_key == idempotency_key)
    ).scalar_one_or_none()
    if existing:
        return existing, False

    if interview.status == InterviewStatus.completed and interview.result:
        return interview.result, False

    res = InterviewResult(
        interview_id=interview.id,
        scores=scores,
        model_version=model_version,
        media_uri=media_uri,
        idempotency_key=idempotency_key,
    )
    session.add(res)
    interview.status = InterviewStatus.completed
    session.flush()
    return res, True


def mark_interview_no_show(session: Session, interview: Interview) -> None:
    interview.status = InterviewStatus.no_show


def withdraw_application(session: Session, app: Application) -> None:
    app.status = ApplicationStatus.withdrawn


def replace_shortlist(
    session: Session,
    job_id: uuid.UUID,
    ranked_application_ids: List[uuid.UUID],
) -> None:
    apps = list(session.execute(select(Application).where(Application.job_id == job_id)).scalars().all())
    app_ids = [a.id for a in apps]
    if app_ids:
        session.execute(delete(ShortlistEntry).where(ShortlistEntry.application_id.in_(app_ids)))
    session.flush()
    shortlisted_set = set(ranked_application_ids)
    for rank, app_id in enumerate(ranked_application_ids, start=1):
        session.add(ShortlistEntry(application_id=app_id, rank=rank))
        app = session.get(Application, app_id)
        if app:
            app.status = ApplicationStatus.shortlisted
    for a in apps:
        if a.id in shortlisted_set:
            continue
        if a.status in (
            ApplicationStatus.prescreen_failed,
            ApplicationStatus.withdrawn,
            ApplicationStatus.offer_sent,
            ApplicationStatus.offer_accepted,
            ApplicationStatus.offer_declined,
            ApplicationStatus.hired,
        ):
            continue
        if a.status in (
            ApplicationStatus.interview_completed,
            ApplicationStatus.prescreen_passed,
            ApplicationStatus.interview_link_sent,
            ApplicationStatus.interview_scheduled,
        ):
            a.status = ApplicationStatus.not_selected
    session.flush()


def create_offer(
    session: Session,
    application: Application,
    body: dict,
    idempotency_key: str,
) -> Optional[Offer]:
    existing = session.execute(
        select(Offer).where(Offer.idempotency_key == idempotency_key)
    ).scalar_one_or_none()
    if existing:
        return existing
    row = Offer(
        application_id=application.id,
        status=OfferStatus.sent,
        body=body,
        sent_at=datetime.now(timezone.utc),
        idempotency_key=idempotency_key,
    )
    session.add(row)
    session.flush()
    application.status = ApplicationStatus.offer_sent
    return row


def list_pending_screening_applications(session: Session, job_id: uuid.UUID) -> List[Application]:
    return list(
        session.scalars(
            select(Application).where(
                Application.job_id == job_id,
                Application.status == ApplicationStatus.submitted,
            )
        ).all()
    )


def mark_offer_response(session: Session, offer: Offer, accepted: bool) -> None:
    offer.response_at = datetime.now(timezone.utc)
    offer.status = OfferStatus.accepted if accepted else OfferStatus.declined
    if offer.application:
        offer.application.status = (
            ApplicationStatus.offer_accepted if accepted else ApplicationStatus.offer_declined
        )


def mark_job_needs_review(session: Session, job: Job, reason: str) -> None:
    job.status = JobStatus.needs_review
    logger.error("Job %s needs_review: %s", job.id, reason)


def get_org_settings(session: Session) -> OrgSettings:
    row = session.get(OrgSettings, _ORG_SETTINGS_ID)
    if row is None:
        row = OrgSettings(id=_ORG_SETTINGS_ID)
        session.add(row)
        session.flush()
    return row


def upsert_policy(session: Session, s3_key: Optional[str], filename: Optional[str]) -> OrgSettings:
    row = get_org_settings(session)
    row.policy_s3_key = s3_key
    row.policy_filename = filename
    session.flush()
    return row

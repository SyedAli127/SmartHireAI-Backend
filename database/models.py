import enum
import uuid
from datetime import datetime, timezone
from typing import Optional


def _utcnow():
    return datetime.now(timezone.utc)

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class JobStatus(str, enum.Enum):
    draft = "draft"
    jd_ready = "jd_ready"
    ads_ready = "ads_ready"
    posted = "posted"
    collecting = "collecting"
    closed = "closed"
    needs_review = "needs_review"


class ApplicationStatus(str, enum.Enum):
    submitted = "submitted"
    prescreen_failed = "prescreen_failed"
    prescreen_passed = "prescreen_passed"
    interview_scheduled = "interview_scheduled"
    interview_link_sent = "interview_link_sent"
    interview_completed = "interview_completed"
    interview_no_show = "interview_no_show"
    withdrawn = "withdrawn"
    shortlisted = "shortlisted"
    not_selected = "not_selected"
    offer_sent = "offer_sent"
    offer_accepted = "offer_accepted"
    offer_declined = "offer_declined"
    hired = "hired"
    needs_review = "needs_review"


class InterviewStatus(str, enum.Enum):
    scheduled = "scheduled"
    link_sent = "link_sent"
    completed = "completed"
    no_show = "no_show"
    cancelled = "cancelled"


class OfferStatus(str, enum.Enum):
    draft = "draft"
    sent = "sent"
    accepted = "accepted"
    declined = "declined"
    withdrawn = "withdrawn"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    department: Mapped[Optional[str]] = mapped_column(String(256))
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.draft)
    correlation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    thread_id: Mapped[Optional[str]] = mapped_column(String(64))
    pass_threshold: Mapped[float] = mapped_column(default=0.5)
    min_applications_threshold: Mapped[int] = mapped_column(BigInteger, default=1)
    current_applications_count: Mapped[int] = mapped_column(BigInteger, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    jd_versions = relationship("JDVersion", back_populates="job", order_by="JDVersion.version")
    applications = relationship("Application", back_populates="job")
    ad_posts = relationship("AdPost", back_populates="job")


class JDVersion(Base):
    __tablename__ = "jd_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(BigInteger, nullable=False)
    jd_text: Mapped[str] = mapped_column(Text, nullable=False)
    requirements_snapshot: Mapped[Optional[dict]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    job = relationship("Job", back_populates="jd_versions")
    applications = relationship("Application", back_populates="jd_version")


class AdPost(Base):
    __tablename__ = "ad_posts"
    __table_args__ = (UniqueConstraint("job_id", "platform", "idempotency_key", name="uq_ad_post_idempotent"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    platform: Mapped[str] = mapped_column(String(64), nullable=False)
    external_post_id: Mapped[Optional[str]] = mapped_column(String(256))
    url: Mapped[Optional[str]] = mapped_column(Text)
    payload: Mapped[Optional[dict]] = mapped_column(JSONB)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    job = relationship("Job", back_populates="ad_posts")


class Candidate(Base):
    __tablename__ = "candidates"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(256))
    phone: Mapped[Optional[str]] = mapped_column(String(64))
    pii: Mapped[Optional[dict]] = mapped_column(JSONB)
    retention_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    applications = relationship("Application", back_populates="candidate")


class Application(Base):
    __tablename__ = "applications"
    __table_args__ = (UniqueConstraint("job_id", "candidate_id", "reapplication_seq", name="uq_app_job_candidate_seq"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    candidate_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"), index=True)
    jd_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("jd_versions.id", ondelete="SET NULL"))
    reapplication_seq: Mapped[int] = mapped_column(BigInteger, default=0)
    status: Mapped[ApplicationStatus] = mapped_column(Enum(ApplicationStatus), default=ApplicationStatus.submitted)
    resume_storage_uri: Mapped[Optional[str]] = mapped_column(Text)
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    job = relationship("Job", back_populates="applications")
    candidate = relationship("Candidate", back_populates="applications")
    jd_version = relationship("JDVersion", back_populates="applications")
    prescreens = relationship("PrescreeningRun", back_populates="application")
    interviews = relationship("Interview", back_populates="application")
    shortlist = relationship("ShortlistEntry", back_populates="application", uselist=False)
    offers = relationship("Offer", back_populates="application")


class PrescreeningRun(Base):
    __tablename__ = "prescreening_runs"
    __table_args__ = (UniqueConstraint("application_id", "idempotency_key", name="uq_prescreen_idempotent"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    application_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("applications.id", ondelete="CASCADE"), index=True)
    model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    scores: Mapped[dict] = mapped_column(JSONB, nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    application = relationship("Application", back_populates="prescreens")


class Interview(Base):
    __tablename__ = "interviews"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    application_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("applications.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    scheduled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    reminder_sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    link_sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    status: Mapped[InterviewStatus] = mapped_column(Enum(InterviewStatus), default=InterviewStatus.scheduled)
    timezone_name: Mapped[Optional[str]] = mapped_column(String(64))
    idempotency_schedule_key: Mapped[Optional[str]] = mapped_column(String(128), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    application = relationship("Application", back_populates="interviews")
    result = relationship("InterviewResult", back_populates="interview", uselist=False)


class InterviewResult(Base):
    __tablename__ = "interview_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    interview_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("interviews.id", ondelete="CASCADE"), unique=True)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    scores: Mapped[dict] = mapped_column(JSONB, nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    media_uri: Mapped[Optional[str]] = mapped_column(Text)
    idempotency_key: Mapped[Optional[str]] = mapped_column(String(128), unique=True)

    interview = relationship("Interview", back_populates="result")


class ShortlistEntry(Base):
    __tablename__ = "shortlist_entries"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    application_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("applications.id", ondelete="CASCADE"), unique=True)
    rank: Mapped[int] = mapped_column(BigInteger, nullable=False)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    application = relationship("Application", back_populates="shortlist")


class Offer(Base):
    __tablename__ = "offers"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    application_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("applications.id", ondelete="CASCADE"), index=True)
    status: Mapped[OfferStatus] = mapped_column(Enum(OfferStatus), default=OfferStatus.draft)
    body: Mapped[dict] = mapped_column(JSONB, nullable=False)
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    response_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    idempotency_key: Mapped[Optional[str]] = mapped_column(String(128), unique=True)

    application = relationship("Application", back_populates="offers")


class PipelineEvent(Base):
    __tablename__ = "pipeline_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    correlation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_id: Mapped[Optional[str]] = mapped_column(String(64))
    from_state: Mapped[Optional[str]] = mapped_column(String(64))
    to_state: Mapped[Optional[str]] = mapped_column(String(64))
    agent_name: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[Optional[dict]] = mapped_column(JSONB)
    idempotency_key: Mapped[Optional[str]] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (
        UniqueConstraint("correlation_id", "idempotency_key", name="uq_pipeline_event_idempotent"),
    )


class HRConversation(Base):
    __tablename__ = "hr_conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    correlation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    application_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("applications.id", ondelete="SET NULL"))
    speaker: Mapped[str] = mapped_column(String(32), nullable=False)  # hr|agent|system
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    meta_json: Mapped[Optional[dict]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AuditFinding(Base):
    __tablename__ = "audit_findings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    correlation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    application_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("applications.id", ondelete="SET NULL"))
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)  # low|medium|high
    finding_type: Mapped[str] = mapped_column(String(64), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[Optional[dict]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class LearningFeedback(Base):
    __tablename__ = "learning_feedback"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    correlation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    job_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("jobs.id", ondelete="SET NULL"), index=True)
    application_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("applications.id", ondelete="SET NULL"), index=True)
    agent_name: Mapped[str] = mapped_column(String(128), nullable=False)
    feedback_type: Mapped[str] = mapped_column(String(64), nullable=False)
    label: Mapped[Optional[str]] = mapped_column(String(128))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    payload: Mapped[Optional[dict]] = mapped_column(JSONB)
    created_by: Mapped[str] = mapped_column(String(64), default="hr")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


_ORG_SETTINGS_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


class OrgSettings(Base):
    """Single-row org-level config. Always use id = _ORG_SETTINGS_ID."""
    __tablename__ = "org_settings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=lambda: _ORG_SETTINGS_ID)
    policy_s3_key: Mapped[Optional[str]] = mapped_column(Text)
    policy_filename: Mapped[Optional[str]] = mapped_column(String(512))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

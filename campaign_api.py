"""
SmartHireAI Campaign API — interactive HR-approval flow.

Stages:
  queued → jd_generating → jd_review → ad_generating → ad_review
  → posting → running → completed | failed

HR must explicitly approve or reject each stage via:
  POST /api/campaigns/{run_id}/jd/decision
  POST /api/campaigns/{run_id}/ads/decision
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import json
import os
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import main_v2 as pipeline_runtime
from services import s3_service, email
from services.email import send_email
from Interview_Agent.interview_router import router as interview_router

UPLOADS_DIR = Path("uploads/resumes")
# Local-fallback images land in local_s3_store/Generated_Ads; fall back to
# the bare Generated_Ads dir for any legacy files.
_LOCAL_ADS_DIR = Path("local_s3_store") / "Generated_Ads"
_LEGACY_ADS_DIR = Path("Generated_Ads")
GENERATED_ADS_DIR = _LOCAL_ADS_DIR if _LOCAL_ADS_DIR.exists() else _LEGACY_ADS_DIR


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    from database.session import init_db
    init_db()
    yield

app = FastAPI(title="SmartHireAI Campaign API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if _LOCAL_ADS_DIR.exists():
    app.mount("/generated_ads", StaticFiles(directory=str(_LOCAL_ADS_DIR)), name="generated_ads")
elif _LEGACY_ADS_DIR.exists():
    app.mount("/generated_ads", StaticFiles(directory=str(_LEGACY_ADS_DIR)), name="generated_ads")

app.include_router(interview_router)

# ─── Models ──────────────────────────────────────────────────────────────────

class CampaignIntake(BaseModel):
    title: str = "AI Engineer"
    department: str = "R&D"
    seniority: str = "Mid-level"
    location: str = "Remote"
    employment_type: str = "Full-time"
    hr_notes: str = ""
    days_open: int = 30
    pass_threshold: float = 6.0
    platforms: List[str] = Field(default_factory=lambda: ["Facebook", "Instagram", "X"])
    shortlist_top_n: int = 3
    min_applications_threshold: int = 5
    poll_seconds: int = 10
    max_wait_application_minutes: int = 60
    max_wait_prescreen_minutes: int = 60
    max_wait_interview_minutes: int = 60
    auto_sync_forms: bool = True


class DecisionRequest(BaseModel):
    approved: bool
    feedback: str = ""


class AdTextDecisionRequest(BaseModel):
    approved: bool
    feedback: str = ""
    # When approved: {platform: {headline, text, cta, hashtags, _selected_variant}}
    approved_texts: Optional[Dict[str, Any]] = None


class AdImageScriptDecisionRequest(BaseModel):
    approved: bool
    feedback: str = ""
    approved_script: Optional[str] = None  # selected script text (professional or creative)


class McqCompletionRequest(BaseModel):
    application_id: str
    passed: bool = True
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    agent_name: str                        # "prescreen" | "interview"
    feedback_type: str                     # e.g. "hr_label"
    label: Optional[str] = None           # "false_positive" | "false_negative" | "correct"
    notes: Optional[str] = None           # free-text HR comment
    application_id: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


class OfferApprovalRequest(BaseModel):
    application_id: str


# ─── Run State ───────────────────────────────────────────────────────────────

# HR review timeout: 30 minutes — pipeline fails if HR doesn't respond
HR_REVIEW_TIMEOUT_SECONDS = 1800
# Offer approval timeout: 24 hours — HR has 1 day to select the candidate
OFFER_REVIEW_TIMEOUT_SECONDS = 86400

@dataclass
class RunState:
    run_id: str
    intake: Optional[Dict[str, Any]] = None

    # lifecycle stage
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # db identifiers
    job_id: Optional[str] = None
    correlation_id: Optional[str] = None

    # HR review payloads
    jd_draft: Optional[str] = None
    ads_draft: Optional[List[Dict[str, Any]]] = None
    text_variants_draft: Optional[Dict[str, Any]] = None     # {platform: {A: {...}, B: {...}}}
    image_script_draft: Optional[Dict[str, str]] = None      # {professional: "...", creative: "..."}
    generated_image_draft: Optional[str] = None              # URL shown to HR for final approval

    # pipeline output + logs
    pipeline_result: Dict[str, Any] = field(default_factory=dict)
    snapshot: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None

    # offer approval candidates (persisted so frontend can display them)
    offer_candidates: Optional[List[Dict[str, Any]]] = None

    # ── runtime-only (not persisted) ──────────────────────────────────────
    _jd_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _jd_decision: Optional[Tuple[bool, str]] = field(default=None, repr=False)
    _ad_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _ad_decision: Optional[Tuple[bool, str]] = field(default=None, repr=False)
    _ad_text_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _ad_text_decision: Optional[Tuple[bool, str, Optional[Dict[str, Any]]]] = field(default=None, repr=False)
    _ad_image_script_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _ad_image_script_decision: Optional[Tuple[bool, str, Optional[str]]] = field(default=None, repr=False)
    _ad_image_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _ad_image_decision: Optional[Tuple[bool, str]] = field(default=None, repr=False)
    _offer_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _offer_decision: Optional[str] = field(default=None, repr=False)
    seen_response_ids: set = field(default_factory=set)
    forms_client_cache: Dict[str, Any] = field(default_factory=dict)


RUNS: Dict[str, RunState] = {}
RUNS_LOCK = threading.Lock()
RUN_STATE_DIR = Path(".runtime/campaign_runs")


# ─── Persistence helpers ──────────────────────────────────────────────────────

def _serialize_run(run: RunState) -> Dict[str, Any]:
    return {
        "run_id": run.run_id,
        "intake": run.intake,
        "status": run.status,
        "created_at": run.created_at,
        "job_id": run.job_id,
        "correlation_id": run.correlation_id,
        "jd_draft": run.jd_draft,
        "ads_draft": run.ads_draft,
        "text_variants_draft": run.text_variants_draft,
        "image_script_draft": run.image_script_draft,
        "generated_image_draft": run.generated_image_draft,
        "pipeline_result": run.pipeline_result,
        "snapshot": run.snapshot,
        "logs": run.logs,
        "error": run.error,
        "offer_candidates": run.offer_candidates,
        "seen_response_ids": sorted(run.seen_response_ids),
    }


def _persist_run(run: RunState) -> None:
    try:
        RUN_STATE_DIR.mkdir(parents=True, exist_ok=True)
        (RUN_STATE_DIR / f"{run.run_id}.json").write_text(
            json.dumps(_serialize_run(run), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _load_run(run_id: str) -> Optional[RunState]:
    path = RUN_STATE_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    try:
        p = json.loads(path.read_text(encoding="utf-8"))
        run = RunState(
            run_id=p.get("run_id", run_id),
            intake=p.get("intake"),
            status=p.get("status", "queued"),
            created_at=p.get("created_at", datetime.utcnow().isoformat()),
            job_id=p.get("job_id"),
            correlation_id=p.get("correlation_id"),
            jd_draft=p.get("jd_draft"),
            ads_draft=p.get("ads_draft"),
            text_variants_draft=p.get("text_variants_draft"),
            image_script_draft=p.get("image_script_draft"),
            generated_image_draft=p.get("generated_image_draft"),
            pipeline_result=p.get("pipeline_result") or {},
            snapshot=p.get("snapshot") or {},
            logs=p.get("logs") or [],
            error=p.get("error"),
            offer_candidates=p.get("offer_candidates"),
            seen_response_ids=set(p.get("seen_response_ids") or []),
        )
        return run
    except Exception:
        return None


def _image_path_to_url(path: str) -> Optional[str]:
    """Return a viewable URL for an image — S3 URLs pass through directly."""
    if not path:
        return None
    if path.startswith("http://") or path.startswith("https://"):
        return path
    try:
        p = Path(path)
        parts = p.parts
        for i, part in enumerate(parts):
            if part.lower() == "generated_ads":
                rel = "/".join(parts[i + 1:])
                return f"/generated_ads/{rel}"
    except Exception:
        pass
    return None


def _log(run: RunState, msg: str) -> None:
    run.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    run.logs = run.logs[-300:]
    _persist_run(run)


# ─── Pipeline helpers ──────────────────────────────────────────────────────────

def _update_snapshot(run: RunState) -> None:
    if not run.job_id:
        return
    try:
        run.snapshot = pipeline_runtime._collect_status_snapshot(
            pipeline_runtime.get_session_factory(),
            uuid.UUID(run.job_id),
        )
    except Exception as exc:
        _log(run, f"Snapshot update failed: {exc}")
    _persist_run(run)


def _threshold_to_hybrid(threshold: float) -> float:
    return max(0.0, min(1.0, float(threshold) / 10.0))


def _normalize_intake(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "max_wait_application_minutes" not in payload and "max_wait_prescreen_minutes" in payload:
        payload["max_wait_application_minutes"] = payload["max_wait_prescreen_minutes"]
    return payload


# ─── Worker ──────────────────────────────────────────────────────────────────

def _worker(run: RunState, intake: CampaignIntake) -> None:
    run.status = "jd_generating"
    _persist_run(run)

    try:
        _log(run, "Initialising backend and database…")
        pipeline_runtime._ensure_backend()
        pipeline_runtime.init_db()

        payload = _normalize_intake(intake.model_dump())
        pipeline = pipeline_runtime.HiringPipeline(
            session_factory=pipeline_runtime.get_session_factory(),
            config=pipeline_runtime.HiringPipelineConfig(
                hr_auto_approve_jd=False,
                hr_auto_approve_ads=False,
                prescreen_min_hybrid=_threshold_to_hybrid(payload.get("pass_threshold", 6.0)),
            ),
        )

        # ── HR decision providers ───────────────────────────────────────────

        def jd_provider(state: Dict[str, Any]) -> Tuple[bool, str]:
            jd_text = (
                (state.get("winner") or {}).get("jd_text")
                or state.get("final_jd")
                or ""
            )
            run.jd_draft = jd_text
            run.status = "jd_review"
            _log(run, "JD generated — awaiting HR approval.")
            _persist_run(run)

            signalled = run._jd_event.wait(timeout=HR_REVIEW_TIMEOUT_SECONDS)
            if not signalled or run._jd_decision is None:
                raise RuntimeError("JD review timed out — no HR decision received within 30 minutes.")
            approved, feedback = run._jd_decision
            run._jd_event.clear()
            run._jd_decision = None
            if approved:
                run.status = "ad_generating"
                _log(run, "JD approved by HR. Generating ads…")
            else:
                run.status = "jd_generating"
                _log(run, f"JD rejected by HR — generating revised JD. Feedback: {feedback or '(none)'}")
            _persist_run(run)
            return approved, feedback

        def ad_provider(state: Dict[str, Any]) -> Tuple[bool, str, Optional[List[dict]]]:
            # Use approved_ads (set by hr_approval_node with image paths) for the draft shown to HR.
            # Falling back to combined_ads would strip generated_image_path from every ad.
            source_ads = state.get("approved_ads") or state.get("combined_ads") or []
            ads_list: List[Dict[str, Any]] = []
            for ad in source_ads:
                ads_list.append(ad.model_dump() if hasattr(ad, "model_dump") else dict(ad))
            run.ads_draft = ads_list
            run.status = "ad_review"
            _log(run, f"Generated {len(ads_list)} ad variant(s) — awaiting HR approval.")
            _persist_run(run)

            signalled = run._ad_event.wait(timeout=HR_REVIEW_TIMEOUT_SECONDS)
            if not signalled or run._ad_decision is None:
                raise RuntimeError("Ad review timed out — no HR decision received within 30 minutes.")
            approved, feedback = run._ad_decision
            run._ad_event.clear()
            run._ad_decision = None
            if approved:
                run.status = "posting"
                _log(run, "Ads approved by HR. Posting to platforms…")
            else:
                run.status = "ad_rejected"
                _log(run, f"Ads rejected by HR: {feedback or '(no feedback)'}")
            _persist_run(run)
            approved_ads = ads_list if approved else None
            return approved, feedback, approved_ads

        def ad_text_provider(state: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
            run.text_variants_draft = state.get("text_variants")
            run.status = "ad_text_review"
            _log(run, f"Ad texts generated for {len(run.text_variants_draft or {})} platform(s) — awaiting HR approval.")
            _persist_run(run)

            signalled = run._ad_text_event.wait(timeout=HR_REVIEW_TIMEOUT_SECONDS)
            if not signalled or run._ad_text_decision is None:
                raise RuntimeError("Ad text review timed out — no HR decision received within 30 minutes.")
            approved, feedback, approved_texts = run._ad_text_decision
            run._ad_text_event.clear()
            run._ad_text_decision = None
            if approved:
                run.status = "ad_generating"
                _log(run, "Ad texts approved by HR. Generating image scripts…")
            else:
                run.status = "ad_generating"
                _log(run, f"Ad texts rejected — regenerating. Feedback: {feedback or '(none)'}")
            _persist_run(run)
            return approved, feedback, approved_texts

        def ad_image_script_provider(state: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
            run.image_script_draft = state.get("image_script_variants")
            run.status = "ad_image_script_review"
            _log(run, "Image script variants generated — awaiting HR approval.")
            _persist_run(run)

            signalled = run._ad_image_script_event.wait(timeout=HR_REVIEW_TIMEOUT_SECONDS)
            if not signalled or run._ad_image_script_decision is None:
                raise RuntimeError("Image script review timed out — no HR decision received within 30 minutes.")
            approved, feedback, approved_script = run._ad_image_script_decision
            run._ad_image_script_event.clear()
            run._ad_image_script_decision = None
            if approved:
                run.status = "ad_generating"
                _log(run, "Image script approved by HR. Generating image…")
            else:
                run.status = "ad_generating"
                _log(run, f"Image script rejected — regenerating. Feedback: {feedback or '(none)'}")
            _persist_run(run)
            return approved, feedback, approved_script

        def ad_image_provider(state: Dict[str, Any]) -> Tuple[bool, str]:
            run.generated_image_draft = state.get("generated_image_url")
            run.status = "ad_image_review"
            _log(run, "Ad image generated — awaiting HR final approval.")
            _persist_run(run)

            signalled = run._ad_image_event.wait(timeout=HR_REVIEW_TIMEOUT_SECONDS)
            if not signalled or run._ad_image_decision is None:
                raise RuntimeError("Ad image review timed out — no HR decision received within 30 minutes.")
            approved, feedback = run._ad_image_decision
            run._ad_image_event.clear()
            run._ad_image_decision = None
            if approved:
                run.status = "posting"
                _log(run, "Ad image approved by HR. Finalising and posting…")
            else:
                run.status = "ad_generating"
                _log(run, f"Ad image rejected — regenerating. Feedback: {feedback or '(none)'}")
            _persist_run(run)
            return approved, feedback

        # ── Phase 1+2: JD generation → HR review → ad generation → HR review → post ──
        _log(run, f"Starting JD generation for: {payload['title']}")
        result = pipeline.run_full_pipeline(
            payload,
            candidates=[],
            hr_decision_provider=jd_provider,
            ad_decision_provider=ad_provider,
            ad_text_decision_provider=ad_text_provider,
            ad_image_script_decision_provider=ad_image_script_provider,
            ad_image_decision_provider=ad_image_provider,
        )

        run.job_id = result.get("job_id") or run.job_id
        run.correlation_id = result.get("correlation_id") or run.correlation_id
        run.pipeline_result = {
            k: (v if isinstance(v, (str, int, float, bool, list, dict, type(None))) else str(v))
            for k, v in (result or {}).items()
            if k != "combined_ads"
        }
        _persist_run(run)

        if not run.job_id:
            raise RuntimeError(f"Pipeline did not return a job_id. State: {result.get('status')}")

        job_uuid = uuid.UUID(run.job_id)
        run.status = "running"
        _log(run, f"Job ID: {run.job_id}. Collecting applications…")
        if result.get("job_form_link"):
            _log(run, f"Application form: {result['job_form_link']}")
        _update_snapshot(run)

        # ── Phase 3: collect applications for the full campaign period ──────
        days_open = max(1, int(intake.days_open or 30))
        collection_deadline = time.time() + (days_open * 60)
        collection_extensions = 0
        run.pipeline_result["collection_deadline"] = datetime.utcfromtimestamp(collection_deadline).isoformat()
        run.pipeline_result["collection_extensions"] = 0
        _persist_run(run)

        from datetime import datetime as _dt
        deadline_str = _dt.utcfromtimestamp(collection_deadline).strftime("%Y-%m-%d %H:%M UTC")
        _log(run, f"Collecting applications for {days_open} minute(s). Deadline: {deadline_str}")

        while True:
            if intake.auto_sync_forms:
                pipeline_runtime._ingest_new_form_responses(
                    pipeline,
                    job_uuid,
                    form_link=result.get("job_form_link"),
                    shortlist_top_n=max(1, int(intake.shortlist_top_n)),
                    seen_response_ids=run.seen_response_ids,
                    forms_client_cache=run.forms_client_cache,
                )
            _update_snapshot(run)

            if time.time() >= collection_deadline:
                snap = run.snapshot or {}
                current_apps = int(snap.get("current_applications", 0))
                min_threshold = int(snap.get("threshold", 1))

                if current_apps >= min_threshold:
                    _log(run, f"Collection period ended. {current_apps} application(s) received — threshold met. Starting prescreening.")
                    break
                else:
                    collection_extensions += 1
                    collection_deadline = time.time() + (days_open * 60)
                    run.pipeline_result["collection_extensions"] = collection_extensions
                    run.pipeline_result["collection_deadline"] = _dt.utcfromtimestamp(collection_deadline).strftime("%Y-%m-%d %H:%M UTC")
                    _persist_run(run)
                    new_deadline_str = _dt.utcfromtimestamp(collection_deadline).strftime("%Y-%m-%d %H:%M UTC")
                    _log(run, f"Threshold not met ({current_apps}/{min_threshold} CVs). Extending collection by {days_open} more minute(s) (extension #{collection_extensions}). New deadline: {new_deadline_str}")

            time.sleep(max(1, int(intake.poll_seconds)))

        phase3 = pipeline.process_waiting_applications(job_uuid, shortlist_top_n=max(1, int(intake.shortlist_top_n)))
        run.pipeline_result["phase3"] = phase3
        for row in (phase3.get("prescreen_results") or []):
            row_email = row.get("candidate_email") or "unknown"
            hybrid = row.get("hybrid_score")
            hybrid_str = "n/a" if hybrid is None else f"{float(hybrid):.3f}"
            _log(run, f"Prescreen: {row_email} | status={row.get('status') or '-'} | score={hybrid_str}")

        # ── Phase 4: wait for interviews, then pause for HR offer approval ──
        _update_snapshot(run)
        snap = run.snapshot or {}
        dispatched = int(snap.get("pending_interviews", 0)) + int((snap.get("interview_counts") or {}).get("completed", 0))
        if dispatched > 0:
            _log(run, f"Waiting for {dispatched} interview(s)…")
            interview_deadline = time.time() + (intake.max_wait_interview_minutes * 60)
            while time.time() < interview_deadline:
                _update_snapshot(run)
                done = int((run.snapshot.get("interview_counts") or {}).get("completed", 0))
                if done >= dispatched:
                    break
                time.sleep(max(1, int(intake.poll_seconds)))

        # ── Phase 5: HR offer approval ──────────────────────────────────────
        offer_cands = pipeline.get_offer_candidates(job_uuid, top_n=max(1, int(intake.shortlist_top_n)))
        run.offer_candidates = offer_cands
        run.status = "awaiting_offer_approval"
        _log(run, f"Interviews complete. {len(offer_cands)} candidate(s) shortlisted — awaiting HR offer approval.")
        _persist_run(run)

        signalled = run._offer_event.wait(timeout=OFFER_REVIEW_TIMEOUT_SECONDS)
        if not signalled or run._offer_decision is None:
            raise RuntimeError("Offer approval timed out — no HR decision received within 24 hours.")

        chosen_app_id = run._offer_decision
        run._offer_event.clear()
        run._offer_decision = None

        if chosen_app_id == "NO_HIRE":
            _log(run, "HR decision: No Hire — sending rejection emails to all shortlisted candidates.")
            from Pipeline.communication_adapter import send_rejection_email
            job_title = intake.title or "the position"
            for cand in offer_cands:
                cand_name  = cand.get("candidate_name")  or "Candidate"
                cand_email = cand.get("candidate_email") or ""
                if cand_email:
                    sent = send_rejection_email(
                        candidate_name=cand_name,
                        candidate_email=cand_email,
                        job_title=job_title,
                    )
                    _log(run, f"Rejection email {'sent' if sent else 'failed'} → {cand_email}")
            run.status = "completed"
            run.pipeline_result["offer_decision"] = "no_hire"
            _persist_run(run)
            return
        final = pipeline.send_offer_to_candidate(job_uuid, uuid.UUID(chosen_app_id))
        run.pipeline_result.update(final)
        chosen_email = next((c["candidate_email"] for c in offer_cands if c["application_id"] == chosen_app_id), chosen_app_id)
        _log(run, f"Offer sent to {chosen_email}.")

        # ── Self-Audit pass ────────────────────────────────────────────────
        try:
            _log(run, "Running Self-Audit Agent...")
            from sqlalchemy import select as _select
            from database.repo import add_audit_finding as _add_audit_finding
            sf = pipeline_runtime.get_session_factory()
            with pipeline_runtime.session_scope(sf) as session:
                apps = list(session.scalars(
                    _select(pipeline_runtime.m.Application).where(
                        pipeline_runtime.m.Application.job_id == job_uuid
                    )
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

                corr_id = uuid.UUID(run.correlation_id) if run.correlation_id else uuid.uuid4()

                if prescreen_scores:
                    for finding in pipeline_runtime.audit_batch_prescreen(prescreen_scores):
                        _add_audit_finding(
                            session,
                            correlation_id=corr_id,
                            job_id=job_uuid,
                            stage="batch_prescreen_audit",
                            severity=finding["severity"],
                            finding_type=finding["finding_type"],
                            summary=finding["summary"],
                            details=finding.get("details"),
                        )

                if interview_scores:
                    for finding in pipeline_runtime.audit_batch_interview(interview_scores):
                        _add_audit_finding(
                            session,
                            correlation_id=corr_id,
                            job_id=job_uuid,
                            stage="batch_interview_audit",
                            severity=finding["severity"],
                            finding_type=finding["finding_type"],
                            summary=finding["summary"],
                            details=finding.get("details"),
                        )

                all_findings = list(session.scalars(
                    _select(pipeline_runtime.m.AuditFinding).where(
                        pipeline_runtime.m.AuditFinding.job_id == job_uuid
                    )
                ).all())

                audit_summary = pipeline_runtime.generate_audit_summary([
                    {"severity": f.severity, "finding_type": f.finding_type, "summary": f.summary}
                    for f in all_findings
                ])

            finding_counts = {"high": 0, "medium": 0, "low": 0}
            for f in all_findings:
                finding_counts[f.severity] = finding_counts.get(f.severity, 0) + 1

            run.pipeline_result["audit"] = {
                "total_findings": len(all_findings),
                "finding_counts": finding_counts,
                "prescreen_scores_audited": len(prescreen_scores),
                "interview_scores_audited": len(interview_scores),
                "summary": audit_summary,
                "findings": [
                    {"severity": f.severity, "finding_type": f.finding_type, "summary": f.summary}
                    for f in all_findings
                ],
            }
            _log(run, f"Audit complete: {len(all_findings)} finding(s) — high={finding_counts['high']} medium={finding_counts['medium']} low={finding_counts['low']}")
            _log(run, f"Audit summary: {audit_summary}")
        except Exception as audit_exc:
            _log(run, f"Self-Audit Agent error (non-fatal): {audit_exc}")

        # ── Self-Improvement pass ───────────────────────────────────────────
        try:
            _log(run, "Running Self-Improvement Agent...")
            sf = pipeline_runtime.get_session_factory()
            job_uuid_arg = uuid.UUID(run.job_id) if run.job_id else None
            corr_uuid_arg = uuid.UUID(run.correlation_id) if run.correlation_id else None
            improvement_report = pipeline_runtime.run_learning_pass(
                sf, job_id=job_uuid_arg, correlation_id=corr_uuid_arg
            )
            run.pipeline_result["self_improvement"] = {
                "total_feedback_rows": improvement_report.get("total_feedback_rows", 0),
                "adjustments_applied": len(improvement_report.get("adjustments", [])),
                "held_for_review":     len(improvement_report.get("held_for_review", [])),
                "overall_insight":     improvement_report.get("overall_insight", ""),
                "bias_flags":          improvement_report.get("bias_flags", []),
                "prompt_suggestions":  improvement_report.get("prompt_suggestions", []),
                "ab_recommendation":   improvement_report.get("ab_recommendation"),
                "current_weights":     improvement_report.get("current_weights", {}),
                "adjustments":         improvement_report.get("adjustments", []),
                "held_details":        improvement_report.get("held_for_review", []),
                "llm_used":            improvement_report.get("llm_used", False),
            }
            if improvement_report.get("overall_insight"):
                _log(run, f"Self-Improvement insight: {improvement_report['overall_insight']}")
            if improvement_report.get("adjustments"):
                for adj in improvement_report["adjustments"]:
                    _log(run, f"  Applied [{adj['agent']}]: {adj['recommendation']}")
            if improvement_report.get("held_for_review"):
                _log(run, f"  {len(improvement_report['held_for_review'])} adjustment(s) held for HR review (low confidence).")
            if improvement_report.get("bias_flags"):
                for flag in improvement_report["bias_flags"]:
                    _log(run, f"  Bias flag: {flag}")
        except Exception as imp_exc:
            _log(run, f"Self-Improvement Agent error (non-fatal): {imp_exc}")

        run.status = "completed"
        _log(run, "Campaign pipeline completed successfully.")
        _update_snapshot(run)
        _persist_run(run)

    except Exception as exc:
        run.status = "failed"
        run.error = str(exc)
        _log(run, f"Pipeline error: {exc}")
        _log(run, traceback.format_exc())
        _persist_run(run)


# ─── Routes ──────────────────────────────────────────────────────────────────

def _get_run(run_id: str) -> RunState:
    with RUNS_LOCK:
        run = RUNS.get(run_id)
    if not run:
        run = _load_run(run_id)
        if run:
            with RUNS_LOCK:
                RUNS[run_id] = run
        else:
            raise HTTPException(status_code=404, detail="Campaign not found")
    return run


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/campaigns/start")
def start_campaign(intake: CampaignIntake) -> Dict[str, str]:
    run_id = str(uuid.uuid4())
    run = RunState(run_id=run_id, intake=intake.model_dump())
    with RUNS_LOCK:
        RUNS[run_id] = run
    _persist_run(run)
    thread = threading.Thread(target=_worker, args=(run, intake), daemon=True)
    thread.start()
    return {"run_id": run_id}


@app.get("/api/campaigns/{run_id}")
def get_campaign(run_id: str) -> Dict[str, Any]:
    run = _get_run(run_id)
    _update_snapshot(run)

    # Build viewable image URLs — pipeline_result doesn't carry approved_ads, so
    # fall back to ads_draft which is populated from LangGraph state (with image paths).
    ad_image_urls: Dict[str, str] = {}
    approved_ads = (run.pipeline_result or {}).get("approved_ads") or run.ads_draft or []
    for ad in approved_ads:
        if not isinstance(ad, dict):
            continue
        platform = ad.get("platform")
        img_path = ad.get("generated_image_path")
        if platform and img_path:
            url = _image_path_to_url(img_path)
            if url:
                ad_image_urls[platform] = url

    return {
        "run_id": run.run_id,
        "status": run.status,
        "created_at": run.created_at,
        "intake": run.intake,
        "job_id": run.job_id,
        "correlation_id": run.correlation_id,
        "jd_draft": run.jd_draft,
        "ads_draft": run.ads_draft,
        "text_variants_draft": run.text_variants_draft,
        "image_script_draft": run.image_script_draft,
        "generated_image_draft": _image_path_to_url(run.generated_image_draft or "") or run.generated_image_draft,
        "offer_candidates": run.offer_candidates,
        "ad_image_urls": ad_image_urls,
        "snapshot": run.snapshot,
        "pipeline_result": run.pipeline_result,
        "logs": run.logs,
        "error": run.error,
    }


@app.post("/api/campaigns/{run_id}/jd/decision")
def jd_decision(run_id: str, body: DecisionRequest) -> Dict[str, Any]:
    run = _get_run(run_id)
    if run.status not in ("jd_review",):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot submit JD decision in stage '{run.status}'",
        )
    run._jd_decision = (body.approved, body.feedback)
    run._jd_event.set()
    return {"ok": True, "approved": body.approved}


@app.post("/api/campaigns/{run_id}/ads/decision")
def ads_decision(run_id: str, body: DecisionRequest) -> Dict[str, Any]:
    run = _get_run(run_id)
    if run.status not in ("ad_review", "ad_rejected"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot submit ads decision in stage '{run.status}'",
        )
    run._ad_decision = (body.approved, body.feedback)
    run._ad_event.set()
    return {"ok": True, "approved": body.approved}


@app.post("/api/campaigns/{run_id}/ads/text/decision")
def ads_text_decision(run_id: str, body: AdTextDecisionRequest) -> Dict[str, Any]:
    run = _get_run(run_id)
    if run.status != "ad_text_review":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot submit ad text decision in stage '{run.status}'",
        )
    run._ad_text_decision = (body.approved, body.feedback, body.approved_texts)
    run._ad_text_event.set()
    return {"ok": True, "approved": body.approved}


@app.post("/api/campaigns/{run_id}/ads/image-script/decision")
def ads_image_script_decision(run_id: str, body: AdImageScriptDecisionRequest) -> Dict[str, Any]:
    run = _get_run(run_id)
    if run.status != "ad_image_script_review":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot submit image script decision in stage '{run.status}'",
        )
    run._ad_image_script_decision = (body.approved, body.feedback, body.approved_script)
    run._ad_image_script_event.set()
    return {"ok": True, "approved": body.approved}


@app.post("/api/campaigns/{run_id}/ads/image/decision")
def ads_image_decision(run_id: str, body: DecisionRequest) -> Dict[str, Any]:
    run = _get_run(run_id)
    if run.status != "ad_image_review":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot submit image decision in stage '{run.status}'",
        )
    run._ad_image_decision = (body.approved, body.feedback)
    run._ad_image_event.set()
    return {"ok": True, "approved": body.approved}


@app.post("/api/campaigns/{run_id}/offer/approve")
def approve_offer(run_id: str, body: OfferApprovalRequest) -> Dict[str, Any]:
    run = _get_run(run_id)
    if run.status != "awaiting_offer_approval":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot approve offer in stage '{run.status}'",
        )
    try:
        uuid.UUID(body.application_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid application_id UUID") from exc
    run._offer_decision = body.application_id
    run._offer_event.set()
    return {"ok": True, "application_id": body.application_id}


@app.post("/api/campaigns/{run_id}/offer/no-hire")
def no_hire(run_id: str) -> Dict[str, Any]:
    run = _get_run(run_id)
    if run.status != "awaiting_offer_approval":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot submit No Hire decision in stage '{run.status}'",
        )
    run._offer_decision = "NO_HIRE"
    run._offer_event.set()
    return {"ok": True}


@app.post("/api/campaigns/{run_id}/sync")
def sync_forms(run_id: str) -> Dict[str, Any]:
    run = _get_run(run_id)
    if not run.job_id:
        raise HTTPException(status_code=400, detail="Job not created yet")
    pipeline_runtime._ensure_backend()
    pipeline = pipeline_runtime.HiringPipeline(
        session_factory=pipeline_runtime.get_session_factory(),
    )
    form_link = (run.pipeline_result.get("phase12") or {}).get("job_form_link")
    ingested = pipeline_runtime._ingest_new_form_responses(
        pipeline,
        uuid.UUID(run.job_id),
        form_link=form_link,
        shortlist_top_n=max(1, int((run.intake or {}).get("shortlist_top_n", 3))),
        seen_response_ids=run.seen_response_ids,
        forms_client_cache=run.forms_client_cache,
    )
    _update_snapshot(run)
    _log(run, f"Manual form sync: {ingested} new response(s).")
    return {"ingested": ingested, "snapshot": run.snapshot}


@app.post("/api/campaigns/{run_id}/mcq/complete")
def complete_mcq(run_id: str, body: McqCompletionRequest) -> Dict[str, Any]:
    run = _get_run(run_id)
    if not run.job_id:
        raise HTTPException(status_code=400, detail="Job not created yet")
    try:
        app_id = uuid.UUID(body.application_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid application_id UUID") from exc
    pipeline_runtime._ensure_backend()
    pipeline = pipeline_runtime.HiringPipeline(
        session_factory=pipeline_runtime.get_session_factory(),
    )
    result = pipeline.complete_prescreen_mcq(
        app_id,
        passed=bool(body.passed),
        score=body.score,
        metadata=body.metadata,
    )
    _log(run, f"MCQ completed for app={body.application_id}: {result.get('status')}")
    _update_snapshot(run)
    run.pipeline_result.setdefault("mcq", {})[body.application_id] = result
    _persist_run(run)
    return result


@app.post("/api/campaigns/{run_id}/feedback")
def submit_feedback(run_id: str, body: FeedbackRequest) -> Dict[str, Any]:
    """
    HR submits correction feedback for a candidate scoring decision.
    Persists a LearningFeedback row that the Self-Improvement Agent will
    consume on the next learning pass.
    """
    run = _get_run(run_id)
    if not run.job_id:
        raise HTTPException(status_code=400, detail="Job not created yet")
    pipeline_runtime._ensure_backend()
    sf = pipeline_runtime.get_session_factory()

    app_uuid: Optional[uuid.UUID] = None
    if body.application_id:
        try:
            app_uuid = uuid.UUID(body.application_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid application_id UUID") from exc

    from database.repo import session_scope, add_learning_feedback
    with session_scope(sf) as session:
        row = add_learning_feedback(
            session,
            correlation_id=uuid.UUID(run.correlation_id) if run.correlation_id else uuid.uuid4(),
            job_id=uuid.UUID(run.job_id),
            application_id=app_uuid,
            agent_name=body.agent_name,
            feedback_type=body.feedback_type,
            label=body.label,
            notes=body.notes,
            payload=body.payload,
            created_by="hr",
        )
        feedback_id = str(row.id)

    _log(run, f"HR feedback recorded: agent={body.agent_name} label={body.label} app={body.application_id}")
    return {"ok": True, "feedback_id": feedback_id}


@app.get("/api/campaigns/{run_id}/interviews")
def get_interview_results(run_id: str) -> List[Dict[str, Any]]:
    """Return all completed interviews for this campaign with transcripts and scores."""
    run = _get_run(run_id)
    if not run.job_id:
        return []
    pipeline_runtime._ensure_backend()
    sf = pipeline_runtime.get_session_factory()

    from database.repo import session_scope
    from database.models import Interview, InterviewResult as InterviewResultModel, Application
    from sqlalchemy import select

    results = []
    with session_scope(sf) as session:
        rows = session.execute(
            select(Interview)
            .join(Application, Interview.application_id == Application.id)
            .join(InterviewResultModel, InterviewResultModel.interview_id == Interview.id)
            .where(Application.job_id == uuid.UUID(run.job_id))
        ).scalars().all()

        for inv in rows:
            app = inv.application
            candidate = app.candidate if app else None
            result = inv.result
            scores = result.scores if result else {}
            evaluation = scores.get("evaluation", {})
            history = scores.get("history", [])
            results.append({
                "interview_id": str(inv.id),
                "application_id": str(app.id) if app else None,
                "candidate_name": (candidate.name if candidate else None) or "Unknown",
                "candidate_email": (candidate.email if candidate else None) or "",
                "completed_at": result.completed_at.isoformat() if result and result.completed_at else None,
                "overall": round(scores.get("overall", 0) * 100) if scores.get("overall", 0) <= 1 else scores.get("overall", 0),
                "communication": round(scores.get("communication", 0) * 100) if scores.get("communication", 0) <= 1 else scores.get("communication", 0),
                "technical": round(scores.get("technical", 0) * 100) if scores.get("technical", 0) <= 1 else scores.get("technical", 0),
                "recommendation": evaluation.get("recommendation", ""),
                "strengths": evaluation.get("strengths", []),
                "weaknesses": evaluation.get("weaknesses", []),
                "summary": evaluation.get("summary", ""),
                "answer_scores": evaluation.get("answer_scores", []),
                "jd_skills_assessed": evaluation.get("jd_skills_assessed", []),
                "history": history,
                "hr_score_override": None,
                "snapshot_url": scores.get("snapshot_url"),
            })
    return results


@app.patch("/api/campaigns/{run_id}/interviews/{interview_id}/score")
def override_interview_score(run_id: str, interview_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """HR overrides the AI-generated overall score for an interview."""
    _get_run(run_id)
    pipeline_runtime._ensure_backend()
    sf = pipeline_runtime.get_session_factory()

    from database.repo import session_scope
    from database.models import Interview, InterviewResult

    new_score = body.get("overall")
    hr_note = body.get("note", "")

    if new_score is None or not (0 <= float(new_score) <= 100):
        raise HTTPException(status_code=400, detail="overall must be 0–100")

    with session_scope(sf) as session:
        inv = session.get(Interview, uuid.UUID(interview_id))
        if not inv:
            raise HTTPException(status_code=404, detail="Interview not found")
        if inv.result:
            scores = dict(inv.result.scores or {})
            scores["hr_override"] = {"overall": float(new_score), "note": hr_note}
            scores["overall"] = float(new_score) / 100.0
            inv.result.scores = scores

    return {"ok": True, "interview_id": interview_id, "new_overall": new_score}


@app.post("/api/self-improvement/trigger")
def trigger_self_improvement() -> Dict[str, Any]:
    """
    Manually trigger a Self-Improvement Agent pass across all accumulated
    feedback. Returns the full improvement report.
    """
    pipeline_runtime._ensure_backend()
    sf = pipeline_runtime.get_session_factory()
    report = pipeline_runtime.run_learning_pass(sf)
    return {
        "ok": True,
        "total_feedback_rows": report.get("total_feedback_rows", 0),
        "adjustments_applied": len(report.get("adjustments", [])),
        "held_for_review":     len(report.get("held_for_review", [])),
        "overall_insight":     report.get("overall_insight", ""),
        "bias_flags":          report.get("bias_flags", []),
        "prompt_suggestions":  report.get("prompt_suggestions", []),
        "ab_recommendation":   report.get("ab_recommendation"),
        "current_weights":     report.get("current_weights", {}),
        "adjustments":         report.get("adjustments", []),
        "held_details":        report.get("held_for_review", []),
        "llm_used":            report.get("llm_used", False),
    }


@app.get("/api/self-improvement/report")
def get_self_improvement_report() -> Dict[str, Any]:
    """
    Return the current learned weights and full adjustment history
    without triggering a new pass.
    """
    from Pipeline.learning_service import load_weights, load_history
    return {
        "current_weights": load_weights(),
        "adjustment_history": load_history(),
    }


@app.get("/api/policy")
def get_policy() -> Dict[str, Any]:
    """Return the current org-level hiring policy metadata."""
    from database.repo import get_org_settings, session_scope
    from database.session import get_session_factory
    with session_scope(get_session_factory()) as s:
        row = get_org_settings(s)
        return {
            "filename": row.policy_filename,
            "has_policy": bool(row.policy_s3_key),
        }


@app.post("/api/policy/upload")
async def upload_policy(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload an org-wide hiring policy (PDF or DOCX). Replaces any existing policy."""
    allowed = {".pdf", ".docx"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are accepted. If you have a .doc file, open it in Word and save as .docx first.")

    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 10 MB.")

    s3_key = f"policies/org_policy{suffix}"
    s3_service.upload_bytes(data, s3_key, file.content_type or "application/octet-stream")

    from database.repo import upsert_policy, session_scope
    from database.session import get_session_factory
    with session_scope(get_session_factory()) as s:
        upsert_policy(s, s3_key, file.filename)

    return {"ok": True, "filename": file.filename}


@app.delete("/api/policy")
def delete_policy() -> Dict[str, Any]:
    """Remove the custom policy and revert to the default."""
    from database.repo import upsert_policy, session_scope
    from database.session import get_session_factory
    with session_scope(get_session_factory()) as s:
        upsert_policy(s, None, None)
    return {"ok": True}


@app.get("/api/accounts/status")
def accounts_status() -> Dict[str, bool]:
    """Return which social platform credentials are configured."""
    import os
    return {
        "facebook":  bool(os.getenv("FACEBOOK_PAGE_ID") and os.getenv("FACEBOOK_ACCESS_TOKEN")),
        "instagram": bool(os.getenv("INSTAGRAM_ACCOUNT_ID") and os.getenv("FACEBOOK_ACCESS_TOKEN")),
        "x":         bool(os.getenv("TWITTER_API_KEY") and os.getenv("TWITTER_API_SECRET")),
        "linkedin":  bool(os.getenv("LINKEDIN_ACCESS_TOKEN")),
    }


# ─── Candidate application form (HTML) ───────────────────────────────────────

@app.get("/apply/{job_id}", response_class=HTMLResponse)
def apply_form(job_id: str) -> HTMLResponse:
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Apply — SmartHire</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: system-ui, sans-serif; background: #f5f5f5; color: #111; padding: 2rem 1rem; }}
    .card {{ max-width: 640px; margin: 0 auto; background: #fff; border-radius: 12px;
              box-shadow: 0 2px 12px rgba(0,0,0,.08); padding: 2rem; }}
    h1 {{ font-size: 1.5rem; margin-bottom: .25rem; }}
    .sub {{ color: #666; font-size: .9rem; margin-bottom: 1.75rem; }}
    label {{ display: block; font-size: .85rem; font-weight: 600; margin-bottom: .3rem; margin-top: 1rem; }}
    input, textarea, select {{
      width: 100%; padding: .55rem .75rem; border: 1px solid #d1d5db;
      border-radius: 8px; font-size: .95rem; outline: none;
    }}
    input:focus, textarea:focus, select:focus {{ border-color: #6366f1; box-shadow: 0 0 0 3px rgba(99,102,241,.15); }}
    .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
    .full {{ grid-column: 1 / -1; }}
    .req {{ color: #ef4444; margin-left: 2px; }}
    button {{
      margin-top: 1.5rem; width: 100%; padding: .75rem;
      background: #6366f1; color: #fff; border: none; border-radius: 8px;
      font-size: 1rem; font-weight: 600; cursor: pointer;
    }}
    button:hover {{ background: #4f46e5; }}
    button:disabled {{ background: #a5b4fc; cursor: not-allowed; }}
    #msg {{ margin-top: 1rem; padding: .75rem 1rem; border-radius: 8px; display: none; font-size: .9rem; }}
    #msg.ok  {{ background: #ecfdf5; color: #065f46; border: 1px solid #6ee7b7; }}
    #msg.err {{ background: #fef2f2; color: #991b1b; border: 1px solid #fca5a5; }}
  </style>
</head>
<body>
<div class="card">
  <h1>Submit Your Application</h1>
  <p class="sub">Reference: {job_id}</p>

  <form id="applyForm" enctype="multipart/form-data">
    <div class="row">
      <div class="full">
        <label>Full Name <span class="req">*</span></label>
        <input name="name" required placeholder="e.g. Ayesha Khan"/>
      </div>
      <div>
        <label>Email <span class="req">*</span></label>
        <input name="email" type="email" required placeholder="you@example.com"/>
      </div>
      <div>
        <label>Phone <span class="req">*</span></label>
        <input name="phone" type="tel" required placeholder="+92 300 0000000"/>
      </div>
      <div>
        <label>City / Country</label>
        <input name="city" placeholder="e.g. Karachi, Pakistan"/>
      </div>
      <div>
        <label>Current / Most Recent Title</label>
        <input name="current_title" placeholder="e.g. Software Engineer"/>
      </div>
      <div>
        <label>Years of Experience <span class="req">*</span></label>
        <select name="years_experience" required>
          <option value="">Select…</option>
          <option value="0-1">Less than 1 year</option>
          <option value="1-3">1 – 3 years</option>
          <option value="3-5">3 – 5 years</option>
          <option value="5-8">5 – 8 years</option>
          <option value="8+">8+ years</option>
        </select>
      </div>
      <div>
        <label>Availability <span class="req">*</span></label>
        <select name="availability" required>
          <option value="">Select…</option>
          <option value="immediate">Immediately</option>
          <option value="2_weeks">2 weeks notice</option>
          <option value="1_month">1 month notice</option>
          <option value="2_months">2 months notice</option>
          <option value="3_months">3 months notice</option>
        </select>
      </div>
      <div>
        <label>LinkedIn Profile</label>
        <input name="linkedin_url" type="url" placeholder="https://linkedin.com/in/..."/>
      </div>
      <div class="full">
        <label>Resume / CV (PDF, DOC, DOCX — max 5 MB) <span class="req">*</span></label>
        <input name="resume" type="file" accept=".pdf,.doc,.docx" required/>
      </div>
      <div class="full">
        <label>Cover Letter</label>
        <textarea name="cover_letter" rows="4" placeholder="Tell us why you're a great fit…"></textarea>
      </div>
      <div>
        <label>How did you hear about this role?</label>
        <select name="how_heard">
          <option value="">Select…</option>
          <option value="linkedin">LinkedIn</option>
          <option value="facebook">Facebook</option>
          <option value="instagram">Instagram</option>
          <option value="x">X (Twitter)</option>
          <option value="referral">Referral</option>
          <option value="other">Other</option>
        </select>
      </div>
      <div class="full">
        <label>Anything else you'd like us to know?</label>
        <textarea name="additional_notes" rows="2" placeholder="Certifications, projects, etc."></textarea>
      </div>
    </div>
    <button type="submit" id="submitBtn">Submit Application</button>
  </form>
  <div id="msg"></div>
</div>

<script>
  document.getElementById('applyForm').addEventListener('submit', async (e) => {{
    e.preventDefault();
    const btn = document.getElementById('submitBtn');
    const msg = document.getElementById('msg');
    btn.disabled = true;
    btn.textContent = 'Submitting…';
    msg.style.display = 'none';
    try {{
      const fd = new FormData(e.target);
      const res = await fetch('/api/apply/{job_id}', {{ method: 'POST', body: fd }});
      const data = await res.json().catch(() => ({{}}));
      if (res.ok) {{
        msg.className = 'ok';
        msg.textContent = '✅ Application submitted successfully! We will be in touch soon.';
        e.target.reset();
      }} else {{
        throw new Error(data.detail || 'Submission failed');
      }}
    }} catch (err) {{
      msg.className = 'err';
      msg.textContent = '❌ ' + err.message;
      btn.disabled = false;
      btn.textContent = 'Submit Application';
    }}
    msg.style.display = 'block';
  }});
</script>
</div>
</body>
</html>"""
    return HTMLResponse(content=html)


# ─── Candidate application submission ────────────────────────────────────────

@app.post("/api/apply/{run_or_job_id}")
def submit_application(
    run_or_job_id: str,
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(""),
    city: str = Form(""),
    linkedin_url: str = Form(""),
    years_experience: str = Form(""),
    current_title: str = Form(""),
    availability: str = Form(""),
    cover_letter: str = Form(""),
    how_heard: str = Form(""),
    additional_notes: str = Form(""),
    resume: UploadFile = File(None),
) -> Dict[str, Any]:
    # Accept either a run_id (SmartHire campaign UUID) or a direct job_id.
    # If a run with this ID exists, resolve to its job_id.
    job_id = run_or_job_id
    try:
        run = _get_run(run_or_job_id)
        if run.job_id:
            job_id = run.job_id
    except Exception:
        pass  # not a run_id — treat as a direct job_id

    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id format")

    resume_path = ""
    if resume and resume.filename:
        s3_key = f"resumes/{job_id}/{resume.filename}"
        resume_path = s3_service.upload_bytes(resume.file.read(), s3_key, "application/octet-stream")

    try:
        pipeline_runtime._ensure_backend()
        pipeline_runtime.init_db()
        pipeline = pipeline_runtime.HiringPipeline(
            session_factory=pipeline_runtime.get_session_factory(),
        )
        result = pipeline.submit_application(
            job_id=job_uuid,
            email=email,
            name=name or "Candidate",
            phone=phone or None,
            resume_path=resume_path,
            shortlist_top_n=3,
            auto_process_when_threshold_met=False,
        )
        status = result.get("status", "")
        if status == "job_not_found":
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found in database")
        if not result.get("application_id"):
            raise HTTPException(status_code=500, detail=f"Application registration failed: {status}")
        return {
            "ok": True,
            "status": status,
            "application_id": str(result["application_id"]),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/test-email")
async def test_email():
    send_email(
        to="syed.ali@kodexolabs.com",  # send to yourself for testing
        subject="SES Test Email",
        body_html="<h1>It works!</h1><p>AWS SES is connected 🎉</p>"
    )
    return {"message": "Email sent!"}


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("campaign_api:app", host="0.0.0.0", port=8001, reload=True)

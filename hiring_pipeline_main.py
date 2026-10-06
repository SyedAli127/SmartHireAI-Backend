"""
Run the full hiring pipeline with HR JD approval + applicant-threshold trigger.
Requires DATABASE_URL and API keys used by JD/Ad agents.

Example:
  set DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/smarthire
  python hiring_pipeline_main.py --resume path/to/a.pdf --email test@example.com
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_applications_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping invalid JSONL row: %s", line)
            continue
        if not isinstance(item, dict):
            continue
        if not item.get("email") or not item.get("resume_path"):
            logger.warning("Skipping row missing required fields email/resume_path: %s", item)
            continue
        rows.append(item)
    return rows


def _application_key(item: Dict[str, Any]) -> str:
    return f"{str(item.get('email', '')).strip().lower()}|{_resolve_resume_path(str(item.get('resume_path', '')).strip())}"


def _resolve_resume_path(raw_path: str) -> str:
    """Resolve resume path robustly for CLI usage across common input styles."""
    value = (raw_path or "").strip()
    if not value:
        return value

    root = Path(__file__).resolve().parent
    p = Path(value).expanduser()

    # If already absolute and exists, keep it.
    if p.is_absolute() and p.exists():
        return str(p.resolve())

    # Common mistake from docs/CLI: /CV/file.pdf should map to workspace CV/file.pdf.
    if p.is_absolute():
        parts = p.parts
        if len(parts) >= 2 and parts[1].lower() == "cv":
            candidate = root / Path(*parts[1:])
            if candidate.exists():
                return str(candidate.resolve())
        return str(p)

    # Relative path from current working directory first.
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)

    # Then relative to project root.
    root_candidate = (root / p).resolve()
    if root_candidate.exists():
        return str(root_candidate)

    # Keep predictable absolute output for downstream storage/logging.
    return str(cwd_candidate)


def main() -> None:
    # Load project .env so DATABASE_URL does not need to be exported manually.
    _root = Path(__file__).resolve().parent
    load_dotenv(_root / ".env")

    parser = argparse.ArgumentParser(description="SmartHire full pipeline → PostgreSQL")
    parser.add_argument("--resume", help="Path to a candidate resume file")
    parser.add_argument("--email", default="candidate@example.com")
    parser.add_argument("--name", default="Candidate")
    parser.add_argument("--title", default="AI Engineer", help="Job title for JD intake")
    parser.add_argument("--min-applications-threshold", type=int, default=1)
    parser.add_argument("--shortlist-top-n", type=int, default=3)
    parser.add_argument("--hr-auto-approve-jd", action="store_true")
    parser.add_argument("--hr-auto-approve-ads", action="store_true")
    parser.add_argument("--process-pending-job-id", help="Process queued applications for an existing job UUID")
    parser.add_argument("--complete-interview-token", help="Complete interview for a given raw token")
    parser.add_argument("--finalize-job-id", help="Finalize interview stage and send offer/rejection emails")
    parser.add_argument("--submit-application-job-id", help="Submit one candidate application to an existing job UUID")
    parser.add_argument("--auto-drive", action="store_true", help="Auto-complete dispatched interviews and finalize in one run")
    parser.add_argument("--single-command", action="store_true", help="Run as one command service: create job, watch applications, process/finalize")
    parser.add_argument("--applications-jsonl", help="Path to JSONL file with candidate applications for --single-command")
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--max-applications", type=int, default=0, help="Optional stop limit for applications ingested in --single-command mode (0 = no limit)")
    parser.add_argument("--auto-complete-score-overall", type=float, default=0.82)
    parser.add_argument("--auto-complete-score-communication", type=float, default=0.78)
    parser.add_argument("--auto-complete-score-technical", type=float, default=0.85)
    parser.add_argument("--score-overall", type=float, default=0.75)
    parser.add_argument("--score-communication", type=float, default=0.70)
    parser.add_argument("--score-technical", type=float, default=0.80)
    parser.add_argument("--record-feedback", action="store_true", help="Record learning feedback only")
    parser.add_argument("--learning-summary", action="store_true", help="Print aggregated learning feedback")
    parser.add_argument("--correlation-id", help="Correlation UUID for feedback mode")
    parser.add_argument("--agent-name", default="prescreen")
    parser.add_argument("--feedback-type", default="hr_label")
    parser.add_argument("--label")
    parser.add_argument("--notes")
    parser.add_argument("--job-id")
    parser.add_argument("--application-id")
    parser.add_argument("--dry-skip-llm", action="store_true", help="Skip run (smoke DB only)")
    args = parser.parse_args()

    if not os.getenv("DATABASE_URL"):
        logger.error(
            "DATABASE_URL is missing. Add it to %s or set it in the environment "
            "(e.g. postgresql+psycopg2://user:pass@localhost:5432/smarthire).",
            _root / ".env",
        )
        sys.exit(1)

    from Pipeline.hiring_pipeline import HiringPipeline, HiringPipelineConfig

    pipeline = HiringPipeline(
        config=HiringPipelineConfig(
            prescreen_min_hybrid=float(os.getenv("PRESCREEN_MIN_HYBRID", "0.45")),
            hr_auto_approve_jd=bool(args.hr_auto_approve_jd),
            hr_auto_approve_ads=bool(args.hr_auto_approve_ads),
        )
    )
    pipeline.ensure_schema()

    if args.dry_skip_llm:
        logger.info("Dry skip: schema ensured, exiting.")
        return

    if args.record_feedback:
        if not args.correlation_id:
            logger.error("--record-feedback requires --correlation-id")
            sys.exit(1)
        row_id = pipeline.record_learning_feedback(
            correlation_id=args.correlation_id,
            agent_name=args.agent_name,
            feedback_type=args.feedback_type,
            label=args.label,
            notes=args.notes,
            job_id=args.job_id,
            application_id=args.application_id,
            payload={},
        )
        logger.info("Saved learning feedback id=%s", row_id)
        return

    if args.learning_summary:
        logger.info("Learning summary: %s", pipeline.learning_summary())
        return

    if args.process_pending_job_id:
        out = pipeline.process_waiting_applications(
            uuid.UUID(args.process_pending_job_id),
            shortlist_top_n=max(1, int(args.shortlist_top_n)),
        )
        logger.info("Pending processing result: %s", out)
        return

    if args.submit_application_job_id:
        if not args.resume:
            logger.error("--submit-application-job-id requires --resume")
            sys.exit(1)
        out = pipeline.submit_application(
            job_id=uuid.UUID(args.submit_application_job_id),
            email=args.email,
            name=args.name,
            resume_path=_resolve_resume_path(args.resume),
            shortlist_top_n=max(1, int(args.shortlist_top_n)),
        )
        logger.info("Submit application result: %s", out)
        return

    if args.complete_interview_token:
        ok = pipeline.complete_interview_by_token(
            args.complete_interview_token,
            scores={
                "overall": float(args.score_overall),
                "communication": float(args.score_communication),
                "technical": float(args.score_technical),
                "notes": "Completed via CLI",
            },
        )
        logger.info("Interview completion status: %s", ok)
        return

    if args.finalize_job_id:
        out = pipeline.finalize_after_interviews(
            uuid.UUID(args.finalize_job_id),
            top_n=max(1, int(args.shortlist_top_n)),
        )
        logger.info("Finalize result: %s", out)
        return

    intake = {
        "title": args.title,
        "department": "R&D",
        "seniority": "Mid-level",
        "location": "Remote",
        "employment_type": "Full-time",
        "hr_notes": "Python, ML, APIs.",
        "pass_threshold": 6.0,
        "platforms": ["Facebook", "Instagram", "X"],
        "shortlist_top_n": max(1, int(args.shortlist_top_n)),
        "min_applications_threshold": max(1, int(args.min_applications_threshold)),
    }
    candidates = []
    if args.resume:
        candidates.append(
            {
                "email": args.email,
                "name": args.name,
                "resume_path": _resolve_resume_path(args.resume),
            }
        )

    def hr_decision_provider(state: Dict[str, Any]) -> Tuple[bool, str]:
        if args.hr_auto_approve_jd:
            return True, "Approved via --hr-auto-approve-jd"
        jd_preview = (state.get("winner") or {}).get("jd_text") or ""
        print("\n=== JD awaiting HR approval ===\n")
        print(jd_preview[:3000] if jd_preview else "<No JD text>")
        choice = input("\nApprove this JD? (y/n): ").strip().lower()
        if choice == "y":
            feedback = input("Optional feedback: ").strip()
            return True, feedback or "Approved by HR"
        feedback = input("Please provide revision feedback: ").strip()
        return False, feedback or "Rejected by HR for revision"

    def ad_decision_provider(state: Dict[str, Any]) -> Tuple[bool, str, Optional[List[dict]]]:
        if args.hr_auto_approve_ads:
            return True, "Approved via --hr-auto-approve-ads", None
        ads = state.get("combined_ads") or []
        if not ads:
            return False, "No ads generated to approve.", None
        print("\n=== Ads awaiting HR approval ===")
        by_platform: Dict[str, List[Any]] = {}
        for ad in ads:
            platform = ad.platform if hasattr(ad, "platform") else ad.get("platform")
            by_platform.setdefault(platform, []).append(ad)
        approved_ads: List[dict] = []
        for platform, variants in by_platform.items():
            print(f"\nPlatform: {platform}")
            for v in variants:
                variant = v.variant if hasattr(v, "variant") else v.get("variant")
                headline = v.headline if hasattr(v, "headline") else v.get("headline")
                print(f"  Variant {variant}: {headline}")
            pick = input("Pick variant (A/B) or 'r' to reject all: ").strip().upper()
            if pick == "R":
                return False, "Ad pack rejected by HR", None
            selected = None
            for v in variants:
                variant = v.variant if hasattr(v, "variant") else v.get("variant")
                if variant == pick:
                    selected = v
                    break
            if selected is None:
                selected = variants[0]
            if hasattr(selected, "model_dump"):
                approved_ads.append(selected.model_dump())
            else:
                approved_ads.append(dict(selected))
        return True, "Ad pack approved by HR", approved_ads

    out = pipeline.run_full_pipeline(
        intake,
        candidates,
        hr_decision_provider=hr_decision_provider,
        ad_decision_provider=ad_decision_provider,
    )
    logger.info("Pipeline result: %s", out)

    if args.single_command:
        if not args.applications_jsonl:
            logger.error("--single-command requires --applications-jsonl")
            sys.exit(1)
        job_id = out.get("job_id")
        if not job_id:
            logger.error("Single-command mode failed: missing job_id from pipeline output")
            sys.exit(1)

        logger.info(
            "Single-command mode started for job_id=%s, watching %s",
            job_id,
            args.applications_jsonl,
        )

        seen: set[str] = set()
        finalized = False
        ingested = 0
        while True:
            rows = _load_applications_jsonl(args.applications_jsonl)
            new_rows = []
            for r in rows:
                k = _application_key(r)
                if k in seen:
                    continue
                seen.add(k)
                new_rows.append(r)

            for r in new_rows:
                result = pipeline.submit_application(
                    job_id=uuid.UUID(job_id),
                    email=str(r.get("email")).strip(),
                    name=str(r.get("name") or "Candidate").strip(),
                    resume_path=_resolve_resume_path(str(r.get("resume_path")).strip()),
                    shortlist_top_n=max(1, int(args.shortlist_top_n)),
                )
                ingested += 1
                logger.info("Single-command submitted application result: %s", result)

                processing = result.get("processing") if isinstance(result, dict) else None
                dispatch_rows = (processing or {}).get("interview_dispatch") if isinstance(processing, dict) else None
                if dispatch_rows:
                    completed = 0
                    for row in dispatch_rows:
                        token = row.get("interview_token")
                        if not token:
                            continue
                        ok = pipeline.complete_interview_by_token(
                            token,
                            scores={
                                "overall": float(args.auto_complete_score_overall),
                                "communication": float(args.auto_complete_score_communication),
                                "technical": float(args.auto_complete_score_technical),
                                "notes": "Completed via --single-command",
                            },
                        )
                        if ok:
                            completed += 1
                    final = pipeline.finalize_after_interviews(
                        uuid.UUID(job_id),
                        top_n=max(1, int(args.shortlist_top_n)),
                    )
                    finalized = True
                    logger.info("Single-command completed interviews=%s finalize=%s", completed, final)

            if finalized:
                return
            if args.max_applications > 0 and ingested >= int(args.max_applications):
                logger.info("Single-command stopping after max_applications=%s", args.max_applications)
                return

            time.sleep(max(1, int(args.poll_seconds)))

    if not args.auto_drive:
        return

    if out.get("status") != "awaiting_interviews":
        logger.info(
            "Auto-drive stopped: status=%s (only runs when status is awaiting_interviews)",
            out.get("status"),
        )
        return

    dispatch_rows = out.get("interview_dispatch") or []
    completed = 0
    for row in dispatch_rows:
        token = row.get("interview_token")
        if not token:
            continue
        ok = pipeline.complete_interview_by_token(
            token,
            scores={
                "overall": float(args.auto_complete_score_overall),
                "communication": float(args.auto_complete_score_communication),
                "technical": float(args.auto_complete_score_technical),
                "notes": "Completed via --auto-drive",
            },
        )
        if ok:
            completed += 1

    job_id = out.get("job_id")
    if not job_id:
        logger.info("Auto-drive stopped: missing job_id in pipeline output")
        return

    final = pipeline.finalize_after_interviews(
        uuid.UUID(job_id),
        top_n=max(1, int(args.shortlist_top_n)),
    )
    logger.info("Auto-drive completed interviews=%s finalize=%s", completed, final)


if __name__ == "__main__":
    main()

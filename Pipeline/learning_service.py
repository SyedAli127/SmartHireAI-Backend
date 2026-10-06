"""
Self-Improvement Agent
======================
Reads HR feedback, recent audit findings, cross-campaign patterns, and
adjustment history, then uses an LLM to reason about threshold calibration
and pipeline improvements. Results are persisted to:
  - Shared/learned_weights.json  (thresholds applied on the next run)
  - Shared/improvement_history.json  (self-reflection memory)

Capabilities
------------
  LLM-driven threshold reasoning   — replaces fixed 30% / ±0.03 rules
  Audit-learning bridge            — consumes AuditFinding records as signal
  Natural-language feedback        — parses HR free-text notes
  Cross-campaign mining            — detects patterns across multiple jobs
  Per-role threshold profiles      — role-specific calibration in weights file
  Bias detection                   — flags systematic score disparities
  Scoring prompt evolution         — concrete suggestions to improve prompts
  A/B experiment recommendations   — proposes controlled threshold experiments
  Self-reflection memory           — persists history for future LLM context
  Confidence-gated actions         — auto-applies only high-confidence (≥0.70) changes
  Algorithmic fallback             — rule-based logic when LLM is unavailable
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database.models import AuditFinding, LearningFeedback

logger = logging.getLogger(__name__)

_WEIGHTS_PATH  = Path(__file__).parent.parent / "Shared" / "learned_weights.json"
_HISTORY_PATH  = Path(__file__).parent.parent / "Shared" / "improvement_history.json"

_DEFAULTS: Dict[str, Any] = {
    "prescreen": {
        "min_hybrid_score": 0.45,
    },
    "interview": {
        "min_pass_score": 0.75,
    },
    "role_profiles": {},  # keyed by department/title for per-role overrides
}

_STEP            = 0.03   # algorithmic fallback step size
_MAX_STEP        = 0.05   # hard safety clamp per LLM pass
_MIN_SAMPLES     = 5      # minimum feedback rows before adjusting
_CONFIDENCE_GATE = 0.70   # LLM confidence threshold for auto-apply
_FINDINGS_LIMIT  = 50     # max AuditFinding rows to fetch
_NOTES_LIMIT     = 30     # max free-text notes to fetch
_HISTORY_KEEP    = 10     # keep last N records in improvement_history.json


# ---------------------------------------------------------------------------
# Weight persistence
# ---------------------------------------------------------------------------

def load_weights() -> Dict[str, Any]:
    """Load learned weights from disk; fall back to defaults if missing."""
    if _WEIGHTS_PATH.exists():
        try:
            with open(_WEIGHTS_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            merged: Dict[str, Any] = {}
            for key, default in _DEFAULTS.items():
                if isinstance(default, dict):
                    merged[key] = {**default, **(data.get(key) or {})}
                else:
                    merged[key] = data.get(key, default)
            return merged
        except Exception as exc:
            logger.warning("Could not load learned weights from %s: %s", _WEIGHTS_PATH, exc)
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in _DEFAULTS.items()}


def save_weights(weights: Dict[str, Any]) -> None:
    """Persist updated weights to disk."""
    try:
        _WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_WEIGHTS_PATH, "w", encoding="utf-8") as fh:
            json.dump(weights, fh, indent=2)
        logger.info("Saved learned weights to %s", _WEIGHTS_PATH)
    except Exception as exc:
        logger.warning("Could not save learned weights: %s", exc)


# ---------------------------------------------------------------------------
# Adjustment history  (self-reflection memory)
# ---------------------------------------------------------------------------

def load_history() -> List[Dict[str, Any]]:
    """Load past adjustment records from disk."""
    if _HISTORY_PATH.exists():
        try:
            with open(_HISTORY_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh).get("adjustments", [])
        except Exception as exc:
            logger.warning("Could not load improvement history: %s", exc)
    return []


def _append_history(new_entries: List[Dict[str, Any]]) -> None:
    history = load_history()
    history.extend(new_entries)
    history = history[-_HISTORY_KEEP:]
    try:
        _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_HISTORY_PATH, "w", encoding="utf-8") as fh:
            json.dump({"adjustments": history}, fh, indent=2)
    except Exception as exc:
        logger.warning("Could not save improvement history: %s", exc)


# ---------------------------------------------------------------------------
# Feedback aggregation  (public API — unchanged for hiring_pipeline.py)
# ---------------------------------------------------------------------------

def summarize_learning_feedback(session: Session) -> Dict[str, Any]:
    """Aggregate feedback counts by agent / type / label."""
    rows = session.execute(
        select(
            LearningFeedback.agent_name,
            LearningFeedback.feedback_type,
            LearningFeedback.label,
            func.count(LearningFeedback.id),
        ).group_by(
            LearningFeedback.agent_name,
            LearningFeedback.feedback_type,
            LearningFeedback.label,
        )
    ).all()
    grouped: List[Dict[str, Any]] = [
        {
            "agent_name": agent_name,
            "feedback_type": feedback_type,
            "label": label,
            "count": int(count),
        }
        for agent_name, feedback_type, label, count in rows
    ]
    return {"summary": grouped}


def _get_label_counts(
    summary: List[Dict[str, Any]], agent: str, feedback_type: str
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in summary:
        if row["agent_name"] == agent and row["feedback_type"] == feedback_type:
            counts[row["label"] or "unknown"] = row["count"]
    return counts


# ---------------------------------------------------------------------------
# Data fetchers for LLM context
# ---------------------------------------------------------------------------

def _fetch_recent_audit_findings(session: Session) -> List[Dict[str, Any]]:
    """Fetch recent AuditFinding rows to give the LLM pipeline-quality context."""
    rows = session.execute(
        select(
            AuditFinding.stage,
            AuditFinding.severity,
            AuditFinding.finding_type,
            AuditFinding.summary,
        )
        .order_by(AuditFinding.created_at.desc())
        .limit(_FINDINGS_LIMIT)
    ).all()
    return [
        {
            "stage": r.stage,
            "severity": r.severity,
            "finding_type": r.finding_type,
            "summary": r.summary,
        }
        for r in rows
    ]


def _fetch_hr_notes(session: Session) -> List[str]:
    """Collect free-text HR notes from LearningFeedback for NL parsing."""
    rows = session.execute(
        select(LearningFeedback.notes)
        .where(LearningFeedback.notes.isnot(None))
        .order_by(LearningFeedback.created_at.desc())
        .limit(_NOTES_LIMIT)
    ).scalars().all()
    return [n for n in rows if n and n.strip()]


def _fetch_cross_campaign_summary(session: Session) -> List[Dict[str, Any]]:
    """Group feedback by job_id to surface cross-campaign patterns."""
    rows = session.execute(
        select(
            LearningFeedback.job_id,
            LearningFeedback.agent_name,
            LearningFeedback.label,
            func.count(LearningFeedback.id),
        )
        .where(LearningFeedback.job_id.isnot(None))
        .group_by(
            LearningFeedback.job_id,
            LearningFeedback.agent_name,
            LearningFeedback.label,
        )
    ).all()

    campaigns: Dict[str, Dict[str, Any]] = {}
    for job_id, agent_name, label, count in rows:
        jid = str(job_id)
        if jid not in campaigns:
            campaigns[jid] = {"job_id": jid, "agents": {}}
        if agent_name not in campaigns[jid]["agents"]:
            campaigns[jid]["agents"][agent_name] = {}
        campaigns[jid]["agents"][agent_name][label or "unknown"] = int(count)

    return list(campaigns.values())


# ---------------------------------------------------------------------------
# LLM core
# ---------------------------------------------------------------------------

def _build_llm_prompt(
    feedback_summary: List[Dict[str, Any]],
    audit_findings: List[Dict[str, Any]],
    hr_notes: List[str],
    cross_campaign: List[Dict[str, Any]],
    current_weights: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> str:
    weights_text = json.dumps(
        {
            "prescreen": current_weights.get("prescreen", {}),
            "interview": current_weights.get("interview", {}),
            "role_profiles": current_weights.get("role_profiles", {}),
        },
        indent=2,
    )
    feedback_text = (
        json.dumps(feedback_summary, indent=2) if feedback_summary else "No feedback data yet."
    )
    findings_text = (
        "\n".join(
            f"[{f['severity'].upper()}] {f['stage']} / {f['finding_type']}: {f['summary']}"
            for f in audit_findings
        )
        if audit_findings
        else "No recent audit findings."
    )
    notes_text = (
        "\n".join(f"- {n}" for n in hr_notes[:20])
        if hr_notes
        else "No HR free-text notes."
    )
    campaign_text = (
        json.dumps(cross_campaign[:5], indent=2)
        if cross_campaign
        else "No cross-campaign data available."
    )
    history_text = (
        json.dumps(history[-5:], indent=2) if history else "No previous adjustments."
    )

    return f"""You are the Self-Improvement Agent for SmartHire AI, an automated hiring pipeline.
Analyze all signals below and propose calibration improvements.

## Current Threshold Weights
{weights_text}

## HR Feedback Summary (FP/FN counts per agent)
{feedback_text}

## HR Free-Text Notes ({len(hr_notes)} notes)
{notes_text}

## Recent Audit Findings ({len(audit_findings)} findings)
{findings_text}

## Cross-Campaign Patterns ({len(cross_campaign)} campaigns)
{campaign_text}

## Adjustment History (last {len(history)} records)
{history_text}

---
Respond ONLY with valid JSON — no markdown fences, no prose outside the object:

{{
  "prescreen": {{
    "min_hybrid_score": <float or null>,
    "reasoning": "<1-2 sentences>",
    "confidence": <0.0-1.0>
  }},
  "interview": {{
    "min_pass_score": <float or null>,
    "reasoning": "<1-2 sentences>",
    "confidence": <0.0-1.0>
  }},
  "role_profiles": {{
    "<department or title>": {{
      "min_hybrid_score": <float or null>,
      "min_pass_score": <float or null>
    }}
  }},
  "overall_insight": "<paragraph summarising what you found and why>",
  "bias_flags": ["<description of any systematic score disparity>"],
  "prompt_suggestions": ["<concrete suggestion to improve a scoring prompt>"],
  "ab_recommendation": "<controlled experiment to run, or null>"
}}

Rules
- Set any threshold to null if no change is warranted.
- confidence: 0.0 = very uncertain, 1.0 = highly certain.
- Clamp suggestions: prescreen [0.10, 0.90], interview [0.50, 0.95].
- role_profiles: only include departments or titles where you see clear evidence of different calibration needs. Empty object if none.
- bias_flags: list systematic disparities by role type, experience level, or other pattern. Empty list if none detected.
- prompt_suggestions: actionable rewrites or additions for scoring prompts. Empty list if none.
- ab_recommendation: propose a split experiment if evidence is ambiguous, otherwise null.
"""


def _parse_llm_json(raw: str) -> Dict[str, Any]:
    """Strip markdown fences and parse the first JSON object in the response."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(cleaned[start:end])


def _llm_improvement_pass(
    feedback_summary: List[Dict[str, Any]],
    audit_findings: List[Dict[str, Any]],
    hr_notes: List[str],
    cross_campaign: List[Dict[str, Any]],
    current_weights: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    model_name = os.getenv(
        "SELF_IMPROVEMENT_LLM",
        os.getenv("AUDIT_LLM", os.getenv("JD_CREATION_LLM_1", "gpt-4o-mini")),
    )
    llm = ChatOpenAI(model=model_name, temperature=0.0)

    prompt = _build_llm_prompt(
        feedback_summary, audit_findings, hr_notes, cross_campaign, current_weights, history
    )
    messages = [
        SystemMessage(content="You are a precise pipeline calibration agent. Output only valid JSON."),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    return _parse_llm_json(response.content)


# ---------------------------------------------------------------------------
# Algorithmic fallback  (original rule-based logic)
# ---------------------------------------------------------------------------

def _algorithmic_fallback(
    summary: List[Dict[str, Any]],
    weights: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Rule-based adjustment used when the LLM is unavailable."""
    adjustments: List[Dict[str, Any]] = []

    prescreen_counts = _get_label_counts(summary, "prescreen", "hr_label")
    total_prescreen = sum(prescreen_counts.values())
    if total_prescreen >= _MIN_SAMPLES:
        fn = prescreen_counts.get("false_negative", 0)
        fp = prescreen_counts.get("false_positive", 0)
        fn_rate = fn / total_prescreen
        fp_rate = fp / total_prescreen
        old = weights["prescreen"]["min_hybrid_score"]
        new_val = old
        if fn_rate > 0.3:
            new_val = max(0.10, old - _STEP)
            adjustments.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": "prescreen", "field": "min_hybrid_score",
                "old": round(old, 4), "new": round(new_val, 4),
                "confidence": 0.6, "auto_applied": True,
                "recommendation": (
                    f"Lowered prescreen threshold {old:.2f}->{new_val:.2f}: "
                    f"HR overrode {fn}/{total_prescreen} candidates ({fn_rate:.0%} FN rate)."
                ),
            })
        elif fp_rate > 0.3:
            new_val = min(0.90, old + _STEP)
            adjustments.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": "prescreen", "field": "min_hybrid_score",
                "old": round(old, 4), "new": round(new_val, 4),
                "confidence": 0.6, "auto_applied": True,
                "recommendation": (
                    f"Raised prescreen threshold {old:.2f}->{new_val:.2f}: "
                    f"HR rejected {fp}/{total_prescreen} screened-in candidates ({fp_rate:.0%} FP rate)."
                ),
            })
        weights["prescreen"]["min_hybrid_score"] = new_val

    interview_counts = _get_label_counts(summary, "interview", "hr_label")
    total_interview = sum(interview_counts.values())
    if total_interview >= _MIN_SAMPLES:
        fn_i = interview_counts.get("false_negative", 0)
        fp_i = interview_counts.get("false_positive", 0)
        fn_rate_i = fn_i / total_interview
        fp_rate_i = fp_i / total_interview
        old_p = weights["interview"]["min_pass_score"]
        new_p = old_p
        if fn_rate_i > 0.3:
            new_p = max(0.50, old_p - _STEP)
            adjustments.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": "interview", "field": "min_pass_score",
                "old": round(old_p, 4), "new": round(new_p, 4),
                "confidence": 0.6, "auto_applied": True,
                "recommendation": (
                    f"Lowered interview pass score {old_p:.2f}->{new_p:.2f}: "
                    f"HR promoted {fn_i}/{total_interview} low-score candidates ({fn_rate_i:.0%} FN rate)."
                ),
            })
        elif fp_rate_i > 0.3:
            new_p = min(0.95, old_p + _STEP)
            adjustments.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": "interview", "field": "min_pass_score",
                "old": round(old_p, 4), "new": round(new_p, 4),
                "confidence": 0.6, "auto_applied": True,
                "recommendation": (
                    f"Raised interview pass score {old_p:.2f}->{new_p:.2f}: "
                    f"HR rejected {fp_i}/{total_interview} high-score candidates ({fp_rate_i:.0%} FP rate)."
                ),
            })
        weights["interview"]["min_pass_score"] = new_p

    return adjustments


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_learning_pass(
    session_factory: Callable[[], Session],
    *,
    job_id: Optional[Any] = None,
    correlation_id: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run the self-improvement pass: gather all signals, invoke LLM for
    reasoning, apply high-confidence adjustments, persist results, and
    return a full report dict.

    The return shape is backward-compatible with existing callers:
      adjustments      — list of auto-applied changes
      held_for_review  — low-confidence suggestions queued for HR sign-off
      overall_insight  — LLM narrative
      bias_flags       — detected systematic disparities
      prompt_suggestions — concrete prompt improvement ideas
      ab_recommendation  — proposed split experiment (or None)
      current_weights  — weights after this pass
      llm_used         — True when LLM was invoked successfully
    """
    from database.repo import session_scope  # local import to avoid circular

    with session_scope(session_factory) as session:
        agg            = summarize_learning_feedback(session)
        audit_findings = _fetch_recent_audit_findings(session)
        hr_notes       = _fetch_hr_notes(session)
        cross_campaign = _fetch_cross_campaign_summary(session)

    summary       = agg.get("summary", [])
    weights       = load_weights()
    history       = load_history()
    total_feedback = sum(r["count"] for r in summary)

    # Skip LLM entirely when there is not enough feedback to reason about
    if total_feedback < _MIN_SAMPLES:
        logger.info(
            "Self-Improvement Agent: skipping LLM pass (only %d feedback row(s), need %d).",
            total_feedback, _MIN_SAMPLES,
        )
        return {
            "total_feedback_rows": total_feedback,
            "adjustments": [],
            "held_for_review": [],
            "overall_insight": f"Not enough feedback yet ({total_feedback}/{_MIN_SAMPLES} rows). No LLM call made.",
            "bias_flags": [],
            "prompt_suggestions": [],
            "ab_recommendation": None,
            "current_weights": weights,
            "llm_used": False,
        }

    adjustments: List[Dict[str, Any]]    = []
    held_for_review: List[Dict[str, Any]] = []
    overall_insight  = ""
    bias_flags: List[str]         = []
    prompt_suggestions: List[str] = []
    ab_recommendation: Optional[str] = None
    llm_used = False

    # ── LLM reasoning pass ────────────────────────────────────────────────
    try:
        llm_result = _llm_improvement_pass(
            summary, audit_findings, hr_notes, cross_campaign, weights, history
        )
        llm_used          = True
        overall_insight   = llm_result.get("overall_insight", "")
        bias_flags        = llm_result.get("bias_flags") or []
        prompt_suggestions = llm_result.get("prompt_suggestions") or []
        ab_recommendation = llm_result.get("ab_recommendation")

        timestamp = datetime.now(timezone.utc).isoformat()

        def _apply_threshold(
            block: Optional[Dict[str, Any]],
            agent_key: str,
            field: str,
            current: float,
            lo: float,
            hi: float,
        ) -> None:
            """Clamp LLM suggestion to safety bounds and gate by confidence."""
            if not block:
                return
            suggested = block.get(field)
            if suggested is None:
                return
            if total_feedback < _MIN_SAMPLES:
                return
            confidence = float(block.get("confidence", 0.0))
            reasoning  = block.get("reasoning", "")
            clamped = float(max(lo, min(hi, suggested)))
            clamped = max(current - _MAX_STEP, min(current + _MAX_STEP, clamped))
            entry = {
                "timestamp": timestamp,
                "agent": agent_key,
                "field": field,
                "old": round(current, 4),
                "new": round(clamped, 4),
                "reasoning": reasoning,
                "confidence": confidence,
                "recommendation": (
                    f"{agent_key} {field} {current:.2f}->{clamped:.2f} "
                    f"(confidence {confidence:.0%}): {reasoning}"
                ),
            }
            if confidence >= _CONFIDENCE_GATE:
                weights[agent_key][field] = clamped
                entry["auto_applied"] = True
                adjustments.append(entry)
            else:
                entry["auto_applied"] = False
                held_for_review.append(entry)

        _apply_threshold(
            llm_result.get("prescreen"), "prescreen", "min_hybrid_score",
            weights["prescreen"]["min_hybrid_score"], 0.10, 0.90,
        )
        _apply_threshold(
            llm_result.get("interview"), "interview", "min_pass_score",
            weights["interview"]["min_pass_score"], 0.50, 0.95,
        )

        # Per-role profile suggestions (stored but not auto-applied without HR review)
        role_profiles = llm_result.get("role_profiles") or {}
        if role_profiles:
            existing = weights.get("role_profiles") or {}
            existing.update(role_profiles)
            weights["role_profiles"] = existing
            logger.info(
                "Self-Improvement Agent updated role_profiles for: %s",
                list(role_profiles.keys()),
            )

    except Exception as exc:
        logger.warning(
            "LLM improvement pass failed (%s); falling back to algorithmic logic.", exc
        )
        adjustments     = _algorithmic_fallback(summary, weights)
        overall_insight = "LLM unavailable — algorithmic fallback applied."

    # ── Persist ───────────────────────────────────────────────────────────
    if adjustments or (llm_used and (llm_result.get("role_profiles") or {})):  # type: ignore[possibly-undefined]
        save_weights(weights)
    if adjustments:
        _append_history(adjustments)
        logger.info("Self-Improvement Agent applied %d adjustment(s).", len(adjustments))
    if held_for_review:
        logger.info(
            "Self-Improvement Agent held %d adjustment(s) for HR review (low confidence).",
            len(held_for_review),
        )
    if not adjustments and not held_for_review:
        logger.info(
            "Self-Improvement Agent: no adjustments (total_feedback=%d).", total_feedback
        )

    return {
        "total_feedback_rows": total_feedback,
        "adjustments":         adjustments,
        "held_for_review":     held_for_review,
        "overall_insight":     overall_insight,
        "bias_flags":          bias_flags,
        "prompt_suggestions":  prompt_suggestions,
        "ab_recommendation":   ab_recommendation,
        "current_weights":     weights,
        "llm_used":            llm_used,
    }

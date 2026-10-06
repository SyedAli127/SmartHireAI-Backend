"""
Self-Audit Agent
================
Validates scoring quality and flags anomalies at two levels:

  Per-candidate  : audit_prescreen(scores)  / audit_interview(scores)
  Batch/pipeline : audit_batch_prescreen(scores_list) / audit_batch_interview(scores_list)
  Reporting      : generate_audit_summary(findings)

Batch checks include:
  - Score distribution analysis (suspiciously tight cluster → all same score → flag)
  - Outlier detection (IQR method — extreme highs/lows warrant review)
  - Score calibration consistency (std-dev too low = rubber-stamping)
  - Missing explanation coverage

generate_audit_summary uses an LLM to produce a human-readable report for HR.
Falls back to a plain-text summary when the LLM is unavailable.
"""

from __future__ import annotations

import logging
import os
import statistics
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-candidate validators (existing API kept intact)
# ---------------------------------------------------------------------------

def audit_prescreen(scores: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate a single candidate's prescreen output."""
    findings: List[Dict[str, Any]] = []

    hybrid = float(scores.get("hybrid_score", 0.0))
    if hybrid < 0.2:
        findings.append({
            "severity": "medium",
            "finding_type": "low_fit_score",
            "summary": "Very low prescreen hybrid score.",
            "details": {"hybrid_score": hybrid},
        })

    if not str(scores.get("explanation", "")).strip():
        findings.append({
            "severity": "low",
            "finding_type": "missing_explanation",
            "summary": "Prescreen output is missing explanation text.",
            "details": {"keys": sorted(list(scores.keys()))},
        })

    return findings


def audit_interview(scores: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate a single candidate's interview scores."""
    findings: List[Dict[str, Any]] = []

    overall = float(scores.get("overall", 0.0))
    if overall <= 0.0:
        findings.append({
            "severity": "high",
            "finding_type": "invalid_interview_score",
            "summary": "Interview overall score is zero or missing.",
            "details": {"overall": overall},
        })

    for k in ("communication", "technical"):
        if k not in scores:
            findings.append({
                "severity": "low",
                "finding_type": "missing_dimension",
                "summary": f"Interview score is missing dimension: {k}.",
                "details": {"present_keys": sorted(list(scores.keys()))},
            })

    return findings


# ---------------------------------------------------------------------------
# Batch validators
# ---------------------------------------------------------------------------

def audit_batch_prescreen(scores_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyse prescreen scores across all candidates in a pipeline run.
    Flags: rubber-stamp clustering, outliers, missing explanations.
    """
    findings: List[Dict[str, Any]] = []
    if len(scores_list) < 2:
        return findings

    hybrids = [float(s.get("hybrid_score", 0.0)) for s in scores_list]
    n = len(hybrids)
    mean_h = statistics.mean(hybrids)
    stdev_h = statistics.pstdev(hybrids)  # population stdev

    # --- Rubber-stamp detection: everyone has nearly identical scores ---
    if stdev_h < 0.03 and n >= 3:
        findings.append({
            "severity": "medium",
            "finding_type": "score_clustering",
            "summary": (
                f"All {n} prescreen scores are suspiciously similar "
                f"(mean={mean_h:.3f}, stdev={stdev_h:.4f}). "
                "Possible rubber-stamping or model collapse."
            ),
            "details": {"mean": mean_h, "stdev": stdev_h, "n": n},
        })

    # --- Outlier detection via IQR ---
    sorted_h = sorted(hybrids)
    q1 = sorted_h[n // 4]
    q3 = sorted_h[(3 * n) // 4]
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    outlier_indices = [i for i, v in enumerate(hybrids) if v < lower_fence or v > upper_fence]
    if outlier_indices:
        findings.append({
            "severity": "low",
            "finding_type": "score_outliers",
            "summary": (
                f"{len(outlier_indices)} candidate(s) have outlier prescreen scores "
                f"(outside IQR fence [{lower_fence:.3f}, {upper_fence:.3f}]). Manual review recommended."
            ),
            "details": {
                "outlier_indices": outlier_indices,
                "outlier_scores": [hybrids[i] for i in outlier_indices],
                "q1": q1,
                "q3": q3,
            },
        })

    # --- Missing explanation coverage ---
    missing_expl = sum(
        1 for s in scores_list if not str(s.get("explanation", "")).strip()
    )
    if missing_expl > 0:
        findings.append({
            "severity": "low",
            "finding_type": "missing_explanations",
            "summary": f"{missing_expl}/{n} prescreen results are missing explanations.",
            "details": {"missing_count": missing_expl, "total": n},
        })

    return findings


def audit_batch_interview(scores_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyse interview scores across all candidates in a pipeline run.
    Flags: all-zero scores, calibration issues, missing dimensions, outliers.
    """
    findings: List[Dict[str, Any]] = []
    if len(scores_list) < 2:
        return findings

    overalls = [float(s.get("overall", 0.0)) for s in scores_list]
    n = len(overalls)

    # --- All-zero check ---
    zero_count = sum(1 for v in overalls if v <= 0.0)
    if zero_count > 0:
        findings.append({
            "severity": "high",
            "finding_type": "zero_interview_scores",
            "summary": f"{zero_count}/{n} candidates have zero overall interview scores.",
            "details": {"zero_count": zero_count},
        })

    valid_overalls = [v for v in overalls if v > 0.0]
    if len(valid_overalls) < 2:
        return findings

    mean_o  = statistics.mean(valid_overalls)
    stdev_o = statistics.pstdev(valid_overalls)

    # --- Rubber-stamp detection ---
    if stdev_o < 0.02 and len(valid_overalls) >= 3:
        findings.append({
            "severity": "medium",
            "finding_type": "interview_score_clustering",
            "summary": (
                f"Interview scores are suspiciously uniform "
                f"(mean={mean_o:.3f}, stdev={stdev_o:.4f}). "
                "Possible synthetic or uncalibrated scoring."
            ),
            "details": {"mean": mean_o, "stdev": stdev_o, "n": len(valid_overalls)},
        })

    # --- Score calibration: check communication vs technical variance ---
    for dim in ("communication", "technical"):
        dim_scores = [float(s[dim]) for s in scores_list if dim in s]
        if len(dim_scores) >= 2:
            dim_stdev = statistics.pstdev(dim_scores)
            if dim_stdev < 0.02:
                findings.append({
                    "severity": "low",
                    "finding_type": f"{dim}_score_clustering",
                    "summary": (
                        f"{dim.capitalize()} scores show no differentiation "
                        f"(stdev={dim_stdev:.4f}). Consider re-calibrating the interview rubric."
                    ),
                    "details": {"dimension": dim, "stdev": dim_stdev},
                })

    # --- Missing dimension coverage ---
    for dim in ("communication", "technical"):
        missing = sum(1 for s in scores_list if dim not in s)
        if missing > 0:
            findings.append({
                "severity": "low",
                "finding_type": f"missing_{dim}_dimension",
                "summary": f"{missing}/{n} interview results missing '{dim}' dimension.",
                "details": {"dimension": dim, "missing_count": missing},
            })

    return findings


# ---------------------------------------------------------------------------
# Audit summary generator
# ---------------------------------------------------------------------------

def generate_audit_summary(findings: List[Dict[str, Any]]) -> str:
    """
    Produce a human-readable audit summary for HR.
    Uses LLM when available; falls back to plain-text format.
    """
    if not findings:
        return "No audit findings for this pipeline run. All checks passed."

    # Severity counts
    counts: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for f in findings:
        sev = f.get("severity", "low")
        counts[sev] = counts.get(sev, 0) + 1

    try:
        return _llm_audit_summary(findings, counts)
    except Exception as e:
        logger.warning("LLM audit summary failed (%s); using plain-text fallback.", e)
        return _plain_audit_summary(findings, counts)


def _llm_audit_summary(
    findings: List[Dict[str, Any]],
    counts: Dict[str, int],
) -> str:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    model_name = os.getenv("AUDIT_LLM", os.getenv("JD_CREATION_LLM_1", "gpt-4o-mini"))
    llm = ChatOpenAI(model=model_name, temperature=0.0)

    findings_text = "\n".join(
        f"[{f.get('severity','?').upper()}] {f.get('finding_type','?')}: {f.get('summary','')}"
        for f in findings
    )

    prompt = (
        f"You are an HR audit analyst. Summarise these pipeline audit findings for the HR manager.\n\n"
        f"Finding counts — High: {counts['high']}, Medium: {counts['medium']}, Low: {counts['low']}\n\n"
        f"Findings:\n{findings_text}\n\n"
        "Write a concise (under 200 words) plain-English summary explaining what was found, "
        "what it means for the hiring decision, and any recommended actions. "
        "Do not use bullet points — use short paragraphs."
    )

    messages = [
        SystemMessage(content="You are an objective HR pipeline audit analyst."),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def _plain_audit_summary(
    findings: List[Dict[str, Any]],
    counts: Dict[str, int],
) -> str:
    lines = [
        f"Audit complete — {counts['high']} high, {counts['medium']} medium, {counts['low']} low severity findings.\n"
    ]
    for sev in ("high", "medium", "low"):
        batch = [f for f in findings if f.get("severity") == sev]
        if batch:
            lines.append(f"{'=' * 10} {sev.upper()} {'=' * 10}")
            for f in batch:
                lines.append(f"  [{f['finding_type']}] {f['summary']}")
            lines.append("")
    return "\n".join(lines)

"""
Offer Letter Generator
======================
Generates a professional offer letter using an LLM, then builds a structured
payload that is persisted to the database and emailed to the selected candidate.

Falls back to the template-based `build_offer_payload` if the LLM call fails.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def generate_offer_letter(
    *,
    job_title: str,
    candidate_name: str,
    role: str,
    start_date: str,
    salary_band: str,
    department: Optional[str] = None,
    location: Optional[str] = None,
    employment_type: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a professional offer letter via LLM.

    Returns a structured dict with:
        - letter_text   : full formatted letter (plain text)
        - format        : "llm_offer_v1" or "structured_offer_v1" (fallback)
        - organization  : company name from env OFFER_ORG_NAME
        - candidate_name, role, job_title, start_date, salary_band, ...
    """
    org = os.getenv("OFFER_ORG_NAME", "SmartHire")

    base = {
        "format": "llm_offer_v1",
        "organization": org,
        "candidate_name": candidate_name,
        "role": role or job_title,
        "job_title": job_title,
        "department": department,
        "location": location,
        "employment_type": employment_type or "Full-time",
        "start_date": start_date,
        "salary_band": salary_band,
    }
    if extra:
        base["extra"] = extra

    try:
        letter_text = _llm_generate_letter(
            org=org,
            candidate_name=candidate_name,
            role=role or job_title,
            job_title=job_title,
            department=department,
            location=location,
            employment_type=employment_type or "Full-time",
            start_date=start_date,
            salary_band=salary_band,
        )
        base["letter_text"] = letter_text
        logger.info("Offer letter generated via LLM for %s", candidate_name)
    except Exception as e:
        logger.warning("LLM offer letter generation failed (%s); using template fallback.", e)
        base["format"] = "structured_offer_v1"
        base["letter_text"] = _template_letter(
            org=org,
            candidate_name=candidate_name,
            role=role or job_title,
            start_date=start_date,
            salary_band=salary_band,
        )

    return base


def _llm_generate_letter(
    *,
    org: str,
    candidate_name: str,
    role: str,
    job_title: str,
    department: Optional[str],
    location: Optional[str],
    employment_type: str,
    start_date: str,
    salary_band: str,
) -> str:
    """Call the LLM to produce a professional, warm offer letter."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    model_name = os.getenv("OFFER_LETTER_LLM", os.getenv("JD_CREATION_LLM_1", "gpt-4o-mini"))
    llm = ChatOpenAI(model=model_name, temperature=0.4)

    dept_line   = f"Department   : {department}" if department else ""
    loc_line    = f"Location     : {location}"   if location   else ""
    today       = date.today().strftime("%B %d, %Y")

    prompt = f"""
You are an HR specialist at {org}. Write a professional, warm, and concise offer letter.

Candidate : {candidate_name}
Role      : {role} ({job_title})
{dept_line}
{loc_line}
Employment: {employment_type}
Start Date: {start_date}
Compensation: {salary_band}
Today's Date: {today}

Requirements:
- Use formal but friendly tone
- Start with "Dear {candidate_name},"
- Congratulate the candidate on their selection
- Mention the role, start date, and compensation clearly
- Ask for written acceptance within 5 business days
- End with "Sincerely, {org} Talent Acquisition Team"
- Keep it under 300 words
- Do NOT include any placeholders — use the exact details above
""".strip()

    messages = [
        SystemMessage(content=f"You are the HR offer letter writer for {org}."),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def _template_letter(
    *,
    org: str,
    candidate_name: str,
    role: str,
    start_date: str,
    salary_band: str,
) -> str:
    today = date.today().isoformat()
    return (
        f"{org}\n\n"
        f"{today}\n\n"
        f"Dear {candidate_name},\n\n"
        f"We are pleased to offer you the position of {role}. "
        f"Your proposed start date is {start_date}. "
        f"Compensation: {salary_band}.\n\n"
        f"Please confirm your acceptance within 5 business days.\n\n"
        f"Sincerely,\n{org} Talent Acquisition Team"
    )


# ---------------------------------------------------------------------------
# Public re-export: keep backward compatibility with Pipeline.offer_service
# ---------------------------------------------------------------------------
from Pipeline.offer_service import build_offer_payload  # noqa: F401, E402

__all__ = ["generate_offer_letter", "build_offer_payload"]

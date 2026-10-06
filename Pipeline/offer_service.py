import os
from datetime import date
from typing import Any, Dict, Optional


def build_offer_payload(
    *,
    job_title: str,
    candidate_name: str,
    role: str,
    start_date: str,
    salary_band: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    org = os.getenv("OFFER_ORG_NAME", "SmartHire")
    body = {
        "format": "structured_offer_v1",
        "organization": org,
        "candidate_name": candidate_name,
        "role": role or job_title,
        "job_title": job_title,
        "start_date": start_date,
        "salary_band": salary_band,
        "letter_text": (
            f"{org}\n\n{date.today().isoformat()}\n\n"
            f"Dear {candidate_name},\n\n"
            f"We are pleased to offer you the position of {role or job_title}. "
            f"Proposed start date: {start_date}. Compensation: {salary_band}.\n\n"
            f"Please confirm your acceptance.\n\nSincerely,\n{org} Talent Team"
        ),
    }
    if extra:
        body["extra"] = extra
    return body

import hashlib
import secrets
from typing import Tuple


def mint_interview_token() -> Tuple[str, str]:
    raw = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return raw, token_hash


def hash_interview_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

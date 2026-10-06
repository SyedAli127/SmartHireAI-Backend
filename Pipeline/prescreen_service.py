import os
import re
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import requests

from langchain_openai import OpenAIEmbeddings

from Parser.resume_parser import parse_resume
from Resume_Parser.resumeparserapp import hybrid_semantic_score

PRESCREEN_PROMPT_VERSION = os.getenv("PRESCREEN_PROMPT_VERSION", "hybrid-v1")
RESUME_EMBED_MODEL = os.getenv("RESUME_EMBED_MODEL", "text-embedding-3-small")
PRESCREEN_MODEL_ID = os.getenv("PRESCREEN_MODEL_ID", f"openai:{RESUME_EMBED_MODEL}+gpt-4o-mini")


def _extract_drive_file_id(url: str) -> Optional[str]:
    parsed = urlparse(url)
    if "drive.google.com" not in (parsed.netloc or ""):
        return None

    qs = parse_qs(parsed.query or "")
    if "id" in qs and qs["id"]:
        return qs["id"][0]

    match = re.search(r"/file/d/([^/]+)", parsed.path or "")
    if match:
        return match.group(1)

    return None


def _guess_suffix_from_url(url: str, response: requests.Response) -> str:
    path = urlparse(url).path or ""
    suffix = Path(path).suffix.lower()
    if suffix:
        return suffix

    content_type = (response.headers.get("content-type") or "").lower()
    if "pdf" in content_type:
        return ".pdf"
    if "word" in content_type or "docx" in content_type:
        return ".docx"
    if "text/plain" in content_type:
        return ".txt"
    return ".pdf"


def _download_resume_to_temp(url: str) -> str:
    source_url = url
    drive_file_id = _extract_drive_file_id(url)
    if drive_file_id:
        source_url = f"https://drive.google.com/uc?export=download&id={drive_file_id}"

    resp = requests.get(source_url, timeout=30)
    resp.raise_for_status()

    suffix = _guess_suffix_from_url(url, resp)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(resp.content)
        return tmp.name


def _materialize_resume_path(resume_path: str) -> tuple[str, bool]:
    value = (resume_path or "").strip()
    if value.startswith("http://") or value.startswith("https://"):
        local_path = _download_resume_to_temp(value)
        return local_path, True
    return value, False


def run_prescreen(jd_text: str, resume_path: str) -> dict:
    local_path, is_temp = _materialize_resume_path(resume_path)
    try:
        parsed = parse_resume(local_path)
        resume_text = parsed["cleaned"]
        embeddings = OpenAIEmbeddings(model=RESUME_EMBED_MODEL)
        scores = hybrid_semantic_score(jd_text, resume_text, embeddings)
        return {"resume_text": resume_text, "scores": scores}
    finally:
        if is_temp:
            try:
                os.remove(local_path)
            except OSError:
                pass

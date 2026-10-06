# resume_parser.py
import os
from typing import Iterable, List, Union
import whisper

_whisper_model = whisper.load_model("base")  # or "small", "medium"


import docx2txt

try:
    import pypdf  # type: ignore
except ImportError:
    pypdf = None

try:
    import pdfminer.high_level as _pdfminer  # type: ignore
except ImportError:
    _pdfminer = None

try:
    # OCR dependencies (optional — requires poppler on system)
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
    from pdf2image import convert_from_path  # type: ignore
except ImportError:
    pytesseract = None
    Image = None
    convert_from_path = None

try:
    # Audio dependencies
    import speech_recognition as sr  # type: ignore
    from pydub import AudioSegment  # type: ignore
except ImportError:
    sr = None
    AudioSegment = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".mp4", ".wma"}


def read_resume(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return read_pdf(file_path)

    if ext == ".docx":
        return docx2txt.process(file_path)

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if ext in IMAGE_EXTS:
        return read_image(file_path)

    if ext in AUDIO_EXTS:
        return read_audio(file_path)

    raise Exception(f"Unsupported file format: {ext}")


# -------------------------
# PDF extraction (text-first, OCR fallback)
# -------------------------
def _extract_pdf_text_pypdf(path: str) -> str:
    """Extract text from PDF using pypdf (no external binaries needed)."""
    reader = pypdf.PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"\n\n--- Page {i + 1} ---\n{text}")
    return "".join(pages)


def _extract_pdf_text_pdfminer(path: str) -> str:
    """Extract text from PDF using pdfminer (no external binaries needed)."""
    return _pdfminer.extract_text(path) or ""


def _extract_pdf_ocr(path: str) -> str:
    """Rasterise PDF pages with poppler then OCR with Tesseract."""
    images = convert_from_path(path, dpi=300)
    pages = []
    for idx, img in enumerate(images):
        page_text = pytesseract.image_to_string(img)
        pages.append(f"\n\n--- Page {idx + 1} ---\n{page_text}")
    return "".join(pages)


def read_pdf(path: str) -> str:
    """
    Extract text from a PDF using the best available method:
      1. pypdf  (pure Python, no system deps)
      2. pdfminer.six  (pure Python fallback)
      3. pdf2image + Tesseract OCR  (requires poppler + tesseract — scanned PDFs)
    """
    # 1. pypdf — fast, no dependencies
    if pypdf is not None:
        try:
            text = _extract_pdf_text_pypdf(path)
            if len(text.strip()) > 50:
                return text
        except Exception:
            pass

    # 2. pdfminer — handles more complex layouts
    if _pdfminer is not None:
        try:
            text = _extract_pdf_text_pdfminer(path)
            if len(text.strip()) > 50:
                return text
        except Exception:
            pass

    # 3. OCR fallback (scanned / image-based PDFs) — requires poppler
    if convert_from_path is not None and pytesseract is not None:
        try:
            return _extract_pdf_ocr(path)
        except Exception as e:
            raise Exception(f"All PDF extraction methods failed. Last error: {e}")

    raise Exception(
        "Cannot extract text from PDF: no suitable library found. "
        "Install pypdf (pip install pypdf) or pdfminer.six (pip install pdfminer.six). "
        "For scanned PDFs also install poppler and pytesseract."
    )


# Keep old name as alias so any direct callers don't break
read_pdf_as_images = read_pdf


# -------------------------
# IMAGE OCR
# -------------------------
def read_image(path: str) -> str:
    if pytesseract is None or Image is None:
        raise ImportError(
            "pytesseract and pillow are required for image OCR."
        )

    try:
        img = Image.open(path)
        return pytesseract.image_to_string(img)
    except Exception as e:
        raise Exception(f"Error reading image: {e}")


# -------------------------
# AUDIO → TEXT
# -------------------------
def read_audio(path: str) -> str:
    if sr is None:
        raise ImportError(
            "SpeechRecognition is required. Install: pip install SpeechRecognition pydub"
        )

    recognizer = sr.Recognizer()
    ext = os.path.splitext(path)[1].lower()

    # Convert to WAV if needed
    if ext != ".wav":
        if AudioSegment is None:
            raise ImportError(
                "pydub is required for non-WAV formats. Install ffmpeg."
            )
        audio = AudioSegment.from_file(path)
        wav_path = path.rsplit(".", 1)[0] + "_temp.wav"
        audio.export(wav_path, format="wav")
        path = wav_path

    full_text = []

    with sr.AudioFile(path) as source:
        while True:
            try:
                # Read audio in 30-second chunks
                audio_data = recognizer.record(source, duration=30)
                if not audio_data.frame_data:
                    break

                chunk_text = recognizer.recognize_google(audio_data)
                full_text.append(chunk_text)

            except sr.UnknownValueError:
                # Silence or unclear chunk → skip
                continue
            except sr.RequestError as e:
                raise Exception(f"Speech recognition error: {e}")

    return " ".join(full_text).strip()


# -------------------------
# CLEANING & PARSING
# -------------------------
def clean_text(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(line.strip() for line in lines if line.strip())


def parse_resume(file_path: Union[str, Iterable[str]]):
    if isinstance(file_path, (list, tuple, set)):
        return [parse_resume(p) for p in file_path]

    raw = read_resume(file_path)
    cleaned = clean_text(raw)

    return {
        "raw_text": raw,
        "cleaned": cleaned
    }


def parse_resumes(file_paths: Iterable[str]) -> List[dict]:
    return [parse_resume(path) for path in file_paths]

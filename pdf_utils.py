"""PDF text extraction.

PyMuPDF handles the manuals and oddly-encoded scans better; pypdf is the fallback
so the app still works if PyMuPDF isn't installed.
"""

from __future__ import annotations

from pypdf import PdfReader

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency
    fitz = None


def _with_pymupdf(data: bytes) -> str:
    if fitz is None:
        return ""
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            return "".join(page.get_text("text") or "" for page in doc)
    except Exception:
        return ""


def _with_pypdf(upload) -> str:
    try:
        upload.seek(0)
        reader = PdfReader(upload)
        return "".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def extract_documents(uploads) -> dict[str, str]:
    """Map each uploaded file's name to its extracted text."""
    texts: dict[str, str] = {}
    for upload in uploads or []:
        upload.seek(0)
        text = _with_pymupdf(upload.read())
        if not text.strip():
            text = _with_pypdf(upload)
        texts[upload.name] = text
    return texts

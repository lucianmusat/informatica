"""Text extraction for uploaded documents: PDF, DOCX, plain text and Markdown.

PyMuPDF handles the manuals and oddly-encoded scans better; pypdf is the PDF
fallback so the app still works if PyMuPDF isn't installed.

``extract`` raises ``ExtractError`` with a message fit to show the user.
"""

from __future__ import annotations

import io
from pathlib import PurePath

from pypdf import PdfReader

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency
    fitz = None

FILE_TYPES = ("pdf", "docx", "txt", "md")


class ExtractError(Exception):
    pass


def extension(name: str) -> str:
    return PurePath(name).suffix.lower().lstrip(".")


def _pdf_with_pymupdf(data: bytes) -> str:
    if fitz is None:
        return ""
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            return "".join(page.get_text("text") or "" for page in doc)
    except Exception:
        return ""


def _pdf_with_pypdf(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        return "".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def _pdf(data: bytes) -> str:
    if not data.startswith(b"%PDF"):
        raise ExtractError("not a valid PDF file")
    text = _pdf_with_pymupdf(data)
    if not text.strip():
        text = _pdf_with_pypdf(data)
    if not text.strip():
        raise ExtractError("no text found; is it a scan?")
    return text


def _docx(data: bytes) -> str:
    import docx  # python-docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:  # BadZipFile, PackageNotFoundError, KeyError, ...
        raise ExtractError("not a valid Word (.docx) file") from exc

    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            # Merged cells repeat across a row; keep each once.
            cells = list(dict.fromkeys(cell.text.strip() for cell in row.cells))
            parts.append(" | ".join(c for c in cells if c))
    return "\n".join(p for p in parts if p.strip())


def _txt(data: bytes) -> str:
    if b"\x00" in data[:4096]:
        raise ExtractError("doesn't look like a text file")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


_PARSERS = {"pdf": _pdf, "docx": _docx, "txt": _txt, "md": _txt}  # Markdown is kept as is


def extract(name: str, data: bytes) -> str:
    """The text of one uploaded file. Raises ExtractError if there is none."""
    parser = _PARSERS.get(extension(name))
    if parser is None:
        raise ExtractError(f"unsupported file type; use {', '.join('.' + t for t in FILE_TYPES)}")
    text = parser(data)
    if not text.strip():
        raise ExtractError("the file contains no text")
    return text

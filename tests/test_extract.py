import io

import pytest

from extract import FILE_TYPES, ExtractError, extract


def make_pdf(text):
    import fitz

    doc = fitz.open()
    if text:
        doc.new_page().insert_text((72, 72), text)
    else:
        doc.new_page()  # like a scan: a page with no text layer
    return doc.tobytes()


def make_docx(paragraphs, table=None):
    import docx

    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    if table:
        t = d.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, value in enumerate(row):
                t.cell(r, c).text = value
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def test_pdf():
    assert "Descale every month" in extract("Manual.PDF", make_pdf("Descale every month"))


def test_pdf_without_text_layer():
    with pytest.raises(ExtractError, match="scan"):
        extract("scan.pdf", make_pdf(""))


def test_corrupt_pdf():
    with pytest.raises(ExtractError, match="not a valid PDF"):
        extract("broken.pdf", b"this is not a pdf")
    with pytest.raises(ExtractError):
        extract("truncated.pdf", make_pdf("hello")[:200])


def test_docx_paragraphs_and_tables():
    text = extract("notes.docx", make_docx(["First point", "Second point"],
                                           table=[["Model", "Watts"], ["T1000", "50"]]))
    assert "First point\nSecond point" in text
    assert "Model | Watts" in text and "T1000 | 50" in text


def test_corrupt_docx():
    with pytest.raises(ExtractError, match="Word"):
        extract("notes.docx", b"PK\x03\x04 not really a zip")


def test_txt_encodings():
    assert extract("a.txt", "één café".encode("utf-8")) == "één café"
    assert extract("a.txt", "﻿hi".encode("utf-8")) == "hi"  # BOM dropped
    assert extract("a.txt", "café €5".encode("cp1252")) == "café €5"


def test_binary_named_txt():
    with pytest.raises(ExtractError, match="text file"):
        extract("photo.txt", b"\x89PNG\r\n\x1a\n\x00\x00\x00")


def test_empty_and_unsupported():
    with pytest.raises(ExtractError, match="no text"):
        extract("empty.txt", b"  \n ")
    with pytest.raises(ExtractError, match="unsupported"):
        extract("old.doc", b"\xd0\xcf\x11\xe0")
    with pytest.raises(ExtractError, match="unsupported"):
        extract("noextension", b"hello")


def test_markdown_is_kept_as_text():
    md = "# Setup\n\n- Install **foo**\n- Run `bar`\n"
    assert extract("NOTES.MD", md.encode("utf-8")) == md
    assert "md" in FILE_TYPES

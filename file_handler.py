"""
file_handler.py — Process uploaded files (images, PDFs, DOCX, text) for ZenZone AI Panel
"""

import base64
import io
from fastapi import UploadFile


async def process_upload(file: UploadFile) -> dict:
    """
    Read an uploaded file and return a dict with:
      filename, file_type, is_image, text_content, image_base64 (if image)
    """
    raw = await file.read()
    filename = file.filename or "upload"
    ctype = file.content_type or ""

    result = {
        "filename": filename,
        "file_type": ctype,
        "is_image": False,
        "text_content": None,
        "image_base64": None,
        "image_media_type": None,
    }

    if ctype.startswith("image/") or _ext(filename) in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        result["is_image"] = True
        result["image_base64"] = base64.b64encode(raw).decode()
        result["image_media_type"] = ctype or "image/png"
        result["text_content"] = f"[Image uploaded: {filename}]"

    elif ctype == "application/pdf" or _ext(filename) == ".pdf":
        result["text_content"] = _extract_pdf(raw, filename)

    elif _ext(filename) in (".docx",):
        result["text_content"] = _extract_docx(raw, filename)

    elif ctype.startswith("text/") or _ext(filename) in (".txt", ".csv", ".md", ".json", ".xml", ".yaml", ".yml"):
        try:
            result["text_content"] = raw.decode("utf-8", errors="replace")
        except Exception:
            result["text_content"] = f"[Could not decode text file: {filename}]"

    else:
        result["text_content"] = f"[File uploaded: {filename} — text extraction not supported for this format]"

    return result


def _ext(filename: str) -> str:
    return ("." + filename.rsplit(".", 1)[-1]).lower() if "." in filename else ""


def _extract_pdf(raw: bytes, filename: str) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n\n".join(p for p in pages if p.strip())
        return f"[PDF: {filename}]\n\n{text}" if text else f"[PDF: {filename} — no extractable text found]"
    except ImportError:
        return _extract_pdf_fallback(raw, filename)
    except Exception as e:
        return f"[PDF: {filename} — extraction error: {e}]"


def _extract_pdf_fallback(raw: bytes, filename: str) -> str:
    """Fallback using PyPDF2 if pdfplumber not available."""
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(raw))
        pages = [reader.pages[i].extract_text() or "" for i in range(len(reader.pages))]
        text = "\n\n".join(p for p in pages if p.strip())
        return f"[PDF: {filename}]\n\n{text}" if text else f"[PDF: {filename} — no extractable text]"
    except Exception as e:
        return f"[PDF: {filename} — could not extract text: {e}]"


def _extract_docx(raw: bytes, filename: str) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(raw))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs)
        return f"[Document: {filename}]\n\n{text}" if text else f"[Document: {filename} — no text found]"
    except Exception as e:
        return f"[Document: {filename} — extraction error: {e}]"

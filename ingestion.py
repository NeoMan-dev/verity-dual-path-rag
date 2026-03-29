"""
ingestion.py — Document ingestion pipeline
Validates, extracts, cleans, and chunks uploaded files.
Supports: PDF, CSV, TXT

Changes from v1:
- CHUNK_SIZE: 500 → 1000 chars (avoids splitting product/price rows mid-entry)
- CHUNK_OVERLAP: 50 → 150 chars (more context continuity across boundaries)
- Added lowercase normalization for the embedding text field to eliminate
  case-sensitivity mismatches during cosine similarity search. The original
  display text is preserved separately so citations still show proper casing.
- Improved sentence-boundary snapping: also checks for newlines (better for
  price lists and tables that use newlines as row delimiters).
"""

import csv
import hashlib
import io
import logging
import re
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".pdf", ".csv", ".txt"}
MAX_FILE_SIZE_MB   = 20
CHUNK_SIZE         = 1000  # characters — increased from 500
CHUNK_OVERLAP      = 150   # characters — increased from 50


# ── Validation ────────────────────────────────────────────────────────────────

def validate_file(filename: str, content: bytes) -> tuple[bool, str]:
    """
    Validate file before processing.
    Returns (is_valid, error_message).
    """
    path = Path(filename)

    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type '{ext}'. Allowed: PDF, CSV, TXT."

    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({size_mb:.1f}MB). Max: {MAX_FILE_SIZE_MB}MB."

    if len(content) == 0:
        return False, "File is empty."

    if ext == ".pdf" and not content.startswith(b"%PDF"):
        return False, "File claims to be PDF but is not a valid PDF."

    return True, ""


# ── Text Extraction ───────────────────────────────────────────────────────────

def extract_text_pdf(content: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(content))
        pages  = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except Exception as e:
        raise ValueError(f"PDF extraction failed: {e}")


def extract_text_csv(content: bytes) -> str:
    """Convert CSV to readable text — header + rows as key:value pairs."""
    try:
        text    = content.decode("utf-8", errors="replace")
        reader  = csv.DictReader(io.StringIO(text))
        rows    = []
        headers = None
        for i, row in enumerate(reader):
            if headers is None:
                headers = [str(k) for k in row.keys() if k is not None]
            pairs = ", ".join(f"{k}: {v}" for k, v in row.items() if v is not None and str(v).strip())
            rows.append(f"Row {i+1}: {pairs}")
        header_line = f"Columns: {', '.join(headers)}" if headers else ""
        return header_line + "\n\n" + "\n".join(rows)
    except Exception as e:
        raise ValueError(f"CSV extraction failed: {e}")


def extract_text_txt(content: bytes) -> str:
    """Decode plain text file."""
    try:
        return content.decode("utf-8", errors="replace")
    except Exception as e:
        raise ValueError(f"TXT extraction failed: {e}")


def extract_text(filename: str, content: bytes) -> str:
    """Route to the correct extractor based on file extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_text_pdf(content)
    elif ext == ".csv":
        return extract_text_csv(content)
    elif ext == ".txt":
        return extract_text_txt(content)
    else:
        raise ValueError(f"Cannot extract text from '{ext}' files.")


# ── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalize and clean extracted text."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


# ── Prompt Injection Guard ────────────────────────────────────────────────────

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?instructions?",
    r"you\s+are\s+now\s+(a\s+)?",
    r"act\s+as\s+(if\s+you\s+are\s+)?",
    r"system\s*:\s*you",
    r"<\s*/?system\s*>",
    r"\[INST\]",
    r"###\s*instruction",
]

def sanitize_for_injection(text: str) -> str:
    """
    Neutralize potential prompt injection patterns in document content.
    Replaces dangerous patterns with a safe marker.
    """
    for pattern in _INJECTION_PATTERNS:
        text = re.sub(pattern, "[REDACTED]", text, flags=re.IGNORECASE)
    return text


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    filename: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int    = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Fixed-size chunking with overlap and boundary snapping.

    Each chunk stores two text fields:
    - "text"           : original casing — shown in citations
    - "text_normalized": lowercased      — used for embedding to fix case-sensitivity

    Returns list of chunk dicts with metadata.
    """
    chunks = []
    start  = 0
    idx    = 0

    while start < len(text):
        end = start + chunk_size

        # Snap to nearest natural boundary (newline or sentence) if not at end
        if end < len(text):
            # Prefer newline boundary (good for tables/price lists)
            snap_nl = text.rfind("\n", start, end)
            snap_dot = text.rfind(".", start, end)

            # Pick the latest boundary that's past the halfway point
            halfway = start + chunk_size // 2
            candidates = [s for s in [snap_nl, snap_dot] if s > halfway]
            if candidates:
                end = max(candidates) + 1

        chunk_text_content = text[start:end].strip()

        if chunk_text_content:
            chunk_id = hashlib.md5(
                f"{filename}:{idx}:{chunk_text_content[:50]}".encode()
            ).hexdigest()[:12]

            chunks.append({
                "chunk_id":        chunk_id,
                "document":        filename,
                "chunk_index":     idx,
                "text":            chunk_text_content,                   # original case for display
                "text_normalized": chunk_text_content.lower(),           # lowercased for embedding
                "char_start":      start,
                "char_end":        end,
            })
            idx += 1

        start = max(start + 1, end - overlap)

    logger.info(f"Chunked '{filename}' → {len(chunks)} chunks")
    return chunks


# ── Main Ingestion Entry Point ────────────────────────────────────────────────

def ingest_file(filename: str, content: bytes) -> tuple[list[dict], str]:
    """
    Full ingestion pipeline for a single file.
    Returns (chunks, error_message). If error, chunks is empty.
    """
    valid, err = validate_file(filename, content)
    if not valid:
        return [], err

    try:
        raw_text = extract_text(filename, content)
    except ValueError as e:
        return [], str(e)

    cleaned = clean_text(raw_text)

    if len(cleaned) < 10:
        return [], "File contains no extractable text."

    sanitized = sanitize_for_injection(cleaned)

    chunks = chunk_text(sanitized, filename)

    if not chunks:
        return [], "No usable content could be extracted from this file."

    return chunks, ""
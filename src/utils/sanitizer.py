"""
Input sanitization for all user-facing inputs:
- query strings
- file uploads
- URLs
- free text ingestion
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_QUERY_LENGTH = 2000
MAX_TEXT_LENGTH = 500_000
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}
ALLOWED_URL_SCHEMES = {"http", "https"}

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"you\s+are\s+now\s+",
    r"act\s+as\s+(a\s+)?(?!an?\s+assistant)",
    r"forget\s+(everything|all)",
    r"(system|sys)\s*prompt",
    r"<\s*/?system\s*>",
    r"###\s*instruction",
    r"jailbreak",
    r"\[INST\]",
    r"<\|im_start\|>",
]
COMPILED_INJECTION = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

SQL_PATTERNS = [
    r"(--|;|\/\*|\*\/)",
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|EXEC|UNION|SELECT)\b",
]
COMPILED_SQL = [re.compile(p, re.IGNORECASE) for p in SQL_PATTERNS]


# ── Exceptions ────────────────────────────────────────────────────────────────

class SanitizationError(ValueError):
    """Raised when input fails sanitization checks."""
    pass


# ── Sanitizers ────────────────────────────────────────────────────────────────

def sanitize_query(query: str) -> str:
    if not isinstance(query, str):
        raise SanitizationError("Query must be a string")

    query = query.strip()
    query = re.sub(r"[ \t]+", " ", query)
    query = re.sub(r"\n{3,}", "\n\n", query)

    if not query:
        raise SanitizationError("Query cannot be empty")

    if len(query) > MAX_QUERY_LENGTH:
        raise SanitizationError(f"Query too long: {len(query)} chars (max {MAX_QUERY_LENGTH})")

    for pattern in COMPILED_INJECTION:
        if pattern.search(query):
            logger.warning(f"Prompt injection attempt blocked: {query[:80]}")
            raise SanitizationError("Query contains disallowed content")

    for pattern in COMPILED_SQL:
        if pattern.search(query):
            logger.warning(f"SQL injection pattern in query: {query[:80]}")
            raise SanitizationError("Query contains disallowed SQL patterns")

    return query


def sanitize_url(url: str) -> str:
    if not isinstance(url, str):
        raise SanitizationError("URL must be a string")

    url = url.strip()
    if not url:
        raise SanitizationError("URL cannot be empty")

    try:
        parsed = urlparse(url)
    except Exception:
        raise SanitizationError("Invalid URL format")

    if parsed.scheme not in ALLOWED_URL_SCHEMES:
        raise SanitizationError(f"URL scheme '{parsed.scheme}' not allowed. Use http or https.")

    if not parsed.netloc:
        raise SanitizationError("URL must have a valid domain")

    hostname = parsed.hostname or ""

    if hostname in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
        raise SanitizationError("URLs pointing to localhost are not allowed")

    private_patterns = [
        r"^10\.", r"^172\.(1[6-9]|2\d|3[01])\.",
        r"^192\.168\.", r"^169\.254\.",
        r"^fc00:", r"^fe80:",
    ]
    for p in private_patterns:
        if re.match(p, hostname):
            raise SanitizationError("URLs pointing to private/internal networks are not allowed")

    return url


def sanitize_text(text: str, source_name: str = "manual") -> tuple[str, str]:
    if not isinstance(text, str):
        raise SanitizationError("Text must be a string")

    text = text.strip()
    if not text:
        raise SanitizationError("Text cannot be empty")

    if len(text) > MAX_TEXT_LENGTH:
        raise SanitizationError(f"Text too large: {len(text)} chars (max {MAX_TEXT_LENGTH})")

    source_name = re.sub(r"[^\w\s\-\.]", "", source_name).strip()[:100] or "manual"

    return text, source_name


def sanitize_file(filename: str, file_size_bytes: int) -> str:
    if not filename:
        raise SanitizationError("Filename cannot be empty")

    filename = Path(filename).name
    filename = re.sub(r"[^\w\s\-\.]", "_", filename).strip()

    if not filename:
        raise SanitizationError("Invalid filename after sanitization")

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise SanitizationError(f"File type '{ext}' not allowed. Allowed: {ALLOWED_EXTENSIONS}")

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size_bytes > max_bytes:
        raise SanitizationError(
            f"File too large: {file_size_bytes / 1024 / 1024:.1f}MB (max {MAX_FILE_SIZE_MB}MB)"
        )

    return filename

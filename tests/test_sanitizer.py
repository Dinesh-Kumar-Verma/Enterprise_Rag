"""Tests for input sanitization."""

import pytest
from src.utils.sanitizer import (
    SanitizationError,
    sanitize_query,
    sanitize_url,
    sanitize_text,
    sanitize_file,
)


class TestSanitizeQuery:
    def test_valid_query(self):
        assert sanitize_query("What is the refund policy?") == "What is the refund policy?"

    def test_strips_whitespace(self):
        assert sanitize_query("  hello world  ") == "hello world"

    def test_collapses_spaces(self):
        assert sanitize_query("hello   world") == "hello world"

    def test_empty_raises(self):
        with pytest.raises(SanitizationError, match="empty"):
            sanitize_query("   ")

    def test_too_long_raises(self):
        with pytest.raises(SanitizationError, match="too long"):
            sanitize_query("x" * 2001)

    def test_sql_injection_blocked(self):
        with pytest.raises(SanitizationError, match="SQL"):
            sanitize_query("SELECT * FROM users; DROP TABLE users;")

    def test_sql_union_blocked(self):
        with pytest.raises(SanitizationError, match="SQL"):
            sanitize_query("UNION SELECT password FROM accounts--")

    def test_non_string_raises(self):
        with pytest.raises(SanitizationError):
            sanitize_query(123)  # type: ignore

    def test_normal_greeting_passes(self):
        """Greetings should NOT be blocked by sanitizer — NeMo Guardrails handles intent."""
        assert sanitize_query("Hello, how are you?") == "Hello, how are you?"

    def test_normal_conversation_passes(self):
        """Normal conversational input should pass sanitization."""
        assert sanitize_query("Can you help me with this?") == "Can you help me with this?"


class TestSanitizeURL:
    def test_valid_https(self):
        assert sanitize_url("https://example.com/docs") == "https://example.com/docs"

    def test_valid_http(self):
        assert sanitize_url("http://example.com") == "http://example.com"

    def test_strips_whitespace(self):
        assert sanitize_url("  https://example.com  ") == "https://example.com"

    def test_ftp_blocked(self):
        with pytest.raises(SanitizationError, match="scheme"):
            sanitize_url("ftp://example.com/file")

    def test_javascript_blocked(self):
        with pytest.raises(SanitizationError, match="scheme"):
            sanitize_url("javascript:alert(1)")

    def test_localhost_blocked(self):
        with pytest.raises(SanitizationError, match="localhost"):
            sanitize_url("http://localhost:8080/internal")

    def test_loopback_blocked(self):
        with pytest.raises(SanitizationError, match="localhost"):
            sanitize_url("http://127.0.0.1/secret")

    def test_private_ip_blocked(self):
        with pytest.raises(SanitizationError, match="private"):
            sanitize_url("http://192.168.1.1/admin")

    def test_internal_network_blocked(self):
        with pytest.raises(SanitizationError, match="private"):
            sanitize_url("http://10.0.0.1/internal-api")

    def test_empty_raises(self):
        with pytest.raises(SanitizationError, match="empty"):
            sanitize_url("")


class TestSanitizeFile:
    def test_valid_pdf(self):
        assert sanitize_file("report.pdf", 1024) == "report.pdf"

    def test_valid_docx(self):
        assert sanitize_file("doc.docx", 2048) == "doc.docx"

    def test_unsupported_extension(self):
        with pytest.raises(SanitizationError, match="not allowed"):
            sanitize_file("malware.exe", 100)

    def test_too_large(self):
        with pytest.raises(SanitizationError, match="too large"):
            sanitize_file("big.pdf", 100 * 1024 * 1024)

    def test_path_traversal_stripped(self):
        result = sanitize_file("../../etc/passwd.txt", 100)
        assert "/" not in result
        assert ".." not in result

    def test_empty_filename_raises(self):
        with pytest.raises(SanitizationError):
            sanitize_file("", 100)


class TestSanitizeText:
    def test_valid_text(self):
        text, name = sanitize_text("Some content here.", "docs")
        assert text == "Some content here."
        assert name == "docs"

    def test_strips_whitespace(self):
        text, _ = sanitize_text("  hello  ")
        assert text == "hello"

    def test_empty_raises(self):
        with pytest.raises(SanitizationError, match="empty"):
            sanitize_text("")

    def test_too_large_raises(self):
        with pytest.raises(SanitizationError, match="too large"):
            sanitize_text("x" * 600_000)

    def test_source_name_sanitized(self):
        _, name = sanitize_text("content", "my source <script>")
        assert "<" not in name
        assert ">" not in name

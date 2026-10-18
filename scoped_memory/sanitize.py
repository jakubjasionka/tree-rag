"""Secret scrubbing, workspace slug sanitization, and deterministic content hashing."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping

_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(ghp_[A-Za-z0-9]{20,})\b"), "ghp_***REDACTED***"),
    (re.compile(r"\b(gho_[A-Za-z0-9]{20,})\b"), "gho_***REDACTED***"),
    (re.compile(r"\b(ghu_[A-Za-z0-9]{20,})\b"), "ghu_***REDACTED***"),
    (re.compile(r"\b(ghs_[A-Za-z0-9]{20,})\b"), "ghs_***REDACTED***"),
    (re.compile(r"\b(ghr_[A-Za-z0-9]{20,})\b"), "ghr_***REDACTED***"),
    (re.compile(r"\b(github_pat_[A-Za-z0-9_]{20,})\b"), "github_pat_***REDACTED***"),
    (re.compile(r"\b(sk-[A-Za-z0-9\-_]{20,})\b"), "sk-***REDACTED***"),
    (re.compile(r"\b(ak-[A-Za-z0-9]{14,})\b"), "ak-***REDACTED***"),
    (re.compile(r"\b(as-[A-Za-z0-9]{14,})\b"), "as-***REDACTED***"),
    (
        re.compile(r"(?<![A-Za-z0-9])(AAAAAAAAAAAAAAAAAAAA[A-Za-z0-9%_\-=]{20,})(?![A-Za-z0-9%_\-=])"),
        "x_bearer_***REDACTED***",
    ),
    (re.compile(r"\b([0-9]{8,20}-[A-Za-z0-9]{38,45})\b"), "x_access_token_***REDACTED***"),
    (re.compile(r"(li_at=[A-Za-z0-9_\-%]{25,})"), "li_at=***REDACTED***"),
    (re.compile(r"\b(AQ[A-Za-z0-9][A-Za-z0-9_\-]{30,})\b"), "linkedin_oauth_***REDACTED***"),
    (re.compile(r"\b(xox[baprs]-[A-Za-z0-9\-]{15,})\b"), "xox_***REDACTED***"),
]


def _scrub_secrets(text: str) -> str:
    """Redact known API tokens, OAuth bearer strings, and session cookies before indexing."""
    s = text or ""
    for pat, repl in _SECRET_PATTERNS:
        s = pat.sub(repl, s)
    return s


def scrub_secrets(text: str) -> str:
    """Public alias for `_scrub_secrets`."""
    return _scrub_secrets(text)


def sanitize(name: str, aliases: Mapping[str, str] | None = None) -> str:
    """Normalize a workspace/namespace identifier into a safe canonical directory slug.

    Rules:
    - Lowercase, non-alphanumeric characters collapsed into single `_`.
    - Leading/trailing underscores stripped.
    - Empty inputs resolve deterministically to `"default"`.
    """
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower())
    s = re.sub(r"_+", "_", s).strip("_") or "default"
    if aliases:
        return aliases.get(s, s)
    return s


def _sanitize(name: str) -> str:
    """Internal alias for `sanitize`."""
    return sanitize(name)


def normalize_text(text: str) -> str:
    """Collapse whitespace and scrub sensitive tokens prior to hashing or persistence."""
    return _scrub_secrets(" ".join((text or "").strip().split()))


def _normalize_text(text: str) -> str:
    """Internal alias for `normalize_text`."""
    return normalize_text(text)


def text_hash(text: str) -> str:
    """Compute a deterministic SHA-256 hex digest of normalized and scrubbed text."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _text_hash(text: str) -> str:
    """Internal alias for `text_hash`."""
    return text_hash(text)

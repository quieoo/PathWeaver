"""Shared normalization and content-addressing helpers for local stores."""

from __future__ import annotations

import hashlib
import re
from typing import Any


_SPACE_RE = re.compile(r"\s+")
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_PAREN_CONTENT_RE = re.compile(r"\([^)]*\)")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = _ZERO_WIDTH_RE.sub("", str(value)).strip()
    return _SPACE_RE.sub(" ", text)


def canonical_entity_key(value: Any) -> str:
    """Return the deterministic entity key used before learned resolution exists."""

    text = _PAREN_CONTENT_RE.sub("", normalize_text(value)).casefold()
    key = _SPACE_RE.sub(" ", "".join(char if char.isalnum() else " " for char in text)).strip()
    # Some extracted datasets contain symbol-only nodes such as "?" or "ΣΞ".
    # Preserve a deterministic key instead of dropping the corresponding edge.
    return key or f"symbol:{text}"


def canonical_relation(value: Any) -> str:
    text = normalize_text(value).casefold()
    return _SPACE_RE.sub(" ", text).strip()


def content_hash(*parts: Any) -> str:
    payload = "\x1f".join(normalize_text(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

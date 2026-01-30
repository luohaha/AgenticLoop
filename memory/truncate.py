"""Utilities for truncating large tool outputs before storing in memory."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TruncationResult:
    """Result of a truncation attempt."""

    content: str
    truncated: bool


def truncate_tool_output(
    content: str,
    policy: str,
    max_tokens: int,
    max_bytes: int,
    serialization_buffer: float,
    approx_chars_per_token: int,
) -> TruncationResult:
    """Truncate tool output according to policy.

    Args:
        content: Tool output content.
        policy: "none", "tokens", or "bytes".
        max_tokens: Token limit for truncation (policy="tokens").
        max_bytes: Byte limit for truncation (policy="bytes").
        serialization_buffer: Multiplier to account for JSON overhead.
        approx_chars_per_token: Approximate chars per token for estimation.
    """
    if not content or policy == "none":
        return TruncationResult(content=content, truncated=False)

    policy = policy.lower()
    if policy == "tokens":
        token_budget = max(0, _apply_buffer(max_tokens, serialization_buffer))
        token_count = _estimate_tokens(content, approx_chars_per_token)
        if token_count <= token_budget:
            return TruncationResult(content=content, truncated=False)
        char_budget = max(0, token_budget * max(1, approx_chars_per_token))
        truncated_content = _truncate_with_split(
            content,
            char_budget,
            removed_units=max(0, token_count - token_budget),
            unit_label="tokens",
        )
        return TruncationResult(content=truncated_content, truncated=True)

    if policy == "bytes":
        byte_budget = max(0, _apply_buffer(max_bytes, serialization_buffer))
        content_bytes = content.encode("utf-8")
        if len(content_bytes) <= byte_budget:
            return TruncationResult(content=content, truncated=False)
        truncated_content = _truncate_with_byte_split(content, content_bytes, byte_budget)
        return TruncationResult(content=truncated_content, truncated=True)

    return TruncationResult(content=content, truncated=False)


def _estimate_tokens(content: str, approx_chars_per_token: int) -> int:
    if not content:
        return 0
    divisor = max(1, approx_chars_per_token)
    return math.ceil(len(content) / divisor)


def _apply_buffer(value: int, buffer: float) -> int:
    if value <= 0:
        return 0
    return int(math.ceil(value * max(0.0, buffer)))


def _truncate_with_split(
    content: str,
    max_units: int,
    removed_units: int,
    unit_label: str,
) -> str:
    if max_units <= 0:
        return _format_marker(max(0, removed_units), unit_label)

    if len(content) <= max_units:
        return content

    left_budget = max_units // 2
    right_budget = max_units - left_budget

    left = content[:left_budget]
    right = content[-right_budget:] if right_budget > 0 else ""
    marker = _format_marker(max(0, removed_units), unit_label)
    truncated = f"{left}{marker}{right}"

    total_lines = content.count("\n") + 1 if content else 0
    if total_lines > 1:
        return f"Total output lines: {total_lines}\n\n{truncated}"

    return truncated


def _truncate_with_byte_split(content: str, content_bytes: bytes, max_bytes: int) -> str:
    if max_bytes <= 0:
        return _format_marker(len(content), "chars")

    if len(content_bytes) <= max_bytes:
        return content

    left_budget = max_bytes // 2
    right_budget = max_bytes - left_budget

    left_bytes = content_bytes[:left_budget]
    right_bytes = content_bytes[-right_budget:] if right_budget > 0 else b""

    left = _decode_prefix(left_bytes)
    right = _decode_suffix(right_bytes)
    removed_chars = max(0, len(content) - len(left) - len(right))
    marker = _format_marker(removed_chars, "chars")
    truncated = f"{left}{marker}{right}"

    total_lines = content.count("\n") + 1 if content else 0
    if total_lines > 1:
        return f"Total output lines: {total_lines}\n\n{truncated}"

    return truncated


def _decode_prefix(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")


def _decode_suffix(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")


def _format_marker(removed_units: int, unit_label: str) -> str:
    return f"...{removed_units} {unit_label} truncated..."

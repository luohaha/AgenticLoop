"""Tests for tool output truncation utilities."""

from memory.truncate import truncate_tool_output


class TestToolOutputTruncation:
    """Validate truncation behavior for different policies."""

    def test_bytes_policy_truncates_and_adds_line_count(self):
        """Bytes policy should truncate and include a line-count header."""
        content = "line1\nline2\nline3\nline4\nline5"

        result = truncate_tool_output(
            content=content,
            policy="bytes",
            max_tokens=0,
            max_bytes=20,
            serialization_buffer=1.0,
            approx_chars_per_token=4,
        )

        assert result.truncated
        assert result.content.startswith("Total output lines:")
        assert "chars truncated" in result.content

    def test_bytes_policy_no_truncation_when_within_budget(self):
        """Bytes policy should keep content when within budget."""
        content = "short output"

        result = truncate_tool_output(
            content=content,
            policy="bytes",
            max_tokens=0,
            max_bytes=200,
            serialization_buffer=1.0,
            approx_chars_per_token=4,
        )

        assert not result.truncated
        assert result.content == content

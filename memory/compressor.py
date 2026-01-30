"""Memory compression using LLM-based summarization."""

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

from config import Config
from llm.content_utils import extract_text
from llm.message_types import LLMMessage

from .types import CompressedMemory, CompressionStrategy

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from llm import LiteLLMAdapter


class WorkingMemoryCompressor:
    """Compresses conversation history using LLM summarization."""

    # Tools that should NEVER be compressed - their state must be preserved
    PROTECTED_TOOLS = {"manage_todo_list"}

    # Prefix for summary messages to identify them
    SUMMARY_PREFIX = Config.COMPACT_SUMMARY_PREFIX
    LEGACY_SUMMARY_PREFIX = "[Previous conversation summary]\n"
    TURN_ABORTED_MARKER = "<turn-aborted>"

    COMPRESSION_PROMPT = (
        Config.COMPACT_SUMMARIZATION_PROMPT
        + """

Original messages ({count} messages, ~{tokens} tokens):

{messages}

Provide a concise but comprehensive summary that captures the essential information. Be specific and include concrete details. Target length: {target_tokens} tokens."""
    )

    def __init__(self, llm: "LiteLLMAdapter"):
        """Initialize compressor.

        Args:
            llm: LLM instance to use for summarization
        """
        self.llm = llm
        self.PROTECTED_TOOLS = set(Config.PROTECTED_TOOLS)

    async def compress(
        self,
        messages: List[LLMMessage],
        strategy: str = CompressionStrategy.SLIDING_WINDOW,
        target_tokens: Optional[int] = None,
    ) -> CompressedMemory:
        """Compress messages using specified strategy.

        Args:
            messages: List of messages to compress
            strategy: Compression strategy to use
            target_tokens: Target token count for compressed output

        Returns:
            CompressedMemory object
        """
        if not messages:
            return CompressedMemory(messages=[])

        if target_tokens is None:
            # Calculate target based on config compression ratio
            original_tokens = self._estimate_tokens(messages)
            target_tokens = int(original_tokens * Config.MEMORY_COMPRESSION_RATIO)

        # Select and apply compression strategy
        if strategy == CompressionStrategy.SLIDING_WINDOW:
            return await self._compress_sliding_window(messages, target_tokens)
        elif strategy == CompressionStrategy.SELECTIVE:
            return await self._compress_selective(messages, target_tokens)
        elif strategy == CompressionStrategy.DELETION:
            return self._compress_deletion(messages)
        else:
            logger.warning(f"Unknown strategy {strategy}, using sliding window")
            return await self._compress_sliding_window(messages, target_tokens)

    async def _compress_sliding_window(
        self,
        messages: List[LLMMessage],
        target_tokens: int,
    ) -> CompressedMemory:
        """Compress using sliding window strategy.

        Summarizes all messages into a single summary.

        Args:
            messages: Messages to compress
            target_tokens: Target token count

        Returns:
            CompressedMemory object
        """
        # Format messages for summarization
        formatted = self._format_messages_for_summary(
            [msg for msg in messages if not self.is_summary_message(msg)]
        )
        original_tokens = self._estimate_tokens(messages)

        # Create summarization prompt
        prompt_text = self.COMPRESSION_PROMPT.format(
            count=len(messages),
            tokens=original_tokens,
            messages=formatted,
            target_tokens=target_tokens,
        )

        # Call LLM to generate summary
        try:
            prompt = LLMMessage(role="user", content=prompt_text)
            response = await self.llm.call_async(messages=[prompt], max_tokens=target_tokens * 2)
            summary_text = self.llm.extract_text(response)

            result_messages = self.build_compacted_history(
                initial_context=[m for m in messages if m.role == "system"],
                user_messages=self.select_user_messages(
                    self.collect_user_messages(messages), Config.COMPACT_USER_MESSAGE_MAX_TOKENS
                ),
                summary_text=summary_text,
                protected_messages=self.collect_protected_messages(messages),
                orphaned_tool_calls=self.collect_orphaned_tool_calls(messages),
            )

            # Calculate compression metrics
            compressed_tokens = self._estimate_tokens(result_messages)
            compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 0

            return CompressedMemory(
                messages=result_messages,
                original_message_count=len(messages),
                compressed_tokens=compressed_tokens,
                original_tokens=original_tokens,
                compression_ratio=compression_ratio,
                metadata={"strategy": "sliding_window"},
            )
        except Exception as e:
            logger.error(f"Error during compression: {e}")
            # Fallback: keep system messages + first and last non-system message
            non_system = [m for m in messages if m.role != "system"]
            fallback_other = [non_system[0], non_system[-1]] if len(non_system) > 1 else non_system
            fallback_messages = [m for m in messages if m.role == "system"] + fallback_other
            return CompressedMemory(
                messages=fallback_messages,
                original_message_count=len(messages),
                compressed_tokens=self._estimate_tokens(fallback_messages),
                original_tokens=original_tokens,
                compression_ratio=0.5,
                metadata={"strategy": "sliding_window", "error": str(e)},
            )

    async def _compress_selective(
        self,
        messages: List[LLMMessage],
        target_tokens: int,
    ) -> CompressedMemory:
        """Compress using selective preservation strategy.

        Preserves important messages (tool calls, system prompts) and
        summarizes the rest.

        Args:
            messages: Messages to compress
            target_tokens: Target token count

        Returns:
            CompressedMemory object
        """
        # Separate preserved vs compressible messages
        preserved, to_compress = self._separate_messages(messages)
        original_tokens = self._estimate_tokens(messages)

        # Pre-collect all components that will be in final result per RFC structure:
        # system + user + summary + protected + orphaned
        # This ensures budget calculation matches actual output structure
        system_msgs = [m for m in preserved if m.role == "system"]
        user_messages = self.select_user_messages(
            self.collect_user_messages(messages),
            Config.COMPACT_USER_MESSAGE_MAX_TOKENS,
        )
        protected_messages = self.collect_protected_messages(messages)
        orphaned_tool_calls = self.collect_orphaned_tool_calls(messages)

        if not to_compress:
            # Nothing to compress, use RFC structure without summary
            result_messages = self.build_compacted_history(
                initial_context=system_msgs,
                user_messages=user_messages,
                summary_text="",
                protected_messages=protected_messages,
                orphaned_tool_calls=orphaned_tool_calls,
            )
            compressed_tokens = self._estimate_tokens(result_messages)
            compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
            return CompressedMemory(
                messages=result_messages,
                original_message_count=len(messages),
                compressed_tokens=compressed_tokens,
                original_tokens=original_tokens,
                compression_ratio=compression_ratio,
                metadata={"strategy": "selective"},
            )

        # Calculate budget based on ACTUAL preserved components (not 'preserved' list
        # which includes recent assistant messages that won't be in final output)
        actual_preserved_tokens = (
            self._estimate_tokens(system_msgs)
            + self._estimate_tokens(user_messages)
            + self._estimate_tokens(protected_messages)
            + self._estimate_tokens(orphaned_tool_calls)
        )
        available_for_summary = target_tokens - actual_preserved_tokens

        if available_for_summary > 0:
            formatted = self._format_messages_for_summary(
                [msg for msg in to_compress if not self.is_summary_message(msg)]
            )
            prompt_text = self.COMPRESSION_PROMPT.format(
                count=len(to_compress),
                tokens=self._estimate_tokens(to_compress),
                messages=formatted,
                target_tokens=available_for_summary,
            )

            try:
                prompt = LLMMessage(role="user", content=prompt_text)
                response = await self.llm.call_async(
                    messages=[prompt], max_tokens=available_for_summary * 2
                )
                summary_text = self.llm.extract_text(response)

                result_messages = self.build_compacted_history(
                    initial_context=system_msgs,
                    user_messages=user_messages,
                    summary_text=summary_text,
                    protected_messages=protected_messages,
                    orphaned_tool_calls=orphaned_tool_calls,
                )

                # Calculate metrics based on actual result
                compressed_tokens = self._estimate_tokens(result_messages)
                compression_ratio = (
                    compressed_tokens / original_tokens if original_tokens > 0 else 0
                )

                return CompressedMemory(
                    messages=result_messages,
                    original_message_count=len(messages),
                    compressed_tokens=compressed_tokens,
                    original_tokens=original_tokens,
                    compression_ratio=compression_ratio,
                    metadata={"strategy": "selective", "preserved_count": len(preserved)},
                )
            except Exception as e:
                logger.error(f"Error during selective compression: {e}")

        # Fallback: use RFC structure without summary (not raw 'preserved' list)
        # This ensures consistent structure even when summary budget is exhausted
        result_messages = self.build_compacted_history(
            initial_context=system_msgs,
            user_messages=user_messages,
            summary_text="",
            protected_messages=protected_messages,
            orphaned_tool_calls=orphaned_tool_calls,
        )
        compressed_tokens = self._estimate_tokens(result_messages)
        return CompressedMemory(
            messages=result_messages,
            original_message_count=len(messages),
            compressed_tokens=compressed_tokens,
            original_tokens=original_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            metadata={"strategy": "selective", "preserved_count": len(preserved)},
        )

    def _compress_deletion(self, messages: List[LLMMessage]) -> CompressedMemory:
        """Simple deletion strategy - no compression, just drop old messages.

        Args:
            messages: Messages (will be deleted)

        Returns:
            CompressedMemory with empty messages list
        """
        original_tokens = self._estimate_tokens(messages)

        return CompressedMemory(
            messages=[],
            original_message_count=len(messages),
            compressed_tokens=0,
            original_tokens=original_tokens,
            compression_ratio=0.0,
            metadata={"strategy": "deletion"},
        )

    def _separate_messages(
        self, messages: List[LLMMessage]
    ) -> Tuple[List[LLMMessage], List[LLMMessage]]:
        """Separate messages into preserved and compressible.

        Strategy:
        1. Preserve system messages (if configured)
        2. Preserve orphaned tool_use (waiting for tool_result)
        3. Preserve protected tools (todo list, etc.) - NEVER compress these
        4. Preserve the most recent N messages (MEMORY_SHORT_TERM_MIN_SIZE)
        5. **Critical rule**: Tool pairs (tool_use + tool_result) must stay together
           - If one is preserved, the other must be preserved too
           - If one is compressed, the other must be compressed too

        Args:
            messages: All messages

        Returns:
            Tuple of (preserved, to_compress)
        """
        preserve_indices = set()

        # Step 1: Mark system messages for preservation
        for i, msg in enumerate(messages):
            if Config.MEMORY_PRESERVE_SYSTEM_PROMPTS and msg.role == "system":
                preserve_indices.add(i)

        # Step 2: Find tool pairs and orphaned tool_use messages
        tool_pairs, orphaned_tool_use_indices = self._find_tool_pairs(messages)

        # Step 2a: CRITICAL - Preserve orphaned tool_use (waiting for tool_result)
        # These must NEVER be compressed, or we'll lose the tool_use without its result
        for orphan_idx in orphaned_tool_use_indices:
            preserve_indices.add(orphan_idx)

        # Step 2b: Mark protected tools for preservation (CRITICAL for stateful tools)
        protected_pairs = self._find_protected_tool_pairs(messages, tool_pairs)
        for assistant_idx, user_idx in protected_pairs:
            preserve_indices.add(assistant_idx)
            preserve_indices.add(user_idx)

        # Step 3: Preserve the most recent N messages to maintain conversation continuity
        preserve_count = min(Config.MEMORY_SHORT_TERM_MIN_SIZE, len(messages))
        for i in range(len(messages) - preserve_count, len(messages)):
            if i >= 0:
                preserve_indices.add(i)

        # Step 4: Ensure tool pairs stay together
        for assistant_idx, user_idx in tool_pairs:
            # If either message in the pair is marked for preservation, preserve both
            if assistant_idx in preserve_indices or user_idx in preserve_indices:
                preserve_indices.add(assistant_idx)
                preserve_indices.add(user_idx)
            # Otherwise both will be compressed together

        # Step 5: Build preserved and to_compress lists
        preserved = []
        to_compress = []
        for i, msg in enumerate(messages):
            if i in preserve_indices:
                preserved.append(msg)
            else:
                to_compress.append(msg)

        logger.info(
            f"Separated: {len(preserved)} preserved, {len(to_compress)} to compress "
            f"({len(tool_pairs)} tool pairs, {len(protected_pairs)} protected, "
            f"{len(orphaned_tool_use_indices)} orphaned tool_use, "
            f"{preserve_count} recent)"
        )
        return preserved, to_compress

    def _find_tool_pairs(self, messages: List[LLMMessage]) -> tuple[List[List[int]], List[int]]:
        """Find tool_use/tool_result pairs in messages.

        Handles both:
        - New format: assistant.tool_calls + tool role messages
        - Legacy format: tool_use blocks in assistant content + tool_result blocks in user content

        Returns:
            Tuple of (pairs, orphaned_tool_use_indices)
            - pairs: List of [assistant_index, tool_response_index] for matched pairs
            - orphaned_tool_use_indices: List of message indices with unmatched tool_use
        """
        pairs = []
        pending_tool_uses = {}  # tool_id -> message_index

        for i, msg in enumerate(messages):
            # New format: assistant with tool_calls field
            if msg.role == "assistant" and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    if tool_id:
                        pending_tool_uses[tool_id] = i

            # Legacy format: assistant with tool_use blocks in content
            elif msg.role == "assistant" and isinstance(msg.content, list):
                for block in msg.content:
                    btype = self._get_block_attr(block, "type")
                    if btype == "tool_use":
                        tool_id = self._get_block_attr(block, "id")
                        if tool_id:
                            pending_tool_uses[tool_id] = i

            # New format: tool role message
            elif msg.role == "tool" and hasattr(msg, "tool_call_id") and msg.tool_call_id:
                tool_call_id = msg.tool_call_id
                if tool_call_id in pending_tool_uses:
                    assistant_idx = pending_tool_uses[tool_call_id]
                    pairs.append([assistant_idx, i])
                    del pending_tool_uses[tool_call_id]

            # Legacy format: user with tool_result blocks in content
            elif msg.role == "user" and isinstance(msg.content, list):
                for block in msg.content:
                    btype = self._get_block_attr(block, "type")
                    if btype == "tool_result":
                        tool_use_id = self._get_block_attr(block, "tool_use_id")
                        if tool_use_id in pending_tool_uses:
                            assistant_idx = pending_tool_uses[tool_use_id]
                            pairs.append([assistant_idx, i])
                            del pending_tool_uses[tool_use_id]

        # Remaining items in pending_tool_uses are orphaned (no matching result yet)
        orphaned_indices = list(pending_tool_uses.values())

        if orphaned_indices:
            logger.warning(
                f"Found {len(orphaned_indices)} orphaned tool_use without matching tool_result - "
                f"these will be preserved to wait for results"
            )

        return pairs, orphaned_indices

    def _find_protected_tool_pairs(
        self, messages: List[LLMMessage], tool_pairs: List[List[int]]
    ) -> List[List[int]]:
        """Find tool pairs that use protected tools (must never be compressed).

        Handles both new format (tool_calls field) and legacy format (tool_use blocks).

        Args:
            messages: All messages
            tool_pairs: All tool_use/tool_result pairs

        Returns:
            List of protected tool pairs [assistant_index, tool_response_index]
        """
        protected_pairs = []

        for assistant_idx, response_idx in tool_pairs:
            msg = messages[assistant_idx]

            # New format: check tool_calls field
            if msg.role == "assistant" and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if isinstance(tc, dict):
                        tool_name = tc.get("function", {}).get("name", "")
                    else:
                        tool_name = (
                            getattr(tc.function, "name", "") if hasattr(tc, "function") else ""
                        )
                    if tool_name in self.PROTECTED_TOOLS:
                        protected_pairs.append([assistant_idx, response_idx])
                        logger.debug(
                            f"Protected tool '{tool_name}' at indices [{assistant_idx}, {response_idx}] - will be preserved"
                        )
                        break

            # Legacy format: check tool_use blocks in content
            elif msg.role == "assistant" and isinstance(msg.content, list):
                for block in msg.content:
                    btype = self._get_block_attr(block, "type")
                    if btype == "tool_use":
                        tool_name = self._get_block_attr(block, "name")
                        if tool_name in self.PROTECTED_TOOLS:
                            protected_pairs.append([assistant_idx, response_idx])
                            logger.debug(
                                f"Protected tool '{tool_name}' at indices [{assistant_idx}, {response_idx}] - will be preserved"
                            )
                            break

        return protected_pairs

    def _get_block_attr(self, block, attr: str):
        """Get attribute from block (supports dict and object)."""
        if isinstance(block, dict):
            return block.get(attr)
        return getattr(block, attr, None)

    def _format_messages_for_summary(self, messages: List[LLMMessage]) -> str:
        """Format messages for inclusion in summary prompt.

        Args:
            messages: Messages to format

        Returns:
            Formatted string
        """
        formatted = []
        for i, msg in enumerate(messages, 1):
            role = msg.role.upper()
            content = self._extract_text_content(msg)
            formatted.append(f"[{i}] {role}: {content}")

        return "\n\n".join(formatted)

    def collect_user_messages(self, messages: List[LLMMessage]) -> List[LLMMessage]:
        """Collect user messages while excluding previous summaries and legacy tool_results."""
        return [
            msg
            for msg in messages
            if msg.role == "user"
            and not self.is_summary_message(msg)
            and not self.is_legacy_tool_result(msg)
        ]

    def select_user_messages(self, messages: List[LLMMessage], max_tokens: int) -> List[LLMMessage]:
        """Select user messages, prioritizing recent ones within a token budget."""
        if max_tokens <= 0:
            return []

        selected: List[LLMMessage] = []
        budget = max_tokens

        for msg in reversed(messages):
            if self.is_turn_aborted_message(msg):
                selected.append(msg)
                continue
            tokens = self._estimate_tokens([msg])
            if tokens <= budget:
                selected.append(msg)
                budget -= tokens

        return list(reversed(selected))

    def collect_protected_messages(self, messages: List[LLMMessage]) -> List[LLMMessage]:
        """Collect protected tool call/output pairs to preserve."""
        tool_pairs, _ = self._find_tool_pairs(messages)
        protected_pairs = self._find_protected_tool_pairs(messages, tool_pairs)
        protected_indices = {idx for pair in protected_pairs for idx in pair}
        return [msg for idx, msg in enumerate(messages) if idx in protected_indices]

    def collect_orphaned_tool_calls(self, messages: List[LLMMessage]) -> List[LLMMessage]:
        """Collect orphaned tool calls (calls without matching results).

        These must be preserved in compaction to avoid losing pending tool calls.
        Note: Uses set to deduplicate indices when same assistant has multiple orphan calls.
        """
        _, orphaned_indices = self._find_tool_pairs(messages)
        unique_indices = set(orphaned_indices)
        return [msg for idx, msg in enumerate(messages) if idx in unique_indices]

    def build_compacted_history(
        self,
        initial_context: List[LLMMessage],
        user_messages: List[LLMMessage],
        summary_text: str,
        protected_messages: List[LLMMessage],
        orphaned_tool_calls: Optional[List[LLMMessage]] = None,
    ) -> List[LLMMessage]:
        """Build new history after compaction.

        Order: initial_context + user_messages + summary + protected + orphaned
        Orphaned tool calls go at the end since they're waiting for results.
        """
        result = initial_context + user_messages
        result.append(
            LLMMessage(
                role="user",
                content=f"{self.SUMMARY_PREFIX}{summary_text}",
            )
        )
        result.extend(protected_messages)
        if orphaned_tool_calls:
            result.extend(orphaned_tool_calls)
        return result

    def is_summary_message(self, message: LLMMessage) -> bool:
        """Check if message is a previous summary."""
        if message.role != "user":
            return False
        if not isinstance(message.content, str):
            return False
        return message.content.startswith(self.SUMMARY_PREFIX) or message.content.startswith(
            self.LEGACY_SUMMARY_PREFIX
        )

    def is_turn_aborted_message(self, message: LLMMessage) -> bool:
        """Check if message contains a turn-aborted marker."""
        if message.role != "user":
            return False
        if not isinstance(message.content, str):
            return False
        return self.TURN_ABORTED_MARKER in message.content

    def is_legacy_tool_result(self, message: LLMMessage) -> bool:
        """Check if message is a legacy tool_result (Anthropic format).

        Legacy tool_result messages have role='user' but content is a list
        containing tool_result blocks.
        """
        if message.role != "user":
            return False
        if not isinstance(message.content, list):
            return False
        return any(
            (isinstance(block, dict) and block.get("type") == "tool_result")
            or (hasattr(block, "type") and block.type == "tool_result")
            for block in message.content
        )

    def _extract_text_content(self, message: LLMMessage) -> str:
        """Extract text content from message for token estimation.

        Uses centralized extract_text from content_utils.

        Args:
            message: Message to extract from

        Returns:
            Text content
        """
        # Use centralized extraction
        text = extract_text(message.content)

        # For token estimation, also include tool call info as string representation
        if hasattr(message, "tool_calls") and message.tool_calls:
            text += " " + str(message.tool_calls)

        return text if text else str(message.content)

    def _estimate_tokens(self, messages: List[LLMMessage]) -> int:
        """Estimate token count for messages.

        Args:
            messages: Messages to count

        Returns:
            Estimated token count
        """
        # Improved estimation: account for message structure and content
        total_chars = 0
        for msg in messages:
            # Add overhead for message structure (role, type fields, etc.)
            total_chars += 20  # ~5 tokens for structure

            # Extract and count content
            content = self._extract_text_content(msg)
            total_chars += len(content)

            # For complex content (lists), add overhead for JSON structure
            if isinstance(msg.content, list):
                # Each block has type, id, etc. fields
                total_chars += len(msg.content) * 30  # ~7 tokens per block overhead

        # More accurate ratio: ~3.5 characters per token for mixed content
        # (English text is ~4 chars/token, code/JSON is ~3 chars/token)
        return int(total_chars / 3.5)

"""Core memory manager that orchestrates all memory operations."""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from config import Config
from llm.content_utils import content_has_tool_calls
from llm.message_types import LLMMessage
from utils.runtime import get_db_path

from .compressor import WorkingMemoryCompressor
from .short_term import ShortTermMemory
from .store import MemoryStore
from .token_tracker import TokenTracker
from .truncate import truncate_tool_output
from .types import CompressedMemory, CompressionStrategy

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from llm import LiteLLMAdapter


class MemoryManager:
    """Central memory management system with built-in persistence."""

    def __init__(
        self,
        llm: "LiteLLMAdapter",
        store: Optional[MemoryStore] = None,
        session_id: Optional[str] = None,
        db_path: Optional[str] = None,
    ):
        """Initialize memory manager.

        Args:
            llm: LLM instance for compression
            store: Optional MemoryStore for persistence (if None, creates default store)
            session_id: Optional session ID (if resuming session)
            db_path: Path to database file (default: .aloop/db/memory.db)
        """
        self.llm = llm
        self._db_path = db_path if db_path is not None else get_db_path()

        # Always create/use store for persistence
        if store is None:
            store = MemoryStore(db_path=db_path)
        self.store = store

        # Lazy session creation: only create when first message is added
        # If session_id is provided (resuming), use it immediately
        if session_id is not None:
            self.session_id = session_id
            self._session_created = True
        else:
            self.session_id = None
            self._session_created = False

        # Initialize components using Config directly
        self.short_term = ShortTermMemory(max_size=Config.MEMORY_SHORT_TERM_SIZE)
        self.compressor = WorkingMemoryCompressor(llm)
        self.token_tracker = TokenTracker()

        # Storage for system messages (summaries are now stored as regular messages in short_term)
        self.system_messages: List[LLMMessage] = []

        # State tracking
        self.current_tokens = 0
        self.was_compressed_last_iteration = False
        self.last_compression_savings = 0
        self.compression_count = 0

    @classmethod
    async def from_session(
        cls,
        session_id: str,
        llm: "LiteLLMAdapter",
        store: Optional[MemoryStore] = None,
        db_path: Optional[str] = None,
    ) -> "MemoryManager":
        """Load a MemoryManager from a saved session.

        Args:
            session_id: Session ID to load
            llm: LLM instance for compression
            store: Optional MemoryStore instance (if None, creates default store)
            db_path: Path to database file (default: .aloop/db/memory.db)

        Returns:
            MemoryManager instance with loaded state
        """
        # Create store if not provided
        if store is None:
            store = MemoryStore(db_path=db_path if db_path is not None else get_db_path())

        # Load session data
        session_data = await store.load_session(session_id)
        if not session_data:
            raise ValueError(f"Session {session_id} not found")

        # Create manager (config is now read from Config class directly)
        manager = cls(llm=llm, store=store, session_id=session_id)

        # Restore state
        manager.system_messages = session_data["system_messages"]
        manager.compression_count = session_data["stats"]["compression_count"]

        # Add messages to short-term memory (including any summary messages)
        for msg in session_data["messages"]:
            manager.short_term.add_message(msg)

        # Recalculate tokens
        manager.current_tokens = manager._recalculate_current_tokens()

        logger.info(
            f"Loaded session {session_id}: "
            f"{len(session_data['messages'])} messages, "
            f"{manager.current_tokens} tokens"
        )

        return manager

    async def _ensure_session(self) -> None:
        """Lazily create session when first needed.

        This avoids creating empty sessions when MemoryManager is instantiated
        but no messages are ever added (e.g., user exits before running any task).

        Raises:
            RuntimeError: If session creation fails
        """
        if not self._session_created:
            try:
                self.session_id = await self.store.create_session()
                self._session_created = True
                logger.info(f"Created new session: {self.session_id}")
            except Exception as e:
                logger.error(f"Failed to create session: {e}")
                raise RuntimeError(f"Failed to create memory session: {e}") from e

    async def add_message(self, message: LLMMessage, actual_tokens: Dict[str, int] = None) -> None:
        """Add a message to memory and trigger compression if needed.

        Args:
            message: Message to add
            actual_tokens: Optional dict with actual token counts from LLM response
                          Format: {"input": int, "output": int}
        """
        # Ensure session exists before adding messages
        await self._ensure_session()

        # Track system messages separately
        if message.role == "system":
            self.system_messages.append(message)
            return

        # Truncate large tool outputs before storing
        message = self._maybe_truncate_tool_output(message)

        # Count tokens (use actual if provided, otherwise estimate)
        if actual_tokens:
            # Use actual token counts from LLM response
            # Note: input_tokens includes full context sent to API, not just new content
            input_tokens = actual_tokens.get("input", 0)
            output_tokens = actual_tokens.get("output", 0)

            self.token_tracker.add_input_tokens(input_tokens)
            self.token_tracker.add_output_tokens(output_tokens)

            # Log API usage separately
            logger.debug(
                f"API usage: input={input_tokens}, output={output_tokens}, "
                f"total={input_tokens + output_tokens}"
            )
        else:
            # Estimate token count for non-API messages (tool results, etc.)
            provider = self.llm.provider_name.lower()
            model = self.llm.model
            tokens = self.token_tracker.count_message_tokens(message, provider, model)

            # Update token tracker
            if message.role == "assistant":
                self.token_tracker.add_output_tokens(tokens)
            else:
                self.token_tracker.add_input_tokens(tokens)

        # Add to short-term memory
        self.short_term.add_message(message)

        # Recalculate current tokens based on actual stored content
        # This gives accurate count for compression decisions
        self.current_tokens = self._recalculate_current_tokens()

        # Log memory state (stored content size, not API usage)
        logger.debug(
            f"Memory state: {self.current_tokens} stored tokens, "
            f"{self.short_term.count()}/{Config.MEMORY_SHORT_TERM_SIZE} messages, "
            f"full={self.short_term.is_full()}"
        )

        # Check if compression is needed
        self.was_compressed_last_iteration = False
        should_compress, reason = self._should_compress()
        if should_compress:
            logger.info(f"🗜️  Triggering compression: {reason}")
            await self.compress()
        else:
            # Log compression check details
            logger.debug(
                f"Compression check: stored={self.current_tokens}, "
                f"threshold={Config.MEMORY_COMPRESSION_THRESHOLD}, "
                f"short_term_full={self.short_term.is_full()}"
            )

    def get_context_for_llm(self) -> List[LLMMessage]:
        """Get optimized context for LLM call.

        Returns:
            List of messages: system messages + short-term messages (which includes summaries)
        """
        context = []

        # 1. Add system messages (always included)
        context.extend(self.system_messages)

        # 2. Add short-term memory (includes summary messages and recent messages)
        context.extend(self.short_term.get_messages())

        return self._normalize_for_prompt(context)

    async def compress(self, strategy: str = None) -> Optional[CompressedMemory]:
        """Compress current short-term memory.

        After compression, the compressed messages (including any summary as user message)
        are put back into short_term as regular messages.

        Args:
            strategy: Compression strategy (None = auto-select)

        Returns:
            CompressedMemory object if compression was performed
        """
        messages = self.short_term.get_messages()
        message_count = len(messages)

        if not messages:
            logger.warning("No messages to compress")
            return None

        # Auto-select strategy if not specified
        if strategy is None:
            strategy = self._select_strategy(messages)

        logger.info(f"🗜️  Compressing {message_count} messages using {strategy} strategy")

        try:
            # Perform compression
            # Note: todo state is preserved via PROTECTED_TOOLS (manage_todo_list)
            compressed = await self.compressor.compress(
                messages,
                strategy=strategy,
                target_tokens=self._calculate_target_tokens(),
            )

            # Track compression results
            self.compression_count += 1
            self.was_compressed_last_iteration = True
            self.last_compression_savings = compressed.token_savings

            # Update token tracker
            self.token_tracker.add_compression_savings(compressed.token_savings)
            self.token_tracker.add_compression_cost(compressed.compressed_tokens)

            # Remove compressed messages from short-term memory
            self.short_term.remove_first(message_count)

            # Get any remaining messages (added after compression started)
            remaining_messages = self.short_term.get_messages()
            self.short_term.clear()

            # Add compressed messages (summary + preserved, already combined by compressor)
            for msg in compressed.messages:
                self.short_term.add_message(msg)

            # Add any remaining messages
            for msg in remaining_messages:
                self.short_term.add_message(msg)

            # Update current token count
            old_tokens = self.current_tokens
            self.current_tokens = self._recalculate_current_tokens()

            # Log compression results
            logger.info(
                f"✅ Compression complete: {compressed.original_tokens} → {compressed.compressed_tokens} tokens "
                f"({compressed.savings_percentage:.1f}% saved, ratio: {compressed.compression_ratio:.2f}), "
                f"context: {old_tokens} → {self.current_tokens} tokens, "
                f"short_term now has {self.short_term.count()} messages"
            )

            return compressed

        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return None

    def _should_compress(self) -> tuple[bool, Optional[str]]:
        """Check if compression should be triggered.

        Returns:
            Tuple of (should_compress, reason)
        """
        if not Config.MEMORY_ENABLED:
            return False, "compression_disabled"

        # Hard limit: must compress
        if self.current_tokens > Config.MEMORY_COMPRESSION_THRESHOLD:
            return (
                True,
                f"hard_limit ({self.current_tokens} > {Config.MEMORY_COMPRESSION_THRESHOLD})",
            )

        # CRITICAL: Compress when short-term memory is full to prevent eviction
        # If we don't compress, the next message will cause deque to evict the oldest message,
        # which may break tool_use/tool_result pairs
        if self.short_term.is_full():
            return (
                True,
                f"short_term_full ({self.short_term.count()}/{Config.MEMORY_SHORT_TERM_SIZE} messages, "
                f"current tokens: {self.current_tokens})",
            )

        return False, None

    def _select_strategy(self, messages: List[LLMMessage]) -> str:
        """Auto-select compression strategy based on message characteristics.

        Args:
            messages: Messages to analyze

        Returns:
            Strategy name
        """
        # Check for tool calls
        has_tool_calls = any(self._message_has_tool_calls(msg) for msg in messages)

        # Select strategy
        if has_tool_calls:
            # Preserve tool calls
            return CompressionStrategy.SELECTIVE
        elif len(messages) < 5:
            # Too few messages, just delete
            return CompressionStrategy.DELETION
        else:
            # Default: sliding window
            return CompressionStrategy.SLIDING_WINDOW

    def _message_has_tool_calls(self, message: LLMMessage) -> bool:
        """Check if message contains tool calls.

        Handles both new format (tool_calls field) and legacy format (content blocks).

        Args:
            message: Message to check

        Returns:
            True if contains tool calls
        """
        # New format: check tool_calls field
        if hasattr(message, "tool_calls") and message.tool_calls:
            return True

        # New format: tool role message
        if message.role == "tool":
            return True

        # Legacy/centralized check on content
        return content_has_tool_calls(message.content)

    def _maybe_truncate_tool_output(self, message: LLMMessage) -> LLMMessage:
        """Truncate tool output content if it exceeds configured limits."""
        if message.role != "tool":
            return message
        if not isinstance(message.content, str):
            return message

        result = truncate_tool_output(
            content=message.content,
            policy=Config.TOOL_OUTPUT_TRUNCATION_POLICY,
            max_tokens=Config.TOOL_OUTPUT_MAX_TOKENS,
            max_bytes=Config.TOOL_OUTPUT_MAX_BYTES,
            serialization_buffer=Config.TOOL_OUTPUT_SERIALIZATION_BUFFER,
            approx_chars_per_token=Config.APPROX_CHARS_PER_TOKEN,
        )

        if not result.truncated:
            return message

        return LLMMessage(
            role=message.role,
            content=result.content,
            tool_calls=message.tool_calls,
            tool_call_id=message.tool_call_id,
            name=message.name,
        )

    def remove_oldest_with_pair_integrity(self) -> Optional[LLMMessage]:
        """Remove oldest message and its corresponding tool pair (if any)."""
        messages = self.short_term.get_messages()
        if not messages:
            return None

        oldest = messages[0]
        call_ids = set(self._extract_tool_call_ids(oldest))

        if not call_ids and oldest.role == "tool" and oldest.tool_call_id:
            call_ids.add(oldest.tool_call_id)

        if not call_ids and self._has_legacy_tool_results(oldest):
            call_ids.update(self._extract_tool_result_ids(oldest))

        if call_ids:
            filtered = self._remove_messages_by_tool_call_ids(messages, call_ids)
            if filtered and filtered[0] == oldest:
                filtered = filtered[1:]
        else:
            filtered = messages[1:]

        self.short_term.clear()
        for msg in filtered:
            self.short_term.add_message(msg)

        self.current_tokens = self._recalculate_current_tokens()

        return oldest

    def _remove_messages_by_tool_call_ids(
        self, messages: List[LLMMessage], call_ids: set[str]
    ) -> List[LLMMessage]:
        """Remove all tool calls/results matching given call IDs."""
        filtered: List[LLMMessage] = []
        for msg in messages:
            if msg.role == "assistant" and self._message_has_call_id(msg, call_ids):
                continue
            if msg.role == "tool" and msg.tool_call_id in call_ids:
                continue
            if self._has_legacy_tool_results(msg) and self._message_has_result_id(msg, call_ids):
                continue
            filtered.append(msg)
        return filtered

    def _normalize_for_prompt(self, messages: List[LLMMessage]) -> List[LLMMessage]:
        """Normalize messages to ensure tool call/output integrity before LLM call."""
        normalized = self.ensure_call_outputs_present(messages)
        normalized = self.remove_orphan_outputs(normalized)
        return normalized

    def ensure_call_outputs_present(self, messages: List[LLMMessage]) -> List[LLMMessage]:
        """Add synthetic 'aborted' output for orphaned tool calls."""
        existing_outputs = set()
        for msg in messages:
            if msg.role == "tool" and msg.tool_call_id:
                existing_outputs.add(msg.tool_call_id)
            if self._has_legacy_tool_results(msg):
                existing_outputs.update(self._extract_tool_result_ids(msg))

        normalized: List[LLMMessage] = []
        for msg in messages:
            normalized.append(msg)
            for call_id, tool_name in self._extract_tool_call_id_pairs(msg):
                if call_id in existing_outputs:
                    continue
                normalized.append(
                    LLMMessage(
                        role="tool",
                        content="aborted",
                        tool_call_id=call_id,
                        name=tool_name or None,
                    )
                )
                existing_outputs.add(call_id)

        return normalized

    def remove_orphan_outputs(self, messages: List[LLMMessage]) -> List[LLMMessage]:
        """Remove tool results without matching calls."""
        call_ids = set()
        for msg in messages:
            call_ids.update(self._extract_tool_call_ids(msg))

        filtered: List[LLMMessage] = []
        for msg in messages:
            if msg.role == "tool" and msg.tool_call_id and msg.tool_call_id not in call_ids:
                continue
            if self._has_legacy_tool_results(msg):
                assert isinstance(msg.content, list)  # for type checker
                filtered_blocks = [
                    block
                    for block in msg.content
                    if not self._is_orphan_tool_result_block(block, call_ids)
                ]
                if not filtered_blocks:
                    continue
                # Note: content here is a list, which LLMMessage accepts via Any
                filtered.append(
                    LLMMessage(
                        role=msg.role,
                        content=filtered_blocks,  # type: ignore[arg-type]
                        tool_calls=msg.tool_calls,
                        tool_call_id=msg.tool_call_id,
                        name=msg.name,
                    )
                )
                continue
            filtered.append(msg)

        return filtered

    def _extract_tool_call_id_pairs(self, message: LLMMessage) -> List[tuple[str, str]]:
        pairs: List[tuple[str, str]] = []
        if message.role != "assistant":
            return pairs

        if message.tool_calls:
            for tc in message.tool_calls:
                if isinstance(tc, dict):
                    call_id = tc.get("id")
                    tool_name = tc.get("function", {}).get("name", "")
                else:
                    call_id = getattr(tc, "id", None)
                    tool_name = getattr(getattr(tc, "function", None), "name", "") if tc else ""
                if call_id:
                    pairs.append((call_id, tool_name))
            return pairs

        if isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    call_id = block.get("id")
                    tool_name = block.get("name", "")
                elif hasattr(block, "type") and block.type == "tool_use":
                    call_id = getattr(block, "id", None)
                    tool_name = getattr(block, "name", "")
                else:
                    continue
                if call_id:
                    pairs.append((call_id, tool_name))

        return pairs

    def _extract_tool_call_ids(self, message: LLMMessage) -> List[str]:
        return [call_id for call_id, _ in self._extract_tool_call_id_pairs(message)]

    def _has_legacy_tool_results(self, message: LLMMessage) -> bool:
        return message.role == "user" and isinstance(message.content, list)

    def _extract_tool_result_ids(self, message: LLMMessage) -> List[str]:
        ids: List[str] = []
        if not self._has_legacy_tool_results(message):
            return ids
        assert isinstance(message.content, list)  # for type checker
        for block in message.content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tool_use_id = block.get("tool_use_id")
            elif hasattr(block, "type") and block.type == "tool_result":
                tool_use_id = getattr(block, "tool_use_id", None)
            else:
                continue
            if tool_use_id:
                ids.append(tool_use_id)
        return ids

    def _message_has_call_id(self, message: LLMMessage, call_ids: set[str]) -> bool:
        return any(call_id in call_ids for call_id in self._extract_tool_call_ids(message))

    def _message_has_result_id(self, message: LLMMessage, call_ids: set[str]) -> bool:
        return any(result_id in call_ids for result_id in self._extract_tool_result_ids(message))

    def _is_orphan_tool_result_block(self, block, call_ids: set[str]) -> bool:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            return block.get("tool_use_id") not in call_ids
        if hasattr(block, "type") and block.type == "tool_result":
            return getattr(block, "tool_use_id", None) not in call_ids
        return False

    def _calculate_target_tokens(self) -> int:
        """Calculate target token count for compression.

        Returns:
            Target token count
        """
        original_tokens = self.current_tokens
        target = int(original_tokens * Config.MEMORY_COMPRESSION_RATIO)
        return max(target, 500)  # Minimum 500 tokens for summary

    def _recalculate_current_tokens(self) -> int:
        """Recalculate current token count from scratch.

        Returns:
            Current token count
        """
        provider = self.llm.provider_name.lower()
        model = self.llm.model

        total = 0

        # Count system messages
        for msg in self.system_messages:
            total += self.token_tracker.count_message_tokens(msg, provider, model)

        # Count short-term messages (includes summary messages)
        for msg in self.short_term.get_messages():
            total += self.token_tracker.count_message_tokens(msg, provider, model)

        return total

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dict with statistics
        """
        return {
            "current_tokens": self.current_tokens,
            "total_input_tokens": self.token_tracker.total_input_tokens,
            "total_output_tokens": self.token_tracker.total_output_tokens,
            "compression_count": self.compression_count,
            "total_savings": self.token_tracker.compression_savings,
            "compression_cost": self.token_tracker.compression_cost,
            "net_savings": self.token_tracker.compression_savings
            - self.token_tracker.compression_cost,
            "short_term_count": self.short_term.count(),
            "total_cost": self.token_tracker.get_total_cost(self.llm.model),
        }

    async def save_memory(self):
        """Save current memory state to store.

        This saves the complete memory state including:
        - System messages
        - Short-term messages (which includes summary messages after compression)

        Call this method after completing a task or at key checkpoints.
        """
        # Skip if no session was created (no messages were ever added)
        if not self.store or not self._session_created or not self.session_id:
            logger.debug("Skipping save_memory: no session created")
            return

        messages = self.short_term.get_messages()

        # Skip saving if there are no messages (empty conversation)
        if not messages and not self.system_messages:
            logger.debug(f"Skipping save_memory: no messages to save for session {self.session_id}")
            return

        await self.store.save_memory(
            session_id=self.session_id,
            system_messages=self.system_messages,
            messages=messages,
            summaries=[],  # Summaries are now part of messages
        )
        logger.info(f"Saved memory state for session {self.session_id}")

    def reset(self):
        """Reset memory manager state."""
        self.short_term.clear()
        self.system_messages.clear()
        self.token_tracker.reset()
        self.current_tokens = 0
        self.was_compressed_last_iteration = False
        self.last_compression_savings = 0
        self.compression_count = 0

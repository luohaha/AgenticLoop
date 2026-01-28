# RFC 005: Context Compact System (Intelligent History Management)

- **Status**: Draft
- **Created**: 2026-01-28
- **Author**: AgenticLoop Team

## Abstract

This RFC proposes enhancements to AgenticLoop's memory compression system, inspired by production-proven patterns from OpenAI's Codex agent. The goal is to provide **robust context management** through write-time truncation, automatic overflow recovery, and improved history compaction.

## Motivation

Current AgenticLoop memory system lacks critical features for production use:

1. **Tool outputs can explode context**: A single `read_file` or `shell` command can return 100KB+ output
2. **No automatic recovery from context overflow**: Sessions fail on `context_length_exceeded` errors
3. **Tool pair integrity is fragile**: Removing messages can orphan tool calls/results, causing API errors
4. **No proactive truncation**: Large outputs stored verbatim until compression triggers

## Goals

- **Write-time truncation**: Truncate large tool outputs when added to history
- **Context overflow recovery**: Auto-recover from `context_length_exceeded` by removing oldest messages
- **Tool pair integrity**: Maintain call-output pairs when removing messages
- **User message preservation**: Keep original user messages during compaction
- **Backward compatibility**: Existing behavior continues to work

## Non-Goals

- **Remote compact API** — OpenAI-specific, we use LiteLLM for multi-provider support
- **Ghost snapshot / undo** — Future RFC consideration
- **Changing 4-role message model** — Already sufficient for our needs

## Key Design Decisions

### Decision 1: 4-Role Message Model is Sufficient

AgenticLoop uses `LLMMessage` with 4 roles (`system`, `user`, `assistant`, `tool`), while Codex has 11+ `ResponseItem` variants. Analysis shows our model is sufficient:

| Operation | How It Works |
|-----------|--------------|
| Truncation | Targets `role="tool"` messages only |
| Compact preservation | Identifies `role="user"` messages |
| Tool pair matching | Uses `tool_call_id` field matching |

### Decision 2: Always Inline Compact (No Remote)

Codex supports remote compact via OpenAI's `/responses/compact` endpoint. We choose inline-only because:
- Works with any LLM provider (via LiteLLM)
- More control over summarization
- No vendor lock-in

### Decision 3: 50/50 Truncation Split with Serialization Buffer

Like Codex, preserve both **beginning** and **end** of truncated content:
- 50% from start (context/setup)
- 50% from end (results/conclusions)
- **20% serialization buffer**: Apply `max_tokens * 1.2` to account for JSON overhead
- Marker format: `…N tokens truncated…` or `…N chars truncated…`
- Optional line count header: `Total output lines: X`

> Note: Codex uses `budget / 2` for the split. We can adjust to 60/40 if testing shows better results.

### Decision 4: Prioritize Recent User Messages

During compact, user messages are preserved with recent-first priority (max 20K tokens total).

### Decision 5: Two-Phase Normalization Before LLM Call

Like Codex's `normalize.rs`, ensure tool pair integrity before sending to model:

1. **ensure_call_outputs_present()**: Add synthetic `"aborted"` output for orphaned tool calls
2. **remove_orphan_outputs()**: Remove tool results without matching calls

This prevents API errors from malformed conversation history.

### Decision 6: Normalize on Every LLM Call

Normalization should run in `for_prompt()` (before every LLM call), not just during compression:
1. Call `ensure_call_outputs_present()` to add synthetic outputs
2. Call `remove_orphan_outputs()` to clean up orphans
3. Filter out internal items (e.g., ghost snapshots)

### Decision 7: Protected Tools Never Compressed

Certain tool results must survive compaction (e.g., `manage_todo_list` for task tracking). These are identified by tool name and preserved in rebuilt history.

### Decision 8: Preserve Turn Aborted Markers

When collecting user messages for compact, also preserve `<turn-aborted>` markers that indicate interrupted turns. This maintains context about what was attempted but not completed.

## AgenticLoop Existing Strengths (Keep)

| Feature | Notes |
|---------|-------|
| **Multiple compression strategies** | `deletion`, `sliding_window`, `selective` |
| **Provider-specific token counting** | tiktoken for OpenAI (more accurate than Codex's ~4 chars/token) |
| **Tool pair detection** | `_find_tool_pairs()` handles both OpenAI and Anthropic formats |
| **Orphaned tool handling** | `orphaned_tool_use_indices` preserved during compression |
| **Configurable thresholds** | `MEMORY_COMPRESSION_THRESHOLD`, `MEMORY_SHORT_TERM_SIZE` |

## Gap Analysis Summary

| Feature | Codex | AgenticLoop Current | Proposed |
|---------|-------|---------------------|----------|
| Write-time truncation | ✅ `process_item()` | ❌ None | ✅ Phase 1 |
| Truncation policy config | ✅ Bytes/Tokens | ❌ None | ✅ Phase 1 |
| Context overflow recovery | ✅ Auto-retry | ❌ Fails | ✅ Phase 2 |
| Pair integrity on removal | ✅ `remove_corresponding_for()` | ❌ None | ✅ Phase 3 |
| User message truncation | ✅ 20K limit | ❌ None | ✅ Phase 4 |
| History rebuild | ✅ Structured | ⚠️ Strategy-dependent | ✅ Phase 5 |
| Protected tools | ✅ Yes | ✅ `manage_todo_list` | ✅ Keep |
| Tool pair detection | ✅ `call_id` | ✅ `_find_tool_pairs()` | ✅ Keep |
| Orphan output handling | ✅ `remove_orphan_outputs()` | ❌ None | ✅ Phase 3 |
| Orphan call handling | ✅ `ensure_call_outputs_present()` | ✅ `orphaned_tool_use_indices` | ✅ Keep |

## Compact Flow Design

### Trigger Conditions

| Trigger | Current | Proposed |
|---------|---------|----------|
| Token threshold | ✅ `MEMORY_COMPRESSION_THRESHOLD` | Keep |
| Memory full | ✅ `MEMORY_SHORT_TERM_SIZE` | Keep |
| Context overflow error | ❌ Fails | ✅ Add |
| Manual `/compact` | ❌ None | ⚠️ P2 |

### What Gets Discarded vs Preserved

| Item Type | Action | Notes |
|-----------|--------|-------|
| System prompts | **Keep** | Initial context |
| User messages | **Keep** | Truncated to 20K tokens total |
| Previous summaries | **Discard** | Replaced by new summary |
| Turn aborted markers | **Keep** | Preserve `<turn-aborted>` context |
| Assistant messages | **Discard** | Replaced by summary |
| Tool calls | **Discard** | Not needed after summary |
| Tool results | **Discard** | Not needed after summary |
| Protected tool results | **Keep** | `manage_todo_list`, `read_file` with critical data |

### Protected Tools List

Tools whose results survive compaction (configurable):
- `manage_todo_list` — Task tracking state
- Future: any tool marked with `protected=True`

### Rebuilt History Structure

```
[System Prompts] + [User Messages (truncated)] + [Summary] + [Protected Tools]
```

## Implementation Plan

### Phase 1: Write-Time Truncation (P0)

**Goal**: Truncate tool outputs at `add_message()` time.

**Key interfaces**:
```python
# memory/truncate.py
def truncate_with_split(content: str, max_tokens: int) -> str:
    """50/50 split: preserve beginning and end, remove middle."""

# memory/manager.py
def _maybe_truncate_tool_output(self, message: LLMMessage) -> LLMMessage:
    """Truncate tool message if exceeds TOOL_OUTPUT_MAX_TOKENS."""
```

**Acceptance**:
- Tool outputs > 5000 tokens truncated with marker
- Configurable via `TOOL_OUTPUT_TRUNCATION_POLICY`

### Phase 2: Context Overflow Recovery (P0)

**Goal**: Auto-recover from `context_length_exceeded` errors.

**Key interfaces**:
```python
# llm/retry.py
def is_context_length_error(error: BaseException) -> bool:
    """Detect context overflow from various providers."""

# agent/base.py
async def _call_with_overflow_recovery(self, messages, max_retries=3):
    """Retry LLM call after removing oldest messages."""
```

**Acceptance**:
- Context errors trigger automatic recovery (max 3 retries)
- Removed messages maintain tool pair integrity

### Phase 3: Tool Pair Integrity on Removal (P1)

**Goal**: When removing a message, also remove its counterpart. Add normalization before LLM calls.

**Key interfaces**:
```python
# memory/manager.py
def remove_oldest_with_pair_integrity(self) -> Optional[LLMMessage]:
    """Remove oldest message and its corresponding tool pair."""

def _remove_messages_by_tool_call_ids(self, call_ids: Set[str]) -> None:
    """Remove all tool results matching given call IDs."""

def ensure_call_outputs_present(self, messages: List[LLMMessage]) -> List[LLMMessage]:
    """Add synthetic 'aborted' output for orphaned tool calls."""

def remove_orphan_outputs(self, messages: List[LLMMessage]) -> List[LLMMessage]:
    """Remove tool results without matching calls."""
```

**Acceptance**:
- No orphaned tool calls or results after removal
- Normalization runs before every LLM call

### Phase 4: User Message Truncation During Compact (P1)

**Goal**: Limit total user message tokens during compaction.

**Key interfaces**:
```python
# memory/compressor.py
def select_user_messages(messages: List[str], max_tokens: int = 20000) -> List[str]:
    """Select user messages, prioritizing recent ones."""
```

**Acceptance**:
- User messages capped at 20K tokens (recent-first)

### Phase 5: Inline Compact History Rebuild (P1)

**Goal**: Implement Codex-style history reconstruction.

**Key interfaces**:
```python
# memory/compressor.py
def collect_user_messages(messages: List[LLMMessage]) -> List[str]:
    """Extract user messages, excluding previous summaries."""

def build_compacted_history(
    initial_context: List[LLMMessage],
    user_messages: List[str],
    summary_text: str,
    protected_messages: List[LLMMessage],
) -> List[LLMMessage]:
    """Build new history after compaction."""

def is_summary_message(message: str) -> bool:
    """Check if message is a previous summary."""
```

**Acceptance**:
- History = initial context + user messages + summary + protected tools
- Previous summaries excluded on re-compact

### Phase 6: Manual `/compact` Command (P2, Optional)

**Goal**: User-triggered compression via `/compact` command.

**Acceptance**:
- Reports tokens saved and messages compressed

## Phases at a Glance

| Phase | Priority | Change | New Files |
|------:|----------|--------|-----------|
| 1 | P0 | Write-time truncation (50/50 split) | `memory/truncate.py` |
| 2 | P0 | Context overflow auto-recovery | - |
| 3 | P1 | Tool pair integrity on removal | - |
| 4 | P1 | User message truncation (20K limit) | - |
| 5 | P1 | Inline compact history rebuild | - |
| 6 | P2 | Manual `/compact` command | - |

## Configuration

```python
# Truncation
TOOL_OUTPUT_TRUNCATION_POLICY = "tokens"  # none, bytes, tokens
TOOL_OUTPUT_MAX_TOKENS = 5000
TOOL_OUTPUT_SERIALIZATION_BUFFER = 1.2    # 20% buffer for JSON overhead
APPROX_CHARS_PER_TOKEN = 4

# Compact
COMPACT_USER_MESSAGE_MAX_TOKENS = 20000
CONTEXT_OVERFLOW_MAX_RETRIES = 3

# Protected Tools (results survive compaction)
PROTECTED_TOOLS = ["manage_todo_list"]

# Prompts (customizable)
COMPACT_SUMMARIZATION_PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION.
Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data needed to continue

Be concise and focused on helping the next LLM seamlessly continue."""

COMPACT_SUMMARY_PREFIX = """Another language model started this task and produced
a summary. Use this to build on existing work and avoid duplication:"""
```

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Truncation loses important info | High limit (5000 tokens), preserve start+end |
| Aggressive overflow recovery | Max 3 retries, log removed content |
| Summary quality affects continuity | Use proven Codex prompts, allow customization |
| Long threads cause accuracy loss | Show warning after compaction (like Codex) |

## Open Questions

1. Should truncated content be logged for debugging?
2. Should summary use a smaller/faster model for cost savings?
3. Should we add `is_compact_summary` field to `LLMMessage`? (vs prefix detection)

## Appendix: Compact Flow Diagram

```
TRIGGER
  ├── Token threshold exceeded
  ├── Short-term memory full
  └── Context overflow error (with retry)
        ↓
COLLECT
  ├── initial_context (system prompts)
  ├── user_messages (filter out previous summaries)
  └── protected_tools (manage_todo_list, etc.)
        ↓
SUMMARIZE
  └── Call LLM with COMPACT_SUMMARIZATION_PROMPT
        ↓
REBUILD
  └── initial_context + user_messages + summary + protected_tools
        ↓
REPLACE
  └── Replace history, recompute token usage
```

## References

- Codex compact: `codex-rs/core/src/compact.rs`
- Codex truncation: `codex-rs/core/src/truncate.rs`
- Codex normalization: `codex-rs/core/src/context_manager/normalize.rs`
- AgenticLoop memory: `memory/manager.py`, `memory/compressor.py`

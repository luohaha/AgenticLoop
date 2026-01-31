# Memory Persistence

## Overview

Memory persistence stores conversation sessions as human-readable YAML files under `.aloop/sessions/`. Key features:

1. **Session Management**: Each conversation is saved as a YAML file
2. **Human-Readable**: Session files can be viewed and edited with any text editor
3. **Session Recovery**: Resume previous sessions via CLI (`--resume`) or interactive (`/resume`)
4. **Batch Persistence**: Memory is saved as a batch after task completion (efficient)

## How It Works

Memory is **automatically saved** when `await agent.run()` completes a task.

You can also **manually save** by calling:
```python
await manager.save_memory()  # Saves current state to YAML
```

### Directory Structure

```
.aloop/sessions/
├── .index.yaml                          # UUID → directory mapping (auto-managed)
├── 2025-01-31_a1b2c3d4/
│   └── session.yaml                     # Session data
├── 2025-01-31_e5f6g7h8/
│   └── session.yaml
└── ...
```

### Session YAML Format

```yaml
id: a1b2c3d4-5678-90ab-cdef-1234567890ab
created_at: "2025-01-31T14:30:00"
updated_at: "2025-01-31T15:45:00"

system_messages:
  - role: system
    content: |
      You are a helpful assistant.

messages:
  - role: user
    content: "Hello"
  - role: assistant
    content: null
    tool_calls:
      - id: call_abc123
        type: function
        function:
          name: calculator
          arguments: '{"expression": "2+2"}'
  - role: tool
    content: "4"
    tool_call_id: call_abc123
    name: calculator
```

## Quick Start

### 1. Using with Agent (Automatic Save)

```python
from agent import ReActAgent

# Memory is automatically saved when agent.run() completes
result = await agent.run("Your task here")

print(f"Session ID: {agent.memory.session_id}")
```

### 2. Resume a Previous Session

#### CLI

```bash
# Resume the most recent session
python main.py --resume --task "Continue the previous work"

# Resume a specific session (full ID or prefix)
python main.py --resume a1b2c3d4 --task "Continue from here"
```

#### Interactive Mode

```
> /resume                    # List recent sessions
> /resume a1b2c3d4           # Resume by ID prefix
> /history                   # View all saved sessions
```

### 3. Manual Save (Without Agent)

```python
from memory import MemoryManager
from llm.message_types import LLMMessage

manager = MemoryManager(llm=llm)

await manager.add_message(LLMMessage(role="user", content="Hello"))
await manager.add_message(LLMMessage(role="assistant", content="Hi!"))

await manager.save_memory()
print(f"Session ID: {manager.session_id}")
```

### 4. Restore Existing Session

```python
from memory import MemoryManager

session_id = "your-session-id-here"
manager = await MemoryManager.from_session(session_id=session_id, llm=llm)

# Continue conversation
await manager.add_message(LLMMessage(role="user", content="Continue..."))
await manager.save_memory()
```

### 5. View Historical Sessions

```bash
# List all sessions
python tools/session_manager.py list

# Show specific session details
python tools/session_manager.py show <session_id>

# Show session statistics
python tools/session_manager.py stats <session_id>

# Show session messages
python tools/session_manager.py show <session_id> --messages

# Delete a session
python tools/session_manager.py delete <session_id>
```

## Architecture

Persistence is fully managed internally by `MemoryManager` using a YAML file backend. The store lifecycle is entirely owned by `MemoryManager` — external code should not create or pass in store instances.

## API Documentation

### MemoryManager

```python
from memory import MemoryManager

# Create new session
manager = MemoryManager(llm=llm)

# Load from existing session
manager = await MemoryManager.from_session(session_id="...", llm=llm)

# Session discovery (class methods)
sessions = await MemoryManager.list_sessions(limit=20)
latest_id = await MemoryManager.find_latest_session()
full_id = await MemoryManager.find_session_by_prefix("a1b2")
```

## Notes

1. **File Location**: Sessions are stored under `.aloop/sessions/`. The directory is created automatically.

2. **Session ID**: Each session has a UUID. You can use the full ID or a unique prefix (e.g., first 8 characters) when resuming.

3. **Atomic Writes**: Session files are written atomically (write to `.tmp`, then `os.replace()`) to prevent corruption on crashes.

4. **Index File**: `.aloop/sessions/.index.yaml` maps UUIDs to directory names for fast lookup. It is automatically rebuilt if missing.

5. **Human Editing**: You can manually edit `session.yaml` files. Changes will be picked up on next load.

## Testing

```bash
# Run all memory tests
python -m pytest test/memory/ -v

# Run YAML backend tests specifically
python -m pytest test/memory/test_yaml_backend.py -v

# Run serialization tests
python -m pytest test/memory/test_serialization.py -v
```

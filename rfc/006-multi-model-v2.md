# RFC: Multi-Model Configuration and Runtime Switching (v2)

## Problem Statement

Currently, the AgenticLoop system effectively supports only a single active model configuration. Users cannot:
- Configure multiple models
- Switch between models at runtime
- Easily test different models without editing config files
- Add/remove models dynamically

This limitation makes it difficult to:
- Compare model performance
- Use different models for different tasks
- Quickly switch between models during development
- Manage model configurations in a structured way

## Design Goals

1. **Multiple Models**: Support configuring multiple models
2. **Runtime Switching**: Allow switching between models via `/model` command in interactive mode
3. **Runtime Editing**: Support adding, editing, and removing models at runtime with persistence
4. **Structured Config**: Use YAML for clean, structured configuration
5. **Config Protection**: YAML config should not be committed to git (contains API keys)
6. **No Backward Compatibility**: Clean slate design without legacy support

## Proposed Solution

### 1. Configuration Format (YAML)

New file: `.aloop/models.yaml`

```yaml
# Model Configuration
# This file is gitignored - do not commit to version control
#
# The key under `models` is the LiteLLM model ID: provider/model-name
# The `name` field is optional (purely for display).

models:
  anthropic/claude-3-5-sonnet-20241022:
    name: Claude 3.5 Sonnet
    api_key: sk-ant-...
    timeout: 600
    drop_params: true

  openai/gpt-4o:
    name: GPT-4o
    api_key: sk-...
    timeout: 300

  gemini/gemini-1.5-pro:
    name: Gemini Pro
    api_key: ...

  # Local model example
  ollama/llama2:
    name: Local Llama
    api_base: http://localhost:11434

default: anthropic/claude-3-5-sonnet-20241022
```

### 2. Model Manager Class

Create a new `llm/model_manager.py`:

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml
import os

@dataclass
class ModelProfile:
    """Configuration for a single model."""
    model_id: str              # LiteLLM model ID (provider/model)
    name: str                  # Optional display name
    api_key: Optional[str]
    api_base: Optional[str]
    timeout: int = 600
    drop_params: bool = True

class ModelManager:
    """Manages multiple models with YAML persistence."""

    CONFIG_PATH = ".aloop/models.yaml"

    def __init__(self):
        self.models: Dict[str, ModelProfile] = {}  # key = model_id
        self.default_model_id: Optional[str] = None
        self.current_model_id: Optional[str] = None
        self._load()

    def _load(self) -> None:
        """Load models from YAML config."""
        if not os.path.exists(self.CONFIG_PATH):
            self._create_default_config()

        with open(self.CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

        for model_id, data in config.get('models', {}).items():
            self.models[model_id] = ModelProfile(
                model_id=model_id,
                name=data.get('name', ''),
                api_key=data.get('api_key'),
                api_base=data.get('api_base'),
                timeout=data.get('timeout', 600),
                drop_params=data.get('drop_params', True),
            )

        self.default_model_id = config.get('default')
        if self.default_model_id in self.models:
            self.current_model_id = self.default_model_id
        elif self.models:
            self.default_model_id = next(iter(self.models.keys()))
            self.current_model_id = self.default_model_id

    def _save(self) -> None:
        """Save models to YAML config."""
        config = {
            'models': {},
            'default': self.default_model_id
        }

        for model_id, profile in self.models.items():
            config['models'][model_id] = {
                'name': profile.name,
                'api_key': profile.api_key,
                'api_base': profile.api_base,
                'timeout': profile.timeout,
                'drop_params': profile.drop_params,
            }

        with open(self.CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
```

### 3. Interactive Mode Commands

Enhanced `/model` command in `interactive.py`:

```
/model                          - List all available models
/model <model_id>               - Switch to the specified model
/model add <model_id> <field=value>...     - Add a new model (e.g. name=... api_key=...)
/model edit <model_id> <field=value>...    - Edit model (e.g. /model edit openai/gpt-4o timeout=300)
/model remove <model_id>        - Remove a model (not current)
/model default <model_id>       - Set default model
/model show <model_id>          - Show model details
/model reload                   - Reload models.yaml from disk
```

Example output for `/model`:
```
Available Models:
  [CURRENT] [DEFAULT] Claude 3.5 Sonnet - anthropic/claude-3-5-sonnet-20241022
            GPT-4o - openai/gpt-4o
            Local Llama - ollama/llama2

Use /model <model_id> to switch, /model add to add new
```

### 4. Git Protection

Update `.gitignore`:
```gitignore
# Model configuration (contains API keys)
.aloop/models.yaml
```

### 5. Default Config Creation

When `models.yaml` doesn't exist, create a template:

```python
def _create_default_config(self) -> None:
    """Create default models.yaml template."""
    template = """# Model Configuration
# This file is gitignored - do not commit to version control
#
# Supported fields:
#   - name: Display name
#   - api_key: API key
#   - api_base: Custom base URL (optional)
#   - timeout: Request timeout in seconds (default: 600)
#   - drop_params: Drop unsupported params (default: true)

models:
  anthropic/claude-3-5-sonnet-20241022:
    name: Claude 3.5 Sonnet
    api_key: sk-ant-...
    timeout: 600
    drop_params: true

default: anthropic/claude-3-5-sonnet-20241022
"""
    os.makedirs(os.path.dirname(self.CONFIG_PATH), exist_ok=True)
    with open(self.CONFIG_PATH, 'w') as f:
        f.write(template)
```

### 6. Agent Integration

Modify `agent/base.py`:

```python
class BaseAgent:
    def __init__(self, llm, tools, model_manager: ModelManager, ...):
        self.llm = llm
        self.model_manager = model_manager

    def switch_model(self, model_id: str) -> bool:
        """Switch to a different model."""
        profile = self.model_manager.switch_model(model_id)
        if profile:
            # Reinitialize LLM adapter
            self.llm = LiteLLMAdapter(
                model=profile.model_id,
                api_key=profile.api_key,
                api_base=profile.api_base,
                timeout=profile.timeout,
                drop_params=profile.drop_params,
            )
            return True
        return False
```

### 7. CLI Flag

Support `--model` flag in `main.py`:

```bash
python main.py --task "Hello" --model openai/gpt-4o
```

## Implementation Plan

### Phase 1: Core Infrastructure
1. Create `llm/model_manager.py` with YAML support
2. Update `.gitignore` to protect `models.yaml`
3. Add tests for ModelManager

### Phase 2: Interactive Mode
1. Add `/model` command with all subcommands
2. Update status bar to show current model
3. Add model switching logic

### Phase 3: Agent Integration
1. Update `agent/base.py` to use ModelManager
2. Update `main.py` to load models from YAML
3. Add `--model` CLI flag

### Phase 4: Documentation
1. Update `docs/configuration.md` with YAML format
2. Update `README.md` with new commands
3. Add examples

## Key Design Decisions

### 1. YAML vs Other Formats
- **YAML**: Human-readable, supports comments, standard for config
- **JSON**: No comments, less readable
- **TOML**: Good alternative, but YAML more common in Python ecosystem

### 2. No Environment Variable Substitution
- Store API keys directly in `.aloop/models.yaml`
- Rely on gitignore + file permissions to protect secrets
- On missing/empty config, create a template and guide the user to edit it

### 3. No Backward Compatibility
- Clean break from legacy model configuration via `.aloop/config`
- Simpler code without compatibility layers
- Users migrate manually (one-time cost)

### 4. Persistence Strategy
- All changes immediately saved to YAML
- No separate "save" command needed
- Atomic writes to prevent corruption

### 5. Identifier Choice
- Use the LiteLLM model ID (`provider/model`) as the identifier in config and commands
- `name` is optional and display-only

## Open Questions

1. **Validation**: Should we validate API keys on add/edit?
   - **Decision**: Validate format, but not by making API calls (too slow)

2. **Encryption**: Should API keys be encrypted at rest?
   - **Decision**: No, rely on gitignore and file permissions for now

3. **Import/Export**: Should we support importing/exporting configs?
   - **Decision**: Future enhancement, not in v1

## Success Criteria

- Users can configure multiple models in `.aloop/models.yaml`
- `/model` command lists, switches, adds, edits, removes models
- Changes are persisted to YAML immediately
- Status bar shows current model
- YAML config is gitignored
- Clean, intuitive YAML structure

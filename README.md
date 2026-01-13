# Agentic Loop

General AI Agent System

## Installation

### Option 1: Install from PyPI (Recommended - Coming Soon)

```bash
pip install AgenticLoop
```

### Option 2: Install from Source (Development)

```bash
# Clone the repository
git clone https://github.com/yourusername/AgenticLoop.git
cd AgenticLoop

# Install in development mode
pip install -e .
```

### Option 3: Install from GitHub

```bash
pip install git+https://github.com/yourusername/AgenticLoop.git
```

### Option 4: Docker

```bash
docker pull yourusername/AgenticLoop:latest
docker run -it --rm -e ANTHROPIC_API_KEY=your_key AgenticLoop interactive
```

## Quick Start

### 1. Configuration

Create `.env` file:

```bash
cp .env.example .env
```

Edit `.env` file and configure your LLM provider:

```bash
# LLM Provider (required)
LLM_PROVIDER=gemini  # Options: anthropic, openai, gemini

# API Keys (set the one for your chosen provider)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Model (optional - uses provider defaults if not set)
MODEL=gemini-2.5-flash

# Base URLs (optional - for proxies, Azure, local deployments)
ANTHROPIC_BASE_URL=
OPENAI_BASE_URL=
GEMINI_BASE_URL=

# Agent Configuration
MAX_ITERATIONS=100  # Maximum iteration loops

# Memory Management
MEMORY_MAX_CONTEXT_TOKENS=100000
MEMORY_TARGET_TOKENS=30000
MEMORY_COMPRESSION_THRESHOLD=25000
MEMORY_SHORT_TERM_SIZE=100
MEMORY_COMPRESSION_RATIO=0.3

# Retry Configuration (for handling rate limits)
RETRY_MAX_ATTEMPTS=5
RETRY_INITIAL_DELAY=1.0
RETRY_MAX_DELAY=60.0

# Logging
LOG_DIR=logs
LOG_LEVEL=DEBUG
LOG_TO_FILE=true
LOG_TO_CONSOLE=false
```

**Quick setup for different providers:**

- **Anthropic Claude**: Set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`
- **OpenAI GPT**: Set `LLM_PROVIDER=openai` and `OPENAI_API_KEY`
- **Google Gemini**: Set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY`

### 2. Usage

#### Command Line (After Installation)

```bash
# Interactive mode
aloop

# Single task (ReAct mode)
aloop --mode react "Calculate 123 * 456"

# Single task (Plan-Execute mode)
aloop --mode plan "Build a web scraper"

# Show help
aloop --help
```

#### Direct Python Execution (Development)

If running from source without installation:

**ReAct Mode (Interactive)**

```bash
python main.py --mode react --task "Calculate 123 * 456"
```

**Plan-and-Execute Mode (Planning)**

```bash
python main.py --mode plan --task "Search for Python agent tutorials and summarize top 3 results"
```

**Interactive Input**

```bash
python main.py --mode react
# Then enter your task, press Enter twice to submit
```

## Memory Management

The system includes intelligent memory management that automatically optimizes token usage for long-running tasks:

```bash
python main.py --task "Complex multi-step task with many iterations..."

# Memory statistics shown at the end:
# --- Memory Statistics ---
# Total tokens: 45,234
# Compressions: 3
# Net savings: 15,678 tokens (34.7%)
# Total cost: $0.0234
```

**Key features:**
- Automatic compression when context grows large
- 30-70% token reduction for long conversations
- Multiple compression strategies
- Cost tracking across providers
- Transparent operation (no code changes needed)

See [Memory Management Documentation](docs/memory-management.md) for detailed information.

## Project Structure

```
AgenticLoop/
├── README.md                    # This document
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── config.py                    # Configuration management
├── main.py                      # CLI entry point
├── docs/                        # 📚 Documentation
│   ├── examples.md              # Detailed usage examples
│   ├── configuration.md         # Configuration guide
│   ├── memory-management.md     # Memory system docs
│   ├── advanced-features.md     # Advanced features & optimization
│   └── extending.md             # Extension guide
├── llm/                         # LLM abstraction layer
│   ├── base.py                  # BaseLLM abstract class
│   ├── anthropic_llm.py         # Anthropic Claude adapter
│   ├── openai_llm.py            # OpenAI GPT adapter
│   ├── gemini_llm.py            # Google Gemini adapter
│   └── retry.py                 # Retry logic for rate limits
├── agent/                       # Agent implementations
│   ├── base.py                  # BaseAgent abstract class
│   ├── context.py               # Context injection
│   ├── react_agent.py           # ReAct mode
│   ├── plan_execute_agent.py   # Plan-and-Execute mode
│   ├── tool_executor.py         # Tool execution engine
│   └── todo.py                  # Todo list management
├── memory/                      # 🧠 Memory management system
│   ├── types.py                 # Core data structures
│   ├── manager.py               # Memory orchestrator
│   ├── short_term.py            # Short-term memory
│   ├── compressor.py            # LLM-driven compression
│   └── token_tracker.py         # Token tracking & costs
├── tools/                       # Tool implementations
│   ├── base.py                  # BaseTool abstract class
│   ├── file_ops.py              # File operation tools (read/write/search)
│   ├── advanced_file_ops.py     # Advanced tools (Glob/Grep/Edit)
│   ├── calculator.py            # Code execution/calculator
│   ├── shell.py                 # Shell commands
│   ├── web_search.py            # Web search
│   ├── todo.py                  # Todo list management
│   └── delegation.py            # Sub-agent delegation
├── utils/                       # Utilities
│   └── logger.py                # Logging setup
└── examples/                    # Example code
    ├── react_example.py         # ReAct mode example
    └── plan_execute_example.py  # Plan-Execute example
```

## Documentation

- **[Examples](docs/examples.md)**: Detailed usage examples and patterns
- **[Configuration](docs/configuration.md)**: Complete configuration guide
- **[Memory Management](docs/memory-management.md)**: Memory system documentation
- **[Advanced Features](docs/advanced-features.md)**: Optimization and advanced techniques
- **[Extending](docs/extending.md)**: How to add tools, agents, and LLM providers
- **[Packaging Guide](docs/packaging.md)**: Package and distribute the system

## Configuration Options

See the full configuration template in `.env.example`. Key options:

| Setting | Description | Default |
|---------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (anthropic/openai/gemini) | `anthropic` |
| `MODEL` | Specific model to use | Provider default |
| `MAX_ITERATIONS` | Maximum agent iterations | `100` |
| `MEMORY_MAX_CONTEXT_TOKENS` | Maximum context window | `100000` |
| `MEMORY_TARGET_TOKENS` | Target working memory size | `30000` |
| `MEMORY_COMPRESSION_THRESHOLD` | Compress when exceeded | `25000` |
| `MEMORY_SHORT_TERM_SIZE` | Recent messages to keep | `100` |
| `RETRY_MAX_ATTEMPTS` | Retry attempts for rate limits | `5` |
| `LOG_LEVEL` | Logging level | `DEBUG` |

See [Configuration Guide](docs/configuration.md) for detailed options.

## Testing

Run basic tests:

```bash
source venv/bin/activate
python test_basic.py
```

## Learning Resources

- **ReAct Paper**: [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- **Anthropic API Documentation**: [docs.anthropic.com](https://docs.anthropic.com)
- **Tool Use Guide**: [Tool Use (Function Calling)](https://docs.anthropic.com/en/docs/tool-use)

## Future Improvements

- [ ] Streaming output to display agent thinking process
- [x] Intelligent memory management with compression
- [ ] Parallel tool execution
- [ ] Detailed logging and tracing
- [ ] Human-in-the-loop for dangerous operations
- [ ] Multi-agent collaboration system
- [ ] Persistent memory with session recovery
- [ ] Semantic retrieval with vector database

## License

MIT License

## Development

### Building and Packaging

See the [Packaging Guide](docs/packaging.md) for instructions on:
- Building distributable packages
- Publishing to PyPI
- Creating Docker images
- Generating standalone executables

Quick commands:
```bash
# Install locally for development
./scripts/install_local.sh

# Build distribution packages
./scripts/build.sh

# Publish to PyPI
./scripts/publish.sh
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

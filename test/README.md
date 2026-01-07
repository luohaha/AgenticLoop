# Tests

This directory contains test files for the agentic-loop project.

## Test Files

- `test_basic.py` - Basic functionality tests for tools and imports
- `test_memory.py` - Memory management system demonstration and tests

## Running Tests

### Prerequisites

Make sure you have installed all dependencies:

```bash
pip install -r requirements.txt
```

### Run Individual Tests

From the project root directory:

```bash
# Run basic tests
python3 test/test_basic.py

# Run memory tests
python3 test/test_memory.py
```

### Run All Tests

```bash
# From project root
python3 -m pytest test/

# Or run all test files
for test in test/test_*.py; do python3 "$test"; done
```

## Test Coverage

- **test_basic.py**: Tests basic tool functionality, imports, and API key configuration
- **test_memory.py**: Demonstrates memory compression and token tracking features

## Notes

- Some tests may require environment variables (e.g., `ANTHROPIC_API_KEY`)
- Set up your `.env` file before running tests that require API access
- Memory tests use a mock LLM and don't require API keys

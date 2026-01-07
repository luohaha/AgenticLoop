"""Basic test to verify the agentic loop system works."""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test imports
try:
    from config import Config
    from agent.react_agent import ReActAgent
    from tools.calculator import CalculatorTool
    from tools.file_ops import FileReadTool, FileWriteTool

    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test tool instantiation
try:
    calc = CalculatorTool()
    print(f"✓ Calculator tool created: {calc.name}")

    file_read = FileReadTool()
    print(f"✓ File read tool created: {file_read.name}")

    file_write = FileWriteTool()
    print(f"✓ File write tool created: {file_write.name}")
except Exception as e:
    print(f"✗ Tool instantiation failed: {e}")
    sys.exit(1)

# Test tool execution
try:
    result = calc.execute(code="print(2 + 2)")
    print(f"✓ Calculator execution: {result.strip()}")

    # Test file write and read
    test_file = "test_temp.txt"
    write_result = file_write.execute(file_path=test_file, content="Hello, Agent!")
    print(f"✓ File write: {write_result}")

    read_result = file_read.execute(file_path=test_file)
    print(f"✓ File read: {read_result}")

    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
        print("✓ Cleanup successful")
except Exception as e:
    print(f"✗ Tool execution failed: {e}")
    sys.exit(1)

# Test tool schema generation
try:
    schema = calc.to_anthropic_schema()
    assert "name" in schema
    assert "description" in schema
    assert "input_schema" in schema
    print(f"✓ Tool schema generation successful")
except Exception as e:
    print(f"✗ Schema generation failed: {e}")
    sys.exit(1)

# Check API key
if not Config.ANTHROPIC_API_KEY:
    print("\n⚠ Warning: ANTHROPIC_API_KEY not set")
    print("To use the agent, create a .env file with:")
    print("ANTHROPIC_API_KEY=your_api_key_here")
else:
    print(f"\n✓ API key configured (starts with: {Config.ANTHROPIC_API_KEY[:8]}...)")

print("\n" + "=" * 60)
print("All basic tests passed!")
print("=" * 60)
print("\nTo run the agent:")
print("  source venv/bin/activate")
print("  python main.py --task 'Calculate 123 * 456'")

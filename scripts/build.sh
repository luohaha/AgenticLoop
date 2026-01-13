#!/bin/bash
# Build the package for distribution

set -e  # Exit on error

echo "🔨 Building AgenticLoop package..."

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info

# Install build tools
echo "Installing build tools..."
python3 -m pip install --upgrade build twine

# Build the package
echo "Building package..."
python3 -m build

echo "✅ Build complete! Distribution files are in dist/"
ls -lh dist/

echo ""
echo "Next steps:"
echo "  1. Test locally: pip install dist/agentic_loop-*.whl"
echo "  2. Upload to PyPI: twine upload dist/*"

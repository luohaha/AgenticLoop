#!/bin/bash
# Publish the package to PyPI

set -e

echo "🚀 Publishing AgenticLoop to PyPI..."

# Check if dist/ exists
if [ ! -d "dist" ]; then
    echo "❌ No dist/ directory found. Run ./scripts/build.sh first."
    exit 1
fi

# Install twine if needed
python3 -m pip install --upgrade twine

echo ""
echo "⚠️  This will upload to PyPI (production)."
echo "For testing, use: twine upload --repository testpypi dist/*"
read -p "Continue? (y/N) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Upload to PyPI
    twine upload dist/*
    echo ""
    echo "✅ Published to PyPI!"
    echo "Install with: pip install AgenticLoop"
else
    echo "Cancelled."
fi

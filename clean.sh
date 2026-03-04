rm dump_test* -rf
rm onnxruntime_profile*
rm temp*.onnx
rm test*.onnx

if [ "$1" == "--all" ]; then
    rm -rf dist/
    rm -rf build/
    rm -rf .ruff_cache/
    rm -rf .pytest_cache/
    rm -rf docs/_build/
    rm -rf docs/auto_examples/
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
fi

rm dump_test* -rf
rm onnxruntime_profile*
rm plot*.onnx
rm plot*.png
rm temp*.onnx
rm test*.data
rm test*.xlsx
rm test*.onnx
rm dummy*.onnx
rm *.ep
rm *.ep.graph

if [ "$1" == "--all" ]; then
    rm -rf dist/
    rm -rf build/
    rm -rf .ruff_cache/
    rm -rf .pytest_cache/
    rm -rf docs/_build/
    rm -rf docs/auto*
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
fi

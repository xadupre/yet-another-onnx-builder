# yet-another-onnx-builder

[![Tests](https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/tests.yml/badge.svg)](https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/tests.yml)
[![Documentation](https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/docs.yml/badge.svg)](https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/docs.yml)
[![Style](https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/style.yml/badge.svg)](https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/style.yml)
[![Spelling](https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/spelling.yml/badge.svg)](https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/spelling.yml)
[![codecov](https://codecov.io/gh/xadupre/yet-another-onnx-builder/branch/main/graph/badge.svg)](https://codecov.io/gh/xadupre/yet-another-onnx-builder)
[![GitHub repo size](https://img.shields.io/github/repo-size/xadupre/yet-another-onnx-builder)](https://github.com/xadupre/yet-another-onnx-builder)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Yet another onnx builder, patches, flattening functions...

**[Documentation](https://sdpython.github.io/doc/yet-another-onnx-builder/dev/index.html)**

**yet-another-onnx-builder** (`yobx`) is a toolkit for converting machine learning models
to [ONNX](https://onnx.ai) format from many libraries,
torch, tensorflow, scikit-learn, xgboost, ligthgbm.
It provides:

- A **graph builder API** for constructing and optimizing ONNX graphs, with built-in shape
  inference and a pattern-based graph optimizer : every converter creates nodes into a class
  of your choice as long as it follows one protocol. One GraphBuilder is provided with
  optimization included but other implementations are also made based on onnxscript / ir-py and spox.
- A **symbolic shape expression system** for dynamic shape handling at export time (`yobx.xshape`).
- A **translation tool** that converts ONNX graphs back to executable Python code (`yobx.translate`).

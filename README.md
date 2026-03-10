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

**yet-another-onnx-builder** (`yobx`) proposes a unique API to convert machine learning models
to [ONNX](https://onnx.ai) format from many libraries,
[torch](https://pytorch.org), [tensorflow](https://www.tensorflow.org),
[scikit-learn](https://scikit-learn.org),
[xgboost](https://xgboost.readthedocs.io),
[lightgbm](https://lightgbm.readthedocs.io).
It provides:

- A **graph builder API** for constructing and optimizing ONNX graphs, with built-in shape
  inference and a pattern-based graph optimizer : every converter creates nodes into a class
  of your choice as long as it follows one protocol. One GraphBuilder is provided with
  optimization included but other implementations are also made based on
  [onnxscript](https://microsoft.github.io/onnxscript/) / 
  [ir-py](https://onnx.ai/ir-py/) and [spox](https://spox.readthedocs.io/en/latest/).
- A **symbolic shape expression system** for dynamic shape handling at export time (`yobx.xshape`).
- A **translation tool** that converts ONNX graphs back to executable Python code (`yobx.translate`).
- **Optimization functions** to make the model more efficient.
- It supports multiple opsets and multiple domains.
- It allows the user to directly onnx model with [Spox]() or [onnxscript]()/[ir-py]().

Its unique API:

```python
# the model is called 
expected = model(*args, **kwargs)
onnx_model = to_onnx(model, args, kwargs, **options)
```

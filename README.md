# yet-another-onnx-builder

[![core](https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/ci_core.yml/badge.svg)](https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/ci_core.yml)
[![scikit-learn](https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/ci_sklearn.yml/badge.svg)](https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/ci_sklearn.yml)
[![tensorflow](https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/ci_tensorflow.yml/badge.svg)](https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/ci_tensorflow.yml)
[![pytorch](https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/ci_torch.yml/badge.svg)](https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/ci_torch.yml)
[![Documentation](https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/docs.yml/badge.svg)](https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/docs.yml)
[![Style](https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/style.yml/badge.svg)](https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/style.yml)
[![Spelling](https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/spelling.yml/badge.svg)](https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/spelling.yml)
[![codecov](https://codecov.io/gh/xadupre/yet-another-onnx-builder/branch/main/graph/badge.svg)](https://codecov.io/gh/xadupre/yet-another-onnx-builder)
[![GitHub repo size](https://img.shields.io/github/repo-size/xadupre/yet-another-onnx-builder)](https://github.com/xadupre/yet-another-onnx-builder)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Yet another onnx builder, patches, flattening functions...

**[Documentation](https://sdpython.github.io/doc/yet-another-onnx-builder/dev/index.html)**

**yet-another-onnx-builder** (`yobx`) proposes a unique API and a unique function
``yobx.to_onnx`` to convert machine learning models and other pipelines
to [ONNX](https://onnx.ai) format from many libraries. Each converter relies on a common GraphBuider API
to build the final ONNX model. One default implementation is provided but
it can also be replaced by any implementation of your own
([onnxscript](https://microsoft.github.io/onnxscript/)/[ir-py](https://onnx.ai/ir-py/), [Spox](https://spox.readthedocs.io/en/latest/)).
These API are close to `onnx` API, using `NodeProto` for nodes
and strings for names. This is on purpose: what this API produces is
what you see in the final ONNX model. You can add your own metadata,
choose your own names.

**standard machine learning**

* [category_encoders](https://contrib.scikit-learn.org/category_encoders/)
* [imbalanced-learn](https://imbalanced-learn.org/stable/)
* [lightgbm](https://lightgbm.readthedocs.io)
* [scikit-learn](https://scikit-learn.org)
* [scikit-survival](https://scikit-survival.readthedocs.io)
* [statsmodels](https://www.statsmodels.org)
* [xgboost](https://xgboost.readthedocs.io)

**data manipulation**

This is work in progress.
Many packages produce SQL queries. It starts by converting a SQL
query into ONNX. A lightweight **DataFrame function tracer**
([`dataframe_to_onnx`](https://sdpython.github.io/doc/yet-another-onnx-builder/dev/api/sql/dataframe_to_onnx.html))
records pandas-inspired operations on a virtual DataFrame and compiles them to ONNX:

* [SQL](https://fr.wikipedia.org/wiki/Structured_Query_Language)
* [polars](https://pola.rs/)
* [pandas](https://pandas.pydata.org/)

```python
import numpy as np
from onnxruntime import InferenceSession
from yobx.sql import dataframe_to_onnx
from yobx.reference import ExtendedReferenceEvaluator

def transform(df):
    df = df.filter(df["a"] > 0)
    return df.select([(df["a"] + df["b"]).alias("total")])

artifact = dataframe_to_onnx(transform, {"a": np.float32, "b": np.float32})
ref = InferenceSession(artifact.SerializeToString(), providers=["CPUExecutionProvider"])
(total,) = ref.run(None, {"a": np.array([1., -2., 3.], np.float32),
                           "b": np.array([4.,  5., 6.], np.float32)})
# total == [5., 9.]
```

**deeplearning**

* [litert](https://ai.google.dev/edge/litert/)
* [jax](https://jax.readthedocs.io/en/latest/) *in progress*
* [tensorflow](https://www.tensorflow.org)
* [torch](https://pytorch.org)

Its unique API across all converters:

```python
import torch
import onnxruntime
from yobx import to_onnx

# Define any PyTorch model
class Neuron(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        return torch.relu(self.linear(x))

model = Neuron()
x = torch.randn(3, 4)

# Export to ONNX — dynamic batch dimension
batch_dim = torch.export.Dim("batch", min=1, max=256)
artifact = to_onnx(model, (x,), dynamic_shapes={"x": {0: batch_dim}})

# Run with onnxruntime
sess = onnxruntime.InferenceSession(
    artifact.proto.SerializeToString(), providers=["CPUExecutionProvider"]
)
(result,) = sess.run(None, {"x": x.numpy()})
```

[onnxruntime](https://onnxruntime.ai/) optimizations are triggered with
``target_opset={"": 22, "com.microsoft": 1}``.

## Comparison with existing ONNX conversion tools

| Tool | Scope | Notes |
|------|-------|-------|
| [torch.onnx.export](https://pytorch.org/docs/stable/onnx.html) | PyTorch only | Official PyTorch exporter; `yobx` can delegate to it or use its own FX-based path |
| [onnxscript](https://microsoft.github.io/onnxscript/) | PyTorch (dynamo) | Microsoft's new exporter; `yobx` can use it as a graph-builder backend |
| [sklearn-onnx](https://onnx.ai/sklearn-onnx/) | scikit-learn only | Covers the scikit-learn ecosystem; `yobx` extends this with a unified API |
| [tf2onnx](https://github.com/onnx/tensorflow-onnx) | TensorFlow / Keras | Converts TensorFlow models; `yobx` wraps the same models under one entry point |
| [ModelBuilder](https://onnxruntime.ai/docs/genai/howto/build-model.html) | LLM inference (genai) | ONNX Runtime GenAI builder for large language models; `yobx` can produce models that target the same ORT execution providers |

**Pros of `yobx`**

* **Single entry point** — `yobx.to_onnx` dispatches to the right backend automatically; no need to learn a different API for every framework.
* **Pluggable graph-builder** — the intermediate ONNX graph can be built with the built-in `GraphBuilder`, with [onnxscript](https://microsoft.github.io/onnxscript/)/[ir-py](https://onnx.ai/ir-py/), or with [Spox](https://spox.readthedocs.io/en/latest/), keeping the conversion code framework-agnostic.
* **Transparent names** — node names, initializer names and result names are preserved as-is; what the builder writes is what ends up in the ONNX file.
* **Built-in optimizer** — pattern-based graph rewrites (constant folding, fused ops, …) run automatically before serialization.
* **ORT-specific targets** — passing `target_opset={"": 22, "com.microsoft": 1}` enables `com.microsoft` domain operators consumed directly by [onnxruntime](https://onnxruntime.ai/).
* **Broad framework coverage** — PyTorch, scikit-learn, TensorFlow/Keras, LiteRT, pandas/polars/SQL under one package.

**Cons / limitations**

* Younger project; operator coverage for some exotic models may lag behind the official per-framework exporters.
* The pluggable builder abstraction adds a thin indirection layer that advanced users may want to bypass.

This package was initially starting using [Vibe Coding](https://en.wikipedia.org/wiki/Vibe_coding).

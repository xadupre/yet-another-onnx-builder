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

**[Documentation](https://xadupre.github.io/docs/yet-another-onnx-builder/index.html)**

**yet-another-onnx-builder** (`yobx`) proposes a unique API and a unique function
``yobx.to_onnx`` to convert machine learning models and other pipelines
to [ONNX](https://onnx.ai) format from many libraries. Each converter relies on a common GraphBuilder API
to build the final ONNX model. The default implementation uses the native
`onnx-light` graph builder, pattern optimizer and shape inference engine.
It can also be replaced by an implementation of your own.
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
([`dataframe_to_onnx`](https://xadupre.github.io/docs/yet-another-onnx-builder/api/sql/dataframe_to_onnx.html))
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
import numpy as np
import onnxruntime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from yobx import to_onnx

# A custom numpy function traced to ONNX automatically
def log1p_abs(X):
    return np.log1p(np.abs(X))

pipe = Pipeline([
    ("func", FunctionTransformer(func=log1p_abs)),
    ("scaler", StandardScaler()),
])

X_train = np.random.default_rng(0).standard_normal((80, 4)).astype(np.float32)
pipe.fit(X_train)

# Export the whole pipeline to ONNX in one call
artifact = to_onnx(pipe, (X_train[:1],))

# Run with onnxruntime
sess = onnxruntime.InferenceSession(
    artifact.proto.SerializeToString(), providers=["CPUExecutionProvider"]
)
(result,) = sess.run(None, {"X": X_train})
```

## Native onnx-light backend

`onnx-light` is mandatory; the reference `onnx` package is not a dependency.
`yobx.xbuilder.GraphBuilder` uses onnx-light's native `GraphBuilder`,
pattern optimizer (`GraphGraph`), and symbolic shape inference by default.
The explicit `graph_backend="onnx-light"` selection remains available.
The scikit-learn and PyTorch converters use dedicated subclasses that keep
their converter-specific naming, dynamic-shape, local-function, and FX graph
contracts without adding those APIs to the shared native builder. For direct
graph construction, pass `builder_cls=yobx.builder.onnxlight.OnnxLightGraphBuilder`.

Use the **published wheels**, not an editable onnx-light checkout or a source
build. For example, from this repository on Linux x86-64 with CPython 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: \
  https://github.com/xadupre/onnx-light/releases/download/0.1.26/onnx_light-0.1.26-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl \
  numpy ml_dtypes scipy onnxruntime scikit-learn
export PYTHONPATH="$PWD"
```

For another Python version or platform, select the matching full wheel from
the [onnx-light release](https://github.com/xadupre/onnx-light/releases/tag/0.1.26).
The native backend requires Python 3.12 or newer. Restricting `PYTHONPATH`
to this repository prevents another onnx-light source checkout from shadowing
the installed wheel.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from yobx import to_onnx

X = np.random.default_rng(0).standard_normal((20, 4)).astype(np.float32)
model = LinearRegression().fit(X, X[:, 0] + X[:, 1])
artifact = to_onnx(model, (X,), graph_backend="onnx-light")
artifact.save("regression.onnx")
```

Direct graph construction can stay entirely in onnx-light, including the
returned model, with `to_native()`:

```python
from onnx_light import onnx
from yobx.builder.onnxlight import OnnxLightGraphBuilder, OnnxLightOptimizationOptions

builder = OnnxLightGraphBuilder(
    18,
    ir_version=8,
    optimization_options=OnnxLightOptimizationOptions(patterns=["TransposeTranspose"]),
)
builder.make_tensor_input("X", onnx.TensorProto.FLOAT, ("batch", 3))
transposed = builder.op.Transpose("X", perm=[1, 0])
output = builder.op.Transpose(transposed, perm=[1, 0])
builder.make_tensor_output(output)
native_model = builder.to_native()
```

`to_native()` does not import reference ONNX. Native shapes are available through
`builder.get_shape(name)` and `builder.shapes_context`. `patterns=None` selects
the wheel's standard patterns; `patterns=[]` disables pattern rewrites.
`to_onnx(return_optimize_report=True)` instead returns an `ExportArtifact` with
native timing and rewrite counts in `artifact.report.extra`.

`ExportArtifact` retains native protobufs through export, saving and loading;
it does not convert them to reference ONNX objects. Model shape inference and
symbolic-cost reports use `NativeShapeInference`, backed by `ShapesContext`.
There is no environment-variable switch back to reference ONNX.

Native options and legacy Python optimization options are not interchangeable;
Python pattern groups such as `"default+onnxruntime"` are rejected. The native
converter bridge does not support sequence metadata or inputs with undeclared
rank. The base wheel does not provide all
weight-folding runtime kernels. Extras that installed `onnxscript`, `onnx-ir`,
`spox` or reference shape inference have been removed because they would
reintroduce the reference ONNX dependency.
TensorFlow CI likewise does not install `tf2onnx` or `jax2onnx` for
cross-exporter comparisons, since both require reference ONNX. Its native
converter tests continue to compare results against TensorFlow and ONNX Runtime.

The Python pattern package `yobx.xoptim` and its matching/code-generation APIs
have been removed, not retained as an inactive fallback. Pattern implementations
and their registry come exclusively from the `onnx-light` wheel; YOBX only
selects native pattern names and forwards optimization requests to `GraphGraph`.

The historical `yobx.xbuilder.graph_builder.GraphBuilder` import also resolves
to the native implementation; the Python graph engine has been removed.
Scikit-learn uses `SklearnOnnxLightGraphBuilder`, while PyTorch uses
`TorchOnnxLightGraphBuilder`; both subclasses delegate graph storage,
shape inference, and optimization to onnx-light.
PyTorch export uses `torch.export` and native ONNX lowering, including dynamic
dimensions, linear layers, convolutions, reductions and `torch.cond`. Nested
arguments, input mutations, submodule/ATen function preservation, alternative
tracing frontends and unimplemented ATen operators raise explicit errors.
Standalone ONNX functions and custom conversion dispatchers are supported.
Delegation to `torch.onnx.export` through onnxscript and the Spox/onnxscript
builder bridges are no longer supported.

The project pins the published wheel to **0.1.26**. Local functions, including nested
calls and referenced attributes, are imported directly by the native builder.
Their result descriptors come from native inference, without a separate Python
replay of typed function bodies. Unused initializers are removed by native cleanup.

Native optimization now owns both dependency ordering and lifetime metadata;
the adapter no longer sorts serialized nodes or rebuilds the optimized graph
to repair those analyses. The old Python ordering optimizer has been removed,
and the historical
`graph_builder_opset.Opset` import now resolves to the native converter adapter.

Some legacy cases remain incompatible: the 0.1.26 builder rejects multi-output
`Scan` at opset 22, and subgraphs that redefine names visible in an ancestor
scope are rejected as SSA violations. These errors are surfaced rather than
hidden behind a Python fallback.

## Comparison with existing ONNX conversion tools

**Design choices `yobx`**

* **Single entry point** — `yobx.to_onnx` dispatches to the right backend automatically; no need to learn a different API for every framework.
* **Pluggable graph-builder** — the intermediate ONNX graph uses the native `onnx-light` builder through a common converter API.
* **Transparent names** — node names, initializer names and result names are preserved as-is (unless they are not unique); what the builder writes is what ends up in the ONNX file.
* **Built-in optimizer** — pattern-based graph rewrites (constant folding, fused ops, …) can be run before serialization.

**Comparison with existing tools**

The main new features is the possibility to trace functions written with NumPy, functions operating on DataFrames, and SQL queries.
User can now convert `FunctionTransformer` from scikit-learn or preprocessing through SQL queries or DataFrames.

The implementation was simplified to only handle recent versions of scikit-learn, TensorFlow/Keras, LiteRT. It was extended to other famous packages such `category_encoders`.

One single package for one single repository, one possible source of issues, making it easier for contributors to answer.

| Tool | Scope | Notes |
|------|-------|-------|
| [torch.onnx.export](https://pytorch.org/docs/stable/onnx.html) | PyTorch only | Official PyTorch exporter; `yobx` instead lowers `torch.export` graphs through the native `onnx-light` builder, optimizer and shape engine, without delegating to onnxscript. |
| [sklearn-onnx](https://onnx.ai/sklearn-onnx/) | scikit-learn only | Covers the scikit-learn ecosystem; `yobx` extends this with a unified API and adds support for custom functions written with NumPy via automatic tracing, `yobx` supports new packages such as `category_encoders`, ... |
| [tf2onnx](https://github.com/onnx/tensorflow-onnx) | TensorFlow / Keras | Converts TensorFlow models; `yobx` wraps the same models under one entry point |
| [ModelBuilder](https://onnxruntime.ai/docs/genai/howto/build-model.html) | LLM inference (genai) | ModelBuilder produces models better optimized for `onnxruntime`, `yobx` supports more models but is less efficient for this specific scenario. |

This package was initially starting using [Vibe Coding](https://en.wikipedia.org/wiki/Vibe_coding).

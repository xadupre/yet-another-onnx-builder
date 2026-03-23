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

**yet-another-onnx-builder** (`yobx`) proposes a unique API to convert machine learning models
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
# the model is called 
from yobx import to_onnx

expected = model(*args, **kwargs)
onnx_model = to_onnx(model, args, kwargs, dynamic_shapes, target_opset=22, **options)
```

[onnxruntime](https://onnxruntime.ai/) optimizations are triggered with
``target_opset={"": 22, "com.microsoft": 1}``.

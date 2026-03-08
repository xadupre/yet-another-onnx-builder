"""
.. _l-plot-sklearn-kmeans:

Converting a scikit-learn KMeans to ONNX
=========================================

:func:`yobx.sklearn.to_onnx` converts a fitted
:class:`sklearn.cluster.KMeans` into an
:class:`onnx.ModelProto` that can be executed with any ONNX-compatible
runtime.

The converted model produces two outputs:

* **label** - cluster index for each sample (equivalent to
  :meth:`~sklearn.cluster.KMeans.predict`).
* **distances** - Euclidean distance from each sample to every centroid
  (equivalent to :meth:`~sklearn.cluster.KMeans.transform`).

The workflow is:

1. **Train** a :class:`~sklearn.cluster.KMeans` as usual.
2. Call :func:`yobx.sklearn.to_onnx` with a representative dummy input.
3. **Run** the ONNX model with any ONNX runtime — this example uses
   :epkg:`onnxruntime`.
4. **Verify** that the ONNX outputs match scikit-learn's predictions.
"""

import numpy as np
import onnxruntime
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from yobx.sklearn import to_onnx

# %%
# 1. Train a KMeans model
# -----------------------

rng = np.random.default_rng(0)
X = rng.standard_normal((100, 4)).astype(np.float32)

km = KMeans(n_clusters=3, random_state=0, n_init=10)
km.fit(X)

# %%
# 2. Convert to ONNX
# ------------------

onx = to_onnx(km, (X,))
print(f"ONNX model inputs : {[i.name for i in onx.graph.input]}")
print(f"ONNX model outputs: {[o.name for o in onx.graph.output]}")

# %%
# 3. Run the ONNX model and compare outputs
# ------------------------------------------

ref = onnxruntime.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
label_onnx, distances_onnx = ref.run(None, {"X": X})

label_sk = km.predict(X).astype(np.int64)
distances_sk = km.transform(X).astype(np.float32)

print("\nFirst 5 labels (sklearn):", label_sk[:5])
print("First 5 labels (ONNX)   :", label_onnx[:5])

print("\nFirst 5 distances (sklearn):", distances_sk[:5].round(4))
print("First 5 distances (ONNX)   :", distances_onnx[:5].round(4))

assert (label_sk == label_onnx).all(), "Labels differ!"
assert np.allclose(distances_sk, distances_onnx, atol=1e-4), "Distances differ!"
print("\nAll labels and distances match ✓")

# %%
# 4. KMeans inside a Pipeline
# ----------------------------
#
# KMeans also works as the final step of a :class:`~sklearn.pipeline.Pipeline`.

pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("km", KMeans(n_clusters=3, random_state=0, n_init=10)),
    ]
)
pipe.fit(X)

onx_pipe = to_onnx(pipe, (X,))

ref_pipe = onnxruntime.InferenceSession(
    onx_pipe.SerializeToString(), providers=["CPUExecutionProvider"]
)
label_pipe_onnx, _ = ref_pipe.run(None, {"X": X})
label_pipe_sk = pipe.predict(X).astype(np.int64)

assert (label_pipe_sk == label_pipe_onnx).all(), "Pipeline labels differ!"
print("Pipeline labels match ✓")

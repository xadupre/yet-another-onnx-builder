"""
.. _l-plot-sklearn-affinity-propagation:

Converting a scikit-learn AffinityPropagation to ONNX
======================================================

:func:`yobx.sklearn.to_onnx` converts a fitted
:class:`sklearn.cluster.AffinityPropagation` into an
:class:`onnx.ModelProto` that can be executed with any ONNX-compatible
runtime.

The converted model produces two outputs:

* **label** - cluster index for each sample (equivalent to
  :meth:`~sklearn.cluster.AffinityPropagation.predict`).
* **distances** - Euclidean distance from each sample to every cluster centre.

The workflow is:

1. **Train** a :class:`~sklearn.cluster.AffinityPropagation` as usual.
2. Call :func:`yobx.sklearn.to_onnx` with a representative dummy input.
3. **Run** the ONNX model with any ONNX runtime — this example uses
   :epkg:`onnxruntime`.
4. **Verify** that the ONNX outputs match scikit-learn's predictions.
"""

import numpy as np
import onnxruntime
from sklearn.cluster import AffinityPropagation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from yobx.doc import plot_dot
from yobx.sklearn import to_onnx

# %%
# 1. Train an AffinityPropagation model
# --------------------------------------

rng = np.random.default_rng(0)
X = rng.standard_normal((60, 4)).astype(np.float32)

ap = AffinityPropagation(random_state=0)
ap.fit(X)
print(f"Number of clusters found: {len(ap.cluster_centers_indices_)}")

# %%
# 2. Convert to ONNX
# ------------------

onx = to_onnx(ap, (X,))
print(f"ONNX model inputs : {[i.name for i in onx.graph.input]}")
print(f"ONNX model outputs: {[o.name for o in onx.graph.output]}")

# %%
# 3. Run the ONNX model and compare outputs
# ------------------------------------------

ref = onnxruntime.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
label_onnx, distances_onnx = ref.run(None, {"X": X})

label_sk = ap.predict(X).astype(np.int64)

print("\nFirst 5 labels (sklearn):", label_sk[:5])
print("First 5 labels (ONNX)   :", label_onnx[:5])

assert (label_sk == label_onnx).all(), "Labels differ!"
print("\nAll labels match ✓")

# %%
# 4. AffinityPropagation inside a Pipeline
# -----------------------------------------
#
# AffinityPropagation also works as the final step of a
# :class:`~sklearn.pipeline.Pipeline`.

pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("ap", AffinityPropagation(random_state=0)),
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


# %%
# 5. Visualize the pipeline
# -------------------------

plot_dot(onx_pipe)

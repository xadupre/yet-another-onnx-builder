"""
.. _l-plot-sklearn-pipeline:

Converting a scikit-learn Pipeline to ONNX
==========================================

:func:`yobx.sklearn.to_onnx` converts fitted
:epkg:`scikit-learn` estimators and pipelines into
:class:`onnx.ModelProto` objects that can be executed with any
ONNX-compatible runtime.

The converter covers the following estimators (see
:mod:`yobx.sklearn` for the full registry):

* :class:`sklearn.preprocessing.StandardScaler`
* :class:`sklearn.linear_model.LogisticRegression` /
  :class:`~sklearn.linear_model.LogisticRegressionCV`
* :class:`sklearn.tree.DecisionTreeClassifier` /
  :class:`~sklearn.tree.DecisionTreeRegressor`
* :class:`sklearn.pipeline.Pipeline` — chains the above
  step-by-step

The workflow is:

1. **Train** a scikit-learn estimator (or pipeline) as usual.
2. Call :func:`yobx.sklearn.to_onnx` with a representative dummy
   input to convert the fitted model into an ONNX graph.
3. **Run** the ONNX model with any ONNX runtime — this example uses
   :epkg:`onnxruntime`.
4. **Verify** that the ONNX outputs match scikit-learn's predictions.
"""

import numpy as np
import onnxruntime
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from yobx.sklearn import to_onnx

# %%
# 1. Train a scikit-learn pipeline
# ---------------------------------
#
# We train a simple two-step pipeline:
# ``StandardScaler`` followed by ``LogisticRegression``.
# The dataset has eighty samples and four features with two classes.

rng = np.random.default_rng(0)
X_train = rng.standard_normal((80, 4)).astype(np.float32)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression()),
    ]
)
pipe.fit(X_train, y_train)

print("Pipeline steps:")
for name, step in pipe.steps:
    print(f"  {name}: {step}")

# %%
# 2. Convert to ONNX
# -------------------
#
# :func:`yobx.sklearn.to_onnx` requires a representative *dummy input*
# (a NumPy array) so it can infer the input dtype and shape.
# The first axis is automatically treated as the batch dimension.

onx = to_onnx(pipe, (X_train[:1],))

print(f"\nONNX model opset : {onx.opset_import[0].version}")
print(f"Number of nodes  : {len(onx.graph.node)}")
print("Node op-types    :", [n.op_type for n in onx.graph.node])
print(
    "Graph inputs     :",
    [(inp.name, inp.type.tensor_type.elem_type) for inp in onx.graph.input],
)
print(
    "Graph outputs    :",
    [out.name for out in onx.graph.output],
)

# %%
# 3. Run the ONNX model and compare outputs
# ------------------------------------------
#
# We run the converted model on a held-out test set and verify that
# the ONNX predictions match those produced by scikit-learn.

X_test = rng.standard_normal((20, 4)).astype(np.float32)
y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

ref = onnxruntime.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
label_onnx, proba_onnx = ref.run(None, {"X": X_test})

label_sk = pipe.predict(X_test)
proba_sk = pipe.predict_proba(X_test).astype(np.float32)

print("\nFirst 5 labels  (sklearn):", label_sk[:5])
print("First 5 labels  (ONNX)   :", label_onnx[:5])
print("First 5 probas  (sklearn):", proba_sk[:5])
print("First 5 probas  (ONNX)   :", proba_onnx[:5])

assert np.array_equal(label_sk, label_onnx), "Label mismatch!"
assert np.allclose(proba_sk, proba_onnx, atol=1e-5), "Probability mismatch!"
print("\nAll predictions match ✓")

# %%
# 4. Multiclass pipeline
# -----------------------
#
# The same API works transparently for multiclass problems.
# The ``LogisticRegression`` converter automatically switches from
# a sigmoid-based binary graph to a softmax-based multiclass graph.

X_mc = rng.standard_normal((120, 4)).astype(np.float32)
y_mc = rng.integers(0, 3, size=120)

pipe_mc = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500)),
    ]
)
pipe_mc.fit(X_mc, y_mc)

X_test_mc = rng.standard_normal((30, 4)).astype(np.float32)
onx_mc = to_onnx(pipe_mc, (X_test_mc[:1],))

ref_mc = onnxruntime.InferenceSession(onx_mc.SerializeToString(), providers=["CPUExecutionProvider"])
label_mc_onnx, proba_mc_onnx = ref_mc.run(None, {"X": X_test_mc})

label_mc_sk = pipe_mc.predict(X_test_mc)
proba_mc_sk = pipe_mc.predict_proba(X_test_mc).astype(np.float32)

assert np.array_equal(label_mc_sk, label_mc_onnx), "Multiclass label mismatch!"
assert np.allclose(proba_mc_sk, proba_mc_onnx, atol=1e-5), "Multiclass proba mismatch!"
print("Multiclass predictions match ✓")

# %%
# 5. Visualize the ONNX graph
# ----------------------------
#
# :func:`to_dot <yobx.helpers.dot_helper.to_dot>` converts the
# :class:`onnx.ModelProto` into a DOT string that can be rendered by
# Graphviz.  The graph shows every ONNX node produced by the converter,
# with dtype/shape annotations on each edge.

from yobx.helpers.dot_helper import to_dot  # noqa: E402

dot_src = to_dot(onx)
print(dot_src)

# %%
# Display the graph
# ------------------
#
# The DOT source produced above describes the following graph.
#
# .. gdot::
#     :script: DOT-SECTION
#
#     import numpy as np
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.pipeline import Pipeline
#     from sklearn.preprocessing import StandardScaler
#     from yobx.sklearn import to_onnx
#     from yobx.helpers.dot_helper import to_dot
#
#     rng = np.random.default_rng(0)
#     X_train = rng.standard_normal((80, 4)).astype(np.float32)
#     y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
#     pipe = Pipeline(
#         [("scaler", StandardScaler()), ("clf", LogisticRegression())]
#     )
#     pipe.fit(X_train, y_train)
#     onx = to_onnx(pipe, (X_train[:1],))
#     dot = to_dot(onx)
#     print("DOT-SECTION", dot)

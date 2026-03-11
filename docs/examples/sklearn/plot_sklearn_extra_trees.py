"""
.. _l-plot-sklearn-extra-trees:

Converting scikit-learn ExtraTrees models to ONNX
=================================================

:func:`yobx.sklearn.to_onnx` converts fitted
:class:`sklearn.ensemble.ExtraTreesClassifier` and
:class:`sklearn.ensemble.ExtraTreesRegressor` instances into
:class:`onnx.ModelProto` objects that can be executed with any
ONNX-compatible runtime.

Extra Trees (Extremely Randomized Trees) share the same internal tree
structure as Random Forests, so the converter re-uses the same
``TreeEnsembleClassifier`` / ``TreeEnsembleRegressor`` (legacy path,
``ai.onnx.ml`` opset ≤ 4) or the unified ``TreeEnsemble`` operator
(``ai.onnx.ml`` opset 5) that is used for
:class:`~sklearn.ensemble.RandomForestClassifier` /
:class:`~sklearn.ensemble.RandomForestRegressor`.

**Outputs**:

* **Classifier** — returns ``(label, probabilities)`` matching
  :meth:`~sklearn.ensemble.ExtraTreesClassifier.predict` and
  :meth:`~sklearn.ensemble.ExtraTreesClassifier.predict_proba`.
* **Regressor** — returns the averaged prediction matching
  :meth:`~sklearn.ensemble.ExtraTreesRegressor.predict`.

The workflow is:

1. **Train** an Extra Trees estimator as usual.
2. Call :func:`yobx.sklearn.to_onnx` with a representative dummy input.
3. **Run** the ONNX model with any ONNX runtime — this example uses
   :epkg:`onnxruntime`.
4. **Verify** that the ONNX outputs match scikit-learn's predictions.
"""

import numpy as np
import onnxruntime
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from yobx.doc import plot_dot
from yobx.sklearn import to_onnx

# %%
# 1. Binary classification
# ------------------------
#
# We train a binary :class:`~sklearn.ensemble.ExtraTreesClassifier` and
# verify that the ONNX model produces the same labels and class-probability
# estimates as scikit-learn.

rng = np.random.default_rng(0)
X = rng.standard_normal((120, 4)).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

clf_bin = ExtraTreesClassifier(n_estimators=10, random_state=0)
clf_bin.fit(X, y)

onx_bin = to_onnx(clf_bin, (X,))
print(f"ONNX inputs : {[i.name for i in onx_bin.graph.input]}")
print(f"ONNX outputs: {[o.name for o in onx_bin.graph.output]}")

ref_bin = onnxruntime.InferenceSession(
    onx_bin.SerializeToString(), providers=["CPUExecutionProvider"]
)
label_onnx, proba_onnx = ref_bin.run(None, {"X": X})

label_sk = clf_bin.predict(X).astype(np.int64)
proba_sk = clf_bin.predict_proba(X).astype(np.float32)

print("\nFirst 5 labels  (sklearn):", label_sk[:5])
print("First 5 labels  (ONNX)   :", label_onnx[:5])

assert np.array_equal(label_sk, label_onnx), "Binary labels differ!"
assert np.allclose(proba_sk, proba_onnx, atol=1e-5), "Binary probabilities differ!"
print("\nBinary labels and probabilities match ✓")

# %%
# 2. Multiclass classification
# ----------------------------
#
# Extra Trees works with any number of classes.  Here we use three classes.

X_mc = rng.standard_normal((150, 4)).astype(np.float32)
y_mc = np.digitize(X_mc[:, 0], bins=[-0.5, 0.5]).astype(np.int64)

clf_mc = ExtraTreesClassifier(n_estimators=10, random_state=0)
clf_mc.fit(X_mc, y_mc)

onx_mc = to_onnx(clf_mc, (X_mc,))

ref_mc = onnxruntime.InferenceSession(
    onx_mc.SerializeToString(), providers=["CPUExecutionProvider"]
)
label_mc_onnx, proba_mc_onnx = ref_mc.run(None, {"X": X_mc})

label_mc_sk = clf_mc.predict(X_mc).astype(np.int64)
proba_mc_sk = clf_mc.predict_proba(X_mc).astype(np.float32)

assert np.array_equal(label_mc_sk, label_mc_onnx), "Multiclass labels differ!"
assert np.allclose(proba_mc_sk, proba_mc_onnx, atol=1e-5), "Multiclass probabilities differ!"
print("Multiclass labels and probabilities match ✓")

# %%
# 3. Regression
# -------------
#
# :class:`~sklearn.ensemble.ExtraTreesRegressor` is converted in the same way.
# The output is the averaged prediction as a column vector ``[N, 1]``.

X_r = rng.standard_normal((100, 3)).astype(np.float32)
y_r = (X_r[:, 0] * 2 + X_r[:, 1]).astype(np.float32)

reg = ExtraTreesRegressor(n_estimators=10, random_state=0)
reg.fit(X_r, y_r)

onx_reg = to_onnx(reg, (X_r,))

ref_reg = onnxruntime.InferenceSession(
    onx_reg.SerializeToString(), providers=["CPUExecutionProvider"]
)
preds_onnx = ref_reg.run(None, {"X": X_r})[0]
preds_sk = reg.predict(X_r).astype(np.float32).reshape(-1, 1)

assert np.allclose(preds_sk, preds_onnx, atol=1e-5), "Regression predictions differ!"
print("Regressor predictions match sklearn ✓")

# %%
# 4. Inside a Pipeline
# --------------------
#
# Extra Trees estimators work transparently inside a scikit-learn
# :class:`~sklearn.pipeline.Pipeline`.

pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", ExtraTreesClassifier(n_estimators=10, random_state=0)),
    ]
)
pipe.fit(X, y)

onx_pipe = to_onnx(pipe, (X,))

ref_pipe = onnxruntime.InferenceSession(
    onx_pipe.SerializeToString(), providers=["CPUExecutionProvider"]
)
label_pipe_onnx, proba_pipe_onnx = ref_pipe.run(None, {"X": X})

label_pipe_sk = pipe.predict(X).astype(np.int64)
proba_pipe_sk = pipe.predict_proba(X).astype(np.float32)

assert np.array_equal(label_pipe_sk, label_pipe_onnx), "Pipeline labels differ!"
assert np.allclose(proba_pipe_sk, proba_pipe_onnx, atol=1e-5), "Pipeline probabilities differ!"
print("Pipeline labels and probabilities match ✓")

# %%
# 5. Using the ai.onnx.ml opset 5 TreeEnsemble operator
# -------------------------------------------------------
#
# Pass a dict target opset with ``"ai.onnx.ml": 5`` to emit the unified
# ``TreeEnsemble`` operator instead of the legacy
# ``TreeEnsembleClassifier`` / ``TreeEnsembleRegressor``.  The predictions
# are identical; this path is useful when targeting runtimes that support
# the newer opset.

onx_v5 = to_onnx(clf_bin, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

ml_opset_v5 = {op.domain: op.version for op in onx_v5.opset_import}
op_types_v5 = [n.op_type for n in onx_v5.graph.node]

print(f"\nai.onnx.ml opset: {ml_opset_v5['ai.onnx.ml']}")
print(f"Operators used  : {op_types_v5}")
assert "TreeEnsemble" in op_types_v5
assert "TreeEnsembleClassifier" not in op_types_v5

ref_v5 = onnxruntime.InferenceSession(
    onx_v5.SerializeToString(), providers=["CPUExecutionProvider"]
)
label_v5, proba_v5 = ref_v5.run(None, {"X": X})

assert np.array_equal(label_v5, label_onnx), "v5 labels differ from legacy!"
assert np.allclose(proba_v5, proba_onnx, atol=1e-5), "v5 probabilities differ from legacy!"
print("opset-5 TreeEnsemble predictions match legacy path ✓")

# %%
# 6. Visualize the ONNX graph
# ----------------------------

plot_dot(onx_bin)

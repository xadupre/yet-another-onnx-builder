"""
.. _l-plot-sklearn-pls-float32:

Float32 vs Float64: precision loss with PLSRegression
======================================================

A common source of surprise when deploying scikit-learn models with ONNX
is **numerical discrepancy between scikit-learn and the ONNX runtime output**.
The culprit is almost always a **dtype mismatch**: the model was trained with
``float64`` (the NumPy default), but the ONNX graph was exported with
``float32`` inputs and weights.

With ``scikit-learn>=1.8`` and the initiative to follow
`array API <https://data-apis.org/array-api/latest/>`_,
the computation type is more consistent and less discrepancies
are observed between the original `scikit-learn` model
and its converted ONNX version.

Nevertheless, this shows an example based on
:class:`sklearn.cross_decomposition.PLSRegression` to
illustrate the problem step by step and shows how to fix it.

Why almost correlated data?
-----------------------------

:class:`~sklearn.cross_decomposition.PLSRegression` is designed for datasets
where the **features are strongly correlated** (near-collinear). In such a
setting the model finds a small number of latent directions that capture the
covariance between ``X`` and ``y``.

We therefore generate data with a **low-rank latent-variable structure**:

* A few hidden factors ``Z`` drive both the features ``X`` and the target ``y``.
* Small independent noise is added to each column of ``X`` and to ``y``.

The resulting feature matrix is almost correlated — its columns span roughly
the same low-dimensional subspace — which is the canonical use case for PLS.
This also makes the float32 / float64 precision difference more pronounced:
the weight vectors of PLS point into directions of high variance, and
rounding them to float32 changes those directions non-trivially.

Why does float32 cause discrepancies?
--------------------------------------

Floating-point arithmetic is not exact.  ``float32`` has roughly **7 decimal
digits** of precision, while ``float64`` has **15–16**.  When a model trained
in ``float64`` is exported to ``float32``:

1. The **weight matrices** (``coef_``, ``_x_mean``, ``intercept_``) are cast
   from ``float64`` → ``float32``, losing precision.
2. The **intermediate computations** inside the ONNX graph also run in
   ``float32``.
3. scikit-learn itself always uses ``float64`` internally, regardless of the
   input dtype.

For numerically sensitive operations such as the matrix multiplication at the
core of PLS, even small rounding errors in the weights can accumulate and
produce predictions that differ by an order of magnitude larger than the
tolerance you would expect.

The fix: export with float64
-----------------------------

If your deployment environment supports ``float64`` tensors (all major ONNX
runtimes do), simply pass ``float64`` inputs to
:func:`yobx.sklearn.to_onnx`.  The converter will then keep all weights in
``float64`` and the exported predictions will match scikit-learn up to
floating-point round-off.
"""

import numpy as np
import onnxruntime
from sklearn.cross_decomposition import PLSRegression

from yobx.sklearn import to_onnx

# %%
# 1. Train a PLSRegression in float64 (the default)
# --------------------------------------------------
#
# We generate a dataset with **almost correlated features** using a
# low-rank latent-variable structure: a small number of hidden factors ``Z``
# drive both the feature matrix ``X`` and the target ``y``, with a tiny amount
# of independent noise added on top.  The resulting columns of ``X`` are highly
# correlated — exactly the setting PLS is designed for — and it makes the
# float32 precision loss more visible.

rng = np.random.default_rng(0)
n_samples, n_features, n_latent = 200, 50, 5

# Latent factors shared by X and y
Z = rng.standard_normal((n_samples, n_latent))

# Features: linear combinations of Z plus tiny noise
W = rng.standard_normal((n_latent, n_features))
X = Z @ W + 0.01 * rng.standard_normal((n_samples, n_features))

# Single target: also driven by the latent factors
c = rng.standard_normal(n_latent)
y = Z @ c + 0.01 * rng.standard_normal(n_samples)

# X and y are float64 here
print("X dtype :", X.dtype)  # float64
print("y dtype :", y.dtype)  # float64

pls = PLSRegression(n_components=5)
pls.fit(X, y)
print("coef_ dtype   :", pls.coef_.dtype)  # float64
print("_x_mean dtype :", pls._x_mean.dtype)  # float64

# %%
# 2. Export to ONNX with float32 — the wrong way
# -----------------------------------------------
#
# Passing a ``float32`` dummy input tells the converter to build an ONNX graph
# whose weights are all cast to ``float32``.  scikit-learn, however, will
# still use ``float64`` internally when ``predict`` is called.

X_f32 = X.astype(np.float32)
onx_f32 = to_onnx(pls, (X_f32[:1],))

sess_f32 = onnxruntime.InferenceSession(
    onx_f32.SerializeToString(), providers=["CPUExecutionProvider"]
)
pred_onnx_f32 = sess_f32.run(None, {"X": X_f32})[0]

# scikit-learn always predicts in float64
pred_sk = pls.predict(X).ravel()
pred_sk_f32_input = pls.predict(X_f32).ravel()  # sklearn upcasts to float64

print("\n--- float32 ONNX export ---")
print("sklearn  (float64 input) :", pred_sk[:5])
print("sklearn  (float32 input) :", pred_sk_f32_input[:5])
print("ONNX     (float32)       :", pred_onnx_f32[:5])

max_diff_f32 = float(np.abs(pred_onnx_f32 - pred_sk_f32_input).max())
print(f"Max absolute difference (float32 ONNX vs sklearn): {max_diff_f32:.6e}")

# %%
# 3. Export to ONNX with float64 — the correct way
# ------------------------------------------------
#
# Passing a ``float64`` dummy input keeps all weights in double precision.
# The ONNX graph will now produce predictions that agree with scikit-learn up
# to floating-point round-off (typically < 1e-10 for PLS).

onx_f64 = to_onnx(pls, (X[:1],))  # X is already float64

sess_f64 = onnxruntime.InferenceSession(
    onx_f64.SerializeToString(), providers=["CPUExecutionProvider"]
)
pred_onnx_f64 = sess_f64.run(None, {"X": X})[0]

print("\n--- float64 ONNX export ---")
print("sklearn (float64) :", pred_sk[:5])
print("ONNX    (float64) :", pred_onnx_f64[:5])

max_diff_f64 = float(np.abs(pred_onnx_f64 - pred_sk).max())
print(f"Max absolute difference (float64 ONNX vs sklearn): {max_diff_f64:.6e}")

# %%
# 4. Side-by-side comparison
# ---------------------------
#
# The table below summarises the maximum absolute difference across all 200
# test samples.  The ``float32`` export can introduce errors that are many
# orders of magnitude larger than the ``float64`` export.

print("\nSummary")
print("=" * 50)
print(f"  float32 ONNX max |error|: {max_diff_f32:.4e}")
print(f"  float64 ONNX max |error|: {max_diff_f64:.4e}")
_eps = 1e-300  # guard against division by zero when float64 error is exactly 0
print(f"  Ratio (f32/f64)         : {max_diff_f32 / (max_diff_f64 + _eps):.1f}x")

assert max_diff_f64 < 1e-7, f"float64 export should be near-exact, got {max_diff_f64}"
assert max_diff_f32 > max_diff_f64, "float32 export should introduce more error than float64"
print("\nConclusion: use float64 inputs when the model was trained with float64 ✓")

# %%
# 5. Multi-target PLSRegression with almost correlated data
# ----------------------------------------------------------
#
# The same precision loss applies to multi-target regression.
# We again use a latent-variable structure so that the features are
# almost correlated.

rng2 = np.random.default_rng(1)
n_samples2, n_features2, n_latent2, n_targets2 = 200, 50, 5, 3

Z2 = rng2.standard_normal((n_samples2, n_latent2))
W2 = rng2.standard_normal((n_latent2, n_features2))
X2 = Z2 @ W2 + 0.01 * rng2.standard_normal((n_samples2, n_features2))
C2 = rng2.standard_normal((n_latent2, n_targets2))
Y2 = Z2 @ C2 + 0.01 * rng2.standard_normal((n_samples2, n_targets2))

pls2 = PLSRegression(n_components=5)
pls2.fit(X2, Y2)

# float32 export
X2_f32 = X2.astype(np.float32)
onx2_f32 = to_onnx(pls2, (X2_f32[:1],))
sess2_f32 = onnxruntime.InferenceSession(
    onx2_f32.SerializeToString(), providers=["CPUExecutionProvider"]
)
pred2_onnx_f32 = sess2_f32.run(None, {"X": X2_f32})[0]
pred2_sk = pls2.predict(X2)
diff2_f32 = float(np.abs(pred2_onnx_f32 - pred2_sk).max())

# float64 export
onx2_f64 = to_onnx(pls2, (X2[:1],))
sess2_f64 = onnxruntime.InferenceSession(
    onx2_f64.SerializeToString(), providers=["CPUExecutionProvider"]
)
pred2_onnx_f64 = sess2_f64.run(None, {"X": X2})[0]
diff2_f64 = float(np.abs(pred2_onnx_f64 - pred2_sk).max())

print(f"\nMulti-target PLS — float32 max |error|: {diff2_f32:.4e}")
print(f"Multi-target PLS — float64 max |error|: {diff2_f64:.4e}")

assert diff2_f64 < 1e-7, f"float64 multi-target should be near-exact, got {diff2_f64}"
assert diff2_f32 > diff2_f64
print("Multi-target: same conclusion holds ✓")

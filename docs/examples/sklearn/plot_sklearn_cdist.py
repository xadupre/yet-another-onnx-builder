"""
.. _l-plot-sklearn-cdist:

Using the CDist operator in sklearn.to_onnx
============================================

Several :func:`yobx.sklearn.to_onnx` converters can exploit the
``com.microsoft.CDist`` custom operator that ships with
`ONNX Runtime <https://onnxruntime.ai>`_.

What is ``com.microsoft.CDist``?
---------------------------------

``com.microsoft.CDist`` computes the full pairwise distance matrix between
two matrices *A* (shape ``(N, F)``) and *B* (shape ``(M, F)``), returning
*D* (shape ``(N, M)``) where ``D[i, j] = dist(A[i], B[j])``.  It is
implemented as a fused C++ kernel inside ONNX Runtime and is therefore
faster than the equivalent sequence of standard ONNX operators for large
matrices.

Supported metrics: ``"euclidean"`` and ``"sqeuclidean"``.

Enabling CDist at conversion time
----------------------------------

Pass a ``dict`` as ``target_opset`` that includes ``"com.microsoft": 1``::

    onx = to_onnx(estimator, (X,), target_opset={"": 18, "com.microsoft": 1})

When ``"com.microsoft"`` is absent (or ``target_opset`` is a plain integer),
the converter automatically falls back to a portable standard-ONNX graph that
runs on **any** ONNX-compatible runtime.

Supported estimators
--------------------

The following estimators have a CDist-aware converter in this library:

* :class:`sklearn.neighbors.KNeighborsClassifier` /
  :class:`sklearn.neighbors.KNeighborsRegressor` —
  metrics ``"euclidean"`` and ``"sqeuclidean"``
  (see :ref:`l-plot-sklearn-knn` for a focused example).
* :class:`sklearn.cluster.Birch` — metric ``"euclidean"``.
* :class:`sklearn.cluster.BisectingKMeans` — metric ``"euclidean"``.
* :class:`sklearn.gaussian_process.GaussianProcessRegressor` —
  metric ``"sqeuclidean"`` (used internally for RBF / Matérn kernels).

When to prefer CDist
--------------------

Use the CDist path when **all** of the following hold:

1. You are deploying to **ONNX Runtime** (which ships with CDist).
2. The reference / training matrix is large (many rows *M*).  CDist avoids
   materializing the full ``(N, M)`` intermediate matrix in Python and
   executes the distance loop in a fused C++ kernel.
3. You can accept a model that will not run on runtimes that do not implement
   the ``com.microsoft`` custom domain.

Use the standard ONNX path for maximum portability.
"""

import numpy as np
import onnxruntime
from sklearn.cluster import BisectingKMeans, Birch
from sklearn.neighbors import KNeighborsClassifier
from yobx.sklearn import to_onnx

# %%
# Helper: inspect which op types are present in an ONNX model
# -----------------------------------------------------------


def _op_types(model):
    return {(n.op_type, n.domain or "") for n in model.graph.node}


# %%
# 1. BisectingKMeans
# ------------------
#
# :class:`~sklearn.cluster.BisectingKMeans` produces two outputs: cluster
# labels (``predict``) and Euclidean distances to each cluster centre
# (``transform``).  The distance computation is where CDist is used.

rng = np.random.default_rng(0)
X_bkm = rng.standard_normal((120, 4)).astype(np.float32)

bkm = BisectingKMeans(n_clusters=4, random_state=0, n_init=3)
bkm.fit(X_bkm)

# %%
# Standard ONNX path (default)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

onx_bkm_std = to_onnx(bkm, (X_bkm,))
print("BisectingKMeans standard nodes:", sorted({t for t, _ in _op_types(onx_bkm_std)}))
assert ("CDist", "com.microsoft") not in _op_types(onx_bkm_std)

ref_bkm_std = onnxruntime.InferenceSession(
    onx_bkm_std.SerializeToString(), providers=["CPUExecutionProvider"]
)
labels_bkm_std, dists_bkm_std = ref_bkm_std.run(None, {"X": X_bkm})
assert (labels_bkm_std == bkm.predict(X_bkm).astype(np.int64)).all()
print("BisectingKMeans standard path labels match sklearn ✓")

# %%
# CDist path (com.microsoft)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

onx_bkm_cd = to_onnx(bkm, (X_bkm,), target_opset={"": 18, "com.microsoft": 1})
print("BisectingKMeans CDist nodes:", sorted({t for t, _ in _op_types(onx_bkm_cd)}))
assert ("CDist", "com.microsoft") in _op_types(onx_bkm_cd)

ref_bkm_cd = onnxruntime.InferenceSession(
    onx_bkm_cd.SerializeToString(), providers=["CPUExecutionProvider"]
)
labels_bkm_cd, dists_bkm_cd = ref_bkm_cd.run(None, {"X": X_bkm})
assert (labels_bkm_cd == bkm.predict(X_bkm).astype(np.int64)).all()
print("BisectingKMeans CDist path labels match sklearn ✓")

# %%
# Both paths agree
# ~~~~~~~~~~~~~~~~

assert (labels_bkm_std == labels_bkm_cd).all()
assert np.allclose(dists_bkm_std, dists_bkm_cd, atol=1e-5)
print("BisectingKMeans: both paths agree ✓")

# %%
# 2. Birch
# --------
#
# :class:`~sklearn.cluster.Birch` assigns labels via nearest subcluster
# centre.  It produces two outputs: cluster labels and distances to each
# subcluster centre.  CDist is used for the distance computation.

X_birch = rng.standard_normal((100, 4)).astype(np.float32)

birch = Birch(n_clusters=3, threshold=0.5)
birch.fit(X_birch)

# %%
# Standard ONNX path (default)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

onx_birch_std = to_onnx(birch, (X_birch,))
print("Birch standard nodes:", sorted({t for t, _ in _op_types(onx_birch_std)}))
assert ("CDist", "com.microsoft") not in _op_types(onx_birch_std)

ref_birch_std = onnxruntime.InferenceSession(
    onx_birch_std.SerializeToString(), providers=["CPUExecutionProvider"]
)
labels_birch_std, _ = ref_birch_std.run(None, {"X": X_birch})
assert (labels_birch_std == birch.predict(X_birch).astype(np.int64)).all()
print("Birch standard path labels match sklearn ✓")

# %%
# CDist path (com.microsoft)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

onx_birch_cd = to_onnx(birch, (X_birch,), target_opset={"": 18, "com.microsoft": 1})
print("Birch CDist nodes:", sorted({t for t, _ in _op_types(onx_birch_cd)}))
assert ("CDist", "com.microsoft") in _op_types(onx_birch_cd)

ref_birch_cd = onnxruntime.InferenceSession(
    onx_birch_cd.SerializeToString(), providers=["CPUExecutionProvider"]
)
labels_birch_cd, dists_birch_cd = ref_birch_cd.run(None, {"X": X_birch})
assert (labels_birch_cd == birch.predict(X_birch).astype(np.int64)).all()
print("Birch CDist path labels match sklearn ✓")

# %%
# Both paths agree
# ~~~~~~~~~~~~~~~~

assert (labels_birch_std == labels_birch_cd).all()
print("Birch: both paths agree ✓")

# %%
# 3. KNeighborsClassifier
# -----------------------
#
# :class:`~sklearn.neighbors.KNeighborsClassifier` uses CDist for the
# ``"euclidean"`` and ``"sqeuclidean"`` metrics.  A detailed walk-through of
# the KNeighbors converter (including the regressor and pipeline usage) is
# available at :ref:`l-plot-sklearn-knn`.

X_knn = rng.standard_normal((80, 4)).astype(np.float32)
y_knn = (X_knn[:, 0] > 0).astype(np.int64)

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_knn, y_knn)

onx_knn_std = to_onnx(clf, (X_knn,))
onx_knn_cd = to_onnx(clf, (X_knn,), target_opset={"": 18, "com.microsoft": 1})

assert ("CDist", "com.microsoft") not in _op_types(onx_knn_std)
assert ("CDist", "com.microsoft") in _op_types(onx_knn_cd)

ref_knn_std = onnxruntime.InferenceSession(
    onx_knn_std.SerializeToString(), providers=["CPUExecutionProvider"]
)
ref_knn_cd = onnxruntime.InferenceSession(
    onx_knn_cd.SerializeToString(), providers=["CPUExecutionProvider"]
)
labels_knn_std = ref_knn_std.run(None, {"X": X_knn})[0]
labels_knn_cd = ref_knn_cd.run(None, {"X": X_knn})[0]

assert (labels_knn_std == clf.predict(X_knn).astype(np.int64)).all()
assert (labels_knn_cd == clf.predict(X_knn).astype(np.int64)).all()
assert (labels_knn_std == labels_knn_cd).all()
print("KNeighborsClassifier: both paths agree ✓")

.. _l-design-sklearn-cdist:

==============================================
Using the ``com.microsoft.CDist`` Operator
==============================================

Several converters in :func:`yobx.sklearn.to_onnx` can delegate pairwise
distance computation to the ``com.microsoft.CDist`` custom operator that
ships with `ONNX Runtime <https://onnxruntime.ai>`_.

What is ``com.microsoft.CDist``?
=================================

``com.microsoft.CDist`` computes the full pairwise distance matrix between
two matrices *A* (shape ``(N, F)``) and *B* (shape ``(M, F)``), producing
*D* of shape ``(N, M)`` where ``D[i, j] = dist(A[i], B[j])``.

It is implemented as a fused C++ kernel inside ONNX Runtime and avoids
the intermediate ``(N, M)`` materialization that a naive standard-ONNX
graph would require.  Supported metrics are ``"euclidean"`` and
``"sqeuclidean"``.

The operator lives in the ``com.microsoft`` custom domain and is **not**
part of the standard ONNX specification, so it only runs on runtimes that
ship with it — primarily ONNX Runtime.

Enabling CDist at conversion time
===================================

Pass a ``dict`` as ``target_opset`` that includes ``"com.microsoft": 1``
alongside the standard opset::

    onx = to_onnx(estimator, (X,), target_opset={"": 18, "com.microsoft": 1})

When ``"com.microsoft"`` is absent — or ``target_opset`` is a plain
integer — every CDist-aware converter automatically falls back to a
portable standard-ONNX graph that runs on any ONNX-compatible runtime.

The check inside a converter looks like this:

.. code-block:: python

    if g.has_opset("com.microsoft"):
        # CDist path
        training_data_name = g.make_initializer(f"{name}_training_data", training_data)
        dists = g.make_node(
            "CDist",
            [X, training_data_name],
            domain="com.microsoft",
            metric="euclidean",
            name=f"{name}_cdist",
        )
    else:
        # Standard ONNX fallback
        ...

When to prefer CDist
=====================

Use the CDist path when **all** of the following hold:

1. You are deploying to **ONNX Runtime** (which ships with CDist).
2. The reference / training matrix is large (many rows *M*).  CDist
   executes the distance loop in a fused C++ kernel.
3. You can accept a model that will not run on runtimes that do not
   implement the ``com.microsoft`` custom domain.

Use the standard ONNX path (default) for maximum portability.

Supported estimators
=====================

The following estimators have a CDist-aware converter:

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Estimator
     - Metric
     - Where CDist is used
   * - :class:`~sklearn.neighbors.KNeighborsClassifier` /
       :class:`~sklearn.neighbors.KNeighborsRegressor`
     - ``"euclidean"``, ``"sqeuclidean"``
     - Pairwise distances from query points to all training points
   * - :class:`sklearn.cluster.Birch`
     - ``"euclidean"``
     - Distances from each sample to all subcluster centres
   * - :class:`sklearn.cluster.BisectingKMeans`
     - ``"euclidean"``
     - Distances from each sample to all cluster centres (transform output)
   * - :class:`sklearn.gaussian_process.GaussianProcessRegressor` /
       :class:`~sklearn.gaussian_process.GaussianProcessClassifier`
     - ``"sqeuclidean"``
     - Pairwise squared distances inside RBF / Matérn kernel evaluation

Examples
=========

KNeighborsClassifier
---------------------

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from yobx.sklearn import to_onnx
    from yobx.helpers.onnx_helper import pretty_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)

    clf = KNeighborsClassifier(n_neighbors=3).fit(X, y)

    # Standard ONNX path (portable)
    onx_std = to_onnx(clf, (X,))
    std_nodes = {n.op_type for n in onx_std.graph.node}
    print("Standard path op types:", sorted(std_nodes))

    # CDist path (ONNX Runtime)
    onx_cd = to_onnx(clf, (X,), target_opset={"": 18, "com.microsoft": 1})
    cd_nodes = {(n.op_type, n.domain or "") for n in onx_cd.graph.node}
    print("CDist path op types:", sorted({t for t, _ in cd_nodes}))
    print("CDist present:", ("CDist", "com.microsoft") in cd_nodes)

Birch
------

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.cluster import Birch
    from yobx.sklearn import to_onnx
    from yobx.helpers.onnx_helper import pretty_onnx

    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 3)).astype(np.float32)

    birch = Birch(n_clusters=3, threshold=0.5).fit(X)

    # Standard ONNX path
    onx_std = to_onnx(birch, (X,))
    print("Standard path:", sorted({n.op_type for n in onx_std.graph.node}))

    # CDist path
    onx_cd = to_onnx(birch, (X,), target_opset={"": 18, "com.microsoft": 1})
    print("CDist path:", sorted({n.op_type for n in onx_cd.graph.node}))

BisectingKMeans
----------------

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.cluster import BisectingKMeans
    from yobx.sklearn import to_onnx
    from yobx.helpers.onnx_helper import pretty_onnx

    rng = np.random.default_rng(2)
    X = rng.standard_normal((80, 4)).astype(np.float32)

    bkm = BisectingKMeans(n_clusters=3, random_state=0, n_init=3).fit(X)

    # Standard ONNX path
    onx_std = to_onnx(bkm, (X,))
    print("Standard path:", sorted({n.op_type for n in onx_std.graph.node}))

    # CDist path
    onx_cd = to_onnx(bkm, (X,), target_opset={"": 18, "com.microsoft": 1})
    print("CDist path:", sorted({n.op_type for n in onx_cd.graph.node}))

Both paths produce identical results
--------------------------------------

.. runpython::
    :showcode:

    import numpy as np
    import onnxruntime
    from sklearn.cluster import Birch
    from yobx.sklearn import to_onnx

    rng = np.random.default_rng(3)
    X = rng.standard_normal((50, 3)).astype(np.float32)

    birch = Birch(n_clusters=3, threshold=0.5).fit(X)

    onx_std = to_onnx(birch, (X,))
    onx_cd = to_onnx(birch, (X,), target_opset={"": 18, "com.microsoft": 1})

    ref_std = onnxruntime.InferenceSession(
        onx_std.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    ref_cd = onnxruntime.InferenceSession(
        onx_cd.SerializeToString(), providers=["CPUExecutionProvider"]
    )

    labels_std = ref_std.run(None, {"X": X})[0]
    labels_cd = ref_cd.run(None, {"X": X})[0]
    dists_std = ref_std.run(None, {"X": X})[1]
    dists_cd = ref_cd.run(None, {"X": X})[1]

    print("Labels match:", (labels_std == labels_cd).all())
    print("Distances max diff:", float(np.abs(dists_std - dists_cd).max()))

.. seealso::

    :ref:`l-design-sklearn-converter` — overview of the converter registry
    and the GraphBuilder API.

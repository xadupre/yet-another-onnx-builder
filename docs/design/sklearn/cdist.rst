.. _l-design-sklearn-cdist:

======================================================
ONNX Runtime Contrib Ops (``com.microsoft`` domain)
======================================================

ONNX Runtime ships a set of *contributed operators* in the
``com.microsoft`` custom domain.  These operators go beyond the standard
ONNX specification and provide fused or hardware-accelerated kernels.
They are **not** part of the standard ONNX spec, so they are only
available on runtimes that implement them — primarily ONNX Runtime.

The ``com.microsoft`` domain
=============================

The ``com.microsoft`` domain groups ONNX Runtime's proprietary extensions.
Any model that references nodes in this domain will fail on runtimes that
do not recognise it.  ONNX Runtime is the reference implementation.

Enabling contrib ops
=====================

Pass a ``dict`` as ``target_opset`` that includes ``"com.microsoft": 1``
alongside the standard opset::

    onx = to_onnx(estimator, (X,), target_opset={"": 18, "com.microsoft": 1})

When ``"com.microsoft"`` is absent — or ``target_opset`` is a plain
integer — every contrib-aware converter automatically falls back to a
portable standard-ONNX graph that runs on any ONNX-compatible runtime.

The guard pattern inside any contrib-aware converter looks like this:

.. code-block:: python

    if g.has_opset("com.microsoft"):
        # emit a com.microsoft node
        result = g.make_node(
            "SomeContribOp",
            [input_name],
            domain="com.microsoft",
            name=f"{name}_contrib",
        )
    else:
        # standard ONNX fallback
        ...

Contrib ops available in this package
=======================================

The table below lists the ``com.microsoft`` operators that are referenced
across ``yobx``.

Sklearn converters
------------------

The following estimator converters emit ``com.microsoft.CDist`` when the
domain is registered.  ``CDist`` computes the full pairwise distance matrix
between two matrices *A* (shape ``(N, F)``) and *B* (shape ``(M, F)``),
producing *D* of shape ``(N, M)`` where ``D[i, j] = dist(A[i], B[j])``
as a fused C++ kernel, avoiding intermediate materialisation.

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
   * - :class:`~sklearn.cluster.Birch`
     - ``"euclidean"``
     - Distances from each sample to all subcluster centres
   * - :class:`~sklearn.cluster.BisectingKMeans`
     - ``"euclidean"``
     - Distances from each sample to all cluster centres (transform output)
   * - :class:`~sklearn.gaussian_process.GaussianProcessRegressor` /
       :class:`~sklearn.gaussian_process.GaussianProcessClassifier`
     - ``"sqeuclidean"``
     - Pairwise squared distances inside RBF / Matérn kernel evaluation

Graph optimizer patterns
-------------------------

The optimizer patterns in :mod:`yobx.xoptim.patterns_ort` fuse sequences
of standard ONNX nodes into single ``com.microsoft`` kernels:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Contrib op
     - Replaces / fuses
   * - ``FusedMatMul``
     - ``MatMul`` + optional transpose or scale attributes
   * - ``FusedConv``
     - ``Conv`` followed by ``Relu`` (or other pointwise activations)
   * - ``BiasGelu``
     - ``Add(bias) + Gelu``
   * - ``Gelu``
     - ``Erf``-based GELU approximation
   * - ``FastGelu``
     - ``Tanh``-based GELU approximation
   * - ``QuickGelu``
     - ``Sigmoid``-based GELU approximation (``x * σ(αx)``)
   * - ``SkipSimplifiedLayerNormalization``
     - ``Add(residual) + SimplifiedLayerNorm``
   * - ``RotaryEmbedding``
     - Rotary positional embedding sequence

Reference evaluator
--------------------

:class:`yobx.reference.ExtendedReferenceEvaluator` includes pure-Python
implementations of several ``com.microsoft`` ops so that models using
them can be evaluated without ONNX Runtime during development and testing:

``FusedMatMul``, ``Attention``, ``BiasSoftmax``, ``QLinearAveragePool``,
``QLinearConv``, ``QuickGelu``, ``SkipLayerNormalization``.

When to use contrib ops
========================

Use contrib ops when **all** of the following hold:

1. You are deploying to **ONNX Runtime** (the only common runtime that
   ships with the ``com.microsoft`` domain).
2. The fused kernel provides a meaningful speedup for your workload
   (e.g. large distance matrices for ``CDist``, or repeated MatMul+bias
   patterns for ``FusedMatMul``).
3. You can accept a model that will not run on runtimes that do not
   implement the ``com.microsoft`` custom domain.

Use the standard ONNX path (default) for maximum portability.

CDist example
==============

The following examples show the difference between the standard ONNX
path and the CDist-enabled path for sklearn estimators.

KNeighborsClassifier
---------------------

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from yobx.sklearn import to_onnx

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

    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 3)).astype(np.float32)

    birch = Birch(n_clusters=3, threshold=0.5).fit(X)

    # Standard ONNX path
    onx_std = to_onnx(birch, (X,))
    print("Standard path:", sorted({n.op_type for n in onx_std.graph.node}))

    # CDist path
    onx_cd = to_onnx(birch, (X,), target_opset={"": 18, "com.microsoft": 1})
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

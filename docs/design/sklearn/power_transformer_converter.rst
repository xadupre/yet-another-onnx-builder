.. _l-design-power-transformer-converter:

============================
PowerTransformer Converter
============================

:func:`yobx.sklearn.to_onnx` converts a fitted
:class:`sklearn.preprocessing.PowerTransformer` to ONNX using the standard
registry-based architecture of :mod:`yobx.sklearn`.

The converter is implemented in
:mod:`yobx.sklearn.preprocessing.power_transformer` and supports both
available methods:

* **``'yeo-johnson'``** (default) — works with any real-valued data.
* **``'box-cox'``** — requires strictly positive input data.

The optional post-transformation standardization (``standardize=True``,
default) is inlined as ``Sub`` / ``Div`` nodes using the fitted
:class:`~sklearn.preprocessing.StandardScaler` stored in ``estimator._scaler``.

Yeo-Johnson Transform
=====================

The per-feature formula uses the fitted ``lambdas_`` array.  Four sub-cases
are handled via ``Where`` nodes:

.. code-block:: text

    y >= 0, lam ≠ 0 :  ((y + 1)^lam  - 1) / lam
    y >= 0, lam = 0 :  log(y + 1)
    y < 0,  lam ≠ 2 :  -((-y + 1)^(2-lam) - 1) / (2-lam)
    y < 0,  lam = 2 :  -log(-y + 1)

Division-by-zero for the degenerate cases (``lam = 0`` or ``lam = 2``) is
avoided by replacing those values with 1 in the divisor array and selecting
the log branch with a ``Where`` node.

The resulting graph looks like:

.. code-block:: text

                ┌── Add(1) ── Pow(lam_pos) ── Sub(1) ── Div(lam_pos) ──┐
    X ──┬──────►│                                                        Where ──►
        │       └── Log(x+1) ─────────────────────────────────────────►┘  │
        │                                                                    ↓ (pos)
        │       ┌── Neg ── Add(1) ── Pow(2-lam) ── Sub(1) ── Div(2-lam) ──┐
        └──────►│                                                            Where ──►
                └── Neg ── Log(-x+1) ───────────────────────────────────►┘  │
                                                                              ↓ (neg)
                                               GreaterOrEqual(0) ── Where(pos, neg) ──► result

Box-Cox Transform
=================

.. code-block:: text

    lam ≠ 0 :  (x^lam - 1) / lam
    lam = 0 :  log(x)

The input is assumed to be strictly positive (sklearn enforces this during
:meth:`~sklearn.preprocessing.PowerTransformer.fit`).

Post-transformation Standardization
=====================================

When ``standardize=True`` (the default) the raw transformed tensor is
centred and scaled using the fitted ``_scaler``:

.. code-block:: text

    transformed ──Sub(mean_) ──Div(scale_)──► output

When ``standardize=False`` an ``Identity`` node forwards the raw transformed
tensor to the output.

Example — Yeo-Johnson
=====================

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.preprocessing import PowerTransformer
    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.sklearn import to_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3)).astype(np.float32)

    pt = PowerTransformer()
    pt.fit(X)
    onx = to_onnx(pt, (X,))
    print(pretty_onnx(onx))

Example — Box-Cox
=================

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.preprocessing import PowerTransformer
    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.sklearn import to_onnx

    rng = np.random.default_rng(0)
    X = np.abs(rng.standard_normal((20, 3)).astype(np.float32)) + 0.1

    pt = PowerTransformer(method="box-cox")
    pt.fit(X)
    onx = to_onnx(pt, (X,))
    print(pretty_onnx(onx))

Pipeline Embedding
==================

:class:`~sklearn.preprocessing.PowerTransformer` integrates seamlessly inside
a :class:`sklearn.pipeline.Pipeline`:

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PowerTransformer
    from sklearn.linear_model import LogisticRegression
    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.sklearn import to_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)

    pipe = Pipeline([
        ("pt", PowerTransformer()),
        ("clf", LogisticRegression()),
    ]).fit(X, y)

    onx = to_onnx(pipe, (X,))
    print(pretty_onnx(onx))

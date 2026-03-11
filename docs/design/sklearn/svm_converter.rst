.. _l-design-svm-converters:

==================
SVM Converters
==================

:func:`yobx.sklearn.to_onnx` converts fitted
:class:`sklearn.svm.LinearSVC`, :class:`sklearn.svm.LinearSVR`,
:class:`sklearn.svm.SVC`, :class:`sklearn.svm.NuSVC`,
:class:`sklearn.svm.SVR`, and :class:`sklearn.svm.NuSVR`
estimators to ONNX using the same registry-based architecture as the other
:mod:`yobx.sklearn` converters.

Two families of converters are provided:

* **Linear SVMs** — :class:`~sklearn.svm.LinearSVC` and
  :class:`~sklearn.svm.LinearSVR` use a plain ``Gemm`` node (same pattern as
  :class:`~sklearn.linear_model.LinearRegression` /
  :class:`~sklearn.linear_model.RidgeClassifier`).

* **Kernel SVMs** — :class:`~sklearn.svm.SVC`, :class:`~sklearn.svm.NuSVC`,
  :class:`~sklearn.svm.SVR`, and :class:`~sklearn.svm.NuSVR` use the
  ``SVMClassifier`` / ``SVMRegressor`` operators from the ``ai.onnx.ml``
  domain, which implement the kernel function natively.

Implementations:

* :mod:`yobx.sklearn.svm.linear_svm` — LinearSVC and LinearSVR converters
* :mod:`yobx.sklearn.svm.svm` — SVC, NuSVC, SVR, and NuSVR converters

Linear SVMs
===========

LinearSVC
---------

:func:`~yobx.sklearn.svm.linear_svm.sklearn_linear_svc` converts
:class:`sklearn.svm.LinearSVC`.

:class:`~sklearn.svm.LinearSVC` does not support :meth:`predict_proba`, so
the converter always returns the predicted class label only (one output).

**Binary classification** (``len(classes_) == 2``):

.. code-block:: text

    X  ──Gemm(coef, intercept)──►  decision  [N, 1]
                                        │
                                   Reshape ──►  decision_1d  [N,]
                                        │
                                   Greater(0) ──Cast(INT64)──Gather(classes) ──►  label

**Multiclass** (``len(classes_) > 2``):

.. code-block:: text

    X  ──Gemm(coef, intercept)──►  decision  [N, C]
                                        │
                                   ArgMax ──Cast(INT64)──Gather(classes) ──►  label

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.svm import LinearSVC
    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.sklearn import to_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    clf = LinearSVC(random_state=0, max_iter=5000).fit(X, y)
    onx = to_onnx(clf, (X,))
    print(pretty_onnx(onx))

LinearSVR
---------

:func:`~yobx.sklearn.svm.linear_svm.sklearn_linear_svr` converts
:class:`sklearn.svm.LinearSVR`.

The prediction formula is ``y = X @ coef_.T + intercept_``, identical to
other linear regressors.

.. code-block:: text

    X  ──Gemm(coef, intercept, transB=1)──►  predictions

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.svm import LinearSVR
    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.sklearn import to_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4)).astype(np.float32)
    y = X[:, 0] + 2 * X[:, 1]

    reg = LinearSVR(random_state=0, max_iter=5000).fit(X, y)
    onx = to_onnx(reg, (X,))
    print(pretty_onnx(onx))

Kernel SVMs
===========

The kernel SVM converters use the ONNX ``SVMClassifier`` and
``SVMRegressor`` operators (``ai.onnx.ml`` domain, opset 1).  These operators
evaluate the kernel function directly, so no explicit kernel matrix
computation is needed in the graph.

Supported kernels
-----------------

All four string-named kernels are supported:

==============  ===============================================================
``kernel``      ONNX ``kernel_type``
==============  ===============================================================
``'linear'``    ``LINEAR``
``'poly'``      ``POLY``
``'rbf'``       ``RBF``
``'sigmoid'``   ``SIGMOID``
==============  ===============================================================

The ONNX ``kernel_params`` attribute is set to ``[gamma, coef0, degree]``
where ``gamma`` is read from the fitted ``_gamma`` attribute (i.e. the
resolved scalar value after ``'scale'`` / ``'auto'`` expansion).

Callable kernels (passing a Python function as ``kernel``) raise
:class:`NotImplementedError`.

SVC and NuSVC
-------------

:func:`~yobx.sklearn.svm.svm.sklearn_svc` converts both
:class:`sklearn.svm.SVC` and :class:`sklearn.svm.NuSVC`.

Both classes store the same fitted attributes and are converted identically.

**Coefficient layout**

The ONNX ``SVMClassifier`` stores the dual coefficients grouped by class:

* **Binary** (2 classes): ``coefficients = -dual_coef_.flatten()``,
  ``rho = -intercept_``.
  The negation is required because ONNX treats the first class as positive.
* **Multiclass** (≥ 3 classes, OvO): ``coefficients = dual_coef_.flatten()``,
  ``rho = intercept_``.

The ``vectors_per_class`` attribute is set to ``n_support_`` (number of
support vectors per class) and ``support_vectors`` is the flattened
``support_vectors_`` matrix.

**Probability output**

When ``probability=True``, the Platt scaling parameters (``probA_``,
``probB_``) are embedded in the node as ``prob_a`` / ``prob_b``.  The ONNX
runtime automatically applies the calibration when these attributes are
present, so no extra graph nodes are needed.

.. code-block:: text

    X ──SVMClassifier──► label          (INT64 or STRING)
                     └──► probabilities (FLOAT [N, n_classes], only when probability=True)

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.svm import SVC
    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.sklearn import to_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    clf = SVC(kernel="rbf", probability=True, random_state=0).fit(X, y)
    onx = to_onnx(clf, (X,))
    print(pretty_onnx(onx))

SVR and NuSVR
-------------

:func:`~yobx.sklearn.svm.svm.sklearn_svr` converts both
:class:`sklearn.svm.SVR` and :class:`sklearn.svm.NuSVR`.

The dual coefficients and intercept are mapped directly:
``coefficients = dual_coef_.flatten()``, ``rho = intercept_``.

.. code-block:: text

    X ──SVMRegressor──► predictions  (FLOAT [N, 1])

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.svm import SVR
    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.sklearn import to_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4)).astype(np.float32)
    y = X[:, 0] + 2 * X[:, 1]

    reg = SVR(kernel="rbf").fit(X, y)
    onx = to_onnx(reg, (X,))
    print(pretty_onnx(onx))

Pipeline embedding
==================

All six SVM estimators can be embedded in a :class:`sklearn.pipeline.Pipeline`:

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.sklearn import to_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=0)),
    ]).fit(X, y)

    onx = to_onnx(pipe, (X,))
    print(pretty_onnx(onx))

.. _l-design-export-artifact:

==========================
to_onnx and ExportArtifact
==========================

Overview
========

Every ``to_onnx`` function in *yet-another-onnx-builder* — whether it
converts a scikit-learn estimator, a TensorFlow graph, a LiteRT model,
a PyTorch interpreter trace, or a SQL query — returns a single, uniform
type: :class:`~yobx.container.ExportArtifact`.

This provides a **common interface** regardless of the backend used:
callers always receive the same structured object and never need to check
whether the result is a bare :class:`onnx.ModelProto` or an
:class:`~yobx.container.ExtendedModelContainer`.

.. code-block:: python

    from yobx.sklearn import to_onnx as sklearn_to_onnx
    from yobx.tensorflow import to_onnx as tf_to_onnx
    from yobx.sql import sql_to_onnx
    from yobx.container import ExportArtifact

    # All three return an ExportArtifact:
    artifact_sk  = sklearn_to_onnx(estimator, (X,))
    artifact_tf  = tf_to_onnx(tf_model, (X,))
    artifact_sql = sql_to_onnx("SELECT a + b AS c FROM t", dtypes)

    assert isinstance(artifact_sk,  ExportArtifact)
    assert isinstance(artifact_tf,  ExportArtifact)
    assert isinstance(artifact_sql, ExportArtifact)

proto or container — what's inside?
=====================================

:class:`~yobx.container.ExportArtifact` holds the exported model in one
of two forms, depending on whether ``large_model=True`` was requested
during conversion:

.. list-table::
    :header-rows: 1
    :widths: 20 30 50

    * - Attribute
      - Type
      - When it is set
    * - ``proto``
      - :class:`onnx.ModelProto` | :class:`onnx.GraphProto` |
        :class:`onnx.FunctionProto`
      - **Always** — for standard exports the proto is the fully
        self-contained ONNX model.  For large-model exports it contains
        *external-data* placeholders.
    * - ``container``
      - :class:`~yobx.container.ExtendedModelContainer` | ``None``
      - Set **only** when ``large_model=True``.  The container stores
        large initializers outside the proto (e.g. as separate
        :mod:`numpy` or :mod:`torch` tensors) so that the in-memory
        proto remains small.

The helper method :meth:`~yobx.container.ExportArtifact.get_proto`
abstracts over both cases: it always returns a fully self-contained
:class:`onnx.ModelProto`, embedding weights from the container when
necessary.

.. code-block:: text

    to_onnx(...)
        │
        ▼
    ExportArtifact
        ├── proto      ── ModelProto (always present)
        │                 ↳ small model: weights embedded
        │                 ↳ large model: external-data placeholders
        │
        ├── container  ── ExtendedModelContainer (large_model=True only)
        │                 ↳ stores weight tensors outside the proto
        │
        ├── report     ── ExportReport (stats + extra metadata)
        ├── filename   ── str | None  (set after save())
        └── builder    ── GraphBuilder | None  (optional, for inspection)

Key methods
===========

:meth:`~yobx.container.ExportArtifact.get_proto`
-------------------------------------------------

Returns a fully self-contained :class:`onnx.ModelProto`.  For regular
exports this is just ``artifact.proto``.  For large-model exports
(:attr:`container` is set) the weights are embedded on the fly using
:meth:`~yobx.container.ExtendedModelContainer.to_ir` so the returned
proto can be used with any ONNX runtime without additional files.

.. code-block:: python

    # Works the same regardless of large_model=True/False:
    proto = artifact.get_proto()

    # Skip embedding weights (useful for inspection only):
    proto_no_weights = artifact.get_proto(include_weights=False)

:meth:`~yobx.container.ExportArtifact.save`
--------------------------------------------

Saves the model to a file.  Delegates to
:meth:`~yobx.container.ExtendedModelContainer.save` for large models
(which writes both the ``.onnx`` file and the companion data file) and
to :func:`onnx.save_model` otherwise.  After a successful save the path
is stored in :attr:`~yobx.container.ExportArtifact.filename`.

.. code-block:: python

    artifact.save("model.onnx")
    print(artifact.filename)   # "model.onnx"

:meth:`~yobx.container.ExportArtifact.load`
--------------------------------------------

Class method that loads a previously saved artifact.  It inspects the
file for external-data references and automatically creates an
:class:`~yobx.container.ExtendedModelContainer` when needed.

.. code-block:: python

    loaded = ExportArtifact.load("model.onnx")
    proto  = loaded.get_proto()

Optimization report
===================

:class:`~yobx.container.ExportReport` is available via
``artifact.report`` and accumulates:

* ``report.stats`` — list of per-pattern optimization statistics
  (pattern name, nodes added / removed, time).
* ``report.extra`` — arbitrary key-value metadata added during
  conversion.
* ``report.build_stats`` — :class:`~yobx.container.BuildStats` summary
  when available.

.. code-block:: python

    artifact = to_onnx(estimator, (X,))
    print(artifact.report)
    # ExportReport(n_stats=12, extra=[], has_build_stats=False)

    for s in artifact.report.stats:
        print(s["pattern"], s["removed"], "→", s["added"])

Convenience properties
======================

:class:`~yobx.container.ExportArtifact` also exposes several
pass-through properties that work whether the model is in ``proto`` or
``container``:

* ``artifact.graph`` — :class:`onnx.GraphProto`
* ``artifact.opset_import`` — opset list
* ``artifact.functions`` — local functions
* ``artifact.metadata_props`` — metadata key-value pairs
* ``artifact.ir_version`` — IR version integer
* ``artifact.SerializeToString()`` — raw bytes for ONNX runtimes

Example
=======

.. code-block:: python

    import numpy as np
    from sklearn.linear_model import LinearRegression
    from yobx.sklearn import to_onnx
    from yobx.container import ExportArtifact, ExportReport
    from yobx.reference import ExtendedReferenceEvaluator

    X = np.random.randn(20, 4).astype(np.float32)
    y = X @ np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    reg = LinearRegression().fit(X, y)

    # All to_onnx functions return ExportArtifact:
    artifact = to_onnx(reg, (X,))

    assert isinstance(artifact, ExportArtifact)
    assert isinstance(artifact.report, ExportReport)

    # Standard model → proto is set, container is None:
    assert artifact.proto is not None
    assert artifact.container is None

    # get_proto() works for both standard and large-model exports:
    proto = artifact.get_proto()

    # Save and reload:
    artifact.save("model.onnx")
    loaded = ExportArtifact.load("model.onnx")

    # Run with the reference evaluator:
    ref = ExtendedReferenceEvaluator(artifact)
    (y_pred,) = ref.run(None, {"X": X})


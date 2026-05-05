.. _l-main-to-onnx:

=====================
Main to_onnx function
=====================

:func:`yobx.to_onnx` is the **single entry point** for converting any
supported model to `ONNX <https://onnx.ai>`_ format.  It inspects the
type of the *model* argument at runtime and automatically delegates to
the appropriate backend-specific converter, forwarding all extra keyword
arguments verbatim.

The function always returns an :class:`~yobx.container.ExportArtifact`
regardless of which backend was selected — see
:ref:`l-design-export-artifact` for a full description of that container.

Dispatch rules
==============

:func:`yobx.to_onnx` inspects *model* in this order and calls the
first matching backend:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Model type
     - Backend function
     - Reference
   * - :class:`torch.nn.Module` or :class:`torch.fx.GraphModule`
     - :func:`yobx.torch.to_onnx`
     - :ref:`l-torch-converter`
   * - :class:`sklearn.base.BaseEstimator`
     - :func:`yobx.sklearn.to_onnx`
     - :ref:`l-sklearn-converter`
   * - :class:`tensorflow.Module` (inc. Keras)
     - :func:`yobx.tensorflow.to_onnx`
     - :ref:`l-design-tensorflow-converter`
   * - :class:`bytes` or ``*.tflite`` path
     - :func:`yobx.litert.to_onnx`
     - :ref:`l-design-litert-converter`
   * - SQL :class:`str`, :func:`callable`, or :class:`polars.LazyFrame`
     - :func:`yobx.sql.to_onnx`
     - :ref:`l-design-sql-converter`

.. code-block:: text

    to_onnx(model, args, ...)
         │
         ├── torch.nn.Module / torch.fx.GraphModule ──► yobx.torch.to_onnx
         │
         ├── sklearn.base.BaseEstimator ──────────────► yobx.sklearn.to_onnx
         │
         ├── tensorflow.Module ───────────────────────► yobx.tensorflow.to_onnx
         │
         ├── bytes / *.tflite path ───────────────────► yobx.litert.to_onnx
         │
         └── str / callable / polars.LazyFrame ──────► yobx.sql.to_onnx

If the model type matches none of the above, a :class:`TypeError` is raised
listing all supported types.

Common parameters
=================

All backends accept the following keyword arguments.
Backend-specific parameters can be passed as extra ``**kwargs`` and are
forwarded verbatim to the selected converter.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Parameter
     - Default
     - Description
   * - ``args``
     - ``None``
     - Input arguments forwarded to the selected converter.
       For **torch**: a tuple of :class:`torch.Tensor` objects.
       For **sklearn / tensorflow / litert**: a tuple of
       :class:`numpy.ndarray` objects.
       For **sql**: a ``{column: dtype}`` mapping or a
       :class:`numpy.ndarray` for numpy-function tracing.
   * - ``input_names``
     - ``None``
     - Optional list of names for the ONNX graph input tensors.
       When omitted, names are derived automatically by each backend.
   * - ``dynamic_shapes``
     - ``None``
     - Declares which tensor dimensions are symbolic (variable-length).
       See :ref:`l-main-to-onnx-dynamic-shapes` below for the
       backend-specific formats.
   * - ``target_opset``
     - ``21``
     - ONNX opset version to target.  Either an integer for the default
       domain, or a ``Dict[str, int]`` mapping domain names to opset
       versions (e.g. ``{"": 21, "com.microsoft": 1}``).
   * - ``verbose``
     - ``0``
     - Verbosity level (0 = silent).
   * - ``large_model``
     - ``False``
     - When ``True``, initializers are stored as external data rather than
       embedded in the proto.  Required for models exceeding the 2 GB
       protobuf limit.
   * - ``external_threshold``
     - ``1024``
     - When ``large_model=True``, tensors whose element count exceeds this
       threshold are stored externally.
   * - ``filename``
     - ``None``
     - If set, saves the exported model to this path.
       Not supported by the LiteRT backend (silently ignored).
   * - ``return_optimize_report``
     - ``False``
     - When ``True``, the returned artifact's
       :attr:`~yobx.container.ExportArtifact.report` attribute is
       populated with per-pattern optimization statistics.

.. _l-main-to-onnx-dynamic-shapes:

Dynamic shapes
==============

The meaning of ``None`` and the accepted format for ``dynamic_shapes``
differs by backend:

**PyTorch** follows :func:`torch.export.export` conventions.  Pass a
dict mapping input names to per-axis :class:`torch.export.Dim` objects,
or a tuple with one entry per positional input.  ``None`` means *no*
dynamic dimensions (all shapes are fixed)::

    import torch
    from yobx import to_onnx

    batch = torch.export.Dim("batch", min=1, max=256)
    artifact = to_onnx(model, (x,), dynamic_shapes={"x": {0: batch}})

**scikit-learn / tensorflow / LiteRT** use a tuple of ``{axis: dim_name}``
dicts, one per input.  ``None`` treats **axis 0 as dynamic** (batch
dimension) for every input; all other axes remain static::

    import numpy as np
    from sklearn.linear_model import LinearRegression
    from yobx import to_onnx

    # None → axis 0 is automatically dynamic
    artifact = to_onnx(reg, (X,))

    # explicit dynamic batch for input 0, static for input 1
    artifact = to_onnx(reg, (X,), dynamic_shapes=({0: "batch"},))

**SQL / numpy callable** uses the same ``{axis: dim_name}`` tuple format
as scikit-learn.  ``None`` lets the backend apply its own default
(typically axis 0 dynamic).  Ignored for SQL strings and DataFrame-tracing
callables.

Return value
============

Every call to :func:`yobx.to_onnx` returns an
:class:`~yobx.container.ExportArtifact` instance.
See :ref:`l-design-export-artifact` for the complete API, including how to:

* access the ONNX proto via ``artifact.proto`` or ``artifact.get_proto()``
* save the model to disk with ``artifact.save("model.onnx")``
* reload from disk with ``ExportArtifact.load("model.onnx")``
* inspect per-pattern optimization statistics via ``artifact.report``

Examples
========

scikit-learn — linear regression
---------------------------------

.. code-block:: python

    import numpy as np
    from sklearn.linear_model import LinearRegression
    from yobx import to_onnx

    X = np.random.randn(20, 4).astype(np.float32)
    y = X[:, 0] + X[:, 1]
    reg = LinearRegression().fit(X, y)

    # axis 0 is treated as dynamic (batch) by default
    artifact = to_onnx(reg, (X,))
    print(artifact)

scikit-learn — explicit dynamic batch dimension
------------------------------------------------

.. code-block:: python

    import numpy as np
    from sklearn.linear_model import LinearRegression
    from yobx import to_onnx

    X = np.random.randn(20, 4).astype(np.float32)
    y = X[:, 0] + X[:, 1]
    reg = LinearRegression().fit(X, y)

    # Mark axis 0 of the first input as dynamic:
    artifact = to_onnx(reg, (X,), dynamic_shapes=({0: "batch"},))

PyTorch — dynamic batch dimension
----------------------------------

.. code-block:: python

    import torch
    from yobx import to_onnx

    class Neuron(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)
        def forward(self, x):
            return torch.relu(self.linear(x))

    model = Neuron()
    x = torch.randn(3, 4)
    batch = torch.export.Dim("batch", min=1, max=256)
    artifact = to_onnx(model, (x,), dynamic_shapes={"x": {0: batch}})
    artifact.save("neuron.onnx")

TensorFlow / Keras — simple model
------------------------------------

.. code-block:: python

    import numpy as np
    import tensorflow as tf
    from yobx import to_onnx

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    X = np.random.randn(10, 3).astype(np.float32)
    model(X)   # build the model
    artifact = to_onnx(model, (X,))

SQL query
---------

.. code-block:: python

    import numpy as np
    from yobx import to_onnx

    artifact = to_onnx(
        "SELECT a + b AS total FROM t WHERE a > 0",
        {"a": np.float32, "b": np.float32},
    )

LiteRT / TFLite — from file
-----------------------------

.. code-block:: python

    from yobx import to_onnx

    artifact = to_onnx("model.tflite", ())

LiteRT / TFLite — from raw bytes
----------------------------------

.. code-block:: python

    from yobx import to_onnx

    with open("model.tflite", "rb") as f:
        flatbuffer = f.read()

    artifact = to_onnx(flatbuffer, ())

Saving and running the exported model
======================================

Once you have an :class:`~yobx.container.ExportArtifact`, you can
serialise it and run it with any ONNX-compatible runtime:

.. code-block:: python

    import numpy as np
    import onnxruntime
    from sklearn.linear_model import LinearRegression
    from yobx import to_onnx

    X = np.random.randn(20, 4).astype(np.float32)
    y = X[:, 0] + X[:, 1]
    reg = LinearRegression().fit(X, y)

    artifact = to_onnx(reg, (X,))
    artifact.save("reg.onnx")

    sess = onnxruntime.InferenceSession("reg.onnx")
    (predictions,) = sess.run(None, {"X": X[:5]})

OnnxRuntime operator fusions
==============================

Passing ``"com.microsoft": 1`` in *target_opset* enables operator fusions
(fused attention, layer normalisation, …) that are specific to
:epkg:`onnxruntime`:

.. code-block:: python

    from yobx import to_onnx

    artifact = to_onnx(
        model,
        (x,),
        target_opset={"": 21, "com.microsoft": 1},
    )

See also
========

* :ref:`l-design-export-artifact` — the :class:`~yobx.container.ExportArtifact` container
* :ref:`l-torch-converter` — PyTorch backend details
* :ref:`l-sklearn-converter` — scikit-learn backend details
* :ref:`l-design-tensorflow-converter` — TensorFlow / Keras backend details
* :ref:`l-design-litert-converter` — LiteRT / TFLite backend details
* :ref:`l-design-sql-converter` — SQL / DataFrame backend details
* :ref:`l-design-main-to-onnx` — design notes for :func:`yobx.to_onnx`

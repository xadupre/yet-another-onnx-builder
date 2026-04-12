yet-another-onnx-builder documentation
======================================

.. image:: https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/ci_core.yml/badge.svg
    :target: https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/ci_core.yml
    :alt: core

.. image:: https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/ci_sklearn.yml/badge.svg
    :target: https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/ci_sklearn.yml
    :alt: scikit-learn

.. image:: https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/ci_tensorflow.yml/badge.svg
    :target: https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/ci_tensorflow.yml
    :alt: tensorflow

.. image:: https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/ci_torch.yml/badge.svg
    :target: https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/ci_torch.yml
    :alt: pytorch

.. image:: https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/docs.yml/badge.svg
    :target: https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/docs.yml

.. image:: https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/style.yml/badge.svg
    :target: https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/style.yml

.. image:: https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/spelling.yml/badge.svg
    :target: https://github.com/xadupre/yet-another-onnx-builder/actions/workflows/spelling.yml

.. image:: https://codecov.io/gh/xadupre/yet-another-onnx-builder/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/xadupre/yet-another-onnx-builder

.. image:: https://img.shields.io/github/repo-size/xadupre/yet-another-onnx-builder
    :target: https://github.com/xadupre/yet-another-onnx-builder

`yet-another-onnx-builder on GitHub <https://github.com/xadupre/yet-another-onnx-builder>`_

**yet-another-onnx-builder** (``yobx``) proposes a unique API and a unique function
:func:`yobx.to_onnx` to convert machine learning models
to `ONNX <https://onnx.ai>`_ format and manipulating ONNX graphs programmatically.
It can export from many libraries. Each converters relies on a common GraphBuilder API
(:class:`~yobx.typing.GraphBuilderExtendedProtocol`)
to build the final ONNX model. One default implementation is provided but
it can also be replaced by any implementation of your own.
Any user can implement its own. You can see
:class:`~yobx.builder.onnxscript.bridge_graph_builder.OnnxScriptGraphBuilder`
or :class:`~yobx.builder.spox.SpoxGraphBuilder` for a reference.
These API are close to :epkg:`onnx` API, using `NodeProto` for nodes
and strings for names. This is on purpose: what this API produces is
what you see in the final ONNX model. You can add your own metadata,
choose your own names.

:func:`yobx.to_onnx` is the single entry point for all supported frameworks.
It inspects the type of *model* at runtime and automatically delegates to
the appropriate backend-specific converter:

* a :class:`torch.nn.Module` or :class:`torch.fx.GraphModule` → :func:`yobx.torch.to_onnx`
* a :class:`sklearn.base.BaseEstimator` → :func:`yobx.sklearn.to_onnx`
* a :class:`tensorflow.Module` (including Keras models) → :func:`yobx.tensorflow.to_onnx`
* raw ``.tflite`` bytes or a path ending in ``".tflite"`` → :func:`yobx.litert.to_onnx`
* a SQL string, a Python callable, or a :epkg:`polars.LazyFrame` → :func:`yobx.sql.to_onnx`

All extra keyword arguments are forwarded verbatim to the selected converter,
so the backend-specific parameters (``export_options``, ``function_options``,
``extra_converters``, …) remain fully accessible through the top-level function.

**standard machine learning**

+-------------------------------+--------------------------------+------------------------------+
| :epkg:`category_encoders`     | :func:`yobx.sklearn.to_onnx`   | :ref:`l-sklearn-converter`   |
+-------------------------------+--------------------------------+------------------------------+
| :epkg:`imbalanced-learn`      | :func:`yobx.sklearn.to_onnx`   | :ref:`l-sklearn-converter`   |
+-------------------------------+--------------------------------+------------------------------+
| :epkg:`lightgbm`              | :func:`yobx.sklearn.to_onnx`   | :ref:`l-sklearn-converter`   |
+-------------------------------+--------------------------------+------------------------------+
| :epkg:`scikit-learn`          | :func:`yobx.sklearn.to_onnx`   | :ref:`l-sklearn-converter`   |
+-------------------------------+--------------------------------+------------------------------+
| :epkg:`scikit-survival`       | :func:`yobx.sklearn.to_onnx`   | :ref:`l-sklearn-converter`   |
+-------------------------------+--------------------------------+------------------------------+
| :epkg:`statsmodels`           | :func:`yobx.sklearn.to_onnx`   | :ref:`l-sklearn-converter`   |
+-------------------------------+--------------------------------+------------------------------+
| :epkg:`xgboost`               | :func:`yobx.sklearn.to_onnx`   | :ref:`l-sklearn-converter`   |
+-------------------------------+--------------------------------+------------------------------+

**data manipulations**

This is work in progress.
Many packages produce SQL queries. It starts by converting a SQL
query into ONNX.  A lightweight **DataFrame function tracer**
(:func:`~yobx.sql.dataframe_to_onnx`) records pandas-inspired
operations on a DataFrame and compiles them to ONNX directly.

+-----------------------------+------------------------------------+----------------------------------+
| sql                         | :func:`yobx.sql.to_onnx`           | :ref:`l-design-sql-converter`    |
+-----------------------------+------------------------------------+----------------------------------+
| :epkg:`polars.LazyFrame`    | :func:`yobx.sql.to_onnx`           | :ref:`l-design-sql-polars`       |
+-----------------------------+------------------------------------+----------------------------------+
| :class:`pandas.DataFrame`   | :func:`yobx.sql.to_onnx`           | :ref:`l-design-sql-dataframe`    |
+-----------------------------+------------------------------------+----------------------------------+

**deep learning**

+---------------------------------------+------------------------------------------+----------------------------------------+
| :epkg:`jax` *in progress*             | :func:`yobx.tensorflow.to_onnx`          | :ref:`l-plot-jax-to-onnx`              |
+---------------------------------------+------------------------------------------+----------------------------------------+
| :epkg:`tensorflow`                    | :func:`yobx.tensorflow.to_onnx`          | :ref:`l-design-tensorflow-converter`   |
+---------------------------------------+------------------------------------------+----------------------------------------+
| :epkg:`torch`                         | :func:`yobx.torch.to_onnx`               | :ref:`l-torch-converter`               |
+---------------------------------------+------------------------------------------+----------------------------------------+
| TFLite / :epkg:`LiteRT`               | :func:`yobx.litert.to_onnx`              | :ref:`l-design-litert-converter`       |
+---------------------------------------+------------------------------------------+----------------------------------------+

The package is built upon a single :ref:`graph builder API <l-design-graph-builder>`
for constructing and optimizing ONNX graphs with built-in shape inference
with can also linked to :epkg:`spox` or :epkg:`onnxscript`/:epkg:`ir-py`.
Its unique API:

.. code-block:: python

    # the model is called 
    from yobx import to_onnx
    expected = model(*args, **kwargs)
    onnx_model = to_onnx(model, args, kwargs, dynamic_shapes, **options)

The function returns an :class:`~yobx.container.ExportArtifact` that
wraps the exported ONNX proto together with an
:class:`~yobx.container.ExportReport`. The ONNX model can be retrieved
with `artifact.model_proto` and saved to disk via `artifact.save(path)`.

`options` are different across the libraries producing the model even they
share some of them. The common parameters accepted by all backends are:

* ``target_opset`` — an integer or a dict mapping ONNX domain names to
  their opset version (e.g. ``{"": 18, "com.microsoft": 1}``).  Adding
  ``"com.microsoft"`` will trigger operator fusions specific to
  :epkg:`onnxruntime` such as fused attention and layer normalization.
* ``large_model`` — when ``True`` the weights are stored in a separate
  ``.onnx_data`` file next to the model (ONNX external-data format).
  This is required for models whose size exceeds the 2 GB protobuf limit.
* ``external_threshold`` — size in bytes above which individual initializers
  are stored externally when ``large_model=True`` (default: 1024).
* ``input_names`` — an explicit list of names for the ONNX graph input
  tensors.  When omitted, names are derived automatically.
* ``dynamic_shapes`` — declares which tensor dimensions are symbolic
  (variable-length).  The exact format depends on the backend: torch
  follows :func:`torch.export.export` conventions while the other backends,
  the default is different is different for every library but it is usually
  empty for :epkg:`pytorch` or :epkg:`tensorflow` (so static shape),
  first dimension is batch dimension for :epkg:`scikit-learn`.
  use a tuple of ``{axis: dim_name}`` dicts.
* ``verbose`` — verbosity level (integer, 0 = silent).
* ``return_optimize_report`` — when ``True``, the returned artifact
  has its ``report`` attribute populated with per-pattern optimization
  statistics.

Oother options are specific to every converter and control the way
a model is captured or converted. It is possible to output
the decision path for trees or ensembles in :epkg:`scikit-learn`.

.. toctree::
   :maxdepth: 1

   api/index
   converters
   core
   galleries
   cmds/index
   install
   getting_started
   misc

This package was initially started using :epkg:`vibe coding`.
AI is able to translate an existing code into another one
such as ONNX but it tends sometime to favor ugly functions definition
not friendly to the users.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Older versions
==============

* `0.1.0 <../v0.1.0/index.html>`_

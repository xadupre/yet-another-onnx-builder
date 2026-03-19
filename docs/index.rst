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

**yet-another-onnx-builder** (``yobx``) proposes a unique API to convert machine learning models
to `ONNX <https://onnx.ai>`_ format and manipulating ONNX graphs programmatically.
It can export from many libraries. Each converters relies on a common GraphBuider API
(:class:`~yobx.typing.GraphBuiderExtendedProtocol`)
to build the final ONNX model. One default implementation is provided but
it can also be replaced by any implementation of your own.
Any user can implement its own. You can see
:class:`~yobx.builder.onnxscript.bridge_graph_builder.OnnxScriptGraphBuilder`
or :class:`yobx.builder.spox.gridge_graph_spox.SpoxGraphBuilder` for a reference.
These API are close to :epkg:`onnx` API, using `NodeProto` for nodes
and strings for names. This is on purpose: what this API produces is
what you see in the final ONNX model. You can add your own metadata,
choose your own names.

**standard machine learning**

+---------------------------------------+------------------------------+
| :epkg:`category_encoders`             | :ref:`l-sklearn-converter`   |
+---------------------------------------+------------------------------+
| :epkg:`imbalanced-learn`              | :ref:`l-sklearn-converter`   |
+---------------------------------------+------------------------------+
| :epkg:`lightgbm`                      | :ref:`l-sklearn-converter`   |
+---------------------------------------+------------------------------+
| :epkg:`scikit-learn`                  | :ref:`l-sklearn-converter`   |
+---------------------------------------+------------------------------+
| :epkg:`scikit-survival`               | :ref:`l-sklearn-converter`   |
+---------------------------------------+------------------------------+
| :epkg:`xgboost`                       | :ref:`l-sklearn-converter`   |
+---------------------------------------+------------------------------+

**data manipulations**

This is work in progress.
Many packages produce SQL queries. It starts by converting a SQL
query into ONNX.

* :ref:`l-design-sql`

**deep learning**

+---------------------------------------+----------------------------------------+
| :epkg:`jax` *in progress*             | :ref:`l-plot-jax-to-onnx`              |
+---------------------------------------+----------------------------------------+
| :epkg:`tensorflow`                    | :ref:`l-design-tensorflow-converter`   |
+---------------------------------------+----------------------------------------+
| :epkg:`torch`                         | :ref:`l-torch-converter`               |
+---------------------------------------+----------------------------------------+
| TFLite / :epkg:`LiteRT`               | :ref:`l-design-litert-converter`       |
+---------------------------------------+----------------------------------------+

It also provides:

- A :ref:`graph builder API <l-design-graph-builder>` for constructing and optimizing ONNX graphs,
  with built-in shape inference and a pattern-based
  :ref:`graph optimizer <l-design-pattern-optimizer-patterns>`.
- Converters for **scikit-learn** estimators and pipelines (``yobx.sklearn``).
- Utilities for **PyTorch** export, including model patching and input flattening (``yobx.torch``).
- A symbolic :ref:`shape expression system <l-design-shape>` for dynamic shape handling at export time.
- A :ref:`translation tool <l-design-translate>` that converts ONNX graphs back to executable Python code.
- **Optimization functions** to make the model more efficient.
- It supports multiple opsets and multiple domains.
- It allows the user to directly onnx model with :epkg:`spox` or :epkg:`onnxscript`/:epkg:`ir-py`.

Its unique API:

.. code-block:: python

    # the model is called 
    expected = model(*args, **kwargs)
    onnx_model = to_onnx(model, args, kwargs, dynamic_shapes, **options)

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Older versions
==============

* `0.1.0 <../v0.1.0/index.html>`_

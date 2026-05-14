.. _l-torch-converter:

Torch Export to ONNX
====================

The official converter is :func:`torch.onnx.export`.
This converter has been used to investigate when the first one is failing.
It is designed to quickly fail and offers more tracing options
to capture the :class:`torch.fx.Graph` such as symbolic tracing
or different decomposition tables. First section exposes the differences.
:ref:`l-design-torch-case-coverage` shows some of the differences
on basic examples.

.. toctree::
   :maxdepth: 1

   converter
   flatten
   flatten_list
   patches
   patches_list
   input_observer
   supported_aten_functions
   case_coverage
   coverage/op_coverage

.. note::
    :func:`yobx.torch.interpreter.to_onnx` is **not** :func:`torch.onnx.export`.
    See :ref:`l-not-torch-onnx-export` for a detailed comparison.

This section describes the design of the PyTorch-to-ONNX conversion pipeline.
The entry point is :func:`yobx.torch.interpreter.to_onnx`, which accepts a
:class:`torch.nn.Module` and representative inputs and returns an
:class:`onnx.ModelProto` (or an :class:`onnx.model_container.ModelContainer`
for large models).

The pipeline has three main stages:

1. **Export** — the module is traced into a portable
   :class:`~torch.export.ExportedProgram` (or :class:`~torch.fx.GraphModule`)
   using one of the strategies provided by
   :class:`~yobx.torch.export_options.ExportOptions` (``strict``,
   ``nostrict``, ``tracing``, ``new-tracing``, ``jit``, ``dynamo``, ``fake``, …).
2. **Interpret** — :class:`~yobx.torch.interpreter.interpreter.FxGraphInterpreter`
   walks the FX graph node by node and emits the corresponding ONNX operators
   into a :class:`~yobx.xbuilder.GraphBuilder`.
3. **Optimise** — the accumulated ONNX graph is folded, simplified, and
   serialised by :meth:`~yobx.xbuilder.GraphBuilder.to_onnx`.

The remaining pages in this section document supporting concerns: how custom
pytree nodes must be registered before export (:ref:`l-design-flatten`), which
internal torch/transformers patches are needed for successful tracing
(:ref:`l-design-patches`), and how real forward passes can be used to infer
export arguments and dynamic shapes automatically
(:ref:`l-design-input-observer`).

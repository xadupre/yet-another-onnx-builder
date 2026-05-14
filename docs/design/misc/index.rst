Miscellaneous
=============

This section covers miscellaneous utilities and design topics: the fluent
:ref:`light builder API <l-design-light-api>`, :ref:`ONNX inspection helpers <l-design-onnx-inspection>`,
the :ref:`translate <l-design-translate>` utility to export models as Python code, the
:ref:`evaluators <l-design-evaluator>` (reference, OnnxRuntime, Torch), the
:ref:`ExtendedModelContainer <l-design-container>` for large models,
:ref:`MiniOnnxBuilder and other helpers <l-design-helpers>`,
:ref:`CubeLogs <l-cube>` for structured experiment-log analysis,
:ref:`onnxruntime.SessionOptions <l-design-session-options>` for all
configurable session options, and
alternatives following
:class:`~yobx.typing.GraphBuilderExtendedProtocol` implementation
(:ref:`l-design-graph-builder-extended-protocol`).

.. toctree::
   :maxdepth: 1

   container
   cube
   evaluator
   graph_builder_protocol
   light_api
   helpers
   onnx_inspection
   session_options
   translate

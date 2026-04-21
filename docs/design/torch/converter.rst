.. _l-design-torch-converter:

========================
Torch Converters to ONNX
========================

.. note::
    This section covers the core conversion pipeline that transforms a
    :class:`torch.nn.Module` into an :class:`onnx.ModelProto`.  It is only
    relevant when exporting PyTorch models and has no bearing on ONNX models
    built directly with the builder APIs.

.. _l-not-torch-onnx-export:

Not torch.onnx.export
=====================

:func:`yobx.torch.interpreter.to_onnx` is **not** :func:`torch.onnx.export`.
They share a similar high-level goal — converting a :class:`torch.nn.Module` to
an :class:`onnx.ModelProto` — but are otherwise independent implementations with
different design priorities.

.. list-table:: yobx vs torch.onnx.export
   :widths: 30 35 35
   :header-rows: 1

   * - Aspect
     - :func:`yobx.torch.interpreter.to_onnx`
     - :func:`torch.onnx.export` (dynamo backend)
   * - **ATen → ONNX translation**
     - Custom :class:`~yobx.torch.interpreter.interpreter.FxGraphInterpreter`
       that walks the FX graph node-by-node and emits ONNX ops directly into a
       :class:`~yobx.xbuilder.GraphBuilder`
     - Uses `onnxscript <https://microsoft.github.io/onnxscript/>`_ as the
       intermediate representation for ATen-to-ONNX mappings
   * - **Export strategy**
     - Multiple strategies selectable via
       :class:`~yobx.torch.export_options.ExportOptions`:
       ``strict``, ``nostrict``, ``tracing``, ``fake``, …
       The ``tracing`` strategy uses
       :class:`~yobx.torch.tracing.CustomTracer` (symbolic tracing) and can
       handle models where ``torch.export.export`` fails. It offers
       the possibility to keep modules as leave, and does not decompose them
       into aten operators.
     - Dynamo-based ``torch.export.export``, if the default mode fails, it tries
       with ``strict=True`` a second time.
   * - **Graph decomposition**
     - The graph is **not** decomposed by default (unless
       ``decomposition_table`` is set in
       :class:`~yobx.torch.export_options.ExportOptions`).  Each ATen op is
       translated directly, which keeps the graph compact but means that a
       new op introduced by PyTorch will raise a ``NotImplementedError``
       until a dedicated converter is added.
     - Always decomposes the graph into a fixed set of core ATen ops,
       making the exporter robust to new high-level ops at the cost of
       larger, less readable ONNX graphs
   * - **Pattern Optimization**
     - Supports any kind of pattern matching, multiple outputs, variable number of
       nodes, outputs, types.
     - Supports a fixed number of outputs.
   * - **Dummy dimension equal to 0 or 1**
     - Option :epkg:`guard_size_oblivious` is determined before exporting.
     - Option :epkg:`guard_size_oblivious` is always used even if not needed.
   * - **Shape & type propagation**
     - Every intermediate result is typed and (when possible) given a concrete
       or symbolic shape inside the builder, enabling richer optimization and
       easier debugging
     - Shape information comes from the traced graph; no additional in-builder
       propagation step
   * - **Large-model support**
     - ``large_model=True`` returns an
       :class:`onnx.model_container.ModelContainer`; weights can be stored as
       external data files via ``external_threshold``
     - External data handled by the caller after export
   * - **Debugging**
     - Rich set of environment variables (``ONNXSTOP``, ``ONNXSTOPSHAPE``,
       ``ONNXSTOPTYPE``, ``ONNXSTOPOUTPUT``, …) let you pinpoint exactly
       which node assigns a suspicious shape or type; see
       :ref:`l-design-sklearn-debug-env-vars`
     - Standard Python / PyTorch debugging tools

In short: if you need the officially supported PyTorch exporter, use
:func:`torch.onnx.export`.  If you need finer control over the
ATen → ONNX translation, built-in graph optimization, advanced export
strategies such as fake-tensor mode or automatic dynamic-shape inference,
or rich debugging capabilities (:ref:`l-design-sklearn-debug-env-vars`),
:func:`yobx.torch.interpreter.to_onnx` may be a better fit.

The entry point for converting a PyTorch model to ONNX is
:func:`yobx.torch.interpreter.to_onnx`.  The function orchestrates a
multi-stage pipeline:

1. **Export** — trace the module into a portable graph representation using
   :func:`torch.export.export` (or one of its alternatives).
2. **Interpret** — walk every node of the FX graph and translate it into a
   sequence of ONNX operations via :class:`~yobx.torch.interpreter.interpreter.FxGraphInterpreter`.
3. **Optimise** — run the ONNX graph through the optimiser shipped in
   :class:`~yobx.xbuilder.GraphBuilder` to fold constants, remove redundant
   casts, and simplify shapes.

Pipeline overview
=================

.. code-block:: text

    torch.nn.Module
          │
          │  ExportOptions  (strict / nostrict / tracing / jit / dynamo …)
          ▼
    torch.export.ExportedProgram   ←─── torch.fx.GraphModule (optional)
          │
          │  _make_builder_interpreter()
          ▼
    GraphBuilder  +  FxGraphInterpreter
          │
          │  FxGraphInterpreter.run()  ── node-by-node dispatch
          │        ├── placeholder   →  graph input
          │        ├── get_attr      →  initializer
          │        ├── call_function →  aten_* converter
          │        ├── call_method   →  aten method converter
          │        ├── call_module   →  submodule (recursive)
          │        └── output        →  graph output
          ▼
    GraphBuilder (ONNX ops accumulated)
          │
          │  optimize=True  →  OptimizationOptions applied
          ▼
    onnx.ModelProto  (or ModelContainer for large models)

Key components
==============

to_onnx
-------

:func:`yobx.torch.interpreter.to_onnx` is the public API.  Its most important
parameters are:

* ``mod`` — the :class:`torch.nn.Module` to export.
* ``args`` / ``kwargs`` — representative inputs (used to infer shapes and
  to validate the export when ``validate_onnx`` is set).
* ``dynamic_shapes`` — a nested structure that marks which tensor dimensions
  are symbolic; mirrors the argument accepted by :func:`torch.export.export`.
* ``export_options`` — an :class:`~yobx.torch.export_options.ExportOptions`
  instance (or a short strategy string such as ``"nostrict-dec"``) that
  controls *how* the model is first exported to an FX graph.
* ``target_opset`` — the ONNX opset version to target (default: the latest
  supported opset minus one).
* ``optimize`` — when ``True`` (default) the generated ONNX graph is
  optimised before being returned.
* ``dispatcher`` — an optional :class:`~yobx.torch.interpreter.Dispatcher`
  that can override the default ATen-to-ONNX mapping for specific operators.

Basic usage
~~~~~~~~~~~

.. runpython::
    :showcode:

    import torch
    from yobx.torch.interpreter import to_onnx

    class Neuron(torch.nn.Module):
        def __init__(self, n_dims: int, n_targets: int):
            super().__init__()
            self.linear = torch.nn.Linear(n_dims, n_targets)

        def forward(self, x):
            return torch.relu(self.linear(x))

    model = Neuron(5, 3)
    x = torch.rand(2, 5)
    onx = to_onnx(model, (x,))
    print(onx.graph.node[:3])

ExportOptions
-------------

:class:`~yobx.torch.export_options.ExportOptions` encapsulates the strategy
used to obtain the FX graph from the module.  Several named strategies are
available:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Strategy string
     - Meaning
   * - ``"strict"``
     - ``torch.export.export`` with ``strict=True``
   * - ``"nostrict"``
     - ``torch.export.export`` with ``strict=False`` (default)
   * - ``"nostrict-dec"``
     - ``strict=False`` + default decomposition table
   * - ``"nostrict-decall"``
     - ``strict=False`` + full decomposition table
   * - ``"tracing"``
     - symbolic tracing via :class:`~yobx.torch.tracing.CustomTracer`
   * - ``"jit"``
     - JIT script → FX graph
   * - ``"dynamo"``
     - ``torch._dynamo.export``
   * - ``"dec"``
     - default decomposition table, default strict setting
   * - ``"decall"``
     - full decomposition table, default strict setting
   * - ``"fake"``
     - use :class:`~torch._subclasses.fake_tensor.FakeTensor` inputs instead of real tensors

Fake tensors
~~~~~~~~~~~~

When ``fake=True`` (or ``strategy="fake"``), the export stage replaces every
real input tensor with a :class:`~torch._subclasses.fake_tensor.FakeTensor`
— a lightweight stand-in that carries dtype, shape, and device metadata but
holds no actual data.  This is useful when loading model weights into memory
just to trace the graph would be prohibitively expensive (e.g. very large
language models).

The conversion from real tensors to fake tensors is handled by
:func:`~yobx.torch.fake_tensor_helper.make_fake_with_dynamic_dimensions`, which
also ensures that dimensions sharing the same name in ``dynamic_shapes`` are
mapped to the same symbolic integer.  The
:class:`~yobx.torch.fake_tensor_helper.FakeTensorContext` manages the
underlying :class:`~torch._subclasses.fake_tensor.FakeTensorMode` and the
mapping between concrete dimension values and their symbolic counterparts.

FxGraphInterpreter
------------------

:class:`~yobx.torch.interpreter.interpreter.FxGraphInterpreter` is the heart
of the converter.  It walks the :class:`torch.fx.Graph` node by node and
translates each node into one or more ONNX operators appended to the
:class:`~yobx.xbuilder.GraphBuilder`.

Node kinds and their handlers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - FX node kind
     - Action taken by the interpreter
   * - ``placeholder``
     - Registers a graph input with the correct ONNX type and shape.
   * - ``get_attr``
     - Looks up a weight / buffer / constant via the *retriever* callable and
       registers it as an ONNX initializer.
   * - ``call_function``
     - Looks up the ATen function in the aten-function registry
       (``_aten_functions.py``) or delegates to the :class:`Dispatcher`.
   * - ``call_method``
     - Looks up the ATen method in the method registry
       (``_aten_methods.py``).
   * - ``call_module``
     - Recursively converts a submodule using a nested
       :class:`FxGraphInterpreter` instance.
   * - ``output``
     - Registers graph outputs, applying any output masks produced by the
       export stage.

ATen-to-ONNX converters
~~~~~~~~~~~~~~~~~~~~~~~~

Each ``call_function`` node carries a :class:`torch._ops.OpOverload` such as
``aten.relu.default`` or ``aten.mm.default``.  The interpreter resolves this
to a Python function whose name follows the pattern ``aten_<op>_<overload>``
(e.g. ``aten_relu_default``) defined in one of:

* :mod:`yobx.torch.interpreter._aten_functions` — the bulk of ATen ops.
* :mod:`yobx.torch.interpreter._aten_methods` — tensor methods (e.g.
  ``.view``, ``.contiguous``).
* :mod:`yobx.torch.interpreter._aten_functions_attention` — attention-related
  ops (``scaled_dot_product_attention`` etc.).
* :mod:`yobx.torch.interpreter._prims_functions` — ``torch.ops.prims.*``
  primitives.
* :mod:`yobx.torch.interpreter._math_functions` — ``torch.ops.higher_order``
  and math helpers.
* :mod:`yobx.torch.interpreter._non_aten_functions` — custom and
  non-ATen ops.

Each converter function has the signature::

    def aten_<op>_<overload>(
        g: GraphBuilder,
        sts: Dict[str, Any],   # shape/type state
        outputs: List[str],    # desired output names
        *args,
        **kwargs,
    ) -> str:
        ...

It appends one or more ONNX nodes to ``g`` and returns the name of the
primary output tensor.

Dispatcher
----------

:class:`~yobx.torch.interpreter.Dispatcher` allows callers to override or
extend the built-in ATen converter mapping without modifying the library.
It is especially useful when a model uses custom ops or when the default
conversion of a particular op should be replaced.

.. code-block:: python

    from yobx.torch.interpreter import Dispatcher, to_onnx
    from yobx.xbuilder import GraphBuilder

    def my_relu(g: GraphBuilder, sts, outputs, x):
        return g.op.Relu(x, outputs=outputs)

    dispatcher = Dispatcher({"aten_relu_default": my_relu})
    onx = to_onnx(model, (x,), dispatcher=dispatcher)

:class:`~yobx.torch.interpreter.ForceDispatcher` is a stricter variant that
raises an error when the requested function is not found, making it easier to
discover missing converters during development.

GraphBuilder
------------

The :class:`~yobx.xbuilder.GraphBuilder` accumulates ONNX ops and is
responsible for:

* **Type and shape propagation** — every result tensor is given an ONNX type
  and, when possible, a concrete or symbolic shape.
* **Optimisation** — constant folding, cast elimination, identity removal,
  and other peephole passes controlled by
  :class:`~yobx.xbuilder.OptimizationOptions`.
* **Serialisation** — once all nodes have been appended,
  :meth:`~yobx.xbuilder.GraphBuilder.to_onnx` serialises the accumulated
  state into an :class:`onnx.ModelProto`.

Large models
------------

When ``large_model=True`` the converter returns an
:class:`onnx.model_container.ModelContainer` instead of an
:class:`onnx.ModelProto`.  This defers the decision on whether to embed the
weights inside the protobuf or to store them as external data files.

The ``external_threshold`` parameter (default: 1 024 bytes) controls which
initializers are treated as external when the container is later saved.

Dynamic shapes
==============

The ``dynamic_shapes`` argument is forwarded to :func:`torch.export.export`
and follows the same nested-dict / nested-tuple convention.  Symbolic
dimension variables should be instances of :class:`torch.export.Dim`.

:func:`~yobx.torch.use_dyn_not_str` is a convenience helper that replaces
string-valued dimension annotations (which some helpers return) with
``torch.export.Dim.DYNAMIC``:

.. code-block:: python

    from yobx.torch import use_dyn_not_str
    dynamic_shapes = use_dyn_not_str({"x": {0: "batch", 1: "seq"}})

For more complex scenarios — especially LLMs with prefill/decode phases —
:class:`~yobx.torch.input_observer.InputObserver` can infer the
``dynamic_shapes`` automatically from real forward passes; see
:ref:`l-design-input-observer`.

Submodules as local ONNX functions
===================================

When ``export_modules_as_functions=True`` (or a set of module types), the
converter unfolds the model via :func:`torch.export.unflatten` so that each
submodule becomes a separate ONNX local function.  This preserves the
module hierarchy in the ONNX graph and can reduce model size when the same
submodule type is reused many times.

The granularity is controlled by the ``function_options`` parameter (a
:class:`~yobx.xbuilder.FunctionOptions` instance) which specifies, among
other things, how initializers inside local functions should be represented
(as constants inlined into the function body or as additional function
inputs).

Validation
==========

Setting ``validate_onnx=True`` (or a float tolerance) causes
:func:`to_onnx` to run the exported ONNX model with the same inputs that
were used to export it and compare the outputs against the PyTorch model's
outputs.  :class:`AssertionError` is raised if the maximum absolute
difference exceeds the tolerance (default ``1e-5``).

Environment variables
=====================

Several environment variables alter the converter's behaviour without
requiring code changes:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Variable
     - Effect
   * - ``ONNXVERBOSE=1``
     - Increases verbosity inside :func:`to_onnx`.
   * - ``PRINT_GRAPH_MODULE=1``
     - Prints the FX graph before interpretation.
   * - ``ONNX_BUILDER_PROGRESS=1``
     - Shows a progress bar for large models.
   * - ``PRINT_EXPORTED_PROGRAM=1``
     - Prints the :class:`~torch.export.ExportedProgram` before interpretation.

See also
========

* :ref:`l-design-flatten` — registering custom pytree nodes before export.
* :ref:`l-design-patches` — patching torch / transformers internals for
  successful symbolic tracing.
* :ref:`l-design-input-observer` — automatic inference of export arguments
  and dynamic shapes.
* :ref:`l-graphbuilder-debugging-env` — ``GraphBuilder`` debugging environment
  variables.
* :class:`yobx.xbuilder.GraphBuilder` — the underlying ONNX graph builder.
* :class:`yobx.torch.export_options.ExportOptions` — all export strategy options.
* :func:`yobx.torch.interpreter.to_onnx` — the public conversion API.

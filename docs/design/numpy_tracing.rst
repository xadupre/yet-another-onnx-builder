.. _l-design-function-transformer-tracing:

=============
Numpy-Tracing
=============

This section covers the numpy-tracing infrastructure that converts plain
numpy functions into ONNX graphs — a core building block used by multiple
converters and available as a standalone tool.

``yobx`` can export a :class:`~sklearn.preprocessing.FunctionTransformer`
to ONNX by *tracing* its ``func`` attribute: the function is re-executed with
lightweight proxy objects instead of real numpy arrays.  Every numpy operation
performed on those proxies is recorded as an ONNX node, so the resulting ONNX
graph exactly mirrors the Python code — without any manual operator mapping.

Overview
========

The mechanism consists of two layers:

1. **:class:`~yobx.xtracing.NumpyArray`** — a proxy that wraps an ONNX tensor
   name and an object following the
   :class:`~yobx.typing.GraphBuilderExtendedProtocol`.  It overloads all
   Python arithmetic operators and registers itself as an implementation for
   both the ``__array_ufunc__`` and ``__array_function__`` numpy protocols.
   Whenever numpy (or user code) calls an operation on a
   :class:`~yobx.xtracing.NumpyArray`, the proxy emits the equivalent ONNX
   node into the graph and returns a new :class:`~yobx.xtracing.NumpyArray`
   wrapping the result tensor name.

2. **:func:`~yobx.xtracing.trace_numpy_function`** — the converter-API
   function.  It receives an object following the
   :class:`~yobx.typing.GraphBuilderExtendedProtocol`,
   the desired output names, the callable to trace, and the names of the
   input tensors already registered in that graph.  It wraps those tensors
   as :class:`~yobx.xtracing.NumpyArray` proxies, calls the function, and
   collects the resulting output tensors.

The high-level helper :func:`~yobx.xtracing.trace_numpy_to_onnx` creates a
standalone :class:`onnx.ModelProto` by building a fresh graph, registering
sample-array-derived inputs, and delegating to
:func:`~yobx.xtracing.trace_numpy_function`.

Converter API signature
=======================

:func:`~yobx.xtracing.trace_numpy_function` follows the same convention as
every other converter in this package:

.. code-block:: python

    def trace_numpy_function(
        g: GraphBuilderExtendedProtocol,
        sts: Dict,
        outputs: List[str],
        func: Callable,
        inputs: List[str],
        name: str = "trace",
        kw_args: Optional[Dict[str, Any]] = None,
    ) -> str: ...

===========  ================================================================
Parameter    Description
===========  ================================================================
``g``        :class:`~yobx.typing.GraphBuilderExtendedProtocol` — call
             ``g.op.<Op>(…)`` to emit ONNX nodes.
``sts``      ``Dict`` of metadata (empty ``{}`` in most call sites).
``outputs``  Pre-allocated output tensor names the tracer must write to.
``func``     Python callable that uses numpy operations.
``inputs``   Names of tensors already registered in *g*.
``name``     Node-name prefix.
``kw_args``  Optional keyword arguments forwarded to *func*.
===========  ================================================================

FunctionTransformer converter
==============================

The built-in converter for
:class:`~sklearn.preprocessing.FunctionTransformer` lives in
:mod:`yobx.sklearn.preprocessing.function_transformer`.  It delegates
directly to :func:`~yobx.xtracing.trace_numpy_function`, so all numpy ops
land in the surrounding graph with no sub-model inlining:

.. runpython::
    :showcode:

    from sklearn.preprocessing import FunctionTransformer
    import numpy as np
    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.sklearn import to_onnx

    def my_func(X):
        return np.log1p(np.abs(X))

    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4)).astype(np.float32)

    transformer = FunctionTransformer(func=my_func).fit(X)
    onx = to_onnx(transformer, (X,))
    print(pretty_onnx(onx))
    

When ``func=None`` (the identity transformer) a single ``Identity`` node is
emitted instead of tracing.

Supported numpy operations
==========================

The :class:`~yobx.xtracing.NumpyArray` proxy also supports all Python
arithmetic and comparison operators (``+``, ``-``, ``*``, ``/``, ``//``,
``**``, ``%``, ``@``, ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``,
unary ``-``).

The full list of supported ufuncs and array functions is generated below
directly from the live dispatch tables in
:mod:`yobx.xtracing.numpy_array`.

.. runpython::
    :showcode:
    :rst:

    import numpy as np
    from yobx.xtracing.numpy_array import _UFUNC_TO_ONNX, _HANDLED_FUNCTIONS

    rows_ufunc = []
    for k in sorted(_UFUNC_TO_ONNX.keys(), key=lambda x: x.__name__):
        v = _UFUNC_TO_ONNX[k]
        onnx_op = v[0] if isinstance(v, tuple) else v
        rows_ufunc.append(f"* ``np.{k.__name__}`` → ``{onnx_op}``")

    rows_func = []
    for k in sorted(_HANDLED_FUNCTIONS.keys(), key=lambda x: x.__name__):
        rows_func.append(f"* ``np.{k.__name__}``")

    print("**Ufuncs** (via ``__array_ufunc__``)\n")
    print("\n".join(rows_ufunc))
    print()
    print("**Array functions** (via ``__array_function__``)\n")
    print("\n".join(rows_func))

Standalone usage
================

You can also use the tracing machinery outside of scikit-learn pipelines via
:func:`~yobx.xtracing.trace_numpy_to_onnx`:

.. runpython::
    :showcode:

    import numpy as np
    from yobx.xtracing import trace_numpy_to_onnx
    from yobx.helpers.onnx_helper import pretty_onnx

    def my_func(X):
        return np.sqrt(np.abs(X) + np.float32(1))

    X_sample = np.zeros((4, 3), dtype=np.float32)
    onx = trace_numpy_to_onnx(my_func, X_sample)
    print(pretty_onnx(onx))

Or embedded into a larger graph using
:func:`~yobx.xtracing.trace_numpy_function` directly:

.. runpython::
    :showcode:

    import numpy as np
    from onnx import TensorProto
    from yobx.xbuilder import GraphBuilder
    from yobx.xtracing import trace_numpy_function
    from yobx.helpers.onnx_helper import pretty_onnx

    g = GraphBuilder({"": 21, "ai.onnx.ml": 1})
    g.make_tensor_input("X", TensorProto.FLOAT, ("batch", 3))

    def my_func(X):
        return np.sqrt(np.abs(X) + np.float32(1))

    trace_numpy_function(g, {}, ["output_0"], my_func, ["X"])
    g.make_tensor_output("output_0", indexed=False, allow_untyped_output=True)
    onx, _ = g.to_onnx(return_optimize_report=True)
    print(pretty_onnx(onx))

.. seealso::

    :ref:`l-design-sklearn-custom-converter` — how to write and register a
    custom converter for any scikit-learn estimator.

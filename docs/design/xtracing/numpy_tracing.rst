.. _l-design-function-transformer-tracing:

==========================================
Numpy-Tracing and FunctionTransformer
==========================================

``yobx`` can export a :class:`~sklearn.preprocessing.FunctionTransformer`
to ONNX by *tracing* its ``func`` attribute: the function is re-executed with
lightweight proxy objects instead of real numpy arrays.  Every numpy operation
performed on those proxies is recorded as an ONNX node, so the resulting ONNX
graph exactly mirrors the Python code — without any manual operator mapping.

Overview
========

The mechanism consists of two layers:

1. :class:`~yobx.xtracing.NumpyArray` — a proxy that wraps an ONNX tensor
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
        outputs: Optional[List[str]],
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
:mod:`yobx.xtracing.numpy_array`.  Each numpy name links to the
:epkg:`numpy` reference documentation, and each ONNX op name links to the
:epkg:`ONNX Operators` specification.

**Ufuncs** (via ``__array_ufunc__``)

.. runpython::
    :showcode:
    :rst:

    import numpy as np
    from yobx.xtracing.numpy_array import _UFUNC_TO_ONNX

    _NP_BASE = "https://numpy.org/doc/stable/reference/generated/numpy.{}.html"
    _ONNX_BASE = "https://onnx.ai/onnx/operators/onnx__{}.html"

    # Descriptions for composite (sentinel-string) mappings
    def _onnx_link(op):
        return f"`{op} <{_ONNX_BASE.format(op)}>`_"

    _composite_desc = {
        "floor_divide": f"{_onnx_link('Floor')} ( {_onnx_link('Div')} (a, b) )",
        "not_equal": f"{_onnx_link('Not')} ( {_onnx_link('Equal')} (a, b) )",
        "maximum": f"`Max <{_ONNX_BASE.format('Max')}>`_ (a, b)",
        "minimum": f"`Min <{_ONNX_BASE.format('Min')}>`_ (a, b)",
        "log1p": f"{_onnx_link('Log')} ( {_onnx_link('Add')} (x, 1) )",
        "expm1": f"{_onnx_link('Sub')} ( {_onnx_link('Exp')} (x), 1 )",
    }

    print(".. list-table::")
    print("   :header-rows: 1")
    print("   :widths: 40 60")
    print()
    print("   * - NumPy ufunc")
    print("     - ONNX op")
    for k in sorted(_UFUNC_TO_ONNX.keys(), key=lambda x: x.__name__):
        v = _UFUNC_TO_ONNX[k]
        np_url = _NP_BASE.format(k.__name__)
        np_cell = f"`np.{k.__name__} <{np_url}>`_"
        if isinstance(v, tuple):
            onnx_op = v[0]
            onnx_cell = f"`{onnx_op} <{_ONNX_BASE.format(onnx_op)}>`_"
        else:
            onnx_cell = _composite_desc.get(v, f"*(see source: {v})*")
        print(f"   * - {np_cell}")
        print(f"     - {onnx_cell}")
    print()

**Array functions** (via ``__array_function__``)

.. runpython::
    :showcode:
    :rst:

    import numpy as np
    from yobx.xtracing.numpy_array import _HANDLED_FUNCTIONS

    _NP_BASE = "https://numpy.org/doc/stable/reference/generated/numpy.{}.html"
    _ONNX_BASE = "https://onnx.ai/onnx/operators/onnx__{}.html"

    # Known ONNX mappings for array functions
    _func_onnx = {
        "reshape": "Reshape",
        "transpose": "Transpose",
        "sum": "ReduceSum",
        "mean": "ReduceMean",
        "amax": "ReduceMax",
        "amin": "ReduceMin",
        "prod": "ReduceProd",
        "clip": "Clip",
        "where": "Where",
        "concatenate": "Concat",
        "stack": "Unsqueeze + Concat",
        "expand_dims": "Unsqueeze",
        "squeeze": "Squeeze",
        "matmul": "MatMul",
        "dot": "MatMul",
        "abs": "Abs",
        "sqrt": "Sqrt",
        "exp": "Exp",
        "log": "Log",
        "log1p": "Add + Log",
        "expm1": "Exp + Sub",
    }

    def _onnx_cell(name):
        mapping = _func_onnx.get(name)
        if mapping is None:
            return "*(see source)*"
        # If it's a composite description (contains spaces), return as-is
        if " " in mapping:
            parts = mapping.split(" + ")
            links = [f"`{p} <{_ONNX_BASE.format(p)}>`_" for p in parts]
            return " + ".join(links)
        return f"`{mapping} <{_ONNX_BASE.format(mapping)}>`_"

    print(".. list-table::")
    print("   :header-rows: 1")
    print("   :widths: 40 60")
    print()
    print("   * - NumPy function")
    print("     - ONNX op")
    for k in sorted(_HANDLED_FUNCTIONS.keys(), key=lambda x: x.__name__):
        np_url = _NP_BASE.format(k.__name__)
        np_cell = f"`np.{k.__name__} <{np_url}>`_"
        onnx = _onnx_cell(k.__name__)
        print(f"   * - {np_cell}")
        print(f"     - {onnx}")
    print()

Standalone usage
================

You can also use the tracing machinery outside of scikit-learn pipelines via
:func:`~yobx.xtracing.trace_numpy_to_onnx`:

.. runpython::
    :showcode:

    import numpy as np
    from yobx.sql import trace_numpy_to_onnx
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
    onx = g.to_onnx()
    print(pretty_onnx(onx))

.. seealso::

    :ref:`l-design-sklearn-custom-converter` — how to write and register a
    custom converter for any scikit-learn estimator.

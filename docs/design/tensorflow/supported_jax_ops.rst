.. _l-design-tensorflow-supported-jax-ops:

====================
Supported JAX Ops
====================

When a JAX function is converted to ONNX via
:func:`~yobx.tensorflow.to_onnx`, the JAX computation is first lowered to a
``XlaCallModule`` TensorFlow op whose payload contains a *StableHLO* MLIR
module.  The converter parses that module op-by-op and maps each
``stablehlo.*`` operator to an ONNX node.

The tables below list every ``stablehlo`` op name (after stripping the
``stablehlo.`` prefix) that is currently supported, together with the
corresponding ONNX op or sub-graph it is lowered to.

Direct mappings
---------------

These ``stablehlo`` ops map to a single ONNX op with the same semantics:

.. runpython::
    :showcode:
    :rst:

    from yobx.tensorflow.ops.xla_call_module import _MAPPING_JAX_ONNX

    # Group by ONNX op name for a more readable table
    rows = sorted(_MAPPING_JAX_ONNX.items())  # (jax_op, onnx_op)
    print(".. list-table::")
    print("   :header-rows: 1")
    print("   :widths: 40 60")
    print()
    print("   * - StableHLO op (``stablehlo.<name>``)")
    print("     - ONNX op")
    for jax_op, onnx_op in rows:
        print(f"   * - ``{jax_op}``")
        print(f"     - ``{onnx_op}``")
    print()

Composite mappings
------------------

These ``stablehlo`` ops require more than one ONNX node and are implemented
as small sub-graphs by dedicated factory functions:

.. runpython::
    :showcode:
    :rst:

    from yobx.tensorflow.ops.xla_call_module import _COMPOSITE_JAX_OPS

    _descriptions = {
        "rsqrt": "``Reciprocal(Sqrt(x))``",
        "log_plus_one": "``Log(Add(x, 1))``",
        "exponential_minus_one": "``Sub(Exp(x), 1)``",
    }

    print(".. list-table::")
    print("   :header-rows: 1")
    print("   :widths: 40 60")
    print()
    print("   * - StableHLO op (``stablehlo.<name>``)")
    print("     - ONNX equivalent")
    for jax_op in sorted(_COMPOSITE_JAX_OPS):
        desc = _descriptions.get(jax_op, "*(see source)*")
        print(f"   * - ``{jax_op}``")
        print(f"     - {desc}")
    print()

Adding a new JAX op mapping
----------------------------

To add support for an additional ``stablehlo`` unary op:

1. If the op maps 1-to-1 to an ONNX op, add an entry to
   :data:`_MAPPING_JAX_ONNX` in
   :mod:`yobx.tensorflow.ops.xla_call_module`::

       _MAPPING_JAX_ONNX["cbrt"] = "some_onnx_op"  # if a direct match exists

2. If the op requires multiple ONNX nodes, add a ``_make_<name>`` factory
   function and register it in :data:`_COMPOSITE_JAX_OPS`::

       def _make_cbrt(g):
           import numpy as np
           def _cbrt(*args, **kwargs):
               name = kwargs.pop("name", "cbrt")
               outputs = kwargs.pop("outputs", None)
               (x,) = args
               exp = np.array(1.0 / 3.0, dtype=np.float32)
               kw = {"name": name}
               if outputs is not None:
                   kw["outputs"] = outputs
               return g.op.Pow(x, exp, **kw)
           return _cbrt

       _COMPOSITE_JAX_OPS["cbrt"] = _make_cbrt

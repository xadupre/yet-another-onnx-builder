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
corresponding ONNX op or sub-graph it is lowered to.  Each ONNX op name
links to its entry in the :epkg:`ONNX Operators` specification, and each
StableHLO op name links to the :epkg:`StableHLO` specification.

Direct mappings
---------------

These ``stablehlo`` ops map to a single ONNX op with the same semantics:

.. runpython::
    :showcode:
    :rst:

    from yobx.tensorflow.ops.jax_ops import _MAPPING_JAX_ONNX

    _ONNX_BASE = "https://onnx.ai/onnx/operators/onnx__{}.html"
    _STABLEHLO_BASE = "https://openxla.org/stablehlo/spec#{}"

    def _hlo_anchor(jax_op):
        # compare_EQ/GT/GE/LT/LE/NE are synthetic names for stablehlo.compare
        if jax_op.startswith("compare_"):
            return "compare"
        return jax_op.replace("_", "-")

    # Group by ONNX op name for a more readable table
    rows = sorted(_MAPPING_JAX_ONNX.items())  # (jax_op, onnx_op)
    print(".. list-table::")
    print("   :header-rows: 1")
    print("   :widths: 40 60")
    print()
    print("   * - StableHLO op (``stablehlo.<name>``)")
    print("     - ONNX op")
    for jax_op, onnx_op in rows:
        hlo_url = _STABLEHLO_BASE.format(_hlo_anchor(jax_op))
        onnx_url = _ONNX_BASE.format(onnx_op)
        print(f"   * - `{jax_op} <{hlo_url}>`_")
        print(f"     - `{onnx_op} <{onnx_url}>`_")
    print()

Composite mappings
------------------

These ``stablehlo`` ops require more than one ONNX node and are implemented
as small sub-graphs by dedicated factory functions:

.. runpython::
    :showcode:
    :rst:

    from yobx.tensorflow.ops.jax_ops import _COMPOSITE_JAX_OPS

    _ONNX_BASE = "https://onnx.ai/onnx/operators/onnx__{}.html"
    _STABLEHLO_BASE = "https://openxla.org/stablehlo/spec#{}"

    def _hlo_anchor(jax_op):
        if jax_op.startswith("compare_"):
            return "compare"
        return jax_op.replace("_", "-")

    def _onnx_link(op):
        return f"`{op} <{_ONNX_BASE.format(op)}>`_"

    _descriptions = {
        "rsqrt": f"{_onnx_link('Reciprocal')} ( {_onnx_link('Sqrt')} (x) )",
        "log_plus_one": f"{_onnx_link('Log')} ( {_onnx_link('Add')} (x, 1) )",
        "exponential_minus_one": f"{_onnx_link('Sub')} ( {_onnx_link('Exp')} (x), 1 )",
        "compare_NE": f"{_onnx_link('Not')} ( {_onnx_link('Equal')} (a, b) )",
    }

    print(".. list-table::")
    print("   :header-rows: 1")
    print("   :widths: 40 60")
    print()
    print("   * - StableHLO op (``stablehlo.<name>``)")
    print("     - ONNX equivalent")
    for jax_op in sorted(_COMPOSITE_JAX_OPS):
        hlo_url = _STABLEHLO_BASE.format(_hlo_anchor(jax_op))
        desc = _descriptions.get(jax_op, "*(see source)*")
        print(f"   * - `{jax_op} <{hlo_url}>`_")
        print(f"     - {desc}")
    print()

Structural ops
--------------

The following ``stablehlo`` ops are handled directly by the
``XlaCallModule`` converter in :mod:`yobx.tensorflow.ops.xla_call_module`
and do not go through :func:`~yobx.tensorflow.ops.jax_ops.get_jax_cvt`:

.. runpython::
    :showcode:
    :rst:

    from yobx.tensorflow.ops.xla_call_module import _STRUCTURAL_OPS

    _ONNX_BASE = "https://onnx.ai/onnx/operators/onnx__{}.html"
    _STABLEHLO_BASE = "https://openxla.org/stablehlo/spec#{}"

    def _hlo_anchor(op):
        # Use generic anchors for ops that share a spec section
        _overrides = {
            "reduce_max": "reduce",
            "reduce_sum": "reduce",
        }
        return _overrides.get(op, op.replace("_", "-"))

    print(".. list-table::")
    print("   :header-rows: 1")
    print("   :widths: 40 60")
    print()
    print("   * - StableHLO op")
    print("     - ONNX equivalent")
    for hlo_op, (onnx_op, desc) in sorted(_STRUCTURAL_OPS.items()):
        hlo_url = _STABLEHLO_BASE.format(_hlo_anchor(hlo_op))
        hlo_cell = f"`{hlo_op} <{hlo_url}>`_"
        if onnx_op is not None:
            onnx_url = _ONNX_BASE.format(onnx_op)
            onnx_cell = f"`{onnx_op} <{onnx_url}>`_ — {desc}"
        else:
            onnx_cell = desc
        print(f"   * - {hlo_cell}")
        print(f"     - {onnx_cell}")
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

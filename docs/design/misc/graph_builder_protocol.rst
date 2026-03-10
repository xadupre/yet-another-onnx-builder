.. _l-design-graph-builder-extended-protocol:

========================================
Alternative GraphBuilderExtendedProtocol
========================================

:class:`GraphBuilderExtendedProtocol <yobx.typing.GraphBuilderExtendedProtocol>`
is the interface that every graph builder used by the :mod:`yobx.sklearn`
converters must satisfy.  The package ships with three concrete
implementations and makes it easy to add more:

* :class:`GraphBuilder <yobx.xbuilder.GraphBuilder>` — the default;
  builds graphs using onnx protobuf objects with built-in optimisation passes.
* :class:`OnnxScriptGraphBuilder <yobx.builder.onnxscript.OnnxScriptGraphBuilder>`
  — delegates graph construction to the ``onnxscript`` IR.
* :class:`SpoxGraphBuilder <yobx.builder.spox.SpoxGraphBuilder>`
  — delegates graph construction to the :epkg:`spox` library.

Why provide alternatives?
=========================

Keeping the builders behind a protocol rather than inheriting from a
single base class means that any third-party library can supply its own
builder.  Some reasons for doing so:

* **Better IDE / type support** — :epkg:`spox` and ``onnxscript`` both
  use strongly-typed, opset-versioned Python functions so mistakes are
  caught statically rather than at runtime.
* **Validation on construction** — spox validates the graph structure
  incrementally, so type errors surface when a node is added rather than
  at export time.
* **Integration into an existing IR pipeline** — if the rest of the
  workflow already works with ``onnxscript``'s :class:`ir.Model`, it is
  more convenient to accumulate nodes there directly and avoid a
  round-trip through :class:`onnx.ModelProto`.

Protocol overview
=================

:class:`GraphBuilderExtendedProtocol <yobx.typing.GraphBuilderExtendedProtocol>`
extends :class:`GraphBuilderProtocol <yobx.typing.GraphBuilderProtocol>` with
three additional members required by the converters:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Member
     - Purpose
   * - ``main_opset``
     - Read-only property.  Returns the opset version for the main ONNX
       domain (``""``).
   * - ``op``
     - Returns an :class:`OpsetProtocol <yobx.typing.OpsetProtocol>`
       helper.  Converters use ``g.op.Relu(x)`` as short-hand for
       ``g.make_node("Relu", [x], 1)``.
   * - ``set_type_shape_unary_op(name, input_name, itype)``
     - Propagates type and shape from *input_name* to *name* for
       elementwise unary operators.
   * - ``unique_name(prefix)``
     - Returns a name that starts with *prefix* and is not yet used
       in the graph.
   * - ``get_debug_msg()``
     - Returns a string with diagnostic context that is included in
       exceptions raised during conversion.

See :ref:`l-design-expected-api` for the full list of methods inherited
from :class:`GraphBuilderProtocol <yobx.typing.GraphBuilderProtocol>`.

Using OnnxScriptGraphBuilder
============================

:class:`OnnxScriptGraphBuilder <yobx.builder.onnxscript.OnnxScriptGraphBuilder>`
is a bridge that builds an ``onnxscript`` :class:`ir.Model` internally
while presenting the same string-based API to converters.

.. runpython::
    :showcode:

    import numpy as np
    import onnx
    from sklearn.preprocessing import StandardScaler
    from yobx.sklearn import to_onnx
    from yobx.builder.onnxscript import OnnxScriptGraphBuilder
    from yobx.helpers.onnx_helper import pretty_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4)).astype(np.float32)

    scaler = StandardScaler().fit(X)
    model = to_onnx(scaler, (X,), builder_cls=OnnxScriptGraphBuilder)
    print(pretty_onnx(model))

Using SpoxGraphBuilder
======================

:class:`SpoxGraphBuilder <yobx.builder.spox.SpoxGraphBuilder>` is a bridge
that delegates every operator call to the matching :epkg:`spox` opset
module, providing static type-checking and incremental graph validation.
The only change relative to the default workflow is passing
``builder_cls=SpoxGraphBuilder`` to :func:`yobx.sklearn.to_onnx`:

.. runpython::
    :showcode:

    import numpy as np
    import onnx
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from yobx.sklearn import to_onnx
    from yobx.builder.spox import SpoxGraphBuilder
    from yobx.helpers.onnx_helper import pretty_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipe.fit(X, y)

    model = to_onnx(pipe, (X[:1],), builder_cls=SpoxGraphBuilder)
    print(pretty_onnx(model))

.. seealso::

    :ref:`l-design-expected-api` — the full list of methods and attributes
    every builder must expose.

    :class:`SpoxGraphBuilder <yobx.builder.spox.SpoxGraphBuilder>` — a
    complete, production-quality alternative implementation backed by
    :epkg:`spox`.

    :class:`OnnxScriptGraphBuilder <yobx.builder.onnxscript.OnnxScriptGraphBuilder>`
    — a complete alternative backed by the ``onnxscript`` IR.

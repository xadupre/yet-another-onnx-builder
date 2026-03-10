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

Writing a custom implementation
================================

Any class that satisfies
:class:`GraphBuilderExtendedProtocol <yobx.typing.GraphBuilderExtendedProtocol>`
can be passed as ``builder_cls``.  The minimal skeleton below shows the
structure; see :class:`SpoxGraphBuilder
<yobx.builder.spox.SpoxGraphBuilder>` or :class:`OnnxScriptGraphBuilder
<yobx.builder.onnxscript.OnnxScriptGraphBuilder>` for complete working
examples.

.. code-block:: python

    from typing import Any, Dict, List, Optional, Tuple, Union
    import onnx
    from yobx.typing import GraphBuilderExtendedProtocol, OpsetProtocol

    class MyOpset:
        def __init__(self, builder: "MyGraphBuilder") -> None:
            self._b = builder

        def __getattr__(self, op_type: str):
            def _call(*inputs, **attrs):
                return self._b.make_node(op_type, list(inputs), **attrs)
            return _call

    class MyGraphBuilder(GraphBuilderExtendedProtocol):
        def __init__(self, target_opset: int) -> None:
            self.opsets: Dict[str, int] = {"": target_opset}
            self._input_names: List[str] = []
            self._output_names: List[str] = []
            self._type_map: Dict[str, int] = {}
            self._shape_map: Dict[str, Any] = {}
            self._counter: int = 0
            self._op = MyOpset(self)

        # ---- GraphBuilderExtendedProtocol extras ----

        @property
        def main_opset(self) -> int:
            return self.opsets[""]

        @property
        def op(self) -> OpsetProtocol:
            return self._op  # type: ignore[return-value]

        def unique_name(self, prefix: str) -> str:
            self._counter += 1
            return f"{prefix}_{self._counter}"

        def set_type_shape_unary_op(self, name, input_name, itype=None):
            if self.has_type(input_name):
                self.set_type(name, itype or self.get_type(input_name))
            if self.has_shape(input_name):
                self.set_shape(name, self.get_shape(input_name))

        def get_debug_msg(self) -> str:
            return f"MyGraphBuilder opsets={self.opsets}"

        # ---- GraphBuilderProtocol opset management ----

        @property
        def input_names(self) -> List[str]:
            return list(self._input_names)

        @property
        def output_names(self) -> List[str]:
            return list(self._output_names)

        def get_opset(self, domain: str, exc: bool = True) -> int:
            if exc:
                assert domain in self.opsets, f"Unknown domain {domain!r}."
            return self.opsets.get(domain, 0)

        def set_opset(self, domain: str, version: int = 1) -> None:
            self.opsets[domain] = version

        def has_opset(self, domain: str) -> int:
            return self.opsets.get(domain, 0)

        # ---- type / shape side-channel ----

        def has_name(self, name: str) -> bool:
            return name in self._type_map or name in self._input_names

        def has_type(self, name: str) -> bool:
            return name in self._type_map

        def get_type(self, name: str) -> int:
            return self._type_map[name]

        def set_type(self, name: str, itype: int) -> None:
            self._type_map[name] = itype

        def has_shape(self, name: str) -> bool:
            return name in self._shape_map

        def get_shape(self, name: str) -> Tuple:
            return self._shape_map[name]

        def set_shape(self, name, shape, allow_zero=False) -> None:
            self._shape_map[name] = shape

        # ---- construction API ----

        def make_tensor_input(self, name, elem_type=None, shape=None, device=None):
            self._input_names.append(name)
            if elem_type is not None:
                self.set_type(name, elem_type)
            if shape is not None:
                self.set_shape(name, shape)
            return name

        def make_tensor_output(self, name, elem_type=None, shape=None,
                               indexed=False, allow_untyped_output=False):
            names = [name] if isinstance(name, str) else name
            self._output_names.extend(names)
            return name

        def make_initializer(self, name: str, value: Any) -> str:
            # store *value* as a constant — omitted for brevity
            return name

        def make_node(self, op_type, inputs, outputs=1, domain="",
                      attributes=None, name=None, **kwargs):
            # build and store a node — omitted for brevity
            result = self.unique_name(op_type.lower())
            return result

        def to_onnx(self) -> onnx.ModelProto:
            # assemble the stored nodes into an onnx.ModelProto — omitted
            raise NotImplementedError("implement graph assembly here")

Registering the custom builder for protocol conformance
--------------------------------------------------------

At runtime, :func:`isinstance(g, GraphBuilderExtendedProtocol)` relies on
Python's :mod:`typing.Protocol` structural check.  Because
:class:`GraphBuilderExtendedProtocol <yobx.typing.GraphBuilderExtendedProtocol>`
is decorated with ``@runtime_checkable``, the check passes as long as all
required methods are present (though it only inspects callable members, not
full signatures):

.. code-block:: python

    from yobx.typing import GraphBuilderExtendedProtocol

    g = MyGraphBuilder(18)
    assert isinstance(g, GraphBuilderExtendedProtocol)

.. seealso::

    :ref:`l-design-expected-api` — the full list of methods and attributes
    every builder must expose.

    :class:`SpoxGraphBuilder <yobx.builder.spox.SpoxGraphBuilder>` — a
    complete, production-quality alternative implementation backed by
    :epkg:`spox`.

    :class:`OnnxScriptGraphBuilder <yobx.builder.onnxscript.OnnxScriptGraphBuilder>`
    — a complete alternative backed by the ``onnxscript`` IR.

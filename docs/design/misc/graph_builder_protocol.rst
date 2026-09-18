.. _l-design-graph-builder-extended-protocol:

========================================
Alternative GraphBuilderExtendedProtocol
========================================

:class:`GraphBuilderExtendedProtocol <yobx.typing.GraphBuilderExtendedProtocol>`
is the interface that every graph builder used by the :mod:`yobx.sklearn`
converters must satisfy. The supported implementation is
:class:`GraphBuilder <yobx.xbuilder.GraphBuilder>`, which builds graphs using
native ``onnx-light`` protobuf objects, shape inference, and optimization.

The historical
:class:`OnnxScriptGraphBuilder <yobx.builder.onnxscript.OnnxScriptGraphBuilder>`
and :class:`SpoxGraphBuilder <yobx.builder.spox.SpoxGraphBuilder>` bridges
require the removed reference ONNX dependency. Their constructors now raise
``NotImplementedError`` rather than installing or falling back to that runtime.
Existing callers should select the native builder as shown below.

Why provide alternatives?
=========================

Keeping the builders behind a protocol rather than inheriting from a
single base class means that any third-party library can supply its own
builder. Implementations can provide stronger type checking, validate nodes
as they are added, or integrate an existing graph representation.
They must implement the complete protocol and return native ``onnx-light``
protobufs; the retired bridges are not working alternatives.


Selecting the native builder
============================

The ``builder_cls`` argument selects the implementation explicitly.
Passing :class:`GraphBuilder <yobx.xbuilder.GraphBuilder>` is equivalent
to using the default converter configuration.

.. runpython::
    :showcode:

    import numpy as np
    from yobx._onnx_shim import onnx
    from sklearn.preprocessing import StandardScaler
    from yobx.sklearn import to_onnx
    from yobx.xbuilder import GraphBuilder
    from yobx.helpers.onnx_helper import pretty_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4)).astype(np.float32)

    scaler = StandardScaler().fit(X)
    model = to_onnx(scaler, (X,), builder_cls=GraphBuilder)
    print(pretty_onnx(model))

Converting a pipeline
=====================

The same native implementation handles complete pipelines through
:func:`yobx.sklearn.to_onnx`, without an intermediate external IR:

.. runpython::
    :showcode:

    import numpy as np
    from yobx._onnx_shim import onnx
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from yobx.sklearn import to_onnx
    from yobx.xbuilder import GraphBuilder
    from yobx.helpers.onnx_helper import pretty_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipe.fit(X, y)

    model = to_onnx(pipe, (X[:1],), builder_cls=GraphBuilder)
    print(pretty_onnx(model))

.. seealso::

    :ref:`l-design-expected-api` — the full list of methods and attributes
    every builder must expose.

    :ref:`l-design-graph-builder` — native graph construction and optimization.

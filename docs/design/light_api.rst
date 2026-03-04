
.. _l-design-light-api:

==============================
Fluent ONNX Builder (Light API)
==============================

:mod:`yobx.builder.light` provides a chainable, expression-oriented API for
building ONNX graphs without writing protobuf boilerplate.  It is designed for
small models, test fixtures, and quick experimentation.

Core classes
============

The module exposes three core abstractions:

* :class:`OnnxGraph <yobx.builder.light._graph.OnnxGraph>` — accumulates
  nodes, inputs, outputs, and initializers.  Created via :func:`start
  <yobx.builder.light.start>` (``ModelProto`` output) or :func:`g
  <yobx.builder.light.g>` (``GraphProto`` output for subgraphs).
* :class:`Var <yobx.builder.light._var.Var>` — represents a single tensor
  value (graph input, initializer, or node output).  Supports Python operator
  overloads (``+``, ``-``, ``*``, ``/``, ``@``, etc.) and every standard ONNX
  operator as a method (``Relu()``, ``MatMul()``, …).
* :class:`Vars <yobx.builder.light._var.Vars>` — a tuple of :class:`Var`
  objects returned when a node produces multiple outputs (e.g. ``Split``,
  ``TopK``).

Building a simple model
=======================

The minimal workflow is a single chained expression starting from
:func:`start() <yobx.builder.light.start>`:

.. runpython::
    :showcode:

    from yobx.builder.light import start

    # Build Y = Neg(X)
    onx = start().vin("X").Neg().rename("Y").vout().to_onnx()
    print(f"inputs : {[i.name for i in onx.graph.input]}")
    print(f"outputs: {[o.name for o in onx.graph.output]}")
    print(f"nodes  : {[n.op_type for n in onx.graph.node]}")

Each call in the chain returns either an :class:`OnnxGraph` (for methods
like :meth:`vin <yobx.builder.light._var.BaseVar.vin>`) or a :class:`Var`
(for operator methods and :meth:`rename <yobx.builder.light._var.Var.rename>`).
Calling :meth:`to_onnx <yobx.builder.light._graph.OnnxGraph.to_onnx>` (or its
shortcut on :class:`Var`) finalizes the model.

Two-input models
================

When multiple inputs are needed, call :meth:`vin
<yobx.builder.light._var.BaseVar.vin>` for each input and then combine them
with :meth:`bring <yobx.builder.light._var.BaseVar.bring>` before applying
the operator:

.. runpython::
    :showcode:

    from yobx.builder.light import start

    onx = (
        start()
        .vin("X")
        .vin("Y")
        .bring("X", "Y")
        .Add()
        .rename("Z")
        .vout()
        .to_onnx()
    )
    print(f"inputs : {[i.name for i in onx.graph.input]}")
    print(f"outputs: {[o.name for o in onx.graph.output]}")

Python operator overloads
=========================

:class:`Var` supports the standard Python arithmetic operators so that
expressions closely mirror the underlying math:

.. runpython::
    :showcode:

    import numpy as np
    from yobx.builder.light import start

    gr = start()
    x = gr.vin("X")
    y = gr.vin("Y")
    bias = gr.cst(np.ones(4, dtype=np.float32), "bias")

    # (X * Y) + bias → renamed to Z, declared as graph output
    (x * y + bias).rename("Z").vout()

    onx = gr.to_onnx()
    print(f"nodes  : {[n.op_type for n in onx.graph.node]}")
    print(f"outputs: {[o.name for o in onx.graph.output]}")

Constants and initializers
==========================

:meth:`cst <yobx.builder.light._var.BaseVar.cst>` adds a numpy array as a
graph initializer and returns a :class:`Var` pointing to it:

.. runpython::
    :showcode:

    import numpy as np
    from yobx.builder.light import start

    gr = start()
    x = gr.vin("X")
    w = gr.cst(np.random.randn(4, 2).astype(np.float32), "W")
    (x @ w).rename("Y").vout()

    onx = gr.to_onnx()
    print(f"initializers: {[i.name for i in onx.graph.initializer]}")
    print(f"nodes       : {[n.op_type for n in onx.graph.node]}")

Multiple outputs
================

Operators that produce more than one tensor return a :class:`Vars`
object.  Individual outputs are accessed by indexing.
:meth:`Unique <yobx.builder.light._op_var.OpsVar.Unique>` is an example of
such an operator — it returns unique values, indices, inverse indices, and
counts as four separate tensors:

.. runpython::
    :showcode:

    from yobx.builder.light import start

    gr = start()
    x = gr.vin("X")
    parts = x.Unique(axis=0, sorted=1)  # returns Vars with 4 outputs
    parts[0].rename("vals").vout()      # unique values
    parts[1].rename("inds").vout()      # indices

    onx = gr.to_onnx()
    print(f"outputs: {[o.name for o in onx.graph.output]}")

Subgraphs
=========

Use :func:`g() <yobx.builder.light.g>` to build a ``GraphProto`` for use
inside control-flow operators such as ``If``.  The graph is finalized with
:meth:`to_onnx <yobx.builder.light._graph.OnnxGraph.to_onnx>`, which returns
a :class:`onnx.GraphProto` in this mode:

.. code-block:: python

    from yobx.builder.light import g

    then_branch = g().vin("X").Relu().rename("Y").vout().to_onnx()
    else_branch = g().vin("X").Abs().rename("Y").vout().to_onnx()

.. seealso::

    :ref:`l-design-graph-builder` — the :class:`GraphBuilder
    <yobx.xbuilder.GraphBuilder>` API for building and optimizing ONNX graphs
    programmatically, which is more feature-rich but lower-level.

    :ref:`l-design-translate` — translating an existing ``onnx.ModelProto``
    back to Python source code using the light API or the ``onnx.helper``
    functions.


.. _l-design-onnx-inspection:

========================
ONNX Inspection Helpers
========================

``yobx`` ships a collection of utilities for **reading** and **understanding**
ONNX protobuf structures.  They are scattered across
:mod:`yobx.helpers.onnx_helper` and :mod:`yobx.helpers.helper` and are
primarily used internally by the builder, optimizer, and evaluator, but are
equally useful for debugging, testing, and quick exploration.

Printing models and nodes
=========================

:func:`~yobx.helpers.onnx_helper.pretty_onnx` converts any ONNX protobuf
object to a compact, human-readable text representation.  It works on
:class:`onnx.ModelProto`, :class:`onnx.GraphProto`,
:class:`onnx.FunctionProto`, :class:`onnx.NodeProto`,
:class:`onnx.TensorProto`, :class:`onnx.ValueInfoProto`, and
:class:`onnx.AttributeProto`.

.. runpython::
    :showcode:

    import onnx.helper as oh
    import onnx
    from yobx.helpers.onnx_helper import pretty_onnx

    TFLOAT = onnx.TensorProto.FLOAT
    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Relu", ["X"], ["Y"]),
                oh.make_node("Transpose", ["Y"], ["Z"], perm=[1, 0]),
            ],
            "relu_transpose",
            [oh.make_tensor_value_info("X", TFLOAT, [None, 4])],
            [oh.make_tensor_value_info("Z", TFLOAT, [4, None])],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )
    print(pretty_onnx(model))

For a single node, pass ``with_attributes=True`` to include attribute values:

.. runpython::
    :showcode:

    import onnx.helper as oh
    from yobx.helpers.onnx_helper import pretty_onnx

    node = oh.make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])
    print(pretty_onnx(node, with_attributes=True))

Inspecting shapes and types
===========================

:func:`~yobx.helpers.helper.string_type` produces a concise one-line
description of any Python object including nested structures of
:class:`torch.Tensor` and :class:`numpy.ndarray`.  It is used throughout
the library for error messages and debug logging.

.. runpython::
    :showcode:

    import numpy as np
    from yobx.helpers import string_type

    obj = [
        np.zeros((2, 4), dtype=np.float32),
        np.zeros((2, 4), dtype=np.float32),
    ]
    print(string_type(obj, with_shape=True))

Comparing numerical outputs
===========================

:func:`~yobx.helpers.helper.max_diff` computes the maximum absolute and
relative differences between two nested structures.  It is the primary
validation utility used when checking that an exported ONNX model produces
the same outputs as the original PyTorch model.

.. code-block:: python

    import numpy as np
    from yobx.helpers import max_diff

    ref = {"logits": np.array([[1.0, 2.0, 3.0]])}
    got = {"logits": np.array([[1.0, 2.0001, 3.0]])}

    diff = max_diff(ref, got)
    print("max absolute diff:", diff["abs"])
    print("max relative diff:", diff["rel"])

Walking subgraphs
=================

Several ONNX operators (``If``, ``Loop``, ``Scan``, ``SequenceMap``) embed
*subgraphs* as attributes.  Traversing a model fully therefore requires
recursing into these subgraphs.
:func:`~yobx.helpers.onnx_helper.enumerate_subgraphs` yields every embedded
:class:`onnx.GraphProto` in depth-first order, including the top-level graph:

.. runpython::
    :showcode:

    import onnx
    import onnx.helper as oh
    from yobx.helpers.onnx_helper import enumerate_subgraphs

    TFLOAT = onnx.TensorProto.FLOAT
    then_graph = oh.make_graph(
        [oh.make_node("Relu", ["X"], ["Y"])],
        "then",
        [],
        [oh.make_tensor_value_info("Y", TFLOAT, [None, 4])],
    )
    else_graph = oh.make_graph(
        [oh.make_node("Abs", ["X"], ["Y"])],
        "else",
        [],
        [oh.make_tensor_value_info("Y", TFLOAT, [None, 4])],
    )
    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node(
                    "If",
                    ["cond"],
                    ["Y"],
                    then_branch=then_graph,
                    else_branch=else_graph,
                )
            ],
            "if_model",
            [
                oh.make_tensor_value_info("cond", onnx.TensorProto.BOOL, []),
                oh.make_tensor_value_info("X", TFLOAT, [None, 4]),
            ],
            [oh.make_tensor_value_info("Y", TFLOAT, [None, 4])],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )
    graphs = list(enumerate_subgraphs(model.graph))
    print("number of graphs (including main):", len(graphs))
    print("graph names:", [g.name for g in graphs])

Flattening nested inputs
=========================

:func:`~yobx.helpers.helper.flatten_object` recursively flattens any nested
Python structure (dicts, lists, tuples, ``torch.Tensor``,
``numpy.ndarray``) into a single flat list of leaf tensors.  This is useful
when assembling the flat input list expected by
:class:`onnxruntime.InferenceSession`:

.. runpython::
    :showcode:

    import numpy as np
    from yobx.helpers.helper import flatten_object

    inputs = {
        "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
        "past_kv": [np.zeros((1, 4, 6, 8)), np.zeros((1, 4, 6, 8))],
    }
    flat = flatten_object(inputs, drop_keys=True)
    print("number of leaf tensors:", len(flat))
    for i, t in enumerate(flat):
        print(f"  [{i}]: shape={t.shape}, dtype={t.dtype}")

.. seealso::

    :ref:`l-design-helpers` — :class:`MiniOnnxBuilder
    <yobx.helpers.mini_onnx_builder.MiniOnnxBuilder>` for serializing nested
    tensor structures to ONNX initializers.

    :ref:`l-design-evaluator` — evaluator classes that consume ONNX models
    and produce outputs compatible with the utilities described above.

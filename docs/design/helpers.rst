.. _l-design-helpers:

===================
Interesting Helpers
===================

ONNX Serialization of Nested Structured with Tensors
====================================================

The main goal is to serialize any Python structure into ONNX
format. This relies on :class:`MiniOnnxBuilder <yobx.helpers.mini_onnx_builder.MiniOnnxBuilder>`.
Example :ref:`l-plot-mini-onnx-builder` shows an example.

:class:`MiniOnnxBuilder <yobx.helpers.mini_onnx_builder.MiniOnnxBuilder>`
creates minimal ONNX models whose only purpose is to **store tensors as
initializers** and return them when the model is executed.  The model has
**no inputs** — running it simply replays the stored values.  This is
useful for:

* capturing intermediate activations or model weights for debugging,
* persisting arbitrary nested Python structures (dicts, tuples, lists,
  torch tensors) in a standard, portable format,
* sharing small test fixtures without committing raw binary files.

The class exposes three methods to add outputs:

* :meth:`append_output_initializer
  <yobx.helpers.mini_onnx_builder.MiniOnnxBuilder.append_output_initializer>`
  — stores a single tensor (numpy array or torch tensor).  When
  ``randomize=True`` the values are replaced by a random-number generator
  node, keeping shape and dtype but discarding the original data to reduce
  model size.
* :meth:`append_output_sequence
  <yobx.helpers.mini_onnx_builder.MiniOnnxBuilder.append_output_sequence>`
  — wraps a list of tensors into an ONNX ``Sequence`` output.
* :meth:`append_output_dict
  <yobx.helpers.mini_onnx_builder.MiniOnnxBuilder.append_output_dict>`
  — stores a dict of tensors as two outputs (keys and values).

Two higher-level helpers build on top of
:class:`MiniOnnxBuilder <yobx.helpers.mini_onnx_builder.MiniOnnxBuilder>`
to handle arbitrary nesting automatically:

* :func:`create_onnx_model_from_input_tensors
  <yobx.helpers.mini_onnx_builder.create_onnx_model_from_input_tensors>`
  — serialize any nested structure to an ``onnx.ModelProto``.
* :func:`create_input_tensors_from_onnx_model
  <yobx.helpers.mini_onnx_builder.create_input_tensors_from_onnx_model>`
  — deserialize the model back to the original Python structure.

The following snippet shows a round-trip for a small nested input
dictionary:

.. runpython::
    :showcode:

    import numpy as np
    import torch
    from yobx.helpers.mini_onnx_builder import (
        create_onnx_model_from_input_tensors,
        create_input_tensors_from_onnx_model,
    )

    inputs = {
        "ids": np.array([1, 2, 3], dtype=np.int64),
        "hidden": torch.zeros(2, 4, dtype=torch.float32),
    }
    proto = create_onnx_model_from_input_tensors(inputs)
    restored = create_input_tensors_from_onnx_model(proto)
    print("keys:", list(restored.keys()))
    for k, v in inputs.items():
        arr = v if isinstance(v, np.ndarray) else v.numpy()
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")

The higher-level helpers also handle **deeply nested** structures.  The
next snippet serializes a dict whose values include a flat tensor *and* a
list of tensors (typical for past-key-value caches in transformer models):

.. runpython::
    :showcode:

    import numpy as np
    from yobx.helpers.mini_onnx_builder import (
        create_onnx_model_from_input_tensors,
        create_input_tensors_from_onnx_model,
    )

    inputs = {
        "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
        "past_key_values": [
            np.zeros((1, 4, 6, 8), dtype=np.float32),   # layer 0 keys
            np.zeros((1, 4, 6, 8), dtype=np.float32),   # layer 0 values
        ],
    }

    proto = create_onnx_model_from_input_tensors(inputs)
    restored = create_input_tensors_from_onnx_model(proto)

    print("top-level keys:", list(restored.keys()))
    print("input_ids     :", restored["input_ids"].shape)
    print("past_key_values is a", type(restored["past_key_values"]).__name__,
          "of length", len(restored["past_key_values"]))
    for i, arr in enumerate(restored["past_key_values"]):
        print(f"  [{i}]: shape={arr.shape}")

ONNX Graph Visualization
========================

:func:`to_dot <yobx.helpers.dot_helper.to_dot>` converts an
:class:`onnx.ModelProto` into a `DOT <https://graphviz.org/doc/info/lang.html>`_
string suitable for rendering with `Graphviz <https://graphviz.org/>`_.

.. runpython::
    :showcode:

    import numpy as np
    import onnx
    import onnx.helper as oh
    import onnx.numpy_helper as onh
    from yobx.helpers.dot_helper import to_dot

    TFLOAT = onnx.TensorProto.FLOAT
    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Add", ["X", "Y"], ["added"]),
                oh.make_node("MatMul", ["added", "W"], ["mm"]),
                oh.make_node("Relu", ["mm"], ["Z"]),
            ],
            "add_matmul_relu",
            [
                oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq", 4]),
                oh.make_tensor_value_info("Y", TFLOAT, ["batch", "seq", 4]),
            ],
            [oh.make_tensor_value_info("Z", TFLOAT, ["batch", "seq", 2])],
            [
                onh.from_array(
                    np.zeros((4, 2), dtype=np.float32),
                    name="W",
                )
            ],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )
    dot = to_dot(model)
    print(dot[:200], "...")

The resulting DOT source can be rendered directly in the documentation with the
``gdot`` directive from :epkg:`sphinx-runpython`:

.. gdot::
    :script: DOT-SECTION

    import numpy as np
    import onnx
    import onnx.helper as oh
    import onnx.numpy_helper as onh
    from yobx.helpers.dot_helper import to_dot

    TFLOAT = onnx.TensorProto.FLOAT
    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Add", ["X", "Y"], ["added"]),
                oh.make_node("MatMul", ["added", "W"], ["mm"]),
                oh.make_node("Relu", ["mm"], ["Z"]),
            ],
            "add_matmul_relu",
            [
                oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq", 4]),
                oh.make_tensor_value_info("Y", TFLOAT, ["batch", "seq", 4]),
            ],
            [oh.make_tensor_value_info("Z", TFLOAT, ["batch", "seq", 2])],
            [
                onh.from_array(
                    np.zeros((4, 2), dtype=np.float32),
                    name="W",
                )
            ],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )
    dot = to_dot(model)
    print("DOT-SECTION", dot)

.. seealso::

    :ref:`l-plot-dot-graph` — sphinx-gallery example demonstrating
    :func:`to_dot <yobx.helpers.dot_helper.to_dot>` on a simple hand-built model.


.. _l-design-container:

=========================
ExtendedModelContainer
=========================

:class:`ExtendedModelContainer <yobx.container.model_container.ExtendedModelContainer>`
extends :class:`onnx.model_container.ModelContainer` to handle large ONNX models
whose weight tensors are stored **outside** the main ``.onnx`` file.

Motivation
==========

Standard ONNX files embed every initializer directly in the protobuf.  For
large models — those whose weights reach several gigabytes — this approach
becomes impractical: the protobuf cannot exceed 2 GB and loading everything
into memory before running the model is wasteful.

ONNX addresses this via the *external data* mechanism: an initializer can
reference a separate binary file instead of carrying its bytes inline.
:class:`onnx.model_container.ModelContainer` provides the basic scaffolding
for this pattern; :class:`ExtendedModelContainer
<yobx.container.model_container.ExtendedModelContainer>` builds on top of it
with two important additions:

* **PyTorch tensor support** — large initializers may be
  :class:`torch.Tensor` or :class:`torch.nn.Parameter` objects in addition
  to :class:`numpy.ndarray`.  The serialization path converts them
  transparently so callers do not have to worry about dtype mismatches.
* **Inline local functions** — setting ``container.inline = True`` inlines
  every local function defined in the model before writing it to disk, which
  reduces runtime overhead for exporters that emit function-heavy graphs.

How it works
============

An :class:`ExtendedModelContainer
<yobx.container.model_container.ExtendedModelContainer>` has two main
attributes:

* ``model_proto`` — the :class:`onnx.ModelProto` describing the graph.
  Large initializers appear here with ``data_location = EXTERNAL`` and a
  ``location`` key that acts as a *symbolic handle* (e.g. ``"#weight"``).
* ``large_initializers`` — a plain Python :class:`dict` mapping each
  symbolic handle to the actual tensor data
  (:class:`numpy.ndarray`, :class:`torch.Tensor`, or
  :class:`onnx.TensorProto`).

Save
----

:meth:`save <yobx.container.model_container.ExtendedModelContainer.save>`
iterates over every external tensor in the proto, looks up the corresponding
data in ``large_initializers``, converts it to raw bytes, and writes it to
disk.  Two layouts are supported:

* ``all_tensors_to_one_file=True`` *(default)* — all weights are packed into
  a single ``<model>.onnx.data`` file.  Each tensor's ``offset`` and
  ``length`` fields are updated in the copy of the proto that is saved.
* ``all_tensors_to_one_file=False`` — each weight gets its own
  ``<name>.weight`` file sitting next to the ``.onnx`` file.

The method returns a (possibly modified) copy of ``model_proto`` that
contains the final file locations so the caller can inspect or validate it.

Load
----

:meth:`load <yobx.container.model_container.ExtendedModelContainer.load>`
calls :func:`onnx.load` with ``load_external_data=False`` to read the graph
structure, then calls the parent's ``_load_large_initializers`` to
memory-map / load the weight files referenced by the proto.

Basic example
=============

The snippet below builds a tiny ONNX model that has one external initializer,
saves it through :class:`ExtendedModelContainer
<yobx.container.model_container.ExtendedModelContainer>`, and reloads it.

.. runpython::
    :showcode:

    import os
    import tempfile
    import numpy as np
    import onnx
    import onnx.helper as oh
    from yobx.container import ExtendedModelContainer

    # ---- Build a minimal model with one external initializer ----
    data = np.ones((4, 8), dtype=np.float32)

    x_info = oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [4, 8])
    y_info = oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [4, 8])

    # Declare the initializer as external (symbolic location "#weight")
    init = onnx.TensorProto()
    init.data_type = onnx.TensorProto.FLOAT
    init.name = "weight"
    init.data_location = onnx.TensorProto.EXTERNAL
    for d in data.shape:
        init.dims.append(d)
    ext = init.external_data.add()
    ext.key = "location"
    ext.value = "#weight"

    graph = oh.make_graph(
        [oh.make_node("Add", ["x", "weight"], ["y"])],
        "demo",
        [x_info],
        [y_info],
        initializer=[init],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])

    # ---- Populate the container and save ----
    container = ExtendedModelContainer()
    container.model_proto = model
    container.large_initializers = {"#weight": data}

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "demo.onnx")
        saved_proto = container.save(path)
        print("Saved proto type :", type(saved_proto).__name__)
        print("Files written     :", sorted(os.listdir(tmp)))

        # ---- Round-trip: reload from disk ----
        loaded = ExtendedModelContainer().load(path)
        weight = next(iter(loaded.large_initializers.values()))
        print("Loaded weight shape:", weight.shape)
        print("Loaded weight dtype:", weight.dtype)

PyTorch tensor example
======================

When large initializers are :class:`torch.Tensor` objects the container
serializes them automatically.  This is the typical workflow when exporting
a PyTorch model to ONNX and deferring disk writes to a later stage.

.. runpython::
    :showcode:

    import os
    import tempfile
    import numpy as np
    import onnx
    import onnx.helper as oh

    try:
        import torch
        from yobx.container import ExtendedModelContainer

        data = torch.ones(4, 8, dtype=torch.float32)

        x_info = oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [4, 8])
        y_info = oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [4, 8])

        init = onnx.TensorProto()
        init.data_type = onnx.TensorProto.FLOAT
        init.name = "weight"
        init.data_location = onnx.TensorProto.EXTERNAL
        for d in data.shape:
            init.dims.append(d)
        ext = init.external_data.add()
        ext.key = "location"
        ext.value = "#weight"

        graph = oh.make_graph(
            [oh.make_node("Add", ["x", "weight"], ["y"])],
            "demo_torch",
            [x_info],
            [y_info],
            initializer=[init],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])

        container = ExtendedModelContainer()
        container.model_proto = model
        container.large_initializers = {"#weight": data}

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "demo_torch.onnx")
            container.save(path)
            print("Files written:", sorted(os.listdir(tmp)))

            loaded = ExtendedModelContainer().load(path)
            weight = next(iter(loaded.large_initializers.values()))
            print("Reloaded weight shape:", weight.shape)
    except ImportError:
        print("torch is not available — skipping PyTorch example")

.. seealso::

    :class:`ExtendedModelContainer API reference
    <yobx.container.model_container.ExtendedModelContainer>`

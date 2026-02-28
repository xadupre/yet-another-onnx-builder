yobx.builder.light
==================

Fluent API for building ONNX graphs using a chainable, builder-style syntax.
Inspired by `onnx-array-api/light_api <https://github.com/sdpython/onnx-array-api/tree/main/onnx_array_api/light_api>`_.

.. rubric:: Quick Start

.. code-block:: python

    from yobx.builder.light import start

    # Single-input model: Y = Neg(X)
    onx = start().vin("X").Neg().rename("Y").vout().to_onnx()

    # Two-input model: Z = Add(X, Y)
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

    # Python operator overloads
    import numpy as np
    gr = start()
    x, y = gr.vin("X"), gr.vin("Y")
    (x * y + gr.cst(np.ones(4, dtype=np.float32), "bias")).rename("Z").vout()
    onx = gr.to_onnx()

.. rubric:: Entry Points

.. autofunction:: yobx.builder.light.start

.. autofunction:: yobx.builder.light.g

.. toctree::
    :maxdepth: 1
    :caption: modules

    graph
    var
    op_var
    op_vars

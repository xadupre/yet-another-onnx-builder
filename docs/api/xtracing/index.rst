yobx.xtracing
=============

Lightweight mechanism for tracing numpy functions and exporting them to ONNX.
See :ref:`l-design-function-transformer-tracing` for a full walkthrough.

.. toctree::
    :maxdepth: 1
    :caption: modules

    numpy_array
    tracing

NumpyArray
++++++++++

.. autoclass:: yobx.xtracing.NumpyArray
    :members:
    :no-undoc-members:

trace_numpy_function
++++++++++++++++++++

.. autofunction:: yobx.xtracing.trace_numpy_function

trace_numpy_to_onnx
+++++++++++++++++++

.. autofunction:: yobx.xtracing.trace_numpy_to_onnx

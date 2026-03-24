yobx.xtracing
=============

Lightweight mechanism for tracing numpy and DataFrame functions and exporting them to ONNX.
See :ref:`l-design-function-transformer-tracing` for a full walkthrough.

.. toctree::
    :maxdepth: 1
    :caption: modules

    numpy_array
    tracing
    dataframe_trace

trace_numpy_function
++++++++++++++++++++

.. autofunction:: yobx.xtracing.trace_numpy_function

trace_numpy_to_onnx
+++++++++++++++++++

.. autofunction:: yobx.xtracing.trace_numpy_to_onnx

trace_dataframe
+++++++++++++++

.. autofunction:: yobx.xtracing.trace_dataframe

dataframe_to_onnx
+++++++++++++++++

.. autofunction:: yobx.xtracing.dataframe_to_onnx

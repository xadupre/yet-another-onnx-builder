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
    parse

NumpyArray
++++++++++

.. autoclass:: yobx.xtracing.NumpyArray
    :members:
    :no-undoc-members:

trace_numpy_function
++++++++++++++++++++

.. autofunction:: yobx.xtracing.trace_numpy_function

trace_dataframe
+++++++++++++++

.. autofunction:: yobx.xtracing.trace_dataframe

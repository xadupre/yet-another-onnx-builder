.. _l-design-xtracing:

Numpy Tracing
=============

This section covers the numpy-tracing infrastructure that converts plain
numpy functions into ONNX graphs — a core building block used by multiple
converters (e.g. :class:`~sklearn.preprocessing.FunctionTransformer` and
the SQL converter's custom-function support via :mod:`yobx.sql`) and
available as a standalone tool.

.. toctree::
   :maxdepth: 1

   numpy_tracing

yobx.sql
========

SQL-to-ONNX converter: convert SQL queries into ONNX graphs.

Every column referenced in the query is treated as a **distinct 1-D ONNX
input** tensor.  See :ref:`l-design-sql` for the full design discussion.

.. toctree::
    :maxdepth: 1
    :caption: main functions

    dataframe_to_onnx
    sql_to_onnx
    sql_to_onnx_graph
    parsed_query_to_onnx
    parsed_query_to_onnx_graph
    lazyframe_to_onnx
    trace_numpy_to_onnx
    to_onnx

.. toctree::
    :maxdepth: 1
    :caption: modules

    sql_convert
    polars_convert
    convert
    _expr
    ops/index


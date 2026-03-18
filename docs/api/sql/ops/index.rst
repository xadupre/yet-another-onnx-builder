yobx.sql.ops
============

Registration mechanism and built-in converters for SQL operation types.

Importing :mod:`yobx.sql.ops` automatically registers all built-in SQL op
converters.  Third-party code can add new converters by decorating a function
with :func:`~yobx.sql.ops.register.register_sql_op_converter`.

.. toctree::
    :maxdepth: 1
    :caption: modules

    register
    filter_op
    join_op

SQL_OP_CONVERTERS
+++++++++++++++++

.. autodata:: yobx.sql.ops.register.SQL_OP_CONVERTERS

register_sql_op_converter
+++++++++++++++++++++++++

.. autofunction:: yobx.sql.ops.register.register_sql_op_converter

get_sql_op_converter
++++++++++++++++++++

.. autofunction:: yobx.sql.ops.register.get_sql_op_converter

get_sql_op_converters
+++++++++++++++++++++

.. autofunction:: yobx.sql.ops.register.get_sql_op_converters

convert_filter_op
+++++++++++++++++

.. autofunction:: yobx.sql.ops.filter_op.convert_filter_op

convert_join_op
+++++++++++++++

.. autofunction:: yobx.sql.ops.join_op.convert_join_op

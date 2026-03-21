yobx.sql.ops
============

Registration mechanism and built-in converters for SQL operation types.

Importing :mod:`yobx.sql.ops` automatically registers all built-in SQL op
converters.  Third-party code can add new converters by decorating a function
with :func:`~yobx.sql.ops.register.register_sql_op_converter`.

.. toctree::
    :maxdepth: 1

    register
    filter_op
    join_op

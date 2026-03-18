yobx.sql
========

SQL-to-ONNX converter: convert SQL queries into ONNX graphs.

Every column referenced in the query is treated as a **distinct 1-D ONNX
input** tensor.  See :ref:`l-design-sql` for the full design discussion.

.. toctree::
    :maxdepth: 1
    :caption: modules

    parse
    convert

sql_to_onnx
+++++++++++

.. autofunction:: yobx.sql.sql_to_onnx

parse_sql
+++++++++

.. autofunction:: yobx.sql.parse.parse_sql

ParsedQuery
+++++++++++

.. autoclass:: yobx.sql.parse.ParsedQuery
    :members:
    :no-undoc-members:

SqlOperation subclasses
+++++++++++++++++++++++

.. autoclass:: yobx.sql.parse.SelectOp
    :members:
    :no-undoc-members:

.. autoclass:: yobx.sql.parse.FilterOp
    :members:
    :no-undoc-members:

.. autoclass:: yobx.sql.parse.GroupByOp
    :members:
    :no-undoc-members:

.. autoclass:: yobx.sql.parse.JoinOp
    :members:
    :no-undoc-members:

Expression nodes
++++++++++++++++

.. autoclass:: yobx.sql.parse.SelectItem
    :members:
    :no-undoc-members:

.. autoclass:: yobx.sql.parse.ColumnRef
    :members:
    :no-undoc-members:

.. autoclass:: yobx.sql.parse.Literal
    :members:
    :no-undoc-members:

.. autoclass:: yobx.sql.parse.BinaryExpr
    :members:
    :no-undoc-members:

.. autoclass:: yobx.sql.parse.AggExpr
    :members:
    :no-undoc-members:

.. autoclass:: yobx.sql.parse.Condition
    :members:
    :no-undoc-members:

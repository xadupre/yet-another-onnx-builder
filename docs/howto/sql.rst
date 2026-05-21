.. _l-howto-sql:

SQL conversion
==============

This page answers common *"how do I…"* questions for converting SQL and
:epkg:`polars` lazy plans to ONNX with :func:`yobx.sql.to_onnx`.

----

How to convert a SQL query string
---------------------------------

Pass the SQL query and the input dtypes to :func:`yobx.sql.to_onnx`.

.. runpython::
    :showcode:

    import numpy as np
    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.sql import to_onnx

    dtypes = {"a": np.float32, "b": np.float32}
    onx = to_onnx("SELECT a + b AS total FROM t WHERE a > 0", dtypes)
    print(pretty_onnx(onx))

----

How to convert a SQL JOIN query
-------------------------------

Use :func:`yobx.sql.sql_to_onnx` and pass the right-table dtypes through
``right_input_dtypes``.

.. runpython::
    :showcode:

    import numpy as np
    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.sql import sql_to_onnx

    query = "SELECT a.x, b.y FROM a JOIN b ON a.id = b.rid"
    left_dtypes = {"id": np.int64, "x": np.float32}
    right_dtypes = {"rid": np.int64, "y": np.float32}
    onx = sql_to_onnx(query, left_dtypes, right_input_dtypes=right_dtypes)
    print(pretty_onnx(onx))

----

How to convert a polars LazyFrame
---------------------------------

The same :func:`yobx.sql.to_onnx` entrypoint also accepts a
:class:`polars.LazyFrame`.

.. runpython::
    :showcode:

    import numpy as np
    import polars as pl
    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.sql import to_onnx

    lf = pl.LazyFrame({"a": [1.0, -2.0, 3.0], "b": [4.0, 5.0, 6.0]}).select(
        [(pl.col("a") + pl.col("b")).alias("total")]
    )
    dtypes = {"a": np.float64, "b": np.float64}
    onx = to_onnx(lf, dtypes)
    print(pretty_onnx(onx))

.. seealso::

    :ref:`l-design-sql-converter` — SQL converter design details.

    :ref:`l-design-sql-polars` — polars LazyFrame conversion design.

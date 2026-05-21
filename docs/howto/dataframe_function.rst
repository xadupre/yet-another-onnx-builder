.. _l-howto-dataframe-function:

DataFrame function
==================

This page answers common *"how do I…"* questions for exporting a Python
function that processes one or multiple DataFrames into ONNX.

----

How to export a function processing one DataFrame
-------------------------------------------------

Use :func:`yobx.sql.to_onnx` with a representative :class:`~pandas.DataFrame`.
The function is traced through the DataFrame API and compiled into ONNX.

.. runpython::
    :showcode:

    import numpy as np
    import pandas as pd
    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.sql import to_onnx

    def add_total(df):
        return df.select([(df["a"] + df["b"]).alias("total")])

    df = pd.DataFrame(
        {"a": np.array([1.0, 2.0], dtype=np.float32), "b": np.array([3.0, 4.0], dtype=np.float32)}
    )
    onx = to_onnx(add_total, (df,))
    print(pretty_onnx(onx))

----

How to export a function processing several DataFrames
------------------------------------------------------

Pass a tuple of DataFrames to :func:`yobx.sql.to_onnx`. The callable receives
one traced frame per input DataFrame.

.. runpython::
    :showcode:

    import numpy as np
    import pandas as pd
    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.sql import to_onnx

    def join_like(left, right):
        if hasattr(left, "join"):
            merged = left.join(right, left_key="id", right_key="id")
            return merged.select([(merged["a"] + merged["b"]).alias("total")])
        merged = pd.merge(left, right, on="id")
        return (merged["a"] + merged["b"]).to_frame(name="total")

    left = pd.DataFrame(
        {"id": np.array([0, 1], dtype=np.int64), "a": np.array([1.0, 2.0], dtype=np.float32)}
    )
    right = pd.DataFrame(
        {"id": np.array([0, 1], dtype=np.int64), "b": np.array([10.0, 20.0], dtype=np.float32)}
    )
    onx = to_onnx(join_like, (left, right))
    print(pretty_onnx(onx))

----

How to use this in a scikit-learn Pipeline
-------------------------------------------

Wrap the DataFrame logic in a custom transformer inheriting from
:class:`~yobx.sklearn.TraceableMixin`. Then place that transformer in a normal
:class:`~sklearn.pipeline.Pipeline` and export the fitted pipeline with
:func:`yobx.sklearn.to_onnx`.

.. runpython::
    :showcode:

    import numpy as np
    import pandas as pd
    import onnxruntime
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from yobx.sklearn import TraceableMixin, to_onnx

    class AddTotalTransformer(BaseEstimator, TransformerMixin, TraceableMixin):
        def fit(self, df, y=None):
            return self

        def transform(self, df):
            if hasattr(df, "select"):
                return df.select([(df["a"] + df["b"]).alias("total")])
            return (df["a"] + df["b"]).to_frame(name="total")

    rng = np.random.default_rng(0)
    train_df = pd.DataFrame(
        {
            "a": rng.standard_normal(40).astype(np.float32),
            "b": rng.standard_normal(40).astype(np.float32),
        }
    )
    pipe = Pipeline([("features", AddTotalTransformer()), ("scale", StandardScaler())]).fit(train_df)

    onx = to_onnx(pipe, (train_df,))

    test_df = pd.DataFrame(
        {
            "a": rng.standard_normal(5).astype(np.float32),
            "b": rng.standard_normal(5).astype(np.float32),
        }
    )
    expected = pipe.transform(test_df).astype(np.float32)

    sess = onnxruntime.InferenceSession(
        onx.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (got,) = sess.run(
        None,
        {
            "a": test_df[["a"]].to_numpy().astype(np.float32),
            "b": test_df[["b"]].to_numpy().astype(np.float32),
        },
    )
    assert np.allclose(got, expected, atol=1e-5)
    print("Pipeline outputs match ✓")

.. seealso::

    :ref:`l-howto-sklearn` — additional scikit-learn conversion patterns.

    :ref:`l-design-dataframe-tracing` — detailed design of DataFrame tracing.

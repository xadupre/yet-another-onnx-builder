.. _l-design-dataframe-pipeline:

DataFrame Input to a Pipeline with ColumnTransformer
=====================================================

This page explains how :func:`yobx.sklearn.to_onnx` handles a
:class:`pandas.DataFrame` as the dummy input when the model being converted
contains a :class:`~sklearn.compose.ColumnTransformer`.

Overview
--------

A common :epkg:`scikit-learn` pattern is to build a
:class:`~sklearn.pipeline.Pipeline` whose first step is a
:class:`~sklearn.compose.ColumnTransformer`.  The transformer selects
subsets of columns by **name**, applies a different preprocessing step to
each subset, and concatenates the results.  For example:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    df = pd.DataFrame(
        np.random.randn(100, 4).astype(np.float32),
        columns=["age", "income", "score", "balance"],
    )
    y = (df["age"] > 0).astype(int).to_numpy()

    ct = ColumnTransformer([
        ("std", StandardScaler(),  ["age", "income"]),
        ("mm",  MinMaxScaler(),    ["score", "balance"]),
    ])
    pipe = Pipeline([("preprocessor", ct), ("clf", LogisticRegression())])
    pipe.fit(df, y)

To convert this pipeline to ONNX, pass the fitted :class:`~pandas.DataFrame`
directly as the dummy input:

.. code-block:: python

    from yobx.sklearn import to_onnx

    onx = to_onnx(pipe, (df,))

Per-column ONNX inputs
----------------------

When a :class:`~pandas.DataFrame` is detected, :func:`yobx.sklearn.to_onnx`
expands it column-by-column: each column is registered as a **separate 1-D
ONNX graph input** named after the column.  An ``Unsqueeze`` + ``Concat``
node sequence assembles the per-column tensors back into the 2-D matrix
``(batch, n_cols)`` expected by the converter.

.. code-block:: text

    "age"     ──Unsqueeze──┐
    "income"  ──Unsqueeze──┤ Concat ──► X (batch, 4) ──► pipeline ...
    "score"   ──Unsqueeze──┤
    "balance" ──Unsqueeze──┘

This produces an ONNX model with four separate inputs instead of a single
``X`` matrix, which matches the interface of a serving endpoint that
receives one scalar value per feature.

String column selectors
-----------------------

scikit-learn's :class:`~sklearn.compose.ColumnTransformer` stores the
original column specification in ``transformers_`` after fitting.  When
columns are specified by **name** (strings), those names are preserved in
``transformers_``; only the overall number of features is stored in
``n_features_in_``.

The ONNX converter resolves string column names to integer positions using
``feature_names_in_``, which scikit-learn automatically sets on the
:class:`~sklearn.compose.ColumnTransformer` when it is fitted on a
DataFrame:

.. code-block:: python

    # After fitting:
    print(ct.feature_names_in_)
    # ['age' 'income' 'score' 'balance']

    print(ct.transformers_)
    # [('std', StandardScaler(), ['age', 'income']),
    #  ('mm',  MinMaxScaler(),   ['score', 'balance'])]

The mapping ``name → index`` is computed once and reused for every
transformer entry in ``transformers_``.

Inference
---------

At inference time the ONNX model expects one **1-D array** per column,
matching the graph inputs created during conversion:

.. code-block:: python

    import onnxruntime

    sess = onnxruntime.InferenceSession(onx.proto.SerializeToString())

    # Pass a dict with one key per column:
    feed = {col: df_test[col].to_numpy() for col in df.columns}
    labels, probas = sess.run(None, feed)

Full working example
--------------------

See :ref:`l-plot-sklearn-dataframe-pipeline` for a complete runnable
example that trains the pipeline, converts it to ONNX, runs inference with
:epkg:`onnxruntime`, and verifies that the outputs match scikit-learn.

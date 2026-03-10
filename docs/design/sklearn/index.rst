scikit-learn Export to ONNX
===========================

A basic :epkg:`scikit-learn` model may look like the following,
a scaler following by an estimator.

.. runpython::
    :rst:

    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression()),
    ]).fit(X, y)

    html = pipe._repr_html_()
    print(".. raw:: html")
    print()
    for line in html.split("\n"):
        print(f"    {line}")

Models based on :epkg:`scikit-learn` are made of a custom collection of known
transformers or estimators. The main function functions has to call
every converter for every piece and assembles the result into a single ONNX
model.

The custom collection is mainly implemented through the classes
:class:`sklearn.pipeline.Pipeline`, :class:`sklearn.pipeline.FeatureUnion` and
:class:`sklearn.compose.ColumnTransformer`. Everything else is well defined
and can be mapped to its converted ONNX code. It also implemented through
meta-estimators combining others such as :class:`sklearn.multiclass.OneVsRestClassifier`
or :class:`sklearn.ensemble.VotingClassifier`.

**Common API**

Putting ONNX node together in a model is not difficult but almost everybody
already implemented its own way of doing, :epkg:`ir-py`, :epkg:`onnxscript`,
:epkg:`spox`. Every converting library has also its own: :epkg:`sklearn-onnx`,
:epkg:`tensorflow-onnx`, :epkg:`onnxmltools`... The choice was made not to create
a new one but more to define what the converters expect to find in a class
classed ``GraphBuilder``. It then becomes possible to create a bridge
such as :class:`yobx.builder.onnxscript.OnnxScriptGraphBuilder` which implements
this API for every known way. See :ref:`l-design-expected-api` for further details.

**AI**

Known LLMs now provides a good first draft when it comes to implement a new converter
for a model not already covered but this library. This package includes
a function able to query *Copilot* to get that first draft.
That saves quite some time.

.. toctree::
   :maxdepth: 1

   expected_api
   sklearn_converter
   lightgbm_converter
   supported_converters
   custom_converter
   copilot_draft
   debug

Finally, the example given at the top of the page would be converted
into the mode which follows.

.. code-block:: python

    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from yobx.sklearn import to_onnx
    from yobx.helpers.dot_helper import to_dot

    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression()),
    ]).fit(X, y)

    model = to_onnx(pipe, (X,))
    print("DOT-SECTION", to_dot(model))    

.. gdot::
    :script: DOT-SECTION

    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from yobx.sklearn import to_onnx
    from yobx.helpers.dot_helper import to_dot

    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression()),
    ]).fit(X, y)

    model = to_onnx(pipe, (X,))
    print("DOT-SECTION", to_dot(model))
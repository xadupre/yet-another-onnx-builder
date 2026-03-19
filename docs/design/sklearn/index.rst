.. _l-sklearn-converter:

scikit-learn Export to ONNX
===========================

.. toctree::
   :maxdepth: 1

   expected_api
   sklearn_converter
   sklearn_like_converter
   supported_converters
   custom_converter
   contrib_ops
   debug

.. seealso::

    :ref:`l-design-function-transformer-tracing` — the numpy-tracing
    mechanism used by :class:`~sklearn.preprocessing.FunctionTransformer`
    is documented in the core design section.

A basic :epkg:`scikit-learn` model may look like the following,
a scaler following by an estimator. Every model can be converter
with model :func:`~yobx.sklearn.to_onnx`.

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

**Opsets**

:func:`yobx.sklearn.to_onnx` converts scikit-learn models into
ONNX. The function exposes arguments **target_opset**.
The conversion is done for opset 18 if ``target_opset==18``.
The conversion may includes optimized kernels for :epkg:`onnxruntime`
if ``target_opsets={'': 18, 'com.microsoft': 1}`` (see :ref:`l-design-sklearn-contrib-ops`).

**Discrepancies**

`scikit-learn==1.8` is more strict with computation types and
the number of discrepancies is reduced. Switch to float32 in a matrix
multiplication when the order of magnitude of the coefficient is quite
large usually introduces discrepancies. That is often the case when
a matrix is the inverse of another one (see :ref:`l-plot-sklearn-pls-float32`).
Prior to that, it was not rare the see huge difference when using a model just
after normalizing the data. The normalizer was implicitly switching the type
to float64 while ONNX was keeping float32.
If followed by a tree, a small difference could make the model
choose a different decision path and produce a very different output.

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
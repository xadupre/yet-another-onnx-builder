Scikit-learn Export to ONNX
===========================

@copilot include a picture of a pipeline

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
this API for every known way.

**AI**

Known LLMs now provides a good first draft when it comes to implement a new converter
for a model not already covered but this library. This package includes
a function able to query *Copilot* to get that first draft.
That saves quite some time.

.. toctree::
   :maxdepth: 1

   expected_api
   sklearn_converter
   custom_converter
   copilot_draft

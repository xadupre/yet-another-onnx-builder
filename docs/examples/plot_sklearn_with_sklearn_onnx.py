"""
.. _l-plot-sklearn-with-sklearn-onnx:

Leverage sklearn-onnx to extend the existing sklearn converters
===============================================================
"""

from typing import Dict, List, Tuple
import numpy as np
import onnx
import onnxruntime
from sklearn.neural_network import MLPClassifier
from yobx import doc
from yobx.xbuilder import GraphBuilder, FunctionOptions
from yobx.sklearn import to_onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import DoubleTensorType, FloatTensorType


def to_skl2onnx_input_type(elem_type: int, n_features: int):
    if elem_type == onnx.TensorProto.FLOAT:
        return FloatTensorType([None, n_features])
    if elem_type == onnx.TensorProto.DOUBLE:
        return DoubleTensorType([None, n_features])
    raise NotImplementedError(
        f"Input elem_type {elem_type} is not supported. "
        "Only FLOAT (1) and DOUBLE (11) are supported by the skl2onnx MLP converter."
    )


def convert_sklearn_mlp_classifier(
    g: GraphBuilder,
    sts: Dict,
    outputs: List[str],
    estimator: MLPClassifier,
    X: str,
    name: str = "mlp_classifier",
) -> Tuple[str, str]:
    assert isinstance(
        estimator, MLPClassifier
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    n_features = estimator.coefs_[0].shape[0]

    onx = convert_sklearn(
        estimator,
        initial_types=[("X", to_skl2onnx_input_type(itype, n_features))],
        options={"zipmap": False},
        target_opset=g.main_opset,
    )
    # sklearn chooses the lowest opset equivalent to the current ones
    # given the nodes it contains. We need to overwrite that.
    del onx.opset_import[:]
    d = onx.opset_import.add()
    d.domain = ""
    d.version = g.main_opset
    builder = g.__class__(onx)

    f_options = FunctionOptions(
        export_as_function=True,
        name=g.unique_function_name("MLPClassifier"),
        domain="sklean_onnx_functions",
        move_initializer_to_constant=True,
    )
    g.make_local_function(builder, f_options)
    return g.make_node(f_options.name, [X], outputs, domain=f_options.domain, name=name)


X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
y = np.array([0, 0, 1, 1])
mlp = MLPClassifier(hidden_layer_sizes=(4,), activation="relu", random_state=0, max_iter=2000)
mlp.fit(X, y)

onx = to_onnx(mlp, (X,), extra_converters={MLPClassifier: convert_sklearn_mlp_classifier})


op_types = [n.op_type for n in onx.graph.node]
ref = onnxruntime.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
results = ref.run(None, {"X": X})

np.testing.assert_allclose(mlp.predict(X), results[0], atol=1e-5)
np.testing.assert_allclose(mlp.predict_proba(X), results[1], atol=1e-5)


# %%
doc.plot_dot(onx)

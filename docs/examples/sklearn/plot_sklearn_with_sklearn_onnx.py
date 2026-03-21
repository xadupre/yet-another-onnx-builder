"""
.. _l-plot-sklearn-with-sklearn-onnx:

Using sklearn-onnx to convert any scikit-learn estimator
=========================================================

:func:`yobx.sklearn.to_onnx` ships with built-in converters for a curated
set of :epkg:`scikit-learn` estimators (``StandardScaler``,
``LogisticRegression``, ``DecisionTree*``, ``MLP*``, and ``Pipeline``).
For any estimator that is *not* covered by the built-in registry, you can
supply your own converter through the ``extra_converters`` keyword argument.

This example shows two ways to write such a custom converter for
:class:`~sklearn.neural_network.MLPClassifier` using
:epkg:`sklearn-onnx` (``skl2onnx``) as the conversion back-end.

**Option A — low-level**: write the converter function by hand, directly
calling the skl2onnx converter function with mocked ``Scope``, ``Operator``
and ``ModelComponentContainer`` objects and injecting the resulting nodes
into the enclosing :class:`~yobx.xbuilder.GraphBuilder`.

**Option B — factory** *(recommended)*: use
:func:`~yobx.sklearn.make_skl2onnx_converter` to generate the converter
automatically.  Option B requires only one line of setup code.

The strategy
-------------

1. Look up the converter registered in :epkg:`sklearn-onnx` for the
   estimator's type via :data:`skl2onnx._supported_operators.sklearn_operator_name_map`
   and :func:`skl2onnx.common._registration.get_converter`.
2. Create lightweight mock objects
   (:class:`~skl2onnx.common._topology.Scope`,
   :class:`~skl2onnx.common._topology.Operator`,
   :class:`~skl2onnx.common._topology.Variable`) whose names match the
   input and output tensors already registered in the enclosing
   :class:`~yobx.xbuilder.GraphBuilder`.
3. Create a :class:`~skl2onnx.common._container.ModelComponentContainer`
   that accumulates ONNX nodes and initializers.
4. Call the skl2onnx converter function directly:
   ``converter(scope, operator, container)``.
5. Inject all collected nodes and initializers straight into the enclosing
   :class:`~yobx.xbuilder.GraphBuilder`.
"""

from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime
from sklearn.neural_network import MLPClassifier
from skl2onnx._supported_operators import sklearn_operator_name_map
from skl2onnx.common._container import ModelComponentContainer
from skl2onnx.common._registration import _converter_pool, get_converter
from skl2onnx.common._topology import Operator, Scope, Variable
from skl2onnx.common.data_types import DoubleTensorType, FloatTensorType

from yobx import doc
from yobx.typing import GraphBuilderExtendedProtocol
from yobx.sklearn import to_onnx, make_skl2onnx_converter

# %%
# Option A — low-level custom converter
# --------------------------------------
#
# A yobx *converter* is a plain Python function that receives the active
# :class:`~yobx.xbuilder.GraphBuilder`, a shape-tracking dict (*sts*), the
# desired output tensor names, the fitted estimator, and the name of the
# input tensor already present in the graph.  It must emit all necessary
# ONNX nodes into *g* and return the name(s) of the produced output tensors.
#
# Here we implement this contract by:
#
# 1. Looking up the skl2onnx converter registered for the estimator type.
# 2. Creating lightweight ``Scope``, ``Operator``, ``Variable`` mock objects
#    whose names match the actual tensors in the enclosing GraphBuilder.
# 3. Running ``converter(scope, operator, container)`` to accumulate nodes.
# 4. Injecting those nodes and initializers into the GraphBuilder directly.


def convert_sklearn_mlp_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: MLPClassifier,
    X: str,
    name: str = "mlp_classifier",
) -> Tuple[str, str]:
    """
    Convert a fitted :class:`~sklearn.neural_network.MLPClassifier` into
    ONNX nodes inside an existing :class:`~yobx.xbuilder.GraphBuilder`.

    The conversion invokes the skl2onnx converter directly through mock
    ``Scope``, ``Operator``, and ``ModelComponentContainer`` objects, then
    injects the resulting nodes and initializers straight into *g*.

    :param g: the :class:`~yobx.xbuilder.GraphBuilder` that is currently
        being populated.
    :param sts: shape-tracking dictionary populated by the yobx converter
        framework (passed through unchanged).
    :param outputs: list of desired output tensor names
        ``[label, probabilities]``.
    :param estimator: a fitted ``MLPClassifier`` instance.
    :param X: name of the input tensor already present in *g*.
    :param name: base name used for the emitted call-node.
    :return: tuple ``(label_name, probabilities_name)`` — the names of the
        two output tensors in *g*.
    """
    assert isinstance(
        estimator, MLPClassifier
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    if hasattr(estimator, "n_features_in_"):
        n_features = estimator.n_features_in_
    elif g.has_shape(X):
        shape = g.get_shape(X)
        n_features = int(shape[1]) if len(shape) > 1 and isinstance(shape[1], int) else None
    else:
        n_features = None

    if itype == onnx.TensorProto.FLOAT:
        skl2onnx_type = FloatTensorType([None, n_features])
    elif itype == onnx.TensorProto.DOUBLE:
        skl2onnx_type = DoubleTensorType([None, n_features])
    else:
        raise NotImplementedError(f"Unsupported elem_type {itype!r}.")

    op_name = sklearn_operator_name_map[type(estimator)]
    converter = get_converter(op_name)
    registered_models = {"aliases": sklearn_operator_name_map, "conv": _converter_pool}

    scope = Scope("root", target_opset=g.main_opset)

    input_var = Variable(X, X, scope="root", type=skl2onnx_type)
    input_var.init_status(is_fed=True, is_root=True, is_leaf=False)

    output_vars = []
    for out_name in outputs:
        var = Variable(out_name, out_name, scope="root")
        var.init_status(is_fed=False, is_root=False, is_leaf=True)
        output_vars.append(var)

    operator = Operator(f"{name}_op", "root", op_name, estimator, g.main_opset, scope)
    operator.inputs.append(input_var)
    for var in output_vars:
        operator.outputs.append(var)

    container = ModelComponentContainer(g.main_opset, registered_models=registered_models)
    converter(scope, operator, container)

    for init in container.initializers:
        g.make_initializer(init.name, init)
    for node in container.nodes:
        g.make_node(
            node.op_type,
            list(node.input),
            list(node.output),
            domain=node.domain,
            name=node.name,
            attributes=list(node.attribute),
        )

    return tuple(outputs)


# %%
# 1. Train a simple MLPClassifier
# --------------------------------
#
# We use a tiny dataset (four samples, two features, binary labels) so the
# example runs instantly.  The same API works for larger datasets and deeper
# networks.

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
y = np.array([0, 0, 1, 1])
mlp = MLPClassifier(hidden_layer_sizes=(4,), activation="relu", random_state=0, max_iter=2000)
mlp.fit(X, y)

print("MLPClassifier architecture:")
print(f"  hidden layers : {mlp.hidden_layer_sizes}")
print(f"  activation    : {mlp.activation}")
print(f"  n_layers_     : {mlp.n_layers_}")

# %%
# 2a. Convert to ONNX using the low-level custom converter (Option A)
# -------------------------------------------------------------------
#
# We pass our custom converter function through ``extra_converters``.
# The yobx framework will call it whenever it encounters an
# ``MLPClassifier`` instance while traversing the estimator (or pipeline).

onx = to_onnx(mlp, (X,), extra_converters={MLPClassifier: convert_sklearn_mlp_classifier})

print("\nTop-level graph nodes:")
for node in onx.graph.node:
    print(f"  op_type={node.op_type!r}  domain={node.domain!r}  name={node.name!r}")

# %%
# 3. Run with ONNX Runtime and verify
# ------------------------------------
#
# We execute the exported model with :epkg:`onnxruntime` and compare its
# predictions to scikit-learn's.  Both the class labels and the
# class-probability matrix must match within floating-point tolerance.

ref = onnxruntime.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
results = ref.run(None, {"X": X})
label_onnx, proba_onnx = results[0], results[1]

label_sk = mlp.predict(X)
proba_sk = mlp.predict_proba(X)

print("\nPredicted labels  (sklearn):", label_sk)
print("Predicted labels  (ONNX)   :", label_onnx)

np.testing.assert_array_equal(label_sk, label_onnx)
np.testing.assert_allclose(proba_sk, proba_onnx, atol=1e-5)
print("\nAll predictions match ✓")

# %%
# 2b. Convert to ONNX using the factory helper (Option B — recommended)
# ----------------------------------------------------------------------
#
# :func:`~yobx.sklearn.make_skl2onnx_converter` creates the boilerplate
# converter automatically.  The factory handles type detection, feature
# inference, and node injection transparently.

converter = make_skl2onnx_converter()
onx_b = to_onnx(mlp, (X,), extra_converters={MLPClassifier: converter})

ref_b = onnxruntime.InferenceSession(
    onx_b.SerializeToString(), providers=["CPUExecutionProvider"]
)
results_b = ref_b.run(None, {"X": X})
label_b, proba_b = results_b[0], results_b[1]

np.testing.assert_array_equal(label_sk, label_b)
np.testing.assert_allclose(proba_sk, proba_b, atol=1e-5)
print("Option B predictions also match ✓")

# %%
# 4. Visualise the exported ONNX graph
# -------------------------------------
#
# :func:`yobx.doc.plot_dot` renders the :class:`onnx.ModelProto` as a
# directed graph using Graphviz.  The nodes emitted by the skl2onnx
# converter are injected directly into the enclosing graph.

doc.plot_dot(onx)

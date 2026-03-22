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

**Option A — low-level**: write the converter function by hand.  Obtain
the skl2onnx converter function from skl2onnx's registry, pass pure-Python
mock objects for ``Scope``, ``Operator``, and ``ModelComponentContainer``
(provided by :mod:`yobx.sklearn.skl2onnx_converter`), and inject the
resulting nodes into the enclosing :class:`~yobx.xbuilder.GraphBuilder`.

**Option B — factory** *(recommended)*: use
:func:`~yobx.sklearn.make_skl2onnx_converter`.  One line of setup after
looking up the skl2onnx converter function.

The strategy
-------------

1. Look up the converter registered in :epkg:`sklearn-onnx` for the
   estimator's type.
2. Create lightweight mock objects
   (:class:`~yobx.sklearn.skl2onnx_converter._MockScope`,
   :class:`~yobx.sklearn.skl2onnx_converter._MockOperator`,
   :class:`~yobx.sklearn.skl2onnx_converter._MockVariable`,
   :class:`~yobx.sklearn.skl2onnx_converter._MockContainer`)
   whose tensor names match the ones already registered in the enclosing
   :class:`~yobx.xbuilder.GraphBuilder`.
3. Call the skl2onnx converter function: ``converter(scope, operator, container)``.
4. Inject all collected nodes and initializers straight into the enclosing
   :class:`~yobx.xbuilder.GraphBuilder`.

These mock classes are **pure Python** — :mod:`yobx.sklearn.skl2onnx_converter`
contains no skl2onnx imports.
"""

from typing import Dict, List, Tuple

import numpy as np
import onnxruntime
from sklearn.neural_network import MLPClassifier
from skl2onnx._supported_operators import sklearn_operator_name_map
from skl2onnx.common._registration import get_converter
from skl2onnx.common.data_types import FloatTensorType

from yobx import doc
from yobx.typing import GraphBuilderExtendedProtocol
from yobx.sklearn import to_onnx, make_skl2onnx_converter
from yobx.sklearn.skl2onnx_converter import (
    MockContainer,
    MockOperator,
    MockScope,
    MockVariable,
)

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
# 1. Looking up the skl2onnx converter function from skl2onnx's registry.
# 2. Creating lightweight mock objects whose names match the actual tensors
#    in the enclosing GraphBuilder.
# 3. Running ``converter(scope, operator, container)`` — the container
#    delegates each emitted node/initializer directly to the GraphBuilder.


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

    Uses :mod:`yobx`'s pure-Python mock objects to call the skl2onnx
    converter function directly, then injects the resulting nodes and
    initializers straight into *g*.

    :param g: the :class:`~yobx.xbuilder.GraphBuilder` that is currently
        being populated.
    :param sts: shape-tracking dictionary populated by the yobx converter
        framework (passed through unchanged).
    :param outputs: list of desired output tensor names
        ``[label, probabilities]``.
    :param estimator: a fitted ``MLPClassifier`` instance.
    :param X: name of the input tensor already present in *g*.
    :param name: base name used for the emitted nodes.
    :return: tuple ``(label_name, probabilities_name)`` — the names of the
        two output tensors in *g*.
    """
    assert isinstance(
        estimator, MLPClassifier
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    # Look up the registered skl2onnx converter for MLPClassifier.
    op_name = sklearn_operator_name_map[type(estimator)]
    skl2onnx_fn = get_converter(op_name)

    # Build mock objects with the correct tensor names.
    scope = MockScope(g)

    input_var = MockVariable(X, X)
    output_vars = [MockVariable(out, out) for out in outputs]

    operator = MockOperator(estimator, op_name, f"{name}_op", scope)
    operator.inputs.append(input_var)
    for var in output_vars:
        operator.outputs.append(var)

    container = MockContainer(g)
    skl2onnx_fn(scope, operator, container)

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
# :func:`~yobx.sklearn.make_skl2onnx_converter` eliminates the mock
# boilerplate.  Look up the skl2onnx converter function, pass it to the
# factory, and you're done.

skl2onnx_mlp_fn = get_converter(sklearn_operator_name_map[MLPClassifier])
converter = make_skl2onnx_converter(skl2onnx_mlp_fn, FloatTensorType([None, None]))
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

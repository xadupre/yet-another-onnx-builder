"""
.. _l-plot-sklearn-with-spox:

Converting a scikit-learn model to ONNX with spox
==================================================

:epkg:`spox` is a Python library for constructing ONNX graphs that exposes
strongly-typed, opset-versioned operator functions rather than the raw
``onnx.helper`` API.  Each operator is a Python function (e.g.
``op.MatMul(A, B)``) so the construction code is type-safe, IDE-friendly,
and always produces a graph that is valid for the chosen opset.

:class:`~yobx.builder.spox.SpoxGraphBuilder` implements the same
:class:`~yobx.typing.GraphBuilderExtendedProtocol` as the default
:class:`~yobx.xbuilder.GraphBuilder`, but delegates every operator
construction call to the corresponding :epkg:`spox` opset module.
Existing :mod:`yobx.sklearn` converters work without modification: the
only change is passing ``builder_cls=SpoxGraphBuilder`` to
:func:`yobx.sklearn.to_onnx`.

Covered in this example:

1. Binary classification pipeline (``StandardScaler`` + ``LogisticRegression``)
2. Multiclass classification
3. ``DecisionTreeClassifier`` — uses the ``ai.onnx.ml`` domain, so it
   exercises :class:`~yobx.builder.spox.SpoxGraphBuilder`'s secondary-domain
   dispatch.
4. Visualising the exported ONNX graph.
"""

# %%
# Guard: skip this example if spox is not installed.
try:
    import spox  # noqa: F401
except ImportError:
    import sys

    sys.exit(0)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from yobx.builder.spox import SpoxGraphBuilder
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx

# %%
# 1. Train a binary classification pipeline
# ------------------------------------------
#
# We train a two-step pipeline: ``StandardScaler`` followed by
# ``LogisticRegression`` on a small synthetic dataset (80 samples,
# 4 features, 2 classes).

rng = np.random.default_rng(0)
X_train = rng.standard_normal((80, 4)).astype(np.float32)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression()),
    ]
)
pipe.fit(X_train, y_train)

print("Binary pipeline steps:")
for name, step in pipe.steps:
    print(f"  {name}: {step}")

# %%
# 2. Convert to ONNX using SpoxGraphBuilder
# ------------------------------------------
#
# The only difference from a plain :func:`~yobx.sklearn.to_onnx` call is
# passing ``builder_cls=SpoxGraphBuilder``.  The converters for each
# step in the pipeline will use :epkg:`spox` opset modules to emit
# the ONNX nodes.

onx = to_onnx(pipe, (X_train[:1],), builder_cls=SpoxGraphBuilder)

print(f"\nONNX opset  : {onx.opset_import[0].version}")
print("Node types  :", [n.op_type for n in onx.graph.node])
print(
    "Graph input : ",
    [(inp.name, inp.type.tensor_type.elem_type) for inp in onx.graph.input],
)
print("Graph outputs:", [out.name for out in onx.graph.output])

# %%
# 3. Run and verify — binary classification
# ------------------------------------------
#
# We run the exported model with
# :class:`~yobx.reference.ExtendedReferenceEvaluator` and check that
# class labels and probabilities match scikit-learn's predictions.

X_test = rng.standard_normal((20, 4)).astype(np.float32)

ref = ExtendedReferenceEvaluator(onx)
label_onnx, proba_onnx = ref.run(None, {"X": X_test})

label_sk = pipe.predict(X_test)
proba_sk = pipe.predict_proba(X_test).astype(np.float32)

print("\nFirst 5 labels (sklearn):", label_sk[:5])
print("First 5 labels (ONNX)   :", label_onnx[:5])

assert np.array_equal(label_sk, label_onnx), "Label mismatch!"
assert np.allclose(proba_sk, proba_onnx, atol=1e-5), "Probability mismatch!"
print("\nBinary predictions match ✓")

# %%
# 4. Multiclass classification
# -----------------------------
#
# The same workflow applies to multiclass problems.  The
# ``LogisticRegression`` converter switches from a sigmoid-based binary
# graph to a softmax-based multiclass graph automatically.

X_mc = rng.standard_normal((120, 4)).astype(np.float32)
y_mc = rng.integers(0, 3, size=120)

pipe_mc = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500)),
    ]
)
pipe_mc.fit(X_mc, y_mc)

X_test_mc = rng.standard_normal((30, 4)).astype(np.float32)
onx_mc = to_onnx(pipe_mc, (X_test_mc[:1],), builder_cls=SpoxGraphBuilder)

ref_mc = ExtendedReferenceEvaluator(onx_mc)
label_mc_onnx, proba_mc_onnx = ref_mc.run(None, {"X": X_test_mc})

label_mc_sk = pipe_mc.predict(X_test_mc)
proba_mc_sk = pipe_mc.predict_proba(X_test_mc).astype(np.float32)

assert np.array_equal(label_mc_sk, label_mc_onnx), "Multiclass label mismatch!"
assert np.allclose(proba_mc_sk, proba_mc_onnx, atol=1e-5), "Multiclass proba mismatch!"
print("Multiclass predictions match ✓")

# %%
# 5. DecisionTreeClassifier — the ``ai.onnx.ml`` domain
# -------------------------------------------------------
#
# Decision-tree models map to the ``TreeEnsembleClassifier`` operator in
# the ``ai.onnx.ml`` domain.
# :class:`~yobx.builder.spox.SpoxGraphBuilder` resolves this domain to
# ``spox.opset.ai.onnx.ml.v3`` automatically, so no extra configuration
# is needed.

from sklearn.tree import DecisionTreeClassifier  # noqa: E402

X_dt = rng.standard_normal((60, 4)).astype(np.float32)
y_dt = rng.integers(0, 3, size=60)

dt = DecisionTreeClassifier(max_depth=4, random_state=0)
dt.fit(X_dt, y_dt)

onx_dt = to_onnx(dt, (X_dt[:1],), builder_cls=SpoxGraphBuilder)

print("\nDecisionTree node types:", [n.op_type for n in onx_dt.graph.node])
print("Domains used:", list({n.domain for n in onx_dt.graph.node}))

X_test_dt = rng.standard_normal((20, 4)).astype(np.float32)
ref_dt = ExtendedReferenceEvaluator(onx_dt)
label_dt_onnx, _ = ref_dt.run(None, {"X": X_test_dt})

assert np.array_equal(dt.predict(X_test_dt), label_dt_onnx), "Decision tree mismatch!"
print("DecisionTree predictions match ✓")

# %%
# 6. Visualise the ONNX graph
# ----------------------------
#
# :func:`yobx.helpers.dot_helper.to_dot` converts the
# :class:`onnx.ModelProto` into a DOT string.  The graph below shows
# the binary pipeline produced by :class:`~yobx.builder.spox.SpoxGraphBuilder`.

from yobx.helpers.dot_helper import to_dot  # noqa: E402

dot_src = to_dot(onx)
print(dot_src)

# %%
# Display the graph
# ------------------
#
# The DOT source produced above describes the following graph.
#
# .. gdot::
#     :script: DOT-SECTION
#
#     import numpy as np
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.pipeline import Pipeline
#     from sklearn.preprocessing import StandardScaler
#     from yobx.builder.spox import SpoxGraphBuilder
#     from yobx.sklearn import to_onnx
#     from yobx.helpers.dot_helper import to_dot
#
#     rng = np.random.default_rng(0)
#     X_train = rng.standard_normal((80, 4)).astype(np.float32)
#     y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
#     pipe = Pipeline(
#         [("scaler", StandardScaler()), ("clf", LogisticRegression())]
#     )
#     pipe.fit(X_train, y_train)
#     onx = to_onnx(pipe, (X_train[:1],), builder_cls=SpoxGraphBuilder)
#     dot = to_dot(onx)
#     print("DOT-SECTION", dot)

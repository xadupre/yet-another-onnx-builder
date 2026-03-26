from typing import Dict, List

from sklearn.ensemble import RandomTreesEmbedding

from ..register import get_sklearn_converter, register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ..tree.decision_tree import _make_leaf_id_node


@register_sklearn_converter((RandomTreesEmbedding,))
def sklearn_random_trees_embedding(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: RandomTreesEmbedding,
    X: str,
    name: str = "random_trees_embedding",
) -> str:
    """
    Converts a :class:`sklearn.ensemble.RandomTreesEmbedding` into ONNX.

    :class:`~sklearn.ensemble.RandomTreesEmbedding` maps inputs through a
    forest of totally random trees.  Each tree returns the leaf node id for a
    given sample; the leaf ids from all trees are then one-hot encoded and
    concatenated, yielding a high-dimensional binary embedding.

    The conversion mirrors the scikit-learn ``transform`` implementation:

    1. For each fitted :class:`~sklearn.tree.ExtraTreeRegressor` in
       ``estimators_``, emit a
       :func:`_make_leaf_id_node
       <yobx.sklearn.tree.decision_tree._make_leaf_id_node>` sub-graph that
       outputs the (float) leaf node id for every sample – shape ``(N, 1)``.
    2. Concatenate the per-tree leaf-id columns along ``axis=1`` to produce
       a matrix of shape ``(N, n_estimators)``.
    3. Apply the fitted ``one_hot_encoder_`` via the registered
       :class:`~sklearn.preprocessing.OneHotEncoder` converter to obtain the
       final ``(N, total_leaves)`` embedding.

    Both ``float32`` and ``float64`` inputs are handled correctly: the
    leaf-id tensors inherit the input dtype (``ai.onnx.ml`` opset 5 path),
    and the one-hot indicator cast respects that dtype too, so the output
    dtype matches the input dtype.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names
    :param estimator: a fitted
        :class:`~sklearn.ensemble.RandomTreesEmbedding`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: name of the output tensor
    """
    assert isinstance(
        estimator, RandomTreesEmbedding
    ), f"Unexpected type {type(estimator)} for estimator."

    itype = g.get_type(X)
    estimators = estimator.estimators_

    # 1. Compute leaf node IDs for each tree: list of (N, 1) float tensors.
    #    _make_leaf_id_node uses TreeEnsembleRegressor in the legacy path
    #    (ai.onnx.ml opset <= 4), which always outputs float32.  Cast back to
    #    the input dtype so that the OneHotEncoder comparison uses matching
    #    types for both float32 and float64 inputs.
    leaf_id_cols: List[str] = []
    for i, base_est in enumerate(estimators):
        tree = base_est.tree_
        leaf_id = _make_leaf_id_node(g, tree, X, f"{name}_tree{i}")
        if g.get_type(leaf_id) != itype:
            leaf_id = g.op.Cast(leaf_id, to=itype, name=f"{name}_tree{i}_cast")
        leaf_id_cols.append(leaf_id)

    # 2. Concatenate leaf ids along axis=1 → (N, n_estimators).
    if len(leaf_id_cols) == 1:
        leaf_id_matrix = g.op.Identity(leaf_id_cols[0], name=f"{name}_leaf_concat")
    else:
        leaf_id_matrix = g.op.Concat(*leaf_id_cols, axis=1, name=f"{name}_leaf_concat")

    # 3. Apply the fitted one_hot_encoder_ to produce the final embedding.
    ohe_converter = get_sklearn_converter(type(estimator.one_hot_encoder_))
    return ohe_converter(
        g, sts, outputs, estimator.one_hot_encoder_, leaf_id_matrix, name=f"{name}_ohe"
    )

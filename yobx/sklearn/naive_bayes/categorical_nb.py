from typing import Dict, List, Optional, Tuple
import numpy as np
import onnx
from sklearn.naive_bayes import CategoricalNB
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from .nb import _emit_label_and_proba


@register_sklearn_converter(CategoricalNB)
def sklearn_categorical_nb(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: CategoricalNB,
    X: str,
    name: str = "categorical_nb",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.naive_bayes.CategoricalNB` into ONNX.

    The joint log-likelihood for class *c* is:

    .. code-block:: text

        jll[n, c] = class_log_prior_[c]
                    + Σ_f  feature_log_prob_[f][c, X[n, f]]

    For each feature *f*, ``feature_log_prob_[f]`` has shape ``(C, n_categories[f])``.
    The contribution is looked up via a Gather on the transposed table
    ``feature_log_prob_[f].T`` (shape ``n_categories[f] × C``) using the
    integer feature column ``X[:, f]`` as indices.

    probabilities  ← Softmax(jll, axis=1)
    label          ← classes_[ArgMax(jll, axis=1)]

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired names (label, probabilities)
    :param estimator: a fitted ``CategoricalNB``
    :param X: input tensor name (integer category indices)
    :param name: prefix for added node names
    :return: tuple ``(label_result_name, proba_result_name)``
    """
    assert isinstance(
        estimator, CategoricalNB
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    class_log_prior = estimator.class_log_prior_.astype(dtype)  # (C,)
    n_features = len(estimator.feature_log_prob_)

    # Cast X to int64 for Gather (feature values are integer category indices)
    X_int = g.op.Cast(X, to=onnx.TensorProto.INT64, name=f"{name}_cast_int")

    # Accumulate contributions from each feature
    jll: Optional[str] = None
    for f in range(n_features):
        # feature_log_prob_[f] shape: (C, n_categories[f])
        # Transpose to (n_categories[f], C) for row-wise Gather
        table = estimator.feature_log_prob_[f].T.astype(dtype)  # (n_cat_f, C)

        # Extract column f as a 1-D integer vector of shape (N,)
        # Gather with a scalar index on axis=1 gives shape (N,)
        X_f = g.op.Gather(
            X_int,
            np.array(f, dtype=np.int64),
            axis=1,
            name=f"{name}_col_{f}",
        )

        # Look up log-probs: contrib shape (N, C)
        contrib = g.op.Gather(table, X_f, axis=0, name=f"{name}_contrib_{f}")

        if jll is None:
            jll = contrib  # type: ignore
        else:
            jll = g.op.Add(jll, contrib, name=f"{name}_add_{f}")  # type: ignore

    # Add class log-prior
    jll = g.op.Add(jll, class_log_prior, name=f"{name}_jll")  # type: ignore

    return _emit_label_and_proba(g, sts, jll, estimator.classes_, dtype, name, outputs, itype)  # type: ignore

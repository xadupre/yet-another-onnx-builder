from typing import Dict, List, Tuple, Union
import numpy as np
import onnx
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


def _emit_label_and_proba(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    jll: str,
    classes: np.ndarray,
    dtype,
    name: str,
    outputs: List[str],
    itype: int,
) -> Tuple[str, str]:
    """Shared helper: argmax(jll) → label, softmax(jll) → proba."""
    proba = g.op.Softmax(jll, axis=1, name=name, outputs=outputs[1:])
    assert isinstance(proba, str)

    label_idx = g.op.ArgMax(jll, axis=1, keepdims=0, name=name)
    label_idx_cast = g.op.Cast(label_idx, to=onnx.TensorProto.INT64, name=name)

    if np.issubdtype(classes.dtype, np.integer):
        classes_arr = classes.astype(np.int64)
        label = g.op.Gather(
            classes_arr, label_idx_cast, axis=0, name=f"{name}_label", outputs=outputs[:1]
        )
        assert isinstance(label, str)
        g.set_type(label, onnx.TensorProto.INT64)
    else:
        classes_arr = np.array(classes.astype(str))
        label = g.op.Gather(
            classes_arr, label_idx_cast, axis=0, name=f"{name}_label_string", outputs=outputs[:1]
        )
        assert isinstance(label, str)
        g.set_type(label, onnx.TensorProto.STRING)

    return label, proba


@register_sklearn_converter(GaussianNB)
def sklearn_gaussian_nb(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: GaussianNB,
    X: str,
    name: str = "gaussian_nb",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.naive_bayes.GaussianNB` into ONNX.

    The joint log-likelihood for class *c* is:

    .. code-block:: text

        jll[n, c] = log(prior[c])
                    - 0.5 * Σ_f  log(2π·var[c,f])
                    - 0.5 * Σ_f  (x[n,f] - θ[c,f])² / var[c,f]

    Rewritten as a quadratic form to avoid per-class broadcast:

    .. code-block:: text

        A  = -0.5 / var_          (C x F)
        B  = θ_ / var_            (C x F)
        K  = log(prior)
             - 0.5 · Σ_f log(2π·var)
             - 0.5 · Σ_f θ²/var   (C,)

        jll = (X² @ Aᵀ) + (X @ Bᵀ) + K    (N x C)

    probabilities  ← Softmax(jll, axis=1)
    label          ← classes_[ArgMax(jll, axis=1)]

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired names (label, probabilities)
    :param estimator: a fitted ``GaussianNB``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: tuple ``(label_result_name, proba_result_name)``
    """
    assert isinstance(estimator, GaussianNB), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    var = estimator.var_.astype(dtype)  # (C, F)
    theta = estimator.theta_.astype(dtype)  # (C, F)
    log_prior = np.log(estimator.class_prior_).astype(dtype)  # (C,)

    # Precomputed constants
    A = (-0.5 / var).astype(dtype)  # (C, F)
    B = (theta / var).astype(dtype)  # (C, F)
    K = (
        log_prior
        - 0.5 * np.sum(np.log(2.0 * np.pi * var), axis=1)
        - 0.5 * np.sum(theta**2 / var, axis=1)
    ).astype(
        dtype
    )  # (C,)

    # X^2
    X2 = g.op.Mul(X, X, name=f"{name}_x2")

    # (X^2 @ A.T)  +  (X @ B.T)  +  K
    quad = g.op.MatMul(X2, A.T, name=f"{name}_quad")
    lin = g.op.MatMul(X, B.T, name=f"{name}_lin")
    jll = g.op.Add(g.op.Add(quad, lin, name=f"{name}_ql"), K, name=f"{name}_jll")

    return _emit_label_and_proba(g, sts, jll, estimator.classes_, dtype, name, outputs, itype)  # type: ignore


@register_sklearn_converter((MultinomialNB, ComplementNB))
def sklearn_multinomial_nb(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: Union[MultinomialNB, ComplementNB],
    X: str,
    name: str = "multinomial_nb",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.naive_bayes.MultinomialNB` or
    :class:`sklearn.naive_bayes.ComplementNB` into ONNX.

    **MultinomialNB**:

    .. code-block:: text

        jll = X @ feature_log_prob_ᵀ + class_log_prior_    (N x C)

    **ComplementNB** (multi-class, no ``class_log_prior_`` added):

    .. code-block:: text

        jll = X @ feature_log_prob_ᵀ                       (N x C)

    probabilities  ← Softmax(jll, axis=1)
    label          ← classes_[ArgMax(jll, axis=1)]

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired names (label, probabilities)
    :param estimator: a fitted ``MultinomialNB`` or ``ComplementNB``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: tuple ``(label_result_name, proba_result_name)``
    """
    assert isinstance(
        estimator, (MultinomialNB, ComplementNB)
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    feature_log_prob = estimator.feature_log_prob_.astype(dtype)  # (C, F)

    jll = g.op.MatMul(X, feature_log_prob.T, name=f"{name}_jll_inner")

    # ComplementNB omits class_log_prior_ when n_classes > 1
    is_complement = isinstance(estimator, ComplementNB)
    if not is_complement or len(estimator.classes_) == 1:
        class_log_prior = estimator.class_log_prior_.astype(dtype)  # (C,)
        jll = g.op.Add(jll, class_log_prior, name=f"{name}_jll")

    return _emit_label_and_proba(g, sts, jll, estimator.classes_, dtype, name, outputs, itype)  # type: ignore


@register_sklearn_converter(BernoulliNB)
def sklearn_bernoulli_nb(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: BernoulliNB,
    X: str,
    name: str = "bernoulli_nb",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.naive_bayes.BernoulliNB` into ONNX.

    The joint log-likelihood follows the sklearn formula:

    .. code-block:: text

        neg_prob  = log1p(-exp(feature_log_prob_))   (C x F)  — precomputed
        diff      = feature_log_prob_ - neg_prob       (C x F)  — precomputed
        neg_sum   = Σ_f neg_prob                       (C,)     — precomputed

        X_bin     = (X > binarize).astype(dtype)       if binarize is not None
        jll       = X_bin @ diffᵀ  +  class_log_prior_  +  neg_sum   (N x C)

    probabilities  ← Softmax(jll, axis=1)
    label          ← classes_[ArgMax(jll, axis=1)]

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired names (label, probabilities)
    :param estimator: a fitted ``BernoulliNB``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: tuple ``(label_result_name, proba_result_name)``
    """
    assert isinstance(estimator, BernoulliNB), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    feature_log_prob = estimator.feature_log_prob_.astype(dtype)  # (C, F)
    class_log_prior = estimator.class_log_prior_.astype(dtype)  # (C,)

    # Precompute constants
    neg_prob = np.log1p(-np.exp(feature_log_prob)).astype(dtype)  # (C, F)
    diff = (feature_log_prob - neg_prob).astype(dtype)  # (C, F)
    bias = (class_log_prior + neg_prob.sum(axis=1)).astype(dtype)  # (C,)

    # Optionally binarize input
    if estimator.binarize is not None:
        threshold = np.array(estimator.binarize, dtype=dtype)
        X_bin = g.op.Cast(
            g.op.Greater(X, threshold, name=f"{name}_greater"), to=itype, name=f"{name}_binarize"
        )
    else:
        X_bin = X

    jll = g.op.Add(g.op.MatMul(X_bin, diff.T, name=f"{name}_jll_inner"), bias, name=f"{name}_jll")

    return _emit_label_and_proba(g, sts, jll, estimator.classes_, dtype, name, outputs, itype)  # type: ignore

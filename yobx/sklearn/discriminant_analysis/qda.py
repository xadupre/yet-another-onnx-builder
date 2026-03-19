from typing import Dict, List, Tuple

import numpy as np
import onnx
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(QuadraticDiscriminantAnalysis)
def sklearn_quadratic_discriminant_analysis(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: QuadraticDiscriminantAnalysis,
    X: str,
    name: str = "qda",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`
    into ONNX.

    For each class *k* the log-likelihood (decision function) is computed from
    the per-class SVD stored in ``rotations_`` and ``scalings_`` (eigenvalues
    of the class covariance matrix):

    .. code-block:: text

        W_k     = R_k * S_k^(-0.5)                  # scaled rotation (F, r_k) – constant
        offset_k = mean_k @ W_k                      # (r_k,) – constant
        const_k  = -0.5 * sum(log S_k) + log(prior_k)  # scalar – constant

        z_k     = X @ W_k - offset_k                # (N, r_k)
        norm2_k = ReduceSum(z_k * z_k, axis=1)      # (N,)
        dec_k   = -0.5 * norm2_k + const_k          # (N,)

    When all classes share the same SVD rank *r*, all classes are processed in
    a single batched ``MatMul`` following the same pattern as the
    :func:`~yobx.sklearn.mixture.gaussian_mixture.sklearn_gaussian_mixture`
    ``'full'`` covariance path:

    .. code-block:: text

        W_2d   = hstack(W_0, …, W_{C-1})            # (F, C*r) – constant
        b_flat = hstack(offset_0, …, offset_{C-1})  # (1, C*r) – constant
        consts = [-0.5*logdet_k + logprior_k …]     # (C,) – constant

        XW     = MatMul(X, W_2d)                    # (N, C*r)
        diff   = XW - b_flat                         # (N, C*r)
        diff3d = Reshape(diff, [-1, C, r])           # (N, C, r)
        norm2  = ReduceSum(diff3d * diff3d, axis=2)  # (N, C)
        dec    = -0.5 * norm2 + consts               # (N, C)

    When classes have different SVD ranks (degenerate covariances), the same
    computation is performed **per class** and the resulting ``(N, 1)`` columns
    are concatenated.

    In both cases probabilities and labels are obtained as:

    .. code-block:: text

        proba = Softmax(dec, axis=1)                 # (N, C) – output
        label = Gather(classes_, ArgMax(proba, 1))   # (N,)   – output

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``QuadraticDiscriminantAnalysis``
    :param outputs: desired names (label, probabilities)
    :param X: input tensor name
    :param name: prefix names for the added nodes
    :return: tuple ``(label_result_name, proba_result_name)``
    """
    assert isinstance(
        estimator, QuadraticDiscriminantAnalysis
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    classes = estimator.classes_
    n_classes = len(classes)
    ranks = [estimator.rotations_[k].shape[1] for k in range(n_classes)]
    all_equal_rank = len(set(ranks)) == 1

    # Precompute per-class constants
    W_list = []
    offset_list = []
    const_list = []
    for k in range(n_classes):
        R_k = estimator.rotations_[k].astype(dtype)  # (F, r_k)
        S_k = estimator.scalings_[k].astype(dtype)  # (r_k,)
        m_k = estimator.means_[k].astype(dtype)  # (F,)

        # Scaled rotation: W_k = R_k * S_k^(-0.5)  (each column scaled by 1/sqrt(eigenvalue))
        W_k = R_k * (S_k ** (-0.5))  # (F, r_k)
        # Bias correction: offset = mean @ W = mean @ (R * S^(-0.5))
        offset_k = m_k @ W_k  # (r_k,)
        # Scalar log-likelihood constant: -0.5 * log|Σ_k| + log(prior_k)
        # where log|Σ_k| = sum(log(eigenvalues)) = sum(log(S_k))
        log_det_k = float(np.sum(np.log(S_k)))
        log_prior_k = float(np.log(estimator.priors_[k]))
        const_k = -0.5 * log_det_k + log_prior_k

        W_list.append(W_k)
        offset_list.append(offset_k)
        const_list.append(const_k)

    # --- Batched path (all classes share the same SVD rank) ---
    if all_equal_rank:
        r = ranks[0]
        # W_2d: (F, C*r) — columns of all W_k concatenated
        W_2d = np.concatenate(W_list, axis=1).astype(dtype)  # (F, C*r)
        # b_flat: (1, C*r) — offsets concatenated
        b_flat = np.concatenate(offset_list).reshape(1, -1).astype(dtype)  # (1, C*r)
        # consts: (C,) — per-class scalar constants
        consts = np.array(const_list, dtype=dtype)  # (C,)

        # (N, F) @ (F, C*r) → (N, C*r)
        XW = g.op.MatMul(X, W_2d, name=f"{name}_XW")
        diff = g.op.Sub(XW, b_flat, name=f"{name}_diff")  # (N, C*r)
        diff_sq = g.op.Mul(diff, diff, name=f"{name}_diff_sq")  # (N, C*r)
        # Reshape to (N, C, r) then sum over last axis
        diff_sq_3d = g.op.Reshape(
            diff_sq, np.array([-1, n_classes, r], dtype=np.int64), name=f"{name}_diff_sq_3d"
        )  # (N, C, r)
        norm2 = g.op.ReduceSum(
            diff_sq_3d, np.array([2], dtype=np.int64), keepdims=0, name=f"{name}_norm2"
        )  # (N, C)
        # dec = -0.5 * norm2 + consts
        dec = g.op.Add(
            g.op.Mul(np.array([-0.5], dtype=dtype), norm2, name=f"{name}_half_norm2"),
            consts,
            name=f"{name}_dec",
        )  # (N, C)

    # --- Per-class path (variable SVD ranks) ---
    else:
        dec_parts = []
        for k in range(n_classes):
            W_k = W_list[k].astype(dtype)
            b_k = (-offset_list[k]).astype(
                dtype
            )  # Gemm computes A @ B + C, so pass -offset to achieve X @ W_k - offset_k
            const_k = np.array([const_list[k]], dtype=dtype)

            # z_k = X @ W_k - offset_k  (Gemm: A @ B + C  with C = -offset)
            z_k = g.op.Gemm(X, W_k, b_k, name=f"{name}_z{k}")  # (N, r_k)
            z_sq = g.op.Mul(z_k, z_k, name=f"{name}_zsq{k}")  # (N, r_k)
            norm2_k = g.op.ReduceSum(
                z_sq, np.array([1], dtype=np.int64), keepdims=0, name=f"{name}_norm2_{k}"
            )  # (N,)
            # decision_k = -0.5 * norm2_k + const_k
            dec_k = g.op.Add(
                g.op.Mul(np.array([-0.5], dtype=dtype), norm2_k, name=f"{name}_half_{k}"),
                const_k,
                name=f"{name}_dec_{k}",
            )  # (N,)
            dec_col = g.op.Unsqueeze(
                dec_k, np.array([1], dtype=np.int64), name=f"{name}_col_{k}"
            )  # (N, 1)
            dec_parts.append(dec_col)
        dec = g.op.Concat(*dec_parts, axis=1, name=f"{name}_dec")  # (N, C)

    # Softmax → probabilities
    proba = g.op.Softmax(dec, axis=1, name=name, outputs=outputs[1:])

    # Label: ArgMax → Gather(classes_)
    label_idx = g.op.ArgMax(proba, axis=1, keepdims=0, name=name)
    label_idx_cast = g.op.Cast(label_idx, to=onnx.TensorProto.INT64, name=name)

    if np.issubdtype(classes.dtype, np.integer):
        classes_arr = classes.astype(np.int64)
        label = g.op.Gather(
            classes_arr, label_idx_cast, axis=0, name=f"{name}_label", outputs=outputs[:1]
        )
        g.set_type(label, onnx.TensorProto.INT64)
    else:
        classes_arr = np.array(classes.astype(str))
        label = g.op.Gather(
            classes_arr, label_idx_cast, axis=0, name=f"{name}_label_string", outputs=outputs[:1]
        )
        g.set_type(label, onnx.TensorProto.STRING)

    return label, proba

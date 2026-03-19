import numpy as np
from typing import Dict, List

from sklearn.decomposition import LatentDirichletAllocation

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


def _build_digamma(g: GraphBuilderExtendedProtocol, x: str, name: str, dtype: np.dtype) -> str:
    """
    Build ONNX nodes that approximate the digamma (psi) function.

    Uses the recurrence relation ``psi(x) = psi(x+N) - sum_{k=0}^{N-1} 1/(x+k)``
    together with an asymptotic expansion at ``x+N`` for large arguments:

    .. code-block:: text

        psi(x+N) ≈ ln(x+N) - 1/(2*(x+N)) - 1/(12*(x+N)^2)
                            + 1/(120*(x+N)^4) - 1/(252*(x+N)^6)

    With ``N=8`` the approximation error is below 1e-9 for all ``x > 0``,
    which is well within ``float32`` precision.

    :param g: graph builder
    :param x: name of the input tensor (any shape, all elements > 0)
    :param name: node name prefix
    :param dtype: numpy float dtype (float32 or float64)
    :return: output tensor name (same shape as *x*)
    """
    N = 8  # recurrence steps; shifts x to >= 9 when x >= 1

    # --- recurrence sum: sum_{k=0}^{N-1} 1/(x + k) ---
    recurrence = None
    for k in range(N):
        x_k = g.op.Add(x, np.array(k, dtype=dtype), name=f"{name}_dgk{k}")
        inv_xk = g.op.Reciprocal(x_k, name=f"{name}_dgrec{k}")
        if recurrence is None:
            recurrence = inv_xk
        else:
            recurrence = g.op.Add(recurrence, inv_xk, name=f"{name}_dgsum{k}")

    # --- asymptotic expansion at x_large = x + N ---
    x_large = g.op.Add(x, np.array(N, dtype=dtype), name=f"{name}_dg_xl")
    log_xl = g.op.Log(x_large, name=f"{name}_dg_log")
    inv_xl = g.op.Reciprocal(x_large, name=f"{name}_dg_inv")
    inv_xl2 = g.op.Mul(inv_xl, inv_xl, name=f"{name}_dg_inv2")
    inv_xl4 = g.op.Mul(inv_xl2, inv_xl2, name=f"{name}_dg_inv4")
    inv_xl6 = g.op.Mul(inv_xl4, inv_xl2, name=f"{name}_dg_inv6")

    t1 = g.op.Mul(np.array(0.5, dtype=dtype), inv_xl, name=f"{name}_dg_t1")
    t2 = g.op.Mul(np.array(1.0 / 12.0, dtype=dtype), inv_xl2, name=f"{name}_dg_t2")
    t3 = g.op.Mul(np.array(1.0 / 120.0, dtype=dtype), inv_xl4, name=f"{name}_dg_t3")
    t4 = g.op.Mul(np.array(1.0 / 252.0, dtype=dtype), inv_xl6, name=f"{name}_dg_t4")

    # psi(x_large) = ln(x_large) - t1 - t2 + t3 - t4
    psi_xl = g.op.Sub(log_xl, t1, name=f"{name}_dg_a")
    psi_xl = g.op.Sub(psi_xl, t2, name=f"{name}_dg_b")
    psi_xl = g.op.Add(psi_xl, t3, name=f"{name}_dg_c")
    psi_xl = g.op.Sub(psi_xl, t4, name=f"{name}_dg_d")

    # psi(x) = psi(x_large) - recurrence_sum
    return g.op.Sub(psi_xl, recurrence, name=f"{name}_dg_out")


def _build_dirichlet_expectation(
    g: GraphBuilderExtendedProtocol, gamma: str, name: str, dtype: np.dtype
) -> str:
    """
    Build ONNX nodes computing ``exp(digamma(gamma) - digamma(rowsum(gamma)))``.

    This is the Dirichlet expectation ``E[theta | gamma]`` used in the
    variational E-step of LDA.

    :param g: graph builder
    :param gamma: name of the 2-D tensor ``(n_samples, n_topics)``
    :param name: node name prefix
    :param dtype: numpy float dtype
    :return: output tensor name ``(n_samples, n_topics)``
    """
    psi_gamma = _build_digamma(g, gamma, name=f"{name}_psi_g", dtype=dtype)
    gamma_sum = g.op.ReduceSum(
        gamma, np.array([1], dtype=np.int64), keepdims=1, name=f"{name}_gsum"
    )
    psi_sum = _build_digamma(g, gamma_sum, name=f"{name}_psi_s", dtype=dtype)
    diff = g.op.Sub(psi_gamma, psi_sum, name=f"{name}_psi_diff")
    return g.op.Exp(diff, name=f"{name}_exp_dt")


@register_sklearn_converter(LatentDirichletAllocation)
def sklearn_latent_dirichlet_allocation(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: LatentDirichletAllocation,
    X: str,
    name: str = "lda",
) -> str:
    """
    Converts a :class:`sklearn.decomposition.LatentDirichletAllocation`
    into ONNX.

    The converter implements the variational E-step used by
    :meth:`~sklearn.decomposition.LatentDirichletAllocation.transform`.
    Starting from a uniform document-topic distribution, it iterates
    ``max_doc_update_iter`` times (no early-stopping tolerance check):

    .. code-block:: text

        gamma  ←  ones((N, K))
        exp_dt ←  exp(digamma(gamma) − digamma(rowsum(gamma)))

        for _ in range(max_doc_update_iter):
            norm_phi ←  exp_dt @ exp_W + ε         (N, F)
            gamma    ←  exp_dt * (X / norm_phi @ exp_Wᵀ) + α   (N, K)
            exp_dt   ←  exp(digamma(gamma) − digamma(rowsum(gamma)))

        output ←  gamma / rowsum(gamma)              (N, K)

    where ``exp_W`` is ``exp_dirichlet_component_`` (K × F), ``α`` is
    ``doc_topic_prior_``, and ``ε`` is the floating-point machine epsilon.

    .. note::
        The Digamma function is approximated via the asymptotic expansion
        ``ψ(x) ≈ ln(x) − 1/(2x) − 1/(12x²) + 1/(120x⁴) − 1/(252x⁶)``
        after 8 recurrence steps.  The approximation error is below 1e-9
        for all positive inputs, comfortably within float32 precision.

    .. note::
        Unlike sklearn's sparse implementation, this converter processes
        all word features densely.  For documents with many zero counts the
        zero entries contribute nothing to the update, so the results are
        numerically identical.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names (document-topic distribution)
    :param estimator: a fitted ``LatentDirichletAllocation``
    :param X: input tensor name – word-count matrix ``(N, n_features)``
    :param name: prefix name for the added nodes
    :return: output tensor name ``(N, n_components)``
    """
    assert isinstance(
        estimator, LatentDirichletAllocation
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    n_topics = estimator.n_components
    exp_W = estimator.exp_dirichlet_component_.astype(dtype)  # (n_topics, n_features)
    alpha = np.array(estimator.doc_topic_prior_, dtype=dtype)
    eps = np.array(np.finfo(dtype).eps, dtype=dtype)

    # --- initialise gamma = ones((batch_size, n_topics)) ---
    x_shape = g.op.Shape(X, name=f"{name}_xshape")
    batch_size = g.op.Slice(
        x_shape,
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        name=f"{name}_batch",
    )  # 1-D tensor [N]

    n_topics_arr = np.array([n_topics], dtype=np.int64)
    gamma_shape = g.op.Concat(batch_size, n_topics_arr, axis=0, name=f"{name}_gshape")
    ones_1k = np.ones((1, n_topics), dtype=dtype)
    gamma = g.op.Expand(ones_1k, gamma_shape, name=f"{name}_gamma0")

    # --- initial Dirichlet expectation ---
    exp_dt = _build_dirichlet_expectation(g, gamma, name=f"{name}_init", dtype=dtype)

    # --- unrolled E-step iterations ---
    for i in range(estimator.max_doc_update_iter):
        iname = f"{name}_it{i}"

        # norm_phi = exp_dt @ exp_W + eps  (N, F)
        norm_phi = g.op.MatMul(exp_dt, exp_W, name=f"{iname}_nphi")
        norm_phi = g.op.Add(norm_phi, eps, name=f"{iname}_nphi_eps")

        # gamma = exp_dt * ((X / norm_phi) @ exp_W.T) + alpha  (N, K)
        ratio = g.op.Div(X, norm_phi, name=f"{iname}_ratio")
        contrib = g.op.MatMul(ratio, exp_W.T, name=f"{iname}_contrib")
        gamma = g.op.Mul(exp_dt, contrib, name=f"{iname}_gam_raw")
        gamma = g.op.Add(gamma, alpha, name=f"{iname}_gamma")

        # update exp_dt
        exp_dt = _build_dirichlet_expectation(g, gamma, name=iname, dtype=dtype)

    # --- normalise rows ---
    gamma_sum = g.op.ReduceSum(
        gamma, np.array([1], dtype=np.int64), keepdims=1, name=f"{name}_final_sum"
    )
    res = g.op.Div(gamma, gamma_sum, name=name, outputs=outputs)
    g.set_type(res, itype)
    return res

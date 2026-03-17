import numpy as np
from typing import Dict, List, Union

from sklearn.decomposition import NMF, MiniBatchNMF

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter((NMF, MiniBatchNMF))
def sklearn_nmf(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: Union[NMF, MiniBatchNMF],
    X: str,
    name: str = "nmf",
) -> str:
    """
    Converts a :class:`sklearn.decomposition.NMF` or
    :class:`sklearn.decomposition.MiniBatchNMF` into ONNX.

    The converter implements the **multiplicative update** (MU) rule used by
    :meth:`~sklearn.decomposition.NMF.transform` for the Frobenius
    (β = 2) loss.  All constant matrices are pre-computed at conversion
    time; only ``XHt = X @ H.T`` is computed at inference time.

    Starting from a uniform initialization
    ``W₀ = sqrt(mean(X) / n_components)`` (matching sklearn's MU
    initialization), the update rule is unrolled for ``max_iter``
    steps:

    .. code-block:: text

        H    = components_                           (k, f)
        HHt  = H @ H.T                              (k, k)  [constant]
        XHt  = X @ H.T                              (N, k)  [runtime]
        W₀   = sqrt(mean(X) / k) · ones(N, k)      (N, k)  [runtime]

        for _ in range(max_iter):
            denom  = max(W @ HHt [+ l1] [+ l2·W],  ε)
            W      = W · (XHt / denom)

    .. note::
        For :class:`~sklearn.decomposition.NMF` this converter only
        supports ``solver='mu'`` with ``beta_loss`` set to
        ``'frobenius'`` or ``2``.  Other solver / loss combinations
        raise :exc:`NotImplementedError`.

        For :class:`~sklearn.decomposition.MiniBatchNMF` the default
        Frobenius loss is always supported (``beta_loss='frobenius'``).

    .. note::
        Because the converter always runs exactly ``max_iter``
        multiplicative steps (no early-stopping tolerance check), the
        output may differ slightly from sklearn when the model converged
        before ``max_iter`` with a non-zero ``tol``.  Set ``tol=0`` on
        the estimator before fitting to obtain bit-exact results.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names (latent representation W)
    :param estimator: a fitted ``NMF`` or ``MiniBatchNMF``
    :param X: input tensor name – non-negative matrix ``(N, n_features)``
    :param name: prefix name for the added nodes
    :return: output tensor name ``(N, n_components)``
    """
    assert isinstance(
        estimator, (NMF, MiniBatchNMF)
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    # --- validate configuration ---
    beta_loss = estimator._beta_loss
    if beta_loss != 2:
        raise NotImplementedError(
            f"NMF ONNX converter only supports beta_loss='frobenius' (2), "
            f"got {beta_loss!r}."
        )

    if isinstance(estimator, NMF) and estimator.solver != "mu":
        raise NotImplementedError(
            f"NMF ONNX converter only supports solver='mu', "
            f"got {estimator.solver!r}."
        )

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)
    eps = np.finfo(dtype).eps

    H = estimator.components_.astype(dtype)  # (k, f)
    n_k = H.shape[0]

    # --- pre-compute constants ---
    HHt = (H @ H.T).astype(dtype)  # (k, k)

    # Regularisation terms (scaled by n_features, always constant).
    # _compute_regularization uses n_features (not n_samples) for W terms.
    n_features = H.shape[1]
    alpha_W = float(estimator.alpha_W)
    l1_ratio = float(estimator.l1_ratio)
    l1_reg_W = n_features * alpha_W * l1_ratio
    l2_reg_W = n_features * alpha_W * (1.0 - l1_ratio)

    # Number of MU iterations for transform.
    if isinstance(estimator, MiniBatchNMF):
        max_iter = int(estimator._transform_max_iter)
    else:
        max_iter = int(estimator.max_iter)

    # --- runtime: XHt = X @ H.T  (N, k) ---
    XHt = g.op.MatMul(X, H.T, name=f"{name}_XHt")  # (N, k)

    # --- runtime: initialise W = sqrt(mean(X) / k) * ones(N, k) ---
    x_mean = g.op.ReduceMean(X, keepdims=0, name=f"{name}_xmean")  # scalar
    inv_k = np.array(1.0 / n_k, dtype=dtype)
    w_scalar = g.op.Sqrt(
        g.op.Mul(x_mean, inv_k, name=f"{name}_mean_div_k"),
        name=f"{name}_w_scalar",
    )  # scalar
    w_11 = g.op.Reshape(w_scalar, np.array([1, 1], dtype=np.int64), name=f"{name}_w_11")

    x_shape = g.op.Shape(X, name=f"{name}_xshape")
    batch_size = g.op.Slice(
        x_shape,
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        name=f"{name}_batch",
    )  # 1-D int64 tensor [N]
    n_k_arr = np.array([n_k], dtype=np.int64)
    w_shape = g.op.Concat(batch_size, n_k_arr, axis=0, name=f"{name}_wshape")
    W = g.op.Expand(w_11, w_shape, name=f"{name}_W0")  # (N, k)

    # --- unrolled MU iterations ---
    eps_arr = np.array(eps, dtype=dtype)
    for i in range(max_iter):
        iname = f"{name}_it{i}"

        # denom = W @ HHt  (N, k)
        denom = g.op.MatMul(W, HHt, name=f"{iname}_denom")

        # add L1 regularisation (constant)
        if l1_reg_W > 0:
            denom = g.op.Add(
                denom, np.array(l1_reg_W, dtype=dtype), name=f"{iname}_l1"
            )

        # add L2 regularisation (W-dependent)
        if l2_reg_W > 0:
            denom = g.op.Add(
                denom,
                g.op.Mul(W, np.array(l2_reg_W, dtype=dtype), name=f"{iname}_l2w"),
                name=f"{iname}_l2",
            )

        # clip denominator to avoid division by zero
        denom = g.op.Max(denom, eps_arr, name=f"{iname}_denom_safe")

        # W = W * (XHt / denom)
        W = g.op.Mul(W, g.op.Div(XHt, denom, name=f"{iname}_ratio"), name=f"{iname}_W")

    # assign output names
    res = g.op.Identity(W, name=name, outputs=outputs)
    assert isinstance(res, str)
    if not sts:
        g.set_type(res, itype)
    return res

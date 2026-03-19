import numpy as np
from typing import Dict, List, Union

from onnx import TensorProto
from onnx.helper import make_graph, make_node, make_tensor_value_info
import onnx.numpy_helper as onh

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
    initialization), the update rule is implemented as an ONNX ``Loop``
    running for ``max_iter`` steps:

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
            f"NMF ONNX converter only supports beta_loss='frobenius' (2), got {beta_loss!r}."
        )

    if isinstance(estimator, NMF) and estimator.solver != "mu":
        raise NotImplementedError(
            f"NMF ONNX converter only supports solver='mu', got {estimator.solver!r}."
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
        g.op.Mul(x_mean, inv_k, name=f"{name}_mean_div_k"), name=f"{name}_w_scalar"
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
    W0 = g.op.Expand(w_11, w_shape, name=f"{name}_W0")  # (N, k)

    # --- ONNX Loop body: one MU update step ---
    # Body-internal names (prefixed to keep them unique across converters)
    bp = f"{name}_b"  # body prefix
    b_iter = f"{bp}_iter"
    b_cond_in = f"{bp}_cond_in"
    b_cond_out = f"{bp}_cond_out"
    b_W_in = f"{bp}_W_in"
    b_W_out = f"{bp}_W_out"
    b_denom = f"{bp}_denom"
    b_denom_l1 = f"{bp}_denom_l1"
    b_denom_l2 = f"{bp}_denom_l2"
    b_l2_term = f"{bp}_l2_term"
    b_denom_safe = f"{bp}_denom_safe"
    b_ratio = f"{bp}_ratio"

    # Constants embedded as body initializers
    body_inits = [
        onh.from_array(HHt, name=f"{bp}_HHt"),
        onh.from_array(np.array(eps, dtype=dtype), name=f"{bp}_eps"),  # machine epsilon scalar
    ]
    if l1_reg_W > 0:
        body_inits.append(onh.from_array(np.array(l1_reg_W, dtype=dtype), name=f"{bp}_l1"))
    if l2_reg_W > 0:
        body_inits.append(onh.from_array(np.array(l2_reg_W, dtype=dtype), name=f"{bp}_l2c"))

    # Build body nodes
    body_nodes = []

    # denom = W_in @ HHt
    body_nodes.append(make_node("MatMul", [b_W_in, f"{bp}_HHt"], [b_denom], name=f"{bp}_mm"))

    denom_cur = b_denom

    # optional L1 regularisation: denom += l1
    if l1_reg_W > 0:
        body_nodes.append(
            make_node("Add", [denom_cur, f"{bp}_l1"], [b_denom_l1], name=f"{bp}_l1_add")
        )
        denom_cur = b_denom_l1

    # optional L2 regularisation: denom += l2 * W_in
    if l2_reg_W > 0:
        body_nodes.append(
            make_node("Mul", [b_W_in, f"{bp}_l2c"], [b_l2_term], name=f"{bp}_l2_mul")
        )
        body_nodes.append(
            make_node("Add", [denom_cur, b_l2_term], [b_denom_l2], name=f"{bp}_l2_add")
        )
        denom_cur = b_denom_l2

    # denom_safe = Max(denom, eps)
    body_nodes.append(
        make_node("Max", [denom_cur, f"{bp}_eps"], [b_denom_safe], name=f"{bp}_max")
    )

    # ratio = XHt / denom_safe  (XHt is an outer-scope tensor)
    body_nodes.append(make_node("Div", [XHt, b_denom_safe], [b_ratio], name=f"{bp}_div"))

    # W_out = W_in * ratio
    body_nodes.append(make_node("Mul", [b_W_in, b_ratio], [b_W_out], name=f"{bp}_mul"))

    # cond_out = cond_in  (always continue; Loop exits after max_iter steps)
    body_nodes.append(make_node("Identity", [b_cond_in], [b_cond_out], name=f"{bp}_cond_pass"))

    loop_body = make_graph(
        body_nodes,
        f"{name}_loop_body",
        [
            make_tensor_value_info(b_iter, TensorProto.INT64, []),
            make_tensor_value_info(b_cond_in, TensorProto.BOOL, []),
            # W has shape (N, k) where N is dynamic at inference time
            make_tensor_value_info(b_W_in, itype, None),
        ],
        [
            make_tensor_value_info(b_cond_out, TensorProto.BOOL, []),
            # W output has the same dynamic shape as the input
            make_tensor_value_info(b_W_out, itype, None),
        ],
        body_inits,
    )

    # Loop control scalars: M = max_iter, initial cond = True
    M_name = g.make_initializer(f"{name}_M", np.array(max_iter, dtype=np.int64))
    cond_init = g.make_initializer(f"{name}_cond", np.array(True, dtype=np.bool_))

    # Create the Loop node; W0 is the initial loop-carried state
    W_final = g.unique_name(f"{name}_W_final")
    g.make_node("Loop", [M_name, cond_init, W0], [W_final], body=loop_body, name=f"{name}_loop")
    g.set_type(W_final, itype)

    res = g.op.Identity(W_final, name=name, outputs=outputs)
    g.set_type(res, itype)
    return res

import numpy as np
from typing import Dict, List
from sklearn.preprocessing import KernelCenterer
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(KernelCenterer)
def sklearn_kernel_centerer(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: KernelCenterer,
    X: str,
    name: str = "kernel_centerer",
) -> str:
    """
    Converts a :class:`sklearn.preprocessing.KernelCenterer` into ONNX.

    ``KernelCenterer`` centres a pre-computed kernel matrix ``K`` of shape
    ``(N, M)`` using statistics gathered from the training kernel matrix.
    The transformation implemented here mirrors
    :meth:`sklearn.preprocessing.KernelCenterer.transform`:

    .. code-block:: text

        K_pred_cols = K.sum(axis=1, keepdims=True) / n_train   → (N, 1)

        K_centered = K - K_fit_rows_              (broadcast: (N,M) - (M,))
                       - K_pred_cols              (broadcast: (N,M) - (N,1))
                       + K_fit_all_               (scalar)

    where:

    * ``K_fit_rows_`` — column means of the *training* kernel matrix,
      shape ``(n_train,)``, stored as a constant initializer.
    * ``K_fit_all_`` — grand mean of the training kernel matrix, scalar.
    * ``n_train`` — ``K_fit_rows_.shape[0]``, used to compute the
      per-sample row mean of the *prediction* kernel matrix.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names
    :param estimator: a fitted ``KernelCenterer``
    :param X: input tensor name (the kernel matrix ``K``)
    :param name: prefix for added node names
    :return: output tensor name
    """
    assert isinstance(
        estimator, KernelCenterer
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    n_train = int(estimator.K_fit_rows_.shape[0])
    n_train_arr = np.array([n_train], dtype=dtype)

    # Column means of the training kernel matrix, shape (n_train,).
    K_fit_rows = estimator.K_fit_rows_.astype(dtype)  # (M,)

    # Grand mean of the training kernel matrix, scalar.
    K_fit_all = np.array([estimator.K_fit_all_], dtype=dtype)  # (1,)

    # Step 1: per-sample row mean of the prediction kernel matrix.
    #   K_row_sum = K.sum(axis=1, keepdims=True)   → (N, 1)
    #   K_pred_cols = K_row_sum / n_train            → (N, 1)
    K_row_sum = g.op.ReduceSum(
        X,
        np.array([1], dtype=np.int64),
        keepdims=1,
        name=f"{name}_row_sum",
    )  # (N, 1)
    K_pred_cols = g.op.Div(K_row_sum, n_train_arr, name=f"{name}_pred_cols")  # (N, 1)

    # Step 2: subtract training column means (K_fit_rows_ broadcast over rows).
    K_sub_rows = g.op.Sub(X, K_fit_rows, name=f"{name}_sub_rows")  # (N, M)

    # Step 3: subtract per-sample row mean (K_pred_cols broadcast over columns).
    K_sub_cols = g.op.Sub(K_sub_rows, K_pred_cols, name=f"{name}_sub_cols")  # (N, M)

    # Step 4: add the grand mean.
    res = g.op.Add(K_sub_cols, K_fit_all, name=name, outputs=outputs)  # (N, M)

    assert isinstance(res, str)  # type happiness
    if not sts:
        g.set_type_shape_unary_op(res, X)
    return res

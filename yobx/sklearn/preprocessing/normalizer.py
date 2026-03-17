import numpy as np
from typing import Dict, List
from sklearn.preprocessing import Normalizer
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(Normalizer)
def sklearn_normalizer(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: Normalizer,
    X: str,
    name: str = "normalizer",
) -> str:
    """
    Converts a :class:`sklearn.preprocessing.Normalizer` into ONNX.

    Each sample (row) is scaled independently to unit norm.  The
    ``norm`` parameter selects which norm is used:

    .. code-block:: text

        norm = ReduceL2/L1/Max(|X|, axis=1, keepdims=True)   → (N, 1)
        safe_norm = Where(norm == 0, 1, norm)                  → (N, 1)
        output = X / safe_norm                                 → (N, F)

    When a row has zero norm it is left unchanged (divided by 1), which
    matches :func:`sklearn.preprocessing.normalize`.

    Supported values for ``norm``: ``'l2'`` (default), ``'l1'``,
    ``'max'``.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted or unfitted ``Normalizer``
    :param outputs: desired output names
    :param X: input tensor name
    :param name: prefix for added node names
    :return: output tensor name
    """
    assert isinstance(
        estimator, Normalizer
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    norm = estimator.norm
    axes = np.array([1], dtype=np.int64)

    if norm == "l2":
        norms = g.op.ReduceL2(X, axes, keepdims=1, name=f"{name}_l2norm")
    elif norm == "l1":
        norms = g.op.ReduceL1(X, axes, keepdims=1, name=f"{name}_l1norm")
    elif norm == "max":
        abs_X = g.op.Abs(X, name=f"{name}_abs")
        norms = g.op.ReduceMax(abs_X, axes, keepdims=1, name=f"{name}_maxnorm")
    else:
        raise ValueError(
            f"Unknown norm={norm!r} for Normalizer, expected 'l1', 'l2', or 'max'."
        )

    # Replace zero norms with 1 so that zero rows are left unchanged.
    zero = np.array([0], dtype=dtype)
    one = np.array([1], dtype=dtype)
    is_zero = g.op.Equal(norms, zero, name=f"{name}_is_zero")
    safe_norms = g.op.Where(is_zero, one, norms, name=f"{name}_safe_norm")

    res = g.op.Div(X, safe_norms, name=name, outputs=outputs)

    assert isinstance(res, str)  # type happiness
    if not sts:
        g.set_type_shape_unary_op(res, X)
    return res

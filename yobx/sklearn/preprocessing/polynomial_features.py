import numpy as np
from typing import Dict, List
from sklearn.preprocessing import PolynomialFeatures
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(PolynomialFeatures)
def sklearn_polynomial_features(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: PolynomialFeatures,
    X: str,
    name: str = "polynomial_features",
) -> str:
    """
    Converts a :class:`sklearn.preprocessing.PolynomialFeatures` into ONNX.

    The output is the matrix of all polynomial combinations of the input
    features up to the given degree. Each output feature corresponds to
    one row of ``estimator.powers_``, where each row defines the exponent
    for each input feature.

    The graph is built as follows:

    .. code-block:: text

        X  ──Unsqueeze(axis=1)──►  X_3d (N, 1, F)
                                       │
                           ┌───────────┤
                   powers_3d           │   (1, K, F) constant
                   zero_mask_3d        │   (1, K, F) constant bool
                           │           │
                        Where(zero_mask_3d, 1.0, X_3d) ──► X_safe (N, K, F)
                                       │
                        Pow(X_safe, powers_3d) ──► powered (N, K, F)
                                       │
                        ReduceProd(axis=-1) ──► output (N, K)

    where *K* = ``n_output_features_`` and *F* = ``n_features_in_``.

    The ``Where`` guard ensures that ``0 ** 0 = 1`` (sklearn convention) is
    satisfied even when the input contains zeros: whenever the exponent is 0,
    the base is replaced by 1 before the ``Pow`` operation.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names
    :param estimator: a fitted ``PolynomialFeatures``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: output tensor name
    """
    assert isinstance(
        estimator, PolynomialFeatures
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # powers_ has shape (n_output_features, n_features_in_)
    powers = estimator.powers_  # int array, shape (K, F)
    n_output_features, n_features_in = powers.shape

    # Reshape powers to (1, K, F) for broadcasting with X (N, 1, F).
    powers_3d = powers.reshape(1, n_output_features, n_features_in).astype(dtype)

    # Boolean mask: True where exponent == 0. Stored as a constant initializer.
    zero_mask_3d = powers.reshape(1, n_output_features, n_features_in) == 0

    # 1. Expand X from (N, F) → (N, 1, F)
    x_3d = g.op.Unsqueeze(X, np.array([1], dtype=np.int64), name=f"{name}_unsq")

    # 2. Replace X with 1 where the exponent is 0 to avoid 0^0 ambiguity.
    #    Where broadcasts: (1, K, F), (1, K, F), (N, 1, F) → (N, K, F)
    ones_3d = np.ones((1, n_output_features, n_features_in), dtype=dtype)
    x_safe = g.op.Where(zero_mask_3d, ones_3d, x_3d, name=f"{name}_safe")

    # 3. Raise each (safe) base to its exponent: (N, K, F)
    powered = g.op.Pow(x_safe, powers_3d, name=f"{name}_pow")

    # 4. Multiply over the feature axis to get (N, K).
    res = g.op.ReduceProdAnyOpset(
        powered,
        np.array([-1], dtype=np.int64),
        keepdims=0,
        name=name,
        outputs=outputs,
    )

    assert isinstance(res, str)  # type happiness
    return res

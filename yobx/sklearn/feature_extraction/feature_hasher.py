from typing import Dict, List, Optional

import numpy as np
import onnx
from sklearn.feature_extraction import FeatureHasher

from ...helpers.onnx_helper import np_dtype_to_tensor_dtype
from ...typing import GraphBuilderExtendedProtocol
from ..register import register_sklearn_converter


def _check_murmurhash(estimator: FeatureHasher) -> None:
    """Raises ``NotImplementedError`` if the estimator does not use murmurhash3_32.

    :class:`~sklearn.feature_extraction.FeatureHasher` always uses
    ``murmurhash3_32`` (it is hard-coded in the C extension), so this guard is
    a future-proof safety check in case a subclass overrides the hasher.
    """
    if type(estimator).__name__ != "FeatureHasher":
        raise NotImplementedError(
            f"Only sklearn.feature_extraction.FeatureHasher (murmurhash3_32) "
            f"is supported; got {type(estimator).__qualname__}."
        )


@register_sklearn_converter(FeatureHasher)
def sklearn_feature_hasher(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: FeatureHasher,
    X: str,
    X_values: Optional[str] = None,
    name: str = "feature_hasher",
) -> str:
    """
    Converts a :class:`sklearn.feature_extraction.FeatureHasher` into ONNX.

    :class:`~sklearn.feature_extraction.FeatureHasher` maps a sequence of
    feature dictionaries (or pairs, or strings) to a fixed-size dense matrix
    via the *hashing trick* — specifically ``murmurhash3_32`` with ``seed=0``
    and signed output (``positive=False``).

    This converter requires the ``com.microsoft`` opset (ONNX Runtime contrib
    ops) because the hashing is performed inline using
    ``com.microsoft.MurmurHash3``, which exactly matches sklearn's
    ``murmurhash3_32``.

    The primary input *X* must be a 2-D **string** tensor of shape ``(N, K)``
    where *K* is the maximum number of features per sample (shorter samples
    padded with ``""``).  A second companion float input *X_values* of the
    same shape carries the feature values (``1.0`` per non-padding entry for
    ``input_type='string'``, ``0.0`` for padding slots).

    .. code-block:: text

        X (N, K) STRING, X_values (N, K) float
          │
          ├── MurmurHash3(X, seed=0, positive=0) ──► hashes (N, K) INT32
          ├── abs(hashes) % n_features ─────────────► indices (N, K) INT64
          ├── where(hashes >= 0, +1, −1) ───────────► signs (N, K) float
          ├── signs * X_values ─────────────────────► weighted (N, K) float
          └── ScatterElements(zeros, indices, weighted,
                              axis=1, reduction='add') → output (N, n_features)

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn` (unused; present for
        interface consistency)
    :param estimator: a ``FeatureHasher`` instance
    :param outputs: desired output names
    :param X: primary input name — a ``STRING`` tensor of shape ``(N, K)``
        containing feature names (shorter samples padded with ``""``).
    :param X_values: companion float input of shape ``(N, K)`` with feature
        values (``1.0`` per non-padding entry, ``0.0`` for padding slots).
    :param name: prefix name for the added nodes
    :return: output name
    :raises NotImplementedError: if the ``com.microsoft`` opset is not
        registered in the graph builder, or if *X* is not a ``STRING`` tensor.
    """
    assert isinstance(
        estimator, FeatureHasher
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    _check_murmurhash(estimator)

    itype = g.get_type(X)

    if not g.has_opset("com.microsoft"):
        raise NotImplementedError(
            "FeatureHasher conversion requires the 'com.microsoft' opset "
            "(ONNX Runtime contrib ops) for the MurmurHash3 operator. "
            "Register it via target_opset={'': 18, 'com.microsoft': 1}."
        )

    if itype != onnx.TensorProto.STRING:
        raise NotImplementedError(
            f"FeatureHasher conversion requires a STRING input tensor (got ONNX "
            f"type {itype}). Pass a 2-D padded string array of feature names as "
            f"input X, together with a float array X_values of the same shape."
        )

    assert X_values is not None, (
        "FeatureHasher conversion requires a companion float input X_values of "
        "shape (N, K) with feature values (1.0 per non-padding feature, 0.0 for "
        f"padding slots). Pass it as the second positional argument to to_onnx()."
        f"{g.get_debug_msg()}"
    )
    assert g.has_type(X_values), f"Missing type for {X_values!r}{g.get_debug_msg()}"

    target_dtype = np.dtype(estimator.dtype)
    target_onnx_type = np_dtype_to_tensor_dtype(target_dtype)
    n_features = estimator.n_features
    alternate_sign = estimator.alternate_sign

    # Step 1: hash feature names
    hashes = g.make_node(
        "MurmurHash3", [X], domain="com.microsoft", seed=0, positive=0, name=f"{name}_mmh3"
    )

    # Step 2: indices = abs(hashes) % n_features  (INT64)
    abs_h = g.op.Abs(hashes, name=f"{name}_abs_hash")
    n_feat_i32 = np.array(n_features, dtype=np.int32)
    mod_out = g.op.Mod(abs_h, n_feat_i32, name=f"{name}_mod")
    indices = g.op.Cast(mod_out, to=onnx.TensorProto.INT64, name=f"{name}_idx")

    # Scalar float constants used as type-cast reference
    one_f = np.array(1.0, dtype=target_dtype)

    # Cast X_values to target dtype
    vals = g.op.CastLike(X_values, one_f, name=f"{name}_vals_cast")

    # Step 3: signed weights
    if alternate_sign:
        zero_i32 = np.array(0, dtype=np.int32)
        pos_mask = g.op.GreaterOrEqual(hashes, zero_i32, name=f"{name}_pos_mask")
        neg_one_f = np.array(-1.0, dtype=target_dtype)
        signs = g.op.Where(pos_mask, one_f, neg_one_f, name=f"{name}_signs")
        weighted = g.op.Mul(signs, vals, name=f"{name}_weighted")
    else:
        weighted = vals

    # Step 4: build zero output matrix (N, n_features)
    input_shape = g.op.Shape(X, name=f"{name}_shape")
    n_rows = g.op.Gather(input_shape, np.array(0, dtype=np.int64), axis=0, name=f"{name}_n_rows")
    n_rows_1d = g.op.Unsqueeze(n_rows, np.array([0], dtype=np.int64), name=f"{name}_n_rows_1d")
    n_feat_i64 = np.array([n_features], dtype=np.int64)
    out_shape = g.op.Concat(n_rows_1d, n_feat_i64, axis=0, name=f"{name}_out_shape")
    zeros = g.op.ConstantOfShape(out_shape, name=f"{name}_zeros")
    zeros_typed = g.op.CastLike(zeros, one_f, name=f"{name}_zeros_typed")

    # Step 5: scatter-add weighted values into the output matrix
    res = g.op.ScatterElements(
        zeros_typed, indices, weighted, axis=1, reduction="add", name=name, outputs=outputs
    )
    g.set_type(res, target_onnx_type)
    if g.has_shape(X):
        g.set_shape(res, (g.get_shape(X)[0], n_features))

    assert isinstance(res, str)
    return res

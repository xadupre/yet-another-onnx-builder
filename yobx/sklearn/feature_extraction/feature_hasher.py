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

    **Two conversion paths**

    *With* ``com.microsoft`` *opset* + string input (MurmurHash3 native path):

    When the ``com.microsoft`` opset is registered **and** *X* is a ``STRING``
    tensor, the converter performs hashing inline using
    ``com.microsoft.MurmurHash3``.  A second companion input *X_values* of
    the same shape must be provided with the corresponding feature values
    (``1.0`` for each non-padding feature when ``input_type='string'``,
    ``0.0`` for padding slots).

    .. code-block:: text

        X (N, K) STRING, X_values (N, K) float
          │
          ├── MurmurHash3(X, seed=0, positive=0) ──► hashes (N, K) INT32
          ├── abs(hashes) % n_features ─────────────► indices (N, K) INT64
          ├── where(hashes >= 0, +1, −1) ───────────► signs (N, K) float
          ├── signs * X_values ─────────────────────► weighted (N, K) float
          └── ScatterElements(zeros, indices, weighted,
                              axis=1, reduction='add') → output (N, n_features)

    *Without* ``com.microsoft`` *opset* (pre-hashed path):

    The converter expects the **pre-hashed dense matrix** (i.e., the output of
    ``feature_hasher.transform(raw_X).toarray()``) as *X*.  The graph emits a
    single ``Identity`` or ``Cast`` node that enforces the output dtype.

    .. code-block:: text

        X (N, n_features) float
          └── Cast(to=estimator.dtype) ──► output (N, n_features)

    .. note::

        To prepare inputs compatible with the pre-hashed path, call::

            X_hashed = feature_hasher.transform(raw_X).toarray()

        and feed the resulting ``numpy.ndarray`` to the ONNX runtime.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn` (unused; present for
        interface consistency)
    :param estimator: a ``FeatureHasher`` instance
    :param outputs: desired output names
    :param X: primary input name.  For the ``com.microsoft`` path this must be
        a ``STRING`` tensor of shape ``(N, K)`` containing feature names
        (shorter samples padded with ``""``).  For the pre-hashed path this is
        the dense float matrix of shape ``(N, n_features)``.
    :param X_values: companion float input of shape ``(N, K)`` with feature
        values.  Required when the ``com.microsoft`` path is used (i.e. *X*
        is a ``STRING`` tensor).  Ignored for the pre-hashed path.
    :param name: prefix name for the added nodes
    :return: output name
    """
    assert isinstance(
        estimator, FeatureHasher
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    _check_murmurhash(estimator)

    target_dtype = np.dtype(estimator.dtype)
    target_onnx_type = np_dtype_to_tensor_dtype(target_dtype)
    n_features = estimator.n_features
    alternate_sign = estimator.alternate_sign
    itype = g.get_type(X)

    # ------------------------------------------------------------------
    # Fast path: com.microsoft.MurmurHash3 available + STRING input
    # ------------------------------------------------------------------
    if g.has_opset("com.microsoft") and itype == onnx.TensorProto.STRING:
        assert X_values is not None, (
            "The com.microsoft MurmurHash3 path requires a companion float input "
            "X_values of shape (N, K) with feature values "
            "(use 1.0 for each non-padding feature, 0.0 for padding slots). "
            f"Pass it as the second positional argument to to_onnx().{g.get_debug_msg()}"
        )
        assert g.has_type(X_values), (
            f"Missing type for {X_values!r}{g.get_debug_msg()}"
        )

        # Step 1: hash feature names
        hashes = g.make_node(
            "MurmurHash3",
            [X],
            domain="com.microsoft",
            seed=0,
            positive=0,
            name=f"{name}_mmh3",
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
        n_rows = g.op.Gather(
            input_shape,
            np.array(0, dtype=np.int64),
            axis=0,
            name=f"{name}_n_rows",
        )
        n_rows_1d = g.op.Unsqueeze(
            n_rows, np.array([0], dtype=np.int64), name=f"{name}_n_rows_1d"
        )
        n_feat_i64 = np.array([n_features], dtype=np.int64)
        out_shape = g.op.Concat(
            n_rows_1d, n_feat_i64, axis=0, name=f"{name}_out_shape"
        )
        zeros = g.op.ConstantOfShape(out_shape, name=f"{name}_zeros")
        zeros_typed = g.op.CastLike(zeros, one_f, name=f"{name}_zeros_typed")

        # Step 5: scatter-add weighted values into the output matrix
        res = g.op.ScatterElements(
            zeros_typed,
            indices,
            weighted,
            axis=1,
            reduction="add",
            name=name,
            outputs=outputs,
        )
        if not sts:
            g.set_type(res, target_onnx_type)
            if g.has_shape(X):
                g.set_shape(res, (g.get_shape(X)[0], n_features))

    # ------------------------------------------------------------------
    # Fallback path: pre-hashed float matrix  (N, n_features)
    # ------------------------------------------------------------------
    else:
        if itype == target_onnx_type:
            res = g.op.Identity(X, name=name, outputs=outputs)
            if not sts:
                g.set_type_shape_unary_op(res, X)
        else:
            res = g.op.Cast(X, to=target_onnx_type, name=name, outputs=outputs)
            if not sts:
                g.set_type(res, target_onnx_type)
                if g.has_shape(X):
                    g.set_shape(res, g.get_shape(X))

    assert isinstance(res, str)
    return res

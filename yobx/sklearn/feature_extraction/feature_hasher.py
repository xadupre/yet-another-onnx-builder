from typing import Dict, List

import numpy as np
import onnx
import onnx.numpy_helper as onh
from sklearn.feature_extraction import FeatureHasher

from ...typing import GraphBuilderExtendedProtocol
from ..register import register_sklearn_converter


@register_sklearn_converter(FeatureHasher)
def sklearn_feature_hasher(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: FeatureHasher,
    X: str,
    name: str = "feature_hasher",
) -> str:
    """
    Converts a :class:`sklearn.feature_extraction.FeatureHasher` into ONNX
    using the ``com.microsoft.MurmurHash3`` operator (ONNX Runtime ≥ 1.10).

    **Input format**

    The input tensor *X* must be a 2-D **string** tensor of shape
    ``(N, max_features_per_sample)`` where each row contains the feature
    names for one sample and shorter rows are padded with empty strings
    ``""`` (which are silently ignored).  This matches the padded string
    array produced by converting a list of feature-name lists into a
    rectangular numpy array.

    .. note::

        Only ``input_type='string'`` is supported.  The ``'dict'`` and
        ``'pair'`` input types require variable-length inputs that cannot be
        represented as fixed-size ONNX tensors.  Empty strings ``""`` are
        treated as padding and are **not** counted as features; users should
        therefore avoid using ``""`` as an actual feature name.

    **Requirements**

    The graph builder must have the ``com.microsoft`` ONNX domain registered
    (pass ``target_opset={'': 18, 'com.microsoft': 1}`` to
    :func:`yobx.sklearn.to_onnx`).

    **Supported options**

    * ``input_type='string'`` — the only supported value.
    * ``n_features`` — any positive integer.
    * ``alternate_sign=True`` (default) or ``False``.
    * ``dtype=numpy.float32`` or ``numpy.float64``.

    **Graph layout**

    .. code-block:: text

        X  (N, max_tokens) STRING
        │
        ├── MurmurHash3(seed=0, positive=0)   →  (N, max_tokens) INT32
        │       [com.microsoft domain]
        ├── Cast → INT64
        ├── GreaterOrEqual(0) → bool mask      (alternate_sign only)
        ├── Where(mask, +1, −1) → sign values
        ├── Equal(0) → is_empty mask
        ├── Where(is_empty, 0, sign) → values
        ├── Abs → abs_hash
        ├── Mod(n_features) → indices  (N, max_tokens) INT64
        ├── ConstantOfShape([N, n_features]) → zeros
        ├── ScatterElements(reduction='add', axis=1) → accumulated
        └── Cast(dtype) → output  (N, n_features) FLOAT/DOUBLE

    :param g: graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn` (unused; present for
        interface consistency)
    :param estimator: a fitted ``FeatureHasher`` instance
    :param outputs: desired output names
    :param X: input tensor name — a ``STRING`` tensor of shape
        ``(N, max_features_per_sample)`` padded with ``""``
    :param name: prefix for added node names
    :return: output tensor name
    :raises NotImplementedError: if ``input_type`` is not ``'string'`` or if
        the input tensor is not of type ``STRING``
    :raises RuntimeError: if the ``com.microsoft`` ONNX domain is not registered
        in the graph builder
    """
    assert isinstance(estimator, FeatureHasher), (
        f"Unexpected type {type(estimator)} for estimator."
    )
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    if estimator.input_type != "string":
        raise NotImplementedError(
            f"FeatureHasher converter only supports input_type='string' "
            f"(got {estimator.input_type!r}). The 'dict' and 'pair' input types "
            f"require variable-length inputs that cannot be represented as "
            f"fixed-size ONNX tensors."
        )

    if not g.has_opset("com.microsoft"):
        raise RuntimeError(
            "FeatureHasher converter requires the 'com.microsoft' ONNX domain for "
            "the MurmurHash3 operator. Pass "
            "target_opset={'': 18, 'com.microsoft': 1} to yobx.sklearn.to_onnx()."
        )

    itype = g.get_type(X)
    if itype != onnx.TensorProto.STRING:
        raise NotImplementedError(
            f"FeatureHasher conversion requires a STRING input tensor (got ONNX "
            f"type {itype}). Pass a 2-D padded string array of feature names as "
            f'input X (shorter rows padded with empty string "").'
        )

    n_features = estimator.n_features
    alternate_sign = estimator.alternate_sign

    if estimator.dtype == np.float32:
        out_dtype = onnx.TensorProto.FLOAT
    elif estimator.dtype == np.float64:
        out_dtype = onnx.TensorProto.DOUBLE
    else:
        raise NotImplementedError(
            f"FeatureHasher converter does not support dtype={estimator.dtype!r}. "
            f"Only numpy.float32 and numpy.float64 are supported."
        )

    # ── Hash strings → signed int32 ──────────────────────────────────────────
    # MurmurHash3 (com.microsoft) hashes each string to a signed 32-bit int.
    # seed=0 and positive=0 match sklearn's murmurhash3_32(token, seed=0, positive=False).
    hashed_i32 = g.make_node(
        "MurmurHash3",
        [X],
        domain="com.microsoft",
        seed=0,
        positive=0,
        name=f"{name}_hash",
    )
    g.set_type(hashed_i32, onnx.TensorProto.INT32)
    if g.has_shape(X):
        g.set_shape(hashed_i32, g.get_shape(X))
    elif g.has_rank(X):
        g.set_rank(hashed_i32, g.get_rank(X))

    # Cast to int64 so that abs(INT32_MIN) does not overflow
    hashed = g.op.Cast(hashed_i32, to=onnx.TensorProto.INT64, name=f"{name}_cast64")

    zero_i64 = np.array(0, dtype=np.int64)
    pos_one = np.array(1, dtype=np.int64)

    # ── Compute sign values ───────────────────────────────────────────────────
    if alternate_sign:
        ge_zero = g.op.GreaterOrEqual(hashed, zero_i64, name=f"{name}_ge0")
        neg_one = np.array(-1, dtype=np.int64)
        raw_values = g.op.Where(ge_zero, pos_one, neg_one, name=f"{name}_sign")
    else:
        # All values are +1: compute as (hashed * 0) + 1 to keep the shape dynamic
        raw_values = g.op.Add(
            g.op.Mul(hashed, zero_i64, name=f"{name}_zero_mul"),
            pos_one,
            name=f"{name}_ones",
        )

    # ── Mask out empty-string padding ─────────────────────────────────────────
    # murmurhash3_32("", seed=0) == 0.  Empty strings are padding, not features.
    is_empty = g.op.Equal(hashed, zero_i64, name=f"{name}_is_empty")
    values = g.op.Where(is_empty, zero_i64, raw_values, name=f"{name}_values")

    # ── Compute bucket indices: abs(hash) % n_features ─────────────────────────
    abs_hash = g.op.Abs(hashed, name=f"{name}_abs")
    nf_const = np.array(n_features, dtype=np.int64)
    indices = g.op.Mod(abs_hash, nf_const, name=f"{name}_idx")

    # ── Build zero data tensor of shape (N, n_features) ──────────────────────
    x_shape = g.op.Shape(X, name=f"{name}_xshape")
    batch_dim = g.op.Slice(
        x_shape,
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        name=f"{name}_batch_dim",
    )
    nf_arr = np.array([n_features], dtype=np.int64)
    data_shape = g.op.Concat(batch_dim, nf_arr, axis=0, name=f"{name}_data_shape")
    zero_val = onh.from_array(np.array([0], dtype=np.int64))
    zero_data = g.op.ConstantOfShape(
        data_shape, value=zero_val, name=f"{name}_zero_data"
    )

    # ── Accumulate contributions via ScatterElements (reduction='add') ────────
    # For each (sample i, token j): output[i, indices[i,j]] += values[i,j]
    scattered = g.op.ScatterElements(
        zero_data, indices, values, reduction="add", axis=1, name=f"{name}_scatter"
    )

    # ── Cast to requested output dtype ────────────────────────────────────────
    res = g.op.Cast(scattered, to=out_dtype, name=f"{name}_out", outputs=outputs)
    res_name = res if isinstance(res, str) else res[0]
    g.set_type(res_name, out_dtype)
    if g.has_shape(X):
        batch_dim_val = g.get_shape(X)[0]
        g.set_shape(res_name, (batch_dim_val, n_features))
    elif g.has_rank(X):
        g.set_rank(res_name, 2)
    return res_name

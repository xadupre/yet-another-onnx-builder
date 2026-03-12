from typing import Dict, List, Optional
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(OneHotEncoder)
def sklearn_one_hot_encoder(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: OneHotEncoder,
    X: str,
    name: str = "one_hot_encoder",
) -> str:
    """
    Converts a :class:`sklearn.preprocessing.OneHotEncoder` into ONNX.

    The converter handles all standard configurations of
    :class:`~sklearn.preprocessing.OneHotEncoder`:

    * **Multiple features** – each feature column is processed independently and
      the resulting one-hot blocks are concatenated along the feature axis.
    * **Drop** – when ``drop`` is not ``None`` (e.g. ``'first'`` or ``'if_binary'``),
      the converter uses the fitted :attr:`~sklearn.preprocessing.OneHotEncoder.drop_idx_`
      attribute to skip the dropped category column for each feature.
    * **Unknown handling** – because the encoding is implemented via element-wise
      comparisons (``Equal``), samples with categories that were not seen during
      training produce an all-zero row for that feature, matching
      ``handle_unknown='ignore'`` behaviour.

    The conversion for a single feature *i* with categories
    ``[c_0, c_1, …, c_{K-1}]`` is:

    .. code-block:: text

        X ──Gather(col i)──► col_i (N×1)
                                │
                          Equal(col_i, [[c_0,…,c_{K-1}]])  ──► (N×K) bool
                                │
                             Cast(float)                    ──► (N×K) float
                                │
                       [Gather(keep_idx)]   (only when drop≠None)
                                │
                             feature_i_out (N×K')

    Each ``feature_i_out`` is concatenated along ``axis=1`` to form the final
    output.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names
    :param estimator: a fitted :class:`~sklearn.preprocessing.OneHotEncoder`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: name of the output tensor
    :raises ValueError: when no output columns remain after applying ``drop``
    """
    assert isinstance(
        estimator, OneHotEncoder
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # Determine the output float type to use for Cast.
    # We always cast to the same floating-point type as the input.
    float_onnx_type = int(itype)

    drop_idx: Optional[np.ndarray] = estimator.drop_idx_

    parts: List[str] = []
    for i, cats in enumerate(estimator.categories_):
        # ------------------------------------------------------------------
        # 1. Extract column i: shape (N, 1)
        # ------------------------------------------------------------------
        col_i = g.op.Gather(X, np.array([i], dtype=np.int64), axis=1, name=f"{name}_{i}")

        # ------------------------------------------------------------------
        # 2. Compare against each known category: shape (N, K)
        # Each row contains True at the matching category position.
        # Unknown values produce an all-False row (→ all-zero after Cast).
        # ------------------------------------------------------------------
        cats_2d = cats.astype(dtype).reshape(1, -1)
        eq = g.op.Equal(col_i, cats_2d, name=f"{name}_eq_{i}")

        # ------------------------------------------------------------------
        # 3. Cast bool → float
        # ------------------------------------------------------------------
        part = g.op.Cast(eq, to=float_onnx_type, name=f"{name}_cast_{i}")

        # ------------------------------------------------------------------
        # 4. Apply drop: remove the column for the dropped category.
        # ------------------------------------------------------------------
        if drop_idx is not None and drop_idx[i] is not None:
            n_cats = len(cats)
            drop_col = int(drop_idx[i])
            keep_indices = np.array([j for j in range(n_cats) if j != drop_col], dtype=np.int64)
            if len(keep_indices) == 0:
                # All categories dropped for this feature – skip entirely.
                continue
            part = g.op.Gather(part, keep_indices, axis=1, name=f"{name}_drop_{i}")

        parts.append(part)

    if not parts:
        raise ValueError(
            f"OneHotEncoder {type(estimator).__name__!r} produces no output: "
            "all feature columns were dropped."
        )

    if len(parts) == 1:
        res = g.op.Identity(parts[0], name=name, outputs=outputs)
    else:
        res = g.op.Concat(*parts, axis=1, name=name, outputs=outputs)

    assert isinstance(res, str)  # type happiness
    return res

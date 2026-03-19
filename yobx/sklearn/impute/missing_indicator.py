import numpy as np
from typing import Dict, List

from sklearn.impute import MissingIndicator

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(MissingIndicator)
def sklearn_missing_indicator(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: MissingIndicator,
    X: str,
    name: str = "missing_indicator",
) -> str:
    """
    Converts a :class:`sklearn.impute.MissingIndicator` into ONNX.

    The transformer produces a boolean matrix that indicates which values
    are missing.  When ``features='missing-only'`` (the default), only the
    columns that had at least one missing value during ``fit`` are returned;
    when ``features='all'``, every column is returned.

    Graph structure when ``missing_values`` is :data:`numpy.nan`
    (the default) and ``features='all'``:

    .. code-block:: text

        X ──IsNaN──► mask [N, F]  ──► output

    When ``features='missing-only'``, a `Gather` node selects only
    the columns recorded in ``estimator.features_``:

    .. code-block:: text

        X ──IsNaN──► mask [N, F]
                          │
        features_ ────────┴──► Gather(axis=1) ──► output [N, len(features_)]

    When ``missing_values`` is a numeric value the ``IsNaN`` node is
    replaced by an ``Equal`` node:

    .. code-block:: text

        X ──Equal(missing_values)──► mask [N, F]

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``MissingIndicator``
    :param outputs: desired output names
    :param X: input name
    :param name: prefix name for the added nodes
    :return: output name
    """
    assert isinstance(
        estimator, MissingIndicator
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # ------------------------------------------------------------------
    # Step 1: build the missing-value mask for all features
    # ------------------------------------------------------------------
    missing_values = estimator.missing_values
    if isinstance(missing_values, float) and np.isnan(missing_values):
        mask = g.op.IsNaN(X, name=f"{name}_isnan")
    else:
        mv = np.array(missing_values, dtype=dtype)
        mask = g.op.Equal(X, mv, name=f"{name}_equal")

    # ------------------------------------------------------------------
    # Step 2: optionally select only the features with missing values
    # ------------------------------------------------------------------
    features_param = getattr(estimator, "features", "missing-only")
    features_ = estimator.features_

    if features_param == "all" or len(features_) == estimator.n_features_in_:
        # All features are already present in the mask; no selection needed.
        if outputs:
            res = g.op.Identity(mask, name=name, outputs=outputs)
        else:
            res = mask
    else:
        # Select only the columns that had missing values during training.
        idx = np.array(features_, dtype=np.int64)
        res = g.op.Gather(mask, idx, axis=1, name=name, outputs=outputs)

    assert isinstance(res, str)  # type happiness
    # Output type is bool (ONNX type 9).
    from onnx import TensorProto

    g.set_type(res, TensorProto.BOOL)
    return res

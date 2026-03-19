import numpy as np
from typing import Dict, List

from sklearn.impute import SimpleImputer

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(SimpleImputer)
def sklearn_simple_imputer(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: SimpleImputer,
    X: str,
    name: str = "simple_imputer",
) -> str:
    """
    Converts a :class:`sklearn.impute.SimpleImputer` into ONNX.

    All four strategies (``mean``, ``median``, ``most_frequent``,
    ``constant``) store the per-feature fill value in
    ``estimator.statistics_`` after fitting, so the ONNX graph only
    needs to detect missing entries and replace them with those constants.

    Graph structure when ``missing_values`` is :data:`numpy.nan`
    (the default):

    .. code-block:: text

        X ──IsNaN──► nan_mask [N, F]
                          │
        statistics_ ──────┼──► Where ──► output
                          │
        X ─────────────────┘

    When ``missing_values`` is a numeric value the ``IsNaN`` node is
    replaced by an ``Equal`` node:

    .. code-block:: text

        X ──Equal(missing_values)──► mask [N, F]
                                         │
        statistics_ ─────────────────────┼──► Where ──► output
                                         │
        X ─────────────────────────────────┘

    ``add_indicator=True`` is not supported and raises
    :class:`NotImplementedError`.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``SimpleImputer``
    :param outputs: desired output names
    :param X: input name
    :param name: prefix name for the added nodes
    :return: output name
    """
    assert isinstance(
        estimator, SimpleImputer
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    if getattr(estimator, "add_indicator", False):
        raise NotImplementedError("SimpleImputer with add_indicator=True is not supported.")

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # statistics_ has shape (n_features,); reshape to (1, n_features) for
    # broadcasting against (N, n_features) inputs.
    stats = estimator.statistics_.astype(dtype).reshape(1, -1)

    missing_values = estimator.missing_values
    if isinstance(missing_values, float) and np.isnan(missing_values):
        mask = g.op.IsNaN(X, name=f"{name}_isnan")
    else:
        mv = np.array(missing_values, dtype=dtype)
        mask = g.op.Equal(X, mv, name=f"{name}_equal")

    res = g.op.Where(mask, stats, X, name=name, outputs=outputs)

    g.set_type_shape_unary_op(res, X)
    return res

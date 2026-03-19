from typing import Dict, List
from sklearn.preprocessing import Binarizer
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
import numpy as np


@register_sklearn_converter(Binarizer)
def sklearn_binarizer(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: Binarizer,
    X: str,
    name: str = "binarizer",
) -> str:
    """
    Converts a :class:`sklearn.preprocessing.Binarizer` into ONNX.

    All feature values greater than ``threshold`` are set to 1, all others
    to 0.  The mapping follows the sklearn definition (strictly greater):

    .. code-block:: text

        X  ──Greater(threshold)──►  mask (bool)  ──Cast(dtype)──►  output

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted or unfitted ``Binarizer``
    :param outputs: desired output names
    :param X: input name
    :param name: prefix name for the added nodes
    :return: output name
    """
    assert isinstance(estimator, Binarizer), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    threshold = np.array(estimator.threshold, dtype=dtype)
    mask = g.op.Greater(X, threshold, name=f"{name}_greater")
    res = g.op.Cast(mask, to=itype, name=name, outputs=outputs)

    g.set_type_shape_unary_op(res, X)
    return res

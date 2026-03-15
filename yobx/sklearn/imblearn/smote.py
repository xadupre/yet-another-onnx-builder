from typing import Dict, List

from imblearn.over_sampling import SMOTE

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol


@register_sklearn_converter(SMOTE)
def imblearn_smote(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: SMOTE,
    X: str,
    name: str = "smote",
) -> str:
    """
    Converts a :class:`imblearn.over_sampling.SMOTE` into ONNX.

    SMOTE (Synthetic Minority Over-sampling Technique) is an over-sampling
    method used **during training** to generate synthetic minority-class
    samples from the neighbourhood of existing ones.  At **inference time**
    no resampling takes place: the input data is returned unchanged.  This
    converter therefore emits a single ``Identity`` node.

    The converter is primarily useful when exporting an
    :class:`imblearn.pipeline.Pipeline` that contains a SMOTE step:
    the pipeline converter calls each step's converter in turn, and the
    SMOTE passthrough ensures the graph is well-formed without discarding
    any data.

    .. code-block:: text

        X ──► Identity ──► X   (unchanged at inference time)

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names
    :param estimator: a fitted :class:`~imblearn.over_sampling.SMOTE`
    :param X: input name (shape ``(N, F)``)
    :param name: prefix name for the added nodes
    :return: output name
    """
    assert isinstance(estimator, SMOTE), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    res = g.op.Identity(X, name=name, outputs=outputs)
    assert isinstance(res, str)
    if not sts:
        g.set_type_shape_unary_op(res, X)
    return res

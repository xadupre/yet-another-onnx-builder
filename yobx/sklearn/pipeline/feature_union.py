from typing import Tuple, Dict, List, Optional, Union
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from ...typing import GraphBuilderExtendedProtocol
from ...xbuilder import FunctionOptions
from ..register import register_sklearn_converter, get_sklearn_converter
from ..sklearn_helper import get_output_names
from ..convert import _wrap_step_as_function


@register_sklearn_converter(FeatureUnion)
def sklearn_feature_union(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: FeatureUnion,
    X: str,
    name: str = "feature_union",
    function_options: Optional[FunctionOptions] = None,
) -> str:
    """
    Converts a :class:`sklearn.pipeline.FeatureUnion` into ONNX.

    The converter applies each transformer in the union to the **same**
    input tensor ``X`` and concatenates their outputs along the feature
    axis (``axis=-1``) using an ONNX ``Concat`` node, mirroring what
    :meth:`sklearn.pipeline.FeatureUnion.transform` does.

    Transformers whose weight is ``0`` (i.e. they were explicitly disabled
    via ``transformer_weights``) are skipped, exactly as scikit-learn does.

    .. code-block:: text

        X ──► transformer_A ──► out_A ──┐
          ──► transformer_B ──► out_B ──┤ Concat(axis=-1) ──► output
          ──► transformer_C ──► out_C ──┘

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`; also carries
        an optional :class:`~yobx.xbuilder.FunctionOptions` value under
        the key :data:`~yobx.sklearn.convert._FUNCTION_OPTIONS_KEY`
    :param outputs: desired output tensor names for the union result
    :param estimator: a fitted :class:`sklearn.pipeline.FeatureUnion`
    :param X: name of the input tensor to the feature union
    :param name: prefix used for names of nodes added by this converter
    :param function_options: to export every transformer as a local function
    :return: name of the output tensor
    :raises ValueError: when all transformers are ``'drop'`` (empty output)
    """
    assert isinstance(estimator, FeatureUnion), (
        f"Unexpected type {type(estimator)} for estimator."
    )
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    transformer_weights = estimator.transformer_weights or {}

    parts: List[str] = []
    for trans_name, transformer in estimator.transformer_list:
        if transformer == "drop":
            continue
        # Skip transformers with zero weight (same logic as sklearn).
        if transformer_weights.get(trans_name, 1.0) == 0:
            continue

        try:
            fct = get_sklearn_converter(type(transformer))
        except ValueError as e:
            raise ValueError(
                f"No ONNX converter registered for transformer {trans_name!r} "
                f"of type {type(transformer)!r} inside {type(estimator).__name__!r}."
            ) from e

        sub_outputs = [g.unique_name(f"{name}__{trans_name}_out")]
        step_node_name = f"{name}__{trans_name}"

        is_container = isinstance(transformer, (Pipeline, ColumnTransformer, FeatureUnion))
        if function_options and function_options.export_as_function and not is_container:
            _wrap_step_as_function(
                g,  # type: ignore
                function_options,
                transformer,
                [X],
                sub_outputs,
                fct,
                step_node_name,
            )
        elif is_container:
            fct(
                g,
                sts,
                sub_outputs,
                transformer,
                X,
                name=step_node_name,
                function_options=function_options,
            )
        else:
            fct(g, sts, sub_outputs, transformer, X, name=step_node_name)
        parts.append(sub_outputs[0])

    if not parts:
        raise ValueError(
            f"FeatureUnion {type(estimator).__name__!r} produces no output: "
            "all transformers are 'drop' or have zero weight."
        )

    if len(parts) == 1:
        res = g.op.Identity(parts[0], name=name, outputs=outputs)
    else:
        res = g.op.Concat(*parts, axis=-1, name=name, outputs=outputs)

    assert isinstance(res, str)
    return res

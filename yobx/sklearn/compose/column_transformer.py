import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from ...typing import GraphBuilderExtendedProtocol
from ...xbuilder import FunctionOptions
from ..register import register_sklearn_converter, get_sklearn_converter
from ..convert import _wrap_step_as_function


def _resolve_columns(columns: Union[list, slice, np.ndarray], n_features: int) -> np.ndarray:
    """
    Converts various column specifications to a 1-D ``int64`` numpy array.

    Supported input formats:

    * A Python :class:`slice` (e.g. ``slice(0, 3)``).
    * A boolean array / list of booleans.
    * A list or array of integer indices.

    :param columns: the column specification as stored in
        :attr:`sklearn.compose.ColumnTransformer.transformers_`.
    :param n_features: total number of input features (used to resolve slices).
    :return: 1-D ``int64`` array of column indices.
    """
    if isinstance(columns, slice):
        return np.arange(*columns.indices(n_features), dtype=np.int64)
    col_array = np.asarray(columns)
    if col_array.dtype == bool:
        return np.where(col_array)[0].astype(np.int64)
    return col_array.astype(np.int64)


def _is_passthrough(transformer) -> bool:
    """
    Returns ``True`` when *transformer* acts as an identity (passthrough).

    :class:`sklearn.compose.ColumnTransformer` internally replaces the
    ``'passthrough'`` string with a
    :class:`~sklearn.preprocessing.FunctionTransformer` whose ``func``
    attribute is ``None`` (identity).  This helper detects both forms.

    :param transformer: the (possibly fitted) transformer object.
    :return: ``True`` if the transformer is a passthrough.
    """
    if transformer == "passthrough":
        return True
    if isinstance(transformer, FunctionTransformer) and transformer.func is None:
        return True
    return False


@register_sklearn_converter(ColumnTransformer)
def sklearn_column_transformer(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: ColumnTransformer,
    X: str,
    name: str = "column_transformer",
    function_options: Optional[FunctionOptions] = None,
) -> str:
    """
    Converts a :class:`sklearn.compose.ColumnTransformer` into ONNX.

    The converter:

    1. Iterates over :attr:`~sklearn.compose.ColumnTransformer.transformers_`
       (the *fitted* transformer list).
    2. For each entry it selects the requested feature columns from the input
       tensor with an ONNX ``Gather`` node (``axis=1``).
    3. Transformers flagged as ``'drop'`` are skipped entirely.
    4. Passthrough transformers (the ``'passthrough'`` string or a
       :class:`~sklearn.preprocessing.FunctionTransformer` with ``func=None``)
       contribute the gathered sub-matrix directly.
    5. All remaining transformers are converted via their own registered
       converter using :func:`~yobx.sklearn.register.get_sklearn_converter`.
    6. All partial outputs are concatenated along the feature axis with an ONNX
       ``Concat`` node (``axis=-1``).

    .. code-block:: text

        X ──Gather(cols_A)──► X_A ──converter_A──► out_A ──┐
          ──Gather(cols_B)──► X_B ──converter_B──► out_B ──┤ Concat ──► output
          ──Gather(cols_C)──► X_C ──(passthrough)──────────┘

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names
    :param estimator: a fitted :class:`~sklearn.compose.ColumnTransformer`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :param function_options: function options
    :return: name of the output tensor
    :raises ValueError: when all transformers are ``'drop'`` (empty output)
    """
    assert isinstance(
        estimator, ColumnTransformer
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    n_features = estimator.n_features_in_

    parts: List[str] = []
    for trans_name, transformer, columns in estimator.transformers_:
        if transformer == "drop":
            continue

        col_indices = _resolve_columns(columns, n_features)

        # Select the subset of features for this transformer.
        X_sub = g.op.Gather(X, col_indices, axis=1, name=f"{name}__{trans_name}")

        if _is_passthrough(transformer):
            assert isinstance(X_sub, str)  # type happiness
            parts.append(X_sub)
        else:
            try:
                fct = get_sklearn_converter(type(transformer))
            except ValueError as e:
                raise ValueError(
                    f"No ONNX converter registered for transformer {trans_name!r} "
                    f"of type {type(transformer)!r} inside {type(estimator).__name__!r}."
                ) from e
            sub_outputs = [g.unique_name(f"{name}__{trans_name}_out")]
            step_node_name = f"{name}__{trans_name}"

            is_container = isinstance(X_sub, (Pipeline, ColumnTransformer, FeatureUnion))
            if function_options and function_options.export_as_function and not is_container:
                assert isinstance(X_sub, str)  # type happiness
                _wrap_step_as_function(
                    g,
                    function_options,
                    transformer,
                    [X_sub],
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
                    X_sub,
                    name=step_node_name,
                    function_options=function_options,
                )
            else:
                fct(g, sts, sub_outputs, transformer, X_sub, name=step_node_name)
            parts.append(sub_outputs[0])

    if not parts:
        raise ValueError(
            f"ColumnTransformer {type(estimator).__name__!r} produces no output: "
            "all transformers are 'drop'."
        )

    if len(parts) == 1:
        res = g.op.Identity(parts[0], name=name, outputs=outputs)
    else:
        res = g.op.Concat(*parts, axis=-1, name=name, outputs=outputs)

    assert isinstance(res, str)
    return res

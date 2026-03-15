from typing import Dict, List
from sklearn.decomposition import FastICA
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(FastICA)
def sklearn_fast_ica(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: FastICA,
    X: str,
    name: str = "fast_ica",
) -> str:
    """
    Converts a :class:`sklearn.decomposition.FastICA` into ONNX.

    The transformation replicates :meth:`sklearn.decomposition.FastICA.transform`:

    * When whitening was applied during fitting (``estimator.whiten`` is not
      ``False``), the per-feature mean stored in ``mean_`` is subtracted first.
    * The data (centred or raw) is then projected onto the independent
      components via a matrix multiplication with ``components_.T``.

    .. code-block:: text

        X  ──Sub(mean_)──►  centered  ──MatMul(components_.T)──►  output
             (only when whiten != False)

    When ``whiten`` is ``False`` the ``Sub`` node is omitted and the
    unmixing matrix is applied directly to *X*.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``FastICA``
    :param outputs: desired output names (independent components)
    :param X: input tensor name
    :param name: prefix name for the added nodes
    :return: output tensor name
    """
    assert isinstance(
        estimator, FastICA
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # Centre the data when whitening was used during fitting.
    if estimator.whiten is not False:
        mean = estimator.mean_.astype(dtype)
        centered = g.op.Sub(X, mean, name=name)
    else:
        centered = X

    # Project onto the independent components.
    # components_ has shape (n_components, n_features); we need (n_features, n_components).
    components_T = estimator.components_.T.astype(dtype)
    res = g.op.MatMul(centered, components_T, name=name, outputs=outputs)
    assert isinstance(res, str)  # type happiness
    if not sts:
        g.set_type(res, itype)
    return res

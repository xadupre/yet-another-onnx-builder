from typing import Dict, List

import numpy as np
from sklearn.feature_extraction import FeatureHasher

from ...helpers.onnx_helper import np_dtype_to_tensor_dtype
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
    Converts a :class:`sklearn.feature_extraction.FeatureHasher` into ONNX.

    :class:`~sklearn.feature_extraction.FeatureHasher` maps a sequence of
    feature dictionaries (or pairs, or strings) to a fixed-size dense matrix
    via the *hashing trick* (:func:`~sklearn.utils.murmurhash.murmurhash3_32`).
    Because ``murmurhash3_32`` is not available as a standard ONNX operator,
    this converter assumes the hashing step has already been applied **before**
    reaching the ONNX graph.  The expected input is therefore the dense
    ``float`` matrix that :meth:`~sklearn.feature_extraction.FeatureHasher.transform`
    would produce (shape ``(n_samples, n_features)``), converted to the
    estimator's ``dtype``.

    .. note::

        To prepare inputs compatible with this ONNX model, call::

            X_hashed = feature_hasher.transform(raw_X).toarray()

        and feed the resulting ``numpy.ndarray`` to the ONNX runtime.

    **Graph layout**

    .. code-block:: text

        X (N, n_features)
          │
          └──Cast(to=dtype)──► output (N, n_features)

    If the input type already matches ``estimator.dtype`` the ``Cast`` is
    replaced by an ``Identity`` and will be optimised away.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn` (unused; present for
        interface consistency)
    :param estimator: a ``FeatureHasher`` instance
    :param outputs: desired output names
    :param X: input name (shape ``(N, n_features)``)
    :param name: prefix name for the added nodes
    :return: output name
    """
    assert isinstance(
        estimator, FeatureHasher
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    target_dtype = np.dtype(estimator.dtype)
    target_onnx_type = np_dtype_to_tensor_dtype(target_dtype)

    itype = g.get_type(X)

    if itype == target_onnx_type:
        res = g.op.Identity(X, name=name, outputs=outputs)
        if not sts:
            g.set_type_shape_unary_op(res, X)
    else:
        res = g.op.Cast(X, to=target_onnx_type, name=name, outputs=outputs)
        if not sts:
            # Type changes: set the output type explicitly and copy only the shape.
            g.set_type(res, target_onnx_type)
            if g.has_shape(X):
                g.set_shape(res, g.get_shape(X))

    assert isinstance(res, str)
    return res

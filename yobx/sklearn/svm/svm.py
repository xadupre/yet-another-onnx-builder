from typing import Dict, List, Tuple, Union
import numpy as np
from sklearn.svm import NuSVC, NuSVR, SVC, SVR
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol

# Mapping from sklearn kernel names to ONNX kernel type strings.
_KERNEL_MAP = {"linear": "LINEAR", "poly": "POLY", "rbf": "RBF", "sigmoid": "SIGMOID"}

_SVC_TYPES = (SVC, NuSVC)
_SVR_TYPES = (SVR, NuSVR)


def _get_kernel_params(estimator) -> Tuple[str, List[float]]:
    """
    Returns the ONNX kernel type string and kernel parameters list.

    ONNX ``kernel_params`` is ``[gamma, coef0, degree]``.
    """
    kernel = estimator.kernel
    if isinstance(kernel, str):
        kernel_type = _KERNEL_MAP.get(kernel)
        if kernel_type is None:
            raise NotImplementedError(
                f"Kernel {kernel!r} is not supported. Supported kernels: {list(_KERNEL_MAP)}."
            )
    else:
        raise NotImplementedError(
            "Callable kernels are not supported. "
            "Use a string kernel name ('linear', 'poly', 'rbf', 'sigmoid')."
        )

    gamma = float(estimator._gamma) if kernel != "linear" else 0.0
    coef0 = float(estimator.coef0)
    degree = float(estimator.degree)
    return kernel_type, [gamma, coef0, degree]


@register_sklearn_converter(_SVC_TYPES)
def sklearn_svc(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: Union[SVC, NuSVC],
    X: str,
    name: str = "svc",
) -> Union[str, Tuple[str, str]]:
    """
    Converts a :class:`sklearn.svm.SVC` or :class:`sklearn.svm.NuSVC` into ONNX
    using the ``SVMClassifier`` operator from the ``ai.onnx.ml`` domain.

    **Supported kernels**: ``'linear'``, ``'poly'``, ``'rbf'``, ``'sigmoid'``.
    Callable kernels are not supported.

    When the estimator was trained with ``probability=True`` (i.e., it exposes
    :meth:`predict_proba`), the converter includes the Platt scaling calibration
    parameters ``prob_a`` / ``prob_b`` in the ``SVMClassifier`` node, which
    causes the ONNX runtime to output calibrated probabilities automatically.
    The output is then the tuple ``(label, probabilities)``.  Without
    ``probability=True``, only the predicted label is returned.

    **Coefficient layout** for the ``SVMClassifier`` node:

    * *Binary* (2 classes): ONNX coefficients = ``-dual_coef_.flatten()``,
      ``rho = -intercept_``.
    * *Multiclass* (≥ 3 classes, OvO): ONNX coefficients = ``dual_coef_.flatten()``,
      ``rho = intercept_``.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names (label only, or label + probabilities)
    :param estimator: a fitted ``SVC`` or ``NuSVC``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: label tensor name, or tuple ``(label, probabilities)``
    """
    assert isinstance(estimator, _SVC_TYPES), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    classes = estimator.classes_
    n_classes = len(classes)
    is_binary = n_classes == 2
    has_proba = hasattr(estimator, "predict_proba") and estimator.probability
    emit_proba = len(outputs) > 1

    kernel_type, kernel_params = _get_kernel_params(estimator)

    # Build coefficient and rho arrays.
    # Binary SVC: sklearn decision function is  dual_coef_[0] @ K + intercept_
    # ONNX SVMClassifier uses the negated convention for binary classifiers.
    if is_binary:
        coefficients = (-estimator.dual_coef_).astype(np.float32).flatten().tolist()
        rho = (-estimator.intercept_).astype(np.float64).tolist()
    else:
        coefficients = estimator.dual_coef_.astype(np.float32).flatten().tolist()
        rho = estimator.intercept_.astype(np.float64).tolist()

    support_vectors = estimator.support_vectors_.astype(np.float32).flatten().tolist()
    vectors_per_class = estimator.n_support_.tolist()

    # Class label attributes.
    if np.issubdtype(classes.dtype, np.integer):
        label_kwargs: Dict = {"classlabels_ints": classes.astype(np.int64).tolist()}
    else:
        label_kwargs = {"classlabels_strings": classes.astype(str).tolist()}

    # Probability attributes.
    if has_proba and emit_proba:
        proba_kwargs: Dict = {
            "prob_a": estimator.probA_.astype(np.float32).tolist(),
            "prob_b": estimator.probB_.astype(np.float32).tolist(),
        }
    else:
        proba_kwargs = {}

    post_transform = "NONE"

    # Emit SVMClassifier node (ai.onnx.ml domain).
    # The node always produces 2 outputs (label + scores/proba). Pass the
    # requested output names when available; fall back to auto-generated names.
    if emit_proba:
        node_outputs: List[str] = list(outputs[:2])
    else:
        node_outputs = [outputs[0], f"{name}_proba_unused"]

    result = g.make_node(
        "SVMClassifier",
        [X],
        outputs=node_outputs,
        domain="ai.onnx.ml",
        name=name,
        kernel_type=kernel_type,
        kernel_params=[float(v) for v in kernel_params],
        support_vectors=support_vectors,
        vectors_per_class=vectors_per_class,
        coefficients=coefficients,
        rho=[float(v) for v in rho],
        post_transform=post_transform,
        **label_kwargs,
        **proba_kwargs,
    )

    if isinstance(result, (list, tuple)):
        label, proba = result[0], result[1]  # type: ignore
    else:
        label = result
        proba = node_outputs[1]

    if emit_proba:
        return label, proba
    return label


@register_sklearn_converter(_SVR_TYPES)
def sklearn_svr(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: Union[SVR, NuSVR],
    X: str,
    name: str = "svr",
) -> str:
    """
    Converts a :class:`sklearn.svm.SVR` or :class:`sklearn.svm.NuSVR` into ONNX
    using the ``SVMRegressor`` operator from the ``ai.onnx.ml`` domain.

    **Supported kernels**: ``'linear'``, ``'poly'``, ``'rbf'``, ``'sigmoid'``.
    Callable kernels are not supported.

    Graph structure:

    .. code-block:: text

        X  ──SVMRegressor──►  predictions (N,1)

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names
    :param estimator: a fitted ``SVR`` or ``NuSVR``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: output tensor name
    """
    assert isinstance(estimator, _SVR_TYPES), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    kernel_type, kernel_params = _get_kernel_params(estimator)

    coefficients = estimator.dual_coef_.astype(np.float32).flatten().tolist()
    support_vectors = estimator.support_vectors_.astype(np.float32).flatten().tolist()
    n_supports = int(estimator.dual_coef_.shape[1])
    rho = estimator.intercept_.astype(np.float64).tolist()

    result = g.make_node(
        "SVMRegressor",
        [X],
        outputs=outputs,
        domain="ai.onnx.ml",
        name=name,
        kernel_type=kernel_type,
        kernel_params=[float(v) for v in kernel_params],
        support_vectors=support_vectors,
        coefficients=coefficients,
        n_supports=n_supports,
        rho=[float(v) for v in rho],
        post_transform="NONE",
        one_class=0,
    )

    assert isinstance(result, str)
    return result

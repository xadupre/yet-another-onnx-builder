from typing import Dict, List, Tuple, Union

from imblearn.ensemble import RUSBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

from ...typing import GraphBuilderExtendedProtocol
from ..register import get_sklearn_converter, register_sklearn_converter


@register_sklearn_converter(RUSBoostClassifier)
def sklearn_rus_boost_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: RUSBoostClassifier,
    X: str,
    name: str = "rus_boost_classifier",
) -> Union[str, Tuple[str, str]]:
    """
    Converts an :class:`imblearn.ensemble.RUSBoostClassifier` into ONNX.

    :class:`~imblearn.ensemble.RUSBoostClassifier` is a boosting classifier
    that combines the SAMME algorithm with Random Under-Sampling (RUS).
    During training, each iteration applies random under-sampling to balance
    the class distribution before fitting the next base estimator.  At
    inference time the resampling step is inactive; the resulting ensemble
    has exactly the same structure as a plain
    :class:`~sklearn.ensemble.AdaBoostClassifier` and is therefore converted
    using the same helper.

    Graph structure (two base estimators as an example):

    .. code-block:: text

        X ──[base est 0]──► label_0 (N,)
        X ──[base est 1]──► label_1 (N,)
            label_i == classes_k ? w_i : -w_i/(C-1)  ──► vote_i (N, C)
                        Add votes ──► decision (N, C)
                    ArgMax(axis=1) ──Cast──Gather(classes_) ──► label
                decision/(C-1) ──Softmax ──► probabilities

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names; two entries for
        ``(label, probabilities)``, one entry for label only
    :param estimator: a fitted :class:`~imblearn.ensemble.RUSBoostClassifier`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: label tensor name, or tuple ``(label, probabilities)``
    """
    assert isinstance(
        estimator, RUSBoostClassifier
    ), f"Unexpected type {type(estimator)} for estimator."

    # RUSBoostClassifier is a direct subclass of AdaBoostClassifier.
    # At inference time the SAMME voting logic is identical; delegate to the
    # registered AdaBoostClassifier converter.
    fct = get_sklearn_converter(AdaBoostClassifier)
    return fct(g, sts, outputs, estimator, X, name=name)

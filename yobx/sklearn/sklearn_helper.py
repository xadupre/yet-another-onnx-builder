from typing import Sequence
from sklearn.base import BaseEstimator, ClusterMixin, is_classifier, is_regressor
from sklearn.mixture._base import BaseMixture
from sklearn.pipeline import Pipeline


def _classifier_has_predict_proba(estimator: BaseEstimator) -> bool:
    """
    Returns True when *estimator* is a fitted classifier that genuinely
    supports :meth:`predict_proba`.

    For most classifiers, checking ``hasattr(estimator, 'predict_proba')``
    is sufficient.  However, :class:`sklearn.linear_model.SGDClassifier`
    (and its subclasses such as :class:`~sklearn.linear_model.Perceptron`)
    defines ``predict_proba`` as a method that raises :exc:`AttributeError`
    at runtime for non-probabilistic loss functions (e.g. ``'hinge'``).
    Because Python's ``hasattr`` catches :exc:`AttributeError`, it returns
    ``False`` in those cases — so the plain ``hasattr`` check already works
    correctly for fitted :class:`~sklearn.linear_model.SGDClassifier`
    instances.
    """
    return hasattr(estimator, "predict_proba")


def get_n_expected_outputs(estimator: BaseEstimator) -> int:
    """Returns the number of expected outputs."""
    if is_classifier(estimator):
        return 2 if _classifier_has_predict_proba(estimator) else 1
    if isinstance(estimator, (ClusterMixin, BaseMixture)):
        return 2
    return 1


def get_output_names(estimator: BaseEstimator) -> Sequence[str]:
    """Returns output names for every estimator."""
    if hasattr(estimator, "get_feature_names_out"):
        if isinstance(estimator, Pipeline):
            last_step = estimator.steps[-1][1]
            if not isinstance(last_step, ClusterMixin) and not is_classifier(last_step):
                try:
                    return post_process_output_names(
                        last_step, list(last_step.get_feature_names_out())
                    )
                except AttributeError:
                    pass
        elif not isinstance(estimator, ClusterMixin) and not is_classifier(estimator):
            try:
                return post_process_output_names(
                    estimator, list(estimator.get_feature_names_out())
                )
            except AttributeError:
                pass
    if is_classifier(estimator):
        if _classifier_has_predict_proba(estimator):
            return ["label", "probabilities"]
        return ["label"]
    if isinstance(estimator, BaseMixture):
        return ["label", "probabilities"]
    if is_regressor(estimator):
        return ["predictions"]
    last = estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator
    if isinstance(last, ClusterMixin):
        return ["label", "distances"]
    if isinstance(last, BaseMixture):
        return ["label", "probabilities"]
    return ["Y"]


def post_process_output_names(
    estimator: BaseEstimator, output_names: Sequence[str]
) -> Sequence[str]:
    """Makes sures the number of outputs is expected."""
    n_outputs = get_n_expected_outputs(estimator)
    if len(output_names) == n_outputs:
        return output_names
    if n_outputs == 1:
        return [longest_prefix(output_names)]
    raise NotImplementedError(
        f"Not implemented with {output_names=}, {n_outputs=} and estimator is {type(estimator)}."
    )


def _longest_prefix(s1: str, s2: str) -> str:
    for i, (c1, c2) in enumerate(zip(s1, s2)):
        if c1 != c2:
            return s1[:i]
    return s1


def longest_prefix(names: Sequence[str]) -> str:
    """Creates a common prefix for all name in names."""
    if not names:
        return ""

    prefix = names[0]
    for name in names[1:]:
        prefix = (
            _longest_prefix(prefix, name[: len(prefix)])
            if len(name) > len(prefix)
            else _longest_prefix(prefix[: len(name)], name)
        )
        if not prefix:
            break
    return prefix or "output"

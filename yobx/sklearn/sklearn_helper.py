from typing import Sequence
from sklearn.base import BaseEstimator, ClusterMixin, OutlierMixin, is_classifier, is_regressor
from sklearn.cluster import FeatureAgglomeration
from sklearn.mixture._base import BaseMixture
from sklearn.pipeline import Pipeline

try:
    from sklearn.feature_selection._base import SelectorMixin
except ImportError:
    SelectorMixin = None  # type: ignore

try:
    from sklearn.base import OutlierMixin
except ImportError:
    OutlierMixin = None  # type: ignore


def _should_use_feature_names(estimator: BaseEstimator) -> bool:
    """Returns True when get_feature_names_out() should drive output naming."""
    return (
        (SelectorMixin is not None and isinstance(estimator, SelectorMixin))
        or isinstance(estimator, FeatureAgglomeration)
        or (not isinstance(estimator, ClusterMixin) and not is_classifier(estimator))
    )


def get_n_expected_outputs(estimator: BaseEstimator) -> int:
    """Returns the number of expected outputs."""
    if SelectorMixin is not None and isinstance(estimator, SelectorMixin):
        return 1
    if is_classifier(estimator):
        return 2 if hasattr(estimator, "predict_proba") else 1
    if isinstance(estimator, FeatureAgglomeration):
        return 1
    if isinstance(estimator, (ClusterMixin, BaseMixture)) or (
        OutlierMixin is not None and isinstance(estimator, OutlierMixin)
    ):
        return 2
    # For Pipelines, check the last step explicitly (mirrors get_output_names fallback).
    if isinstance(estimator, Pipeline):
        last = estimator.steps[-1][1]
        if isinstance(last, FeatureAgglomeration):
            return 1
        if isinstance(last, (ClusterMixin, BaseMixture)) or (
            OutlierMixin is not None and isinstance(last, OutlierMixin)
        ):
            return 2
    return 1


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


def get_output_names(estimator: BaseEstimator) -> Sequence[str]:
    """Returns output names for every estimator."""
    if hasattr(estimator, "get_feature_names_out"):
        if isinstance(estimator, Pipeline):
            last_step = estimator.steps[-1][1]
            if _should_use_feature_names(last_step):
                try:
                    return post_process_output_names(
                        last_step, list(last_step.get_feature_names_out())
                    )
                except AttributeError:
                    pass
        elif _should_use_feature_names(estimator):
            try:
                return post_process_output_names(
                    estimator, list(estimator.get_feature_names_out())
                )
            except AttributeError:
                pass

    if SelectorMixin is not None and isinstance(estimator, SelectorMixin):
        return ["Y"]
    if is_classifier(estimator):
        if hasattr(estimator, "predict_proba"):
            return ["label", "probabilities"]
        return ["label"]
    if OutlierMixin is not None and isinstance(estimator, OutlierMixin):
        return ["label", "scores"]
    if isinstance(estimator, BaseMixture):
        return ["label", "probabilities"]
    if isinstance(estimator, OutlierMixin):
        return ["label", "scores"]
    if is_regressor(estimator):
        return ["predictions"]
    last = estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator
    if OutlierMixin is not None and isinstance(last, OutlierMixin):
        return ["label", "scores"]
    if isinstance(last, ClusterMixin) and not isinstance(last, FeatureAgglomeration):
        return ["label", "distances"]
    if isinstance(last, BaseMixture):
        return ["label", "probabilities"]
    if isinstance(last, OutlierMixin):
        return ["label", "scores"]
    return ["Y"]


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

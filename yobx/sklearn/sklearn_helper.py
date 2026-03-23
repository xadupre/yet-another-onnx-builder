from typing import Optional, Sequence
from sklearn.base import BaseEstimator, ClusterMixin, OutlierMixin, is_classifier, is_regressor
from sklearn.cluster import FeatureAgglomeration
from sklearn.mixture._base import BaseMixture
from sklearn.pipeline import Pipeline
from .convert_options import ConvertOptions

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
    estimator: BaseEstimator,
    output_names: Sequence[str],
    convert_options: Optional[ConvertOptions] = None,
) -> Sequence[str]:
    """Makes sures the number of outputs is expected."""
    n_outputs = get_n_expected_outputs(estimator)
    if n_outputs == 1 and len(output_names) != 1:
        output_names = [longest_prefix(output_names)]
    if len(output_names) != n_outputs:
        raise NotImplementedError(
            f"Not implemented with {output_names=}, {n_outputs=} "
            f"and estimator is {type(estimator)}."
        )
    if convert_options:
        for _extra_opt in ConvertOptions.OPTIONS:
            if convert_options.has(_extra_opt, estimator):
                print("***", _extra_opt)
                output_names.append(_extra_opt)
    return output_names


def get_output_names(
    estimator: BaseEstimator, convert_options: Optional[ConvertOptions] = None
) -> Sequence[str]:
    """Returns output names for every estimator."""

    # Append extra output names requested by convert_options so that converters
    # can detect them via len(outputs) > extra_idx and emit the extra nodes.
    if isinstance(estimator, Pipeline):
        last_step = estimator.steps[-1][1]
    else:
        last_step = estimator
    if hasattr(last_step, "get_feature_names_out") and _should_use_feature_names(last_step):
        outnames = estimator.get_feature_names_out()
        return post_process_output_names(last_step, list(outnames), convert_options)

    if SelectorMixin is not None and isinstance(last_step, SelectorMixin):
        return post_process_output_names(last_step, ["Y"], convert_options)
    if is_classifier(last_step):
        if hasattr(last_step, "predict_proba"):
            return post_process_output_names(
                last_step, ["label", "probabilities"], convert_options
            )
        return post_process_output_names(last_step, ["label"], convert_options)
    if OutlierMixin is not None and isinstance(last_step, OutlierMixin):
        return post_process_output_names(last_step, ["label", "scores"], convert_options)
    if isinstance(last_step, BaseMixture):
        return post_process_output_names(last_step, ["label", "probabilities"], convert_options)
    if isinstance(last_step, OutlierMixin):
        return post_process_output_names(last_step, ["label", "scores"], convert_options)
    if is_regressor(last_step):
        return post_process_output_names(last_step, ["predictions"], convert_options)
    last = estimator.steps[-1][1] if isinstance(last_step, Pipeline) else estimator
    if OutlierMixin is not None and isinstance(last, OutlierMixin):
        return post_process_output_names(last_step, ["label", "scores"], convert_options)
    if isinstance(last, ClusterMixin) and not isinstance(last, FeatureAgglomeration):
        return post_process_output_names(last_step, ["label", "distances"], convert_options)
    if isinstance(last, BaseMixture):
        return post_process_output_names(last_step, ["label", "probabilities"], convert_options)
    if isinstance(last, OutlierMixin):
        return post_process_output_names(last_step, ["label", "scores"], convert_options)
    return post_process_output_names(last_step, ["Y"], convert_options)


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

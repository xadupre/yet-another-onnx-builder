from typing import Sequence
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.pipeline import Pipeline


def get_n_expected_outputs(estimator: BaseEstimator) -> int:
    """Returns the number of expected outputs."""
    if is_classifier(estimator):
        return 2
    return 1


def get_output_names(estimator: BaseEstimator) -> Sequence[str]:
    """Returns output names for every estimator."""
    if hasattr(estimator, "get_feature_names_out"):
        if isinstance(estimator, Pipeline):
            try:
                return post_process_output_names(
                    estimator.steps[-1][1], list(estimator.steps[-1][1].get_feature_names_out())
                )
            except AttributeError:
                pass
        else:
            try:
                return post_process_output_names(
                    estimator, list(estimator.get_feature_names_out())
                )
            except AttributeError:
                pass
    if is_classifier(estimator):
        return ["label", "probabilities"]
    if is_regressor(estimator):
        return ["predictions"]
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

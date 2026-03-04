from __future__ import annotations
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ..patterns_api import PatternOptimization


def get_ml_patterns(
    verbose: int = 0,
) -> List[PatternOptimization]:
    """
    Returns a default list of optimization patterns for ai.onnx.ml.
    It is equal to the following list.

    .. runpython::
        :showcode:
        :rst:

        from yobx.xoptim.patterns_api import pattern_table_doc
        from yobx.xoptim.patterns_ml import get_ml_patterns

        print(pattern_table_doc(get_ml_patterns(), as_rst=True))
        print()
    """
    from .tree_ensemble import (
        TreeEnsembleRegressorConcatPattern,
        TreeEnsembleRegressorMulPattern,
    )

    return [
        TreeEnsembleRegressorConcatPattern(verbose=verbose),
        TreeEnsembleRegressorMulPattern(verbose=verbose),
    ]

"""
ONNX converter for :class:`lightgbm.sklearn.LGBMModel`.

:class:`~lightgbm.sklearn.LGBMModel` is the configurable base class for all
LightGBM sklearn-compatible estimators.  Users instantiate it directly with a
specific ``objective`` and call :meth:`~lightgbm.sklearn.LGBMModel.predict`,
which returns:

* **Regression** objectives (``regression``, ``regression_l1``, ``huber``,
  ``quantile``, ``mape``, …) — predicted values, shape ``[N]``.
* **Log-link regression** (``poisson``, ``tweedie``) — ``exp(margin)``,
  shape ``[N]``.
* **Binary classification** (``binary``) — sigmoid probabilities in [0, 1],
  shape ``[N]``.
* **Multi-class classification** (``multiclass`` / ``softmax`` / …) —
  per-class probability matrix, shape ``[N, n_classes]``.
* **Ranking** (``lambdarank``, ``rank_xendcg``) — raw margin scores,
  shape ``[N]``.

The ONNX model produced by this converter follows the same logic and outputs
a single tensor whose shape mirrors the ndim-normalised sklearn output:
``[N, 1]`` for scalar-per-sample objectives, or ``[N, n_classes]`` for
multi-class objectives.

Both ``ai.onnx.ml`` legacy (opset ≤ 4) and modern (opset ≥ 5) tree encodings
are supported, as well as ``float32`` and ``float64`` inputs.
"""

from typing import Dict, List, Optional, Union
import numpy as np
from lightgbm.sklearn import LGBMModel
from ...typing import GraphBuilderExtendedProtocol
from ..register import register_sklearn_converter
from .lgbm import (
    _RANK_IDENTITY_OBJECTIVES,
    _emit_lgbm_tree_node,
    _get_reg_output_transform,
)

#: LightGBM binary classification objectives.
_CLF_BINARY_OBJECTIVES = frozenset({"binary"})

#: LightGBM multi-class classification objectives.
_CLF_MULTICLASS_OBJECTIVES = frozenset(
    {
        "multiclass",
        "softmax",
        "multiclassova",
        "multiclass_ova",
        "ovr",
        "multiclass_ovr",
    }
)


@register_sklearn_converter(LGBMModel)
def sklearn_lgbm_model(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator,
    X: str,
    name: str = "lgbm_model",
) -> Union[str, List[str]]:
    """Convert an :class:`lightgbm.sklearn.LGBMModel` to ONNX.

    The converter inspects the fitted booster's objective string and dispatches
    to the appropriate conversion logic:

    * **Regression** (``regression``, ``regression_l1``, ``huber``,
      ``quantile``, ``mape``, …) — tree raw margin, optional ``exp`` transform
      for ``poisson`` / ``tweedie``.  Output shape ``[N, 1]``.
    * **Binary classification** (``binary``) — tree raw margin passed through
      sigmoid.  Output shape ``[N, 1]``.
    * **Multi-class classification** (``multiclass`` / ``softmax`` / …) —
      per-class raw margins passed through softmax.  Output shape
      ``[N, n_classes]``.
    * **Ranking** (``lambdarank``, ``rank_xendcg``) — raw margin scores,
      identity link.  Output shape ``[N, 1]``.

    :param g: the graph builder to add nodes to
    :param sts: shapes dict (passed through, not used internally)
    :param outputs: desired output names ``[predictions]``
    :param estimator: a fitted :class:`~lightgbm.sklearn.LGBMModel`
    :param X: input tensor name
    :param name: prefix for node names added to the graph
    :return: output tensor name
    :raises NotImplementedError: if the model's objective is not supported
    """
    booster = estimator.booster_
    model_dict = booster.dump_model()
    objective: str = model_dict["objective"]
    base_obj = objective.split()[0].split(":")[0]

    ml_opset = g.get_opset("ai.onnx.ml")
    itype = g.get_type(X)
    trees = model_dict["tree_info"]
    tree_out_name = f"{outputs[0]}_tree_out"

    if base_obj in _CLF_BINARY_OBJECTIVES:
        return _lgbm_model_binary(
            g, outputs, X, name, trees, ml_opset, itype, tree_out_name
        )

    if base_obj in _CLF_MULTICLASS_OBJECTIVES:
        n_classes: int = int(model_dict.get("num_tree_per_iteration", 1))
        return _lgbm_model_multiclass(
            g, outputs, X, name, trees, ml_opset, itype, tree_out_name, n_classes
        )

    if base_obj in _RANK_IDENTITY_OBJECTIVES:
        return _lgbm_model_ranking(
            g, outputs, X, name, trees, ml_opset, itype, tree_out_name
        )

    # Default: regression (validates objective and raises for unknown objectives)
    out_transform = _get_reg_output_transform(objective)
    return _lgbm_model_regression(
        g, outputs, X, name, trees, ml_opset, itype, tree_out_name, out_transform
    )


# ---------------------------------------------------------------------------
# Internal dispatch helpers — one per objective family
# ---------------------------------------------------------------------------


def _lgbm_model_binary(
    g: GraphBuilderExtendedProtocol,
    outputs: List[str],
    X: str,
    name: str,
    trees: List[dict],
    ml_opset: int,
    itype: int,
    tree_out_name: str,
) -> str:
    """Emit ONNX nodes for a binary-classification :class:`~lightgbm.sklearn.LGBMModel`."""
    raw_scores = _emit_lgbm_tree_node(
        g,
        X,
        name,
        n_targets=1,
        trees=trees,
        ml_opset=ml_opset,
        intermediate_name=tree_out_name,
        itype=itype,
    )
    raw_scores = g.make_node(
        "Cast", [raw_scores], outputs=1, name=f"{name}_tree_cast", to=itype
    )
    proba = g.op.Sigmoid(raw_scores, name=f"{name}_sigmoid")
    return g.make_node("Cast", [proba], outputs=outputs, name=f"{name}_cast", to=itype)


def _lgbm_model_multiclass(
    g: GraphBuilderExtendedProtocol,
    outputs: List[str],
    X: str,
    name: str,
    trees: List[dict],
    ml_opset: int,
    itype: int,
    tree_out_name: str,
    n_classes: int,
) -> str:
    """Emit ONNX nodes for a multi-class :class:`~lightgbm.sklearn.LGBMModel`."""
    raw_scores = _emit_lgbm_tree_node(
        g,
        X,
        name,
        n_targets=n_classes,
        trees=trees,
        ml_opset=ml_opset,
        intermediate_name=tree_out_name,
        itype=itype,
    )
    raw_scores = g.make_node(
        "Cast", [raw_scores], outputs=1, name=f"{name}_tree_cast", to=itype
    )
    proba = g.op.Softmax(raw_scores, axis=1, name=f"{name}_softmax")
    return g.make_node("Cast", [proba], outputs=outputs, name=f"{name}_cast", to=itype)


def _lgbm_model_ranking(
    g: GraphBuilderExtendedProtocol,
    outputs: List[str],
    X: str,
    name: str,
    trees: List[dict],
    ml_opset: int,
    itype: int,
    tree_out_name: str,
) -> str:
    """Emit ONNX nodes for a ranking :class:`~lightgbm.sklearn.LGBMModel`."""
    raw_scores = _emit_lgbm_tree_node(
        g,
        X,
        name,
        n_targets=1,
        trees=trees,
        ml_opset=ml_opset,
        intermediate_name=tree_out_name,
        itype=itype,
    )
    # Identity link — normalise dtype and assign output name in a single Cast.
    return g.make_node(
        "Cast", [raw_scores], outputs=outputs, name=f"{name}_cast", to=itype
    )


def _lgbm_model_regression(
    g: GraphBuilderExtendedProtocol,
    outputs: List[str],
    X: str,
    name: str,
    trees: List[dict],
    ml_opset: int,
    itype: int,
    tree_out_name: str,
    out_transform: Optional[str],
) -> str:
    """Emit ONNX nodes for a regression :class:`~lightgbm.sklearn.LGBMModel`."""
    raw_scores = _emit_lgbm_tree_node(
        g,
        X,
        name,
        n_targets=1,
        trees=trees,
        ml_opset=ml_opset,
        intermediate_name=tree_out_name,
        itype=itype,
    )
    raw_scores = g.make_node(
        "Cast", [raw_scores], outputs=1, name=f"{name}_tree_cast", to=itype
    )
    if out_transform == "exp":
        raw_scores = g.op.Exp(raw_scores, name=f"{name}_exp")
    return g.make_node(
        "Cast", [raw_scores], outputs=outputs, name=f"{name}_cast", to=itype
    )

"""
ONNX converters for :class:`perpetual.PerpetualClassifier`
and :class:`perpetual.PerpetualRegressor`.

The converters read the public JSON tree serialization and build the graph
directly, without Perpetual's reference-ONNX-dependent exporter. Numerical
features, missing-value sentinels and separate missing branches are supported;
categorical mappings and categorical splits raise ``NotImplementedError``.
"""

import json
from typing import Dict, List, Tuple
import numpy
from yobx._onnx_shim import onnx
from perpetual import PerpetualClassifier, PerpetualRegressor
from ...typing import GraphBuilderExtendedProtocol
from ..register import register_sklearn_converter


def _tree_attributes(estimator) -> Dict:
    """Extracts numerical trees, including Perpetual's optional third missing child."""
    dump = json.loads(estimator.json_dump())
    boosters = dump.get("boosters", [dump])
    attrs = {
        key: []
        for key in (
            "nodes_treeids",
            "nodes_nodeids",
            "nodes_featureids",
            "nodes_modes",
            "nodes_values",
            "nodes_truenodeids",
            "nodes_falsenodeids",
            "nodes_missing_value_tracks_true",
            "target_treeids",
            "target_nodeids",
            "target_ids",
            "target_weights",
        )
    }
    tree_id = 0
    for target_id, booster in enumerate(boosters):
        trees = booster["trees"]
        if not trees:
            # A zero-contribution tree keeps constant, unboosted models valid.
            trees = [{"nodes": {"0": {"num": 0, "is_leaf": True, "weight_value": 0.0}}}]
        for tree in trees:
            nodes = sorted(tree["nodes"].values(), key=lambda node: node["num"])
            next_id = max(node["num"] for node in nodes) + 1
            for node in nodes:
                leaf = node["is_leaf"]
                feature = 0 if leaf else int(node["split_feature"])
                value = 0.0 if leaf else node["split_value"]
                if value is None:
                    raise NotImplementedError(
                        "Perpetual JSON does not distinguish non-finite split thresholds."
                    )
                value = float(value)
                left = 0 if leaf else int(node["left_child"])
                right = 0 if leaf else int(node["right_child"])
                missing = 0 if leaf else int(node["missing_node"])
                if not leaf and node.get("left_cats") is not None:
                    raise NotImplementedError(
                        "Perpetual categorical splits are not supported by the converter."
                    )
                if not leaf and missing not in (left, right):
                    # All non-NaN values (including infinity) take the true branch.
                    # The false branch leads to Perpetual's separate missing subtree.
                    split = dict(node, num=next_id, missing_node=right)
                    nodes.append(split)
                    left, right, value = next_id, missing, numpy.inf
                    next_id += 1
                    mode = "BRANCH_LEQ"
                else:
                    mode = "LEAF" if leaf else "BRANCH_LT"
                attrs["nodes_treeids"].append(tree_id)
                attrs["nodes_nodeids"].append(int(node["num"]))
                attrs["nodes_featureids"].append(feature)
                attrs["nodes_modes"].append(mode)
                attrs["nodes_values"].append(value)
                attrs["nodes_truenodeids"].append(left)
                attrs["nodes_falsenodeids"].append(right)
                attrs["nodes_missing_value_tracks_true"].append(int(not leaf and missing == left))
                if leaf:
                    attrs["target_treeids"].append(tree_id)
                    attrs["target_nodeids"].append(int(node["num"]))
                    attrs["target_ids"].append(target_id)
                    attrs["target_weights"].append(float(node["weight_value"]))
            tree_id += 1
    attrs.update(
        n_targets=len(boosters),
        base_values=[float(booster["base_score"]) for booster in boosters],
        aggregate_function="SUM",
        post_transform="NONE",
    )
    return attrs


def _raw_predictions(g: GraphBuilderExtendedProtocol, estimator, X: str, name: str) -> str:
    """Builds raw predictions directly from the public JSON serialization."""
    if not estimator.is_fitted:
        raise ValueError("Perpetual estimator must be fitted before conversion.")
    if estimator.cat_mapping:
        raise NotImplementedError(
            "Perpetual categorical input mappings are not supported by the converter."
        )
    attrs = _tree_attributes(estimator)
    input_type = g.get_type(X)
    if input_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE):
        X = g.op.Cast(X, to=onnx.TensorProto.DOUBLE, name=f"{name}_input")
        input_type = onnx.TensorProto.DOUBLE
    dtype = numpy.float64 if input_type == onnx.TensorProto.DOUBLE else numpy.float32
    if estimator.missing is not None and not numpy.isnan(estimator.missing):
        mask = g.op.Equal(X, numpy.array(estimator.missing, dtype=dtype), name=f"{name}_missing")
        X = g.op.Where(mask, numpy.array(numpy.nan, dtype=dtype), X, name=f"{name}_nan")

    thresholds = numpy.asarray(attrs["nodes_values"], dtype=numpy.float64)
    if input_type == onnx.TensorProto.FLOAT:
        rounded = thresholds.astype(numpy.float32)
        # For strict '<', rounding a boundary down misroutes equality probes.
        strict = numpy.array([mode == "BRANCH_LT" for mode in attrs["nodes_modes"]])
        down = strict & (rounded < thresholds)
        rounded[down] = numpy.nextafter(rounded[down], numpy.float32(numpy.inf))
        thresholds = rounded
    else:
        # Classic tree kernels store float32 thresholds. Evaluating double
        # comparisons first preserves boundaries without tensor attributes,
        # which are not supported by every native tree evaluator.
        branches = [i for i, mode in enumerate(attrs["nodes_modes"]) if mode != "LEAF"]
        if branches:
            features = numpy.array(
                [attrs["nodes_featureids"][i] for i in branches], dtype=numpy.int64
            )
            columns = g.op.Gather(X, features, axis=1, name=f"{name}_features")
            limits = thresholds[branches].reshape((1, -1))
            less = g.op.Less(columns, limits, name=f"{name}_less")
            inclusive = numpy.array(
                [attrs["nodes_modes"][i] == "BRANCH_LEQ" for i in branches], dtype=numpy.bool_
            )
            equal = g.op.Equal(columns, limits, name=f"{name}_equal")
            comparisons = g.op.Or(
                less,
                g.op.And(inclusive, equal, name=f"{name}_inclusive"),
                name=f"{name}_comparisons",
            )
            indicators = g.op.Cast(
                comparisons, to=onnx.TensorProto.FLOAT, name=f"{name}_indicators"
            )
            X = g.op.Where(
                g.op.IsNaN(columns, name=f"{name}_isnan"),
                numpy.array(numpy.nan, dtype=numpy.float32),
                indicators,
                name=f"{name}_tree_input",
            )
            for feature, i in enumerate(branches):
                attrs["nodes_featureids"][i] = feature
                attrs["nodes_modes"][i] = "BRANCH_GT"
                thresholds[i] = 0.5
    attrs["nodes_values"] = thresholds.tolist()
    raw = g.make_node(
        "TreeEnsembleRegressor", [X], domain="ai.onnx.ml", name=f"{name}_trees", **attrs
    )
    g.set_type(raw, onnx.TensorProto.FLOAT)
    g.set_shape(raw, (g.get_shape(X)[0], attrs["n_targets"]))
    return raw


@register_sklearn_converter(PerpetualClassifier)
def sklearn_perpetual_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator,
    X: str,
    name: str = "perpetual_classifier",
) -> Tuple[str, str]:
    """
    Converts a :class:`perpetual.PerpetualClassifier` without its ONNX exporter.

    Binary labels follow Perpetual's ``predict`` (encoded 0/1, even when
    ``classes_`` contains other labels); multiclass labels use ``classes_``.
    """
    raw = _raw_predictions(g, estimator, X, name)
    n_classes = len(estimator.classes_)
    if n_classes < 2:
        raise NotImplementedError("Perpetual classification requires at least two classes.")
    if n_classes == 2:
        positive = g.op.Sigmoid(raw, name=f"{name}_sigmoid")
        negative = g.op.Sub(numpy.array([1], dtype=numpy.float32), positive, name=f"{name}_p0")
        probabilities = g.op.Concat(
            negative, positive, axis=1, name=f"{name}_probabilities", outputs=outputs[1:]
        )
        label = g.op.ArgMax(
            probabilities, axis=1, keepdims=0, name=f"{name}_label", outputs=outputs[:1]
        )
    else:
        probabilities = g.op.Softmax(
            raw, axis=1, name=f"{name}_probabilities", outputs=outputs[1:]
        )
        indices = g.op.ArgMax(raw, axis=1, keepdims=0, name=f"{name}_argmax")
        classes = numpy.asarray(estimator.classes_)
        if numpy.issubdtype(classes.dtype, numpy.integer):
            classes = classes.astype(numpy.int64)
        if classes.dtype.kind in "OUS":
            label = g.make_node(
                "LabelEncoder",
                [indices],
                domain="ai.onnx.ml",
                keys_int64s=list(range(n_classes)),
                values_strings=classes.astype(str).tolist(),
                name=f"{name}_label",
                outputs=outputs[:1],
            )
            g.set_type(label, onnx.TensorProto.STRING)
            g.set_shape(label, (g.get_shape(raw)[0],))
        else:
            label = g.op.Gather(
                classes, indices, axis=0, name=f"{name}_label", outputs=outputs[:1]
            )
    return label, probabilities


@register_sklearn_converter(PerpetualRegressor)
def sklearn_perpetual_regressor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator,
    X: str,
    name: str = "perpetual_regressor",
) -> str:
    """Converts a :class:`perpetual.PerpetualRegressor` without its ONNX exporter."""
    raw = _raw_predictions(g, estimator, X, name)
    return g.op.Identity(raw, outputs=outputs, name=name)

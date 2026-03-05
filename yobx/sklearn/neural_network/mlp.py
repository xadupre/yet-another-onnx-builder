import copy
from typing import Tuple, Dict, List

import onnx
import onnx.helper as oh
from onnx import shape_inference
from sklearn.neural_network import MLPClassifier, MLPRegressor

from ..register import register_sklearn_converter
from ...xbuilder import GraphBuilder


# Operator attributes that must always be set explicitly because the yobx
# optimizer patterns cannot handle ONNX-spec defaults (i.e. the absence of
# the attribute).  Maps op_type -> {attr_name: default_value}.
_REQUIRED_ATTRS: Dict[str, Dict[str, int]] = {
    "Softmax": {"axis": -1},
    "LogSoftmax": {"axis": -1},
    "Hardmax": {"axis": -1},
}


def _ensure_required_attrs(node: onnx.NodeProto) -> onnx.NodeProto:
    """
    Return a (possibly modified) copy of *node* with all required
    attributes present.  Some yobx optimizer patterns call
    ``get_attribute(node, attr, exc=True)`` and raise an
    :exc:`AssertionError` when the attribute is missing even though it has
    a well-defined ONNX default value.  Adding the attribute explicitly
    prevents those assertion errors.

    :param node: the original NodeProto (not modified in-place)
    :return: the same node if no attributes needed to be added, otherwise
        a deep copy with the missing attributes appended
    """
    defaults = _REQUIRED_ATTRS.get(node.op_type)
    if not defaults:
        return node
    existing = {a.name for a in node.attribute}
    missing = {k: v for k, v in defaults.items() if k not in existing}
    if not missing:
        return node
    node_copy = copy.deepcopy(node)
    for attr_name, attr_val in missing.items():
        node_copy.attribute.append(oh.make_attribute(attr_name, attr_val))
    return node_copy


def _to_skl2onnx_input_type(elem_type: int, n_features: int):
    """
    Convert an ONNX ``elem_type`` integer into the matching skl2onnx input-type
    object for ``initial_types``.

    :param elem_type: ONNX element type (e.g. ``onnx.TensorProto.FLOAT``)
    :param n_features: number of input features
    :return: ``FloatTensorType`` or ``DoubleTensorType`` instance
    :raises NotImplementedError: for unsupported element types
    """
    from skl2onnx.common.data_types import DoubleTensorType, FloatTensorType

    if elem_type == onnx.TensorProto.FLOAT:
        return FloatTensorType([None, n_features])
    if elem_type == onnx.TensorProto.DOUBLE:
        return DoubleTensorType([None, n_features])
    raise NotImplementedError(
        f"Input elem_type {elem_type} is not supported. "
        "Only FLOAT (1) and DOUBLE (11) are supported by the skl2onnx MLP converter."
    )


def _inject_skl2onnx_nodes(
    g: GraphBuilder,
    onx_model: onnx.ModelProto,
    x_name: str,
    output_mapping: Dict[str, str],
    skip_op_types: set,
    name: str,
) -> None:
    """
    Injects all nodes and initializers from a :epkg:`sklearn-onnx`
    :class:`onnx.ModelProto` into an existing :class:`GraphBuilder`.

    The function performs the following steps:

    1. Runs ONNX shape inference on *onx_model* to populate
       ``graph.value_info`` with element-type information for every
       intermediate result.
    2. Adds all initializers from *onx_model* under prefixed names.
    3. Adds all nodes from *onx_model* (except those whose ``op_type`` is in
       *skip_op_types*), renaming inputs and outputs according to
       *output_mapping* and a freshly generated unique-name prefix.
    4. Propagates element-type information from the shape-inferred
       ``value_info`` so that the yobx optimizer patterns can inspect tensor
       types without raising errors.

    :param g: target graph builder
    :param onx_model: ONNX model produced by :epkg:`sklearn-onnx`
    :param x_name: name of the input tensor already registered in *g*
    :param output_mapping: mapping from skl2onnx tensor names (which should
        map to specific outputs) to the desired result names in *g*
    :param skip_op_types: set of operator type strings to omit (e.g.
        ``{'ZipMap'}`` for classifiers)
    :param name: node-name prefix used for all injected nodes
    """
    # Run ONNX shape inference so that value_info is populated with types.
    onx_model = shape_inference.infer_shapes(onx_model)

    skl_input_name = onx_model.graph.input[0].name

    # Build the renaming table: skl2onnx name → target name in g.
    renaming: Dict[str, str] = {skl_input_name: x_name}
    renaming.update(output_mapping)

    def rename(orig: str) -> str:
        if orig in renaming:
            return renaming[orig]
        new_name = g.unique_name(f"{name}_{orig}")
        renaming[orig] = new_name
        return new_name

    # Collect element-type information for all tensors (after shape inference).
    type_info: Dict[str, int] = {}
    for vi in (
        list(onx_model.graph.value_info)
        + list(onx_model.graph.output)
        + list(onx_model.graph.input)
    ):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type:
            type_info[vi.name] = vi.type.tensor_type.elem_type
    for init in onx_model.graph.initializer:
        type_info[init.name] = init.data_type

    # --- Initializers ---
    for init in onx_model.graph.initializer:
        new_name = rename(init.name)
        init_copy = copy.deepcopy(init)
        init_copy.name = new_name
        g.add_initializer(
            new_name,
            init_copy,
            itype=init.data_type,
            shape=tuple(init.dims),
            source="_inject_skl2onnx_nodes",
        )

    # Register non-default opset domains (e.g. ai.onnx.ml).
    for opset in onx_model.opset_import:
        if opset.domain and opset.domain not in g.opsets:
            g.opsets[opset.domain] = opset.version

    # --- Nodes ---
    for node in onx_model.graph.node:
        if node.op_type in skip_op_types:
            continue
        # Ensure all attributes that must be explicit are present.
        node = _ensure_required_attrs(node)
        new_inputs = [rename(i) if i else "" for i in node.input]
        new_outputs = [rename(o) if o else "" for o in node.output]
        g.make_node(
            node.op_type,
            new_inputs,
            new_outputs,
            domain=node.domain or "",
            attributes=list(node.attribute),
            name=name,
        )
        # Propagate element types so that optimizer patterns can inspect them.
        for orig_o, new_o in zip(node.output, new_outputs):
            if orig_o in type_info and type_info[orig_o]:
                try:
                    g.set_type(new_o, type_info[orig_o])
                except AssertionError:
                    # set_type raises AssertionError when the type is already
                    # known and conflicts; skip silently since GraphBuilder's own
                    # type inference may already have set a consistent type.
                    pass


@register_sklearn_converter((MLPClassifier,))
def sklearn_mlp_classifier(
    g: GraphBuilder,
    sts: Dict,
    outputs: List[str],
    estimator: MLPClassifier,
    X: str,
    name: str = "mlp_classifier",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.neural_network.MLPClassifier` into ONNX
    by delegating to the :epkg:`sklearn-onnx` (``skl2onnx``) converter.

    The sklearn-onnx conversion is used as the reference implementation.
    Its output is injected into the current :class:`GraphBuilder` with
    appropriate input / output name remapping.

    The ``ZipMap`` node (which produces a sequence-of-maps probability
    output used by sklearn-onnx) is discarded; the upstream probability
    tensor (a plain ``[N, n_classes]`` float matrix) is exposed directly
    as ``outputs[1]`` instead.

    :param g: graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names ``[label, probabilities]``
    :param estimator: a fitted ``MLPClassifier``
    :param X: input tensor name
    :param name: node name prefix
    :return: tuple ``(label_result_name, proba_result_name)``
    """
    assert isinstance(
        estimator, MLPClassifier
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    from skl2onnx import convert_sklearn

    itype = g.get_type(X)
    n_features = estimator.coefs_[0].shape[0]

    onx = convert_sklearn(
        estimator,
        initial_types=[("X", _to_skl2onnx_input_type(itype, n_features))],
    )

    # Find the probability tensor (the input consumed by ZipMap).
    prob_tensor = next(
        (node.input[0] for node in onx.graph.node if node.op_type == "ZipMap"),
        None,
    )
    assert prob_tensor is not None, (
        "sklearn-onnx did not produce a ZipMap node for MLPClassifier; "
        "cannot locate the probability tensor."
    )

    _inject_skl2onnx_nodes(
        g,
        onx,
        X,
        output_mapping={
            "output_label": outputs[0],
            prob_tensor: outputs[1],
        },
        skip_op_types={"ZipMap"},
        name=name,
    )

    return outputs[0], outputs[1]


@register_sklearn_converter((MLPRegressor,))
def sklearn_mlp_regressor(
    g: GraphBuilder,
    sts: Dict,
    outputs: List[str],
    estimator: MLPRegressor,
    X: str,
    name: str = "mlp_regressor",
) -> str:
    """
    Converts a :class:`sklearn.neural_network.MLPRegressor` into ONNX
    by delegating to the :epkg:`sklearn-onnx` (``skl2onnx``) converter.

    The sklearn-onnx conversion is used as the reference implementation.
    Its output is injected into the current :class:`GraphBuilder` with
    appropriate input / output name remapping.

    :param g: graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names ``[predictions]``
    :param estimator: a fitted ``MLPRegressor``
    :param X: input tensor name
    :param name: node name prefix
    :return: output tensor name
    """
    assert isinstance(
        estimator, MLPRegressor
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    from skl2onnx import convert_sklearn

    itype = g.get_type(X)
    n_features = estimator.coefs_[0].shape[0]

    onx = convert_sklearn(
        estimator,
        initial_types=[("X", _to_skl2onnx_input_type(itype, n_features))],
    )

    last_output = onx.graph.output[0].name

    _inject_skl2onnx_nodes(
        g,
        onx,
        X,
        output_mapping={last_output: outputs[0]},
        skip_op_types=set(),
        name=name,
    )

    return outputs[0]

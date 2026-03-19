from typing import Dict, List, Tuple

import numpy as np
import onnx
from sklearn.dummy import DummyClassifier, DummyRegressor

from ..register import register_sklearn_converter
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from ...typing import GraphBuilderExtendedProtocol


@register_sklearn_converter(DummyRegressor)
def sklearn_dummy_regressor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: DummyRegressor,
    X: str,
    name: str = "dummy_regressor",
) -> str:
    """
    Converts a :class:`sklearn.dummy.DummyRegressor` into ONNX.

    All supported strategies (``mean``, ``median``, ``quantile``,
    ``constant``) store the prediction constant in ``estimator.constant_``
    after fitting, so the ONNX graph simply broadcasts that constant to
    match the batch dimension of the input.

    Graph structure for single-output models:

    .. code-block:: text

        X  ──Shape──Slice([0:1])──────────────────────► batch_size [N]
                                                             │
        constant_ (1,)  ───────────────────Expand ──────────┘
                                               │
                                           result (N,)

    For multi-output models the constant is ``(1, n_outputs)`` and the
    output shape is ``(N, n_outputs)``.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names
    :param estimator: a fitted ``DummyRegressor``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: output tensor name
    """
    assert isinstance(
        estimator, DummyRegressor
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # constant_ has shape (1, n_outputs) after fitting.
    n_outputs = estimator.n_outputs_

    # Get the batch dimension N from the input shape.
    x_shape = g.op.Shape(X, name=f"{name}_shape")
    batch_size = g.op.Slice(
        x_shape,
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        name=f"{name}_batch_size",
    )  # 1-D tensor [N]

    if n_outputs == 1:
        # Flatten constant_ to shape (1,) and broadcast to (N,).
        const_1d = estimator.constant_.ravel().astype(dtype)
        result = g.op.Expand(const_1d, batch_size, name=name, outputs=outputs)
    else:
        # Broadcast constant_ (1, n_outputs) to (N, n_outputs).
        constant = estimator.constant_.astype(dtype)  # (1, n_outputs)
        n_out_tensor = np.array([n_outputs], dtype=np.int64)
        target_shape = g.op.Concat(batch_size, n_out_tensor, axis=0, name=f"{name}_target_shape")
        result = g.op.Expand(constant, target_shape, name=name, outputs=outputs)

    g.set_type(result, itype)
    return result


@register_sklearn_converter(DummyClassifier)
def sklearn_dummy_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: DummyClassifier,
    X: str,
    name: str = "dummy_classifier",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.dummy.DummyClassifier` into ONNX.

    The converter supports the deterministic strategies
    ``most_frequent``, ``prior``, and ``constant``.  The randomised
    strategies ``stratified`` and ``uniform`` are not supported because
    they produce non-deterministic outputs that cannot be expressed as a
    static ONNX graph.

    For the supported strategies the predicted probability row and the
    predicted class are constant for every sample, so the graph simply
    broadcasts those constants to match the batch dimension:

    .. code-block:: text

        X  ──Shape──Slice([0:1])──────────────────────────► batch_size [N]
                                                                 │
        proba_row (1, n_classes) ──────────Expand ───────────────┤
                                               │                 │
                                           proba (N, n_classes)  │
                                                             Expand
        class_const (1,) ──────────────────────────────────────► label (N,)

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names ``[label, probabilities]``
    :param estimator: a fitted ``DummyClassifier``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: tuple ``(label_name, proba_name)``
    :raises NotImplementedError: when ``strategy`` is ``"stratified"`` or
        ``"uniform"``
    :raises NotImplementedError: when the estimator has more than one output
    """
    if estimator.strategy in ("stratified", "uniform"):
        raise NotImplementedError(
            f"DummyClassifier with strategy={estimator.strategy!r} cannot be "
            "converted to ONNX because it produces non-deterministic output."
        )

    if estimator.n_outputs_ > 1:
        raise NotImplementedError("DummyClassifier with multiple outputs is not yet supported.")

    assert isinstance(
        estimator, DummyClassifier
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    classes = estimator.classes_
    n_classes = len(classes)
    class_prior = estimator.class_prior_

    # Determine the constant probability row and predicted class index.
    if estimator.strategy == "most_frequent":
        predicted_idx = int(np.argmax(class_prior))
        proba_row = np.zeros(n_classes, dtype=dtype)
        proba_row[predicted_idx] = 1.0
    elif estimator.strategy == "prior":
        predicted_idx = int(np.argmax(class_prior))
        proba_row = class_prior.astype(dtype)
    else:  # "constant"
        constant_val = estimator.constant
        # classes_ is always sorted; find the index of the constant value.
        indices = np.where(classes == constant_val)[0]
        if len(indices) == 0:
            raise ValueError(
                f"DummyClassifier constant value {constant_val!r} not found in "
                f"classes_ {classes!r}."
            )
        predicted_idx = int(indices[0])
        proba_row = np.zeros(n_classes, dtype=dtype)
        proba_row[predicted_idx] = 1.0

    # Get the batch dimension N from the input shape.
    x_shape = g.op.Shape(X, name=f"{name}_shape")
    batch_size = g.op.Slice(
        x_shape,
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        name=f"{name}_batch_size",
    )  # 1-D tensor [N]

    # Broadcast proba_row (1, n_classes) → (N, n_classes).
    proba_const = proba_row.reshape(1, n_classes)
    n_classes_tensor = np.array([n_classes], dtype=np.int64)
    proba_shape = g.op.Concat(batch_size, n_classes_tensor, axis=0, name=f"{name}_proba_shape")
    proba = g.op.Expand(
        proba_const, proba_shape, name=f"{name}_expand_proba", outputs=outputs[1:]
    )
    g.set_type(proba, itype)

    # Broadcast the predicted class (1,) → (N,).
    if np.issubdtype(classes.dtype, np.integer):
        class_const = np.array([classes[predicted_idx]], dtype=np.int64)
        label = g.op.Expand(class_const, batch_size, name=f"{name}_label", outputs=outputs[:1])
        g.set_type(label, onnx.TensorProto.INT64)
    else:
        class_const = np.array([str(classes[predicted_idx])])
        label = g.op.Expand(
            class_const, batch_size, name=f"{name}_label_string", outputs=outputs[:1]
        )
        g.set_type(label, onnx.TensorProto.STRING)

    return label, proba

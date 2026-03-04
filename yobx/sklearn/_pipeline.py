"""
Converts a :class:`sklearn.pipeline.Pipeline` into an ONNX graph.
"""
import numpy as np
from onnx import TensorProto
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from ..xbuilder import GraphBuilder
from ._standard_scaler import _add_standard_scaler_nodes
from ._logistic_regression import _add_logistic_regression_nodes


def _get_n_features(estimator):
    """Returns the number of input features for an estimator."""
    if isinstance(estimator, StandardScaler):
        return estimator.mean_.shape[0]
    if isinstance(estimator, LogisticRegression):
        return estimator.coef_.shape[1]
    raise NotImplementedError(
        f"Cannot determine n_features for {type(estimator).__name__!r}."
    )


def convert_pipeline(
    pipeline,
    input_name: str = "X",
    opset: int = 18,
) -> GraphBuilder:
    """
    Converts a fitted :class:`sklearn.pipeline.Pipeline` into a single
    :class:`GraphBuilder`.

    Each step's converter is called in order and the output of one step is
    wired as the input of the next step.  Currently supported step types are
    :class:`sklearn.preprocessing.StandardScaler` and
    :class:`sklearn.linear_model.LogisticRegression`.

    :param pipeline: a fitted ``Pipeline``
    :param input_name: name for the pipeline input
    :param opset: ONNX opset version
    :return: :class:`GraphBuilder` ready to be exported with ``to_onnx()``
    """
    steps = pipeline.steps
    n_steps = len(steps)
    first_estimator = steps[0][1]
    n_features = _get_n_features(first_estimator)

    g = GraphBuilder(opset, ir_version=9)
    g.make_tensor_input(input_name, TensorProto.FLOAT, (None, n_features))

    current_input = input_name
    for i, (step_name, estimator) in enumerate(steps):
        is_last = i == n_steps - 1
        prefix = f"{step_name}_"

        if isinstance(estimator, StandardScaler):
            out_name = "variable" if is_last else f"{step_name}_output"
            out = _add_standard_scaler_nodes(
                g, estimator, current_input, out_name, prefix=prefix
            )
            if is_last:
                n_out = estimator.mean_.shape[0]
                g.make_tensor_output(out, TensorProto.FLOAT, (None, n_out), indexed=False)
            current_input = out

        elif isinstance(estimator, LogisticRegression):
            label, label_dtype, proba_out, n_classes = _add_logistic_regression_nodes(
                g, estimator, current_input, "label", "probabilities", prefix=prefix
            )
            g.make_tensor_output(label, label_dtype, (None,), indexed=False)
            g.make_tensor_output(
                proba_out, TensorProto.FLOAT, (None, n_classes), indexed=False
            )

        else:
            raise NotImplementedError(
                f"No converter registered for step {step_name!r} "
                f"(type {type(estimator).__name__!r}). "
                f"Supported types: StandardScaler, LogisticRegression."
            )

    return g

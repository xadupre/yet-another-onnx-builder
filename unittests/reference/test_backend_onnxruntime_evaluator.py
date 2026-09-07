"""Exercises native backend cases through the explicit ONNX Runtime evaluator."""

from onnx_light.onnx.backend import make_test_class
from yobx.reference.onnxruntime_evaluator import OnnxruntimeEvaluator


def run_model(model, *inputs):
    """Executes a native model through the evaluator's serialization boundary."""
    session = OnnxruntimeEvaluator(model, providers=["CPUExecutionProvider"])
    return session.run(None, dict(zip(session.input_names, inputs)))


TestGeneratedOnnxruntimeBackend = make_test_class(
    run_model,
    include_regex=[
        r"^test_add(?:_bcast)?$",
        r"^test_cc_matmul(?:_|$)",
        r"^test_cc_(scan|loop)_basic_trip_count$",
        r"^test_cc_loop_zero_trip_count$",
    ],
    include_big=False,
    unload=True,
)

"""Exercises arithmetic cases from the native wheel's generated backend corpus."""

from onnx_light.onnx.backend import make_test_class
from yobx.reference import ExtendedReferenceEvaluator


def run_model(model, *inputs):
    """Executes a native backend case with the extended evaluator."""
    session = ExtendedReferenceEvaluator(model)
    return session.run(None, dict(zip(session.input_names, inputs)))


TestGeneratedNativeBackend = make_test_class(
    run_model,
    include_regex=[r"^test_(add(?:_|$)|cc_matmul(?:_|$))"],
    include_big=False,
    unload=True,
)

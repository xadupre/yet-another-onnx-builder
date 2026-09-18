"""Exercises native backend cases through the explicit Torch evaluator."""

import importlib.util
import unittest
from onnx_light.onnx.backend import make_test_class


def run_model(model, *inputs):
    """Executes a native backend case and returns NumPy outputs."""
    import torch
    from yobx.reference.torch_evaluator import TorchReferenceEvaluator

    session = TorchReferenceEvaluator(model)
    outputs = session.run(None, dict(zip(session.input_names, inputs)))
    return [
        value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        for value in outputs
    ]


TestGeneratedTorchBackend = unittest.skipIf(
    importlib.util.find_spec("torch") is None, "Torch is not installed."
)(
    make_test_class(
        run_model,
        include_regex=[r"^test_add(?:_bcast)?$", r"^test_cc_matmul(?:_|$)"],
        include_big=False,
        unload=True,
    )
)

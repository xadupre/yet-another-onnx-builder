import builtins
from unittest.mock import patch
from sklearn.pipeline import Pipeline
from yobx.sklearn import to_onnx


def native_to_onnx(estimator, X, **kwargs):
    """Converts while rejecting Perpetual's exporter and reference ONNX imports."""
    original_import = builtins.__import__
    perpetual_estimator = estimator[-1] if isinstance(estimator, Pipeline) else estimator

    def guarded_import(name, *args, **kwargs):
        if name == "onnx" or name.startswith("onnx."):
            raise AssertionError(f"Reference ONNX import attempted: {name}")
        return original_import(name, *args, **kwargs)

    with (
        patch.object(
            perpetual_estimator,
            "save_as_onnx",
            side_effect=AssertionError("save_as_onnx must not run"),
        ),
        patch("builtins.__import__", side_effect=guarded_import),
    ):
        return to_onnx(estimator, (X,), **kwargs)

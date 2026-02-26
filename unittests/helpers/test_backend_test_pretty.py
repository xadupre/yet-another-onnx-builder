import unittest
from typing import Any, Sequence
import onnx.backend.base
import onnx.backend.test
from onnx import ModelProto
from onnx.backend.base import Device, DeviceType
from yobx.helpers.onnx_helper import pretty_onnx


class PrettyBackendRep(onnx.backend.base.BackendRep):
    def __init__(self, model: onnx.ModelProto):
        self._model = model

    def run(self, inputs, **kwargs):
        text = pretty_onnx(self._model)
        if not text:
            raise AssertionError(f"Unable to print model\n{self._model}")
        return []


class PrettyBackend(onnx.backend.base.Backend):
    @classmethod
    def is_opset_supported(cls, model):  # pylint: disable=unused-argument
        return True, ""

    @classmethod
    def supports_device(cls, device: str) -> bool:
        d = Device(device)
        return d.type == DeviceType.CPU  # type: ignore[no-any-return]

    @classmethod
    def prepare(
        cls, model: onnx.ModelProto, device: str = "CPU", **kwargs: Any
    ) -> PrettyBackendRep:
        assert isinstance(model, ModelProto), f"Unexpected type {type(model)} for model."
        return PrettyBackendRep(model)


class PrettyBackendTest(onnx.backend.test.BackendTest):
    @classmethod
    def assert_similar_outputs(
        cls,
        ref_outputs: Sequence[Any],
        outputs: Sequence[Any],
        rtol: float,
        atol: float,
        model_dir: str | None = None,
    ) -> None:
        return None


backend_test = PrettyBackendTest(PrettyBackend())

backend_test.exclude(
    "(test_bvlc_alexnet"
    "|test_densenet121"
    "|test_inception_v1"
    "|test_inception_v2"
    "|test_resnet50"
    "|test_shufflenet"
    "|test_squeezenet"
    "|test_vgg19"
    "|test_zfnet512)"
)


# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.test_cases)

if __name__ == "__main__":
    res = unittest.main(verbosity=2, exit=False)
    tests_run = res.result.testsRun
    errors = len(res.result.errors)
    skipped = len(res.result.skipped)
    unexpected_successes = len(res.result.unexpectedSuccesses)
    expected_failures = len(res.result.expectedFailures)
    print("---------------------------------")
    print(
        f"tests_run={tests_run} errors={errors} skipped={skipped} "
        f"unexpected_successes={unexpected_successes} "
        f"expected_failures={expected_failures}"
    )

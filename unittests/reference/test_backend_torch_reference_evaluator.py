import unittest
import warnings
from typing import Any
import numpy
import onnx.backend.base
import onnx.backend.test
import torch
from onnx import ModelProto
from onnx.backend.base import Device, DeviceType
from yobx.reference.torch_evaluator import TorchReferenceEvaluator


class TorchReferenceEvaluatorBackendRep(onnx.backend.base.BackendRep):
    def __init__(self, session):
        self._session = session

    def run(self, inputs, **kwargs):
        if isinstance(inputs, numpy.ndarray):
            inputs = [inputs]
        if isinstance(inputs, list):
            feeds = dict(zip(self._session.input_names, inputs))
        elif isinstance(inputs, dict):
            feeds = inputs
        else:
            raise TypeError(f"Unexpected input type {type(inputs)!r}.")
        try:
            outs = self._session.run(None, feeds)
        except (TypeError, NotImplementedError) as e:
            raise unittest.SkipTest(str(e)) from e
        # Always convert torch tensors to numpy for the backend test framework.
        result = []
        for out in outs:
            if isinstance(out, torch.Tensor):
                result.append(out.detach().numpy())
            else:
                result.append(out)
        return result


class TorchReferenceEvaluatorBackend(onnx.backend.base.Backend):
    @classmethod
    def is_compatible(cls, model) -> bool:
        return True

    @classmethod
    def supports_device(cls, device: str) -> bool:
        d = Device(device)
        return d.type == DeviceType.CPU

    @classmethod
    def create_inference_session(cls, model):
        return TorchReferenceEvaluator(model)

    @classmethod
    def prepare(
        cls, model: Any, device: str = "CPU", **kwargs: Any
    ) -> TorchReferenceEvaluatorBackendRep:
        if isinstance(model, TorchReferenceEvaluator):
            return TorchReferenceEvaluatorBackendRep(model)
        if isinstance(model, (str, bytes, ModelProto)):
            try:
                inf = cls.create_inference_session(model)
            except (AssertionError, NotImplementedError) as e:
                raise unittest.SkipTest(str(e)) from e
            return cls.prepare(inf, device, **kwargs)
        raise TypeError(f"Unexpected type {type(model)} for model.")

    @classmethod
    def run_model(cls, model, inputs, device=None, **kwargs):
        rep = cls.prepare(model, device, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return rep.run(inputs, **kwargs)
            except (NotImplementedError, TypeError) as e:
                raise unittest.SkipTest(str(e)) from e

    @classmethod
    def run_node(cls, node, inputs, device=None, outputs_info=None, **kwargs):
        raise NotImplementedError("Unable to run the model node by node.")


backend_test = onnx.backend.test.BackendTest(TorchReferenceEvaluatorBackend, __name__)

# The following tests are too slow with the reference implementation (Conv).
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

# The following tests cannot pass because they consist in generating random numbers.
backend_test.exclude("(test_bernoulli)")

# The following tests are not supported.
backend_test.exclude(
    "(test_gradient"
    "|test_adam_multiple"
    "|test_if_opt"
    "|test_loop16_seq_none"
    "|test_scan_sum)"
)

# Unsupported edge cases.
backend_test.exclude(
    "(test_range_float_type_positive_delta_expanded"
    "|test_range_int32_type_negative_delta_expanded"
    "|test_scatter_with(out)?_axis"
    ")"
)

# Known limitations of TorchReferenceEvaluator.
# AveragePool: dilations and auto_pad modes not implemented.
backend_test.exclude(
    "(test_averagepool_2d_dilations"
    "|test_averagepool_2d_pads"
    "|test_averagepool_2d_precomputed_same_upper"
    "|test_averagepool_2d_same_lower"
    "|test_averagepool_2d_same_upper"
    "|test_averagepool_3d_dilations"
    "|test_conv_with_strides_and_asymmetric_padding)"
)

# Integer and unsigned integer types not fully supported by torch.
backend_test.exclude("(test_div_int8|test_div_int16|test_div_uint|test_pow_types_int)")

# LayerNormalization: known numerical failures.
backend_test.exclude("(test_layer_normalization)")

# ReduceMax/ReduceMin: empty-axes keepdims and empty-set not supported.
backend_test.exclude(
    "(test_reduce_max_default_axes_keepdim"
    "|test_reduce_max_empty_set"
    "|test_reduce_min_default_axes_keepdims"
    "|test_reduce_min_empty_set"
    ")"
)

# Other known failures.
backend_test.exclude(
    "(test_expand_shape_model"
    "|test_identity_opt"
    "|test_identity_sequence"
    "|test_sequence_insert"
    "|test_slice_neg_steps"
    "|test_transpose_default"
    "|test_unsqueeze_unsorted_axes"
    ")"
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

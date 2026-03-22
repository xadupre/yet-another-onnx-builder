import os
import unittest
from typing import Any
import numpy
import onnx.backend.base
import onnx.backend.test
from onnx import ModelProto
from onnx.backend.base import Device, DeviceType
from yobx.helpers import string_type
from yobx.helpers.onnx_helper import pretty_onnx
from yobx.helpers.rt_helper import make_feeds
from yobx.reference import ExtendedReferenceEvaluator
from yobx.xshape import BasicShapeBuilder
from yobx.helpers.onnx_helper import (
    overwrite_shape_in_model_proto,
    replace_static_dimensions_by_strings,
)


class ShapeBuilderTypeRep(onnx.backend.base.BackendRep):
    def __init__(self, model: onnx.ModelProto):
        op_type = model.graph.node[0].op_type
        self._model = overwrite_shape_in_model_proto(
            model,
            n_in=(
                1
                if op_type
                in {
                    "ArgMax",
                    "ArgMin",
                    "BlackmanWindow",
                    "CumSum",
                    "DepthToSpace",
                    "Expand",
                    "Flatten",
                    "GlobalAveragePool",
                    "GlobalMaxPool",
                    "HammingWindow",
                    "HannWindow",
                    "Reshape",
                    "SpaceToDepth",
                    "Squeeze",
                    "Unsqueeze",
                }
                or op_type.startswith("Reduce")
                else None
            ),
        )
        self._session = ExtendedReferenceEvaluator(self._model)
        self._dyn_model, self._mapping = replace_static_dimensions_by_strings(self._model)

    def run(self, inputs, **kwargs):
        if isinstance(inputs, numpy.ndarray):
            inputs = [inputs]
        if isinstance(inputs, list):
            feeds = make_feeds(self._model, inputs)
        elif isinstance(inputs, dict):
            feeds = inputs
        else:
            raise TypeError(f"Unexpected input type {type(inputs)!r}.")
        outs = self._session.run(None, feeds)

        # static
        shape_builder = BasicShapeBuilder(verbose=int(os.environ.get("VERBOSE", "0")))
        shape_builder.run_model(self._dyn_model, exc=True, inference="type")
        try:
            shape_builder.compare_with_true_inputs(feeds, outs, exc=True, do_shape=False)
        except NameError as e:
            raise unittest.SkipTest(  # noqa: B904
                f"shape function was found but only the rank could be set: {e}"
            )
        except Exception as e:
            raise AssertionError(
                f"Unable to handle a model due to {str(e)}\n---\n"
                f"inputs: {string_type(feeds, with_shape=True)}\n---\n"
                f"{shape_builder.get_debug_msg()}\n---\n"
                f"{pretty_onnx(self._dyn_model)}"
            ) from e

        # dynamic
        shape_builder = BasicShapeBuilder(verbose=int(os.environ.get("VERBOSE", "0")))
        shape_builder.run_model(self._model, exc=True, inference="type")
        try:
            shape_builder.compare_with_true_inputs(feeds, outs, exc=True, do_shape=False)
        except NameError as e:
            raise unittest.SkipTest(  # noqa: B904
                f"shape function was found but only the rank could be set: {e}"
            )
        except Exception as e:
            raise AssertionError(
                f"Unable to handle a model due to {str(e)}\n---\n"
                f"inputs: {string_type(feeds, with_shape=True)}\n---\n"
                f"{shape_builder.get_debug_msg()}\n---\n"
                f"{pretty_onnx(self._model)}"
            ) from e

        # ends
        return outs


class ShapeBuilderType(onnx.backend.base.Backend):
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
    ) -> ShapeBuilderTypeRep:
        assert isinstance(model, ModelProto), f"Unexpected type {type(model)} for model."
        return ShapeBuilderTypeRep(model)


backend_test = onnx.backend.test.BackendTest(ShapeBuilderType())

# The following tests are too slow with the reference implementation (Conv).
backend_test.exclude(
    "(test_bvlc_alexnet|test_densenet121|test_inception_v1|test_inception_v2"
    "|test_resnet50|test_shufflenet|test_squeezenet|test_vgg19|test_zfnet512"
    "|test_bernoulli|test_gradient|test_adam_multiple|test_adagrad|test_regex_full_match"
    "|test_adam|test_if_opt|test_loop16_seq_none|test_scan_sum|training|nesterov"
    "|momentum)"
)

# Not implemented yet.
backend_test.exclude(
    "(affine_grid|array_feature_extractor|binarizer|label_encoder|attention"
    "|averagepool"
    "|center_crop|col2im|compress|conv"
    "|det|dft|fft"
    "|gridsample|group_normalization|gru"
    "|hardmax|instancenorm"
    "|lppool|lstm|matmulinteger|maxunpool"
    "|nllloss|optional|pad|quantize|rms"
    "|resize|roialign|sce|sequence|shape_clip"
    "|test_maxpool_|shape_start_greater"
    "|simple_rnn|stft|string|strnorm"
    "|tensorscatter|tfidfvectorizer"
    "|unique|AvgPool|BatchNorm|Conv|Embedding|GLU|LeakyReLU|Linear"
    "|MaxPool|PReLU|ConstantPad"
    "|Reflection|Replication|ZeroPad|test_operator)"
)

# uncommon cases
backend_test.exclude("(expand_dim|_opt_|_if_|_scan|_loop|_image_|_scatter_)")

# broken case
backend_test.exclude(
    "(test_layer_normalization_2d_axis0"
    "|test_layer_normalization_2d_axis1"
    "|test_layer_normalization_2d_axis_negative_"
    "|test_layer_normalization_3d"
    "|test_layer_normalization_4d"
    "|test_layer_normalization_default"
    "|test_range_float_type_positive_delta_expanded"
    "|test_range_int32_type_negative_delta_expanded"
    "|test_rotary_embedding_interleaved_expanded"
    "|test_rotary_embedding_no_position_ids_interleaved_expanded"
    "|test_rotary_embedding_with_interleaved_rotary_dim_expanded"
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

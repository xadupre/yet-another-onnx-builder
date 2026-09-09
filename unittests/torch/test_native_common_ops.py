"""Checks common native ATen lowerings against real PyTorch and ONNX Runtime."""

import unittest

import numpy
import torch
from onnx_light.onnx import checker
from onnxruntime import InferenceSession, SessionOptions

from yobx.torch import to_onnx
from yobx.torch.export_options import ExportOptions


class TestNativeCommonOps(unittest.TestCase):
    def export(self, model, args, **kwargs):
        """Exports and checks a model without decomposing the tested ATen calls."""
        artifact = to_onnx(
            model,
            args,
            input_names=[f"X{i}" for i in range(len(args))],
            export_options=ExportOptions(remove_inplace=False),
            validate_onnx=True,
            **kwargs,
        )
        checker.check_model(artifact.proto)
        return artifact

    def check(self, model, args, cases=None, **kwargs):
        """Compares native execution with PyTorch, including output dtypes."""
        artifact = self.export(model, args, **kwargs)
        options = SessionOptions()
        options.log_severity_level = 3
        session = InferenceSession(
            artifact.proto.SerializeToString(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        for inputs in cases or [args]:
            expected = model(*inputs)
            expected = (expected,) if isinstance(expected, torch.Tensor) else expected
            actual = session.run(None, {f"X{i}": value.numpy() for i, value in enumerate(inputs)})
            self.assertEqual(len(actual), len(expected))
            for result, reference in zip(actual, expected):
                reference = reference.numpy()
                self.assertEqual(result.dtype, reference.dtype)
                numpy.testing.assert_allclose(
                    result, reference, rtol=1e-6, atol=0, equal_nan=True
                )
                if numpy.issubdtype(result.dtype, numpy.floating):
                    zero = (reference == 0) & (result == 0)
                    numpy.testing.assert_array_equal(
                        numpy.signbit(result[zero]), numpy.signbit(reference[zero])
                    )
        return artifact

    def test_remainder_tensor_and_scalars(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return (
                    torch.ops.aten.remainder.Tensor(x, y),
                    torch.ops.aten.remainder.Scalar(x, -3),
                    torch.ops.aten.remainder.Scalar_Tensor(5, y),
                )

        for dtype in (torch.int32, torch.int64, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                x = torch.tensor([[-7, -6, -1, 0, 1, 6, 7]], dtype=dtype)
                y = torch.tensor([[-3], [3]], dtype=dtype)
                self.check(Model(), (x, y))
        self.check(
            Model(),
            (
                torch.tensor([[2**60 + 7, -(2**60) - 7]], dtype=torch.int64),
                torch.tensor([[-3], [3]], dtype=torch.int64),
            ),
        )
        self.check(
            Model(),
            (
                torch.tensor([[-7, -0.0, 0.0, 7, float("nan"), float("inf")]]),
                torch.tensor([[-float("inf")], [float("inf")], [-3.0], [3.0], [0.0]]),
            ),
        )
        self.check(
            Model(),
            (torch.tensor([[-7, 0, 7]], dtype=torch.int32), torch.tensor([[2.5], [-2.5]])),
        )

    def test_division_square_comparison_and_logic(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return (
                    torch.ops.aten.true_divide.Tensor(x, y),
                    torch.ops.aten.true_divide.Scalar(x, 2),
                    torch.ops.aten.square.default(x),
                    torch.ops.aten.ne.Tensor(x, y),
                    torch.ops.aten.ne.Scalar(x, 0.5),
                    torch.ops.aten.logical_xor.default(x, y),
                )

        for dtype in (torch.int32, torch.int64, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                self.check(
                    Model(),
                    (
                        torch.tensor([[-5, 0, 7]], dtype=dtype),
                        torch.tensor([[2], [-3]], dtype=dtype),
                    ),
                )
        self.check(
            Model(),
            (torch.tensor([[float("nan"), 0, -2]]), torch.tensor([[1.0], [float("nan")]])),
        )
        self.check(
            Model(),
            (torch.tensor([[-5, 0, 7]], dtype=torch.int32), torch.tensor([[2.5], [-3.5]])),
        )

    def test_bitwise_and(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return (
                    torch.ops.aten.bitwise_and.Tensor(x, y),
                    torch.ops.aten.bitwise_and.Scalar(x, True),
                )

        for dtype in (torch.bool, torch.uint8, torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                self.check(
                    Model(),
                    (
                        torch.tensor([[0, 1, 7]], dtype=dtype),
                        torch.tensor([[1], [3]], dtype=dtype),
                    ),
                )
        self.check(
            Model(),
            (
                torch.tensor([[-7, 0, 2**60 + 7]], dtype=torch.int64),
                torch.tensor([[-3], [3]], dtype=torch.int32),
            ),
        )
        with self.assertRaisesRegex(NotImplementedError, "opset 18"):
            self.export(Model(), (torch.ones(2, dtype=torch.int64),) * 2, target_opset=17)

    def test_low_precision_operations(self):
        class Model(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.dtype = dtype

            def forward(self, x, y):
                x = torch.ops.aten._to_copy.default(x, dtype=self.dtype)
                y = torch.ops.aten._to_copy.default(y, dtype=self.dtype)
                values = (
                    torch.remainder(x, y),
                    torch.remainder(x, 3.14159),
                    torch.true_divide(x, y),
                    torch.square(x),
                    torch.ne(x, y),
                    torch.ne(x, 3.14159),
                    torch.logical_xor(x, y),
                    torch.zeros_like(x),
                    torch.ones_like(x),
                )
                return tuple(
                    torch.ops.aten._to_copy.default(value, dtype=torch.float32)
                    for value in values
                )

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.check(
                    Model(dtype),
                    (
                        torch.tensor([[-7.125, -0.0, 3.140625, 7.125, float("nan")]]),
                        torch.tensor([[2.5], [-2.5]]),
                    ),
                )

    def test_like_factories_dynamic_nan_and_dtype(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return (
                    torch.zeros_like(x),
                    torch.ones_like(x),
                    torch.zeros_like(x, dtype=torch.int64),
                    torch.ones_like(x, dtype=torch.bool),
                )

        self.check(
            Model(),
            (torch.ones(2, 3),),
            cases=[(torch.full((5, 3), float("nan")),), (torch.full((0, 3), float("inf")),)],
            dynamic_shapes=({0: "batch"},),
        )
        self.check(Model(), (torch.tensor(7, dtype=torch.int64),))

    def test_broadcast_tensors_dynamic_mixed_dtype(self):
        class Model(torch.nn.Module):
            def forward(self, x, y, z):
                return torch.broadcast_tensors(x, y, z)

        self.check(
            Model(),
            (torch.ones(2, 1), torch.arange(3, dtype=torch.int64), torch.tensor(True)),
            cases=[
                (torch.randn(batch, 1), torch.arange(3), torch.tensor(True))
                for batch in (0, 1, 5)
            ],
            dynamic_shapes=({0: "batch"}, None, None),
        )
        self.check(Model(), (torch.tensor(2.0), torch.tensor(3), torch.tensor(False)))

    def test_metadata_assertions(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                torch.ops.aten._assert_tensor_metadata.default(
                    x, None, None, x.dtype, device=x.device, layout=x.layout
                )
                torch.ops.aten._assert_tensor_metadata.default(x, [x.shape[0], 3], [3, 1])
                return x + 1

        artifact = self.check(
            Model(),
            (torch.ones(2, 3),),
            cases=[(torch.randn(5, 3),)],
            dynamic_shapes=({0: "batch"},),
        )
        self.assertNotIn(
            "_assert_tensor_metadata", [node.op_type for node in artifact.proto.graph.node]
        )

    def test_metadata_dtype_assertions(self):
        class Model(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.dtype = dtype

            def forward(self, x):
                value = torch.ops.aten._to_copy.default(x, dtype=self.dtype)
                torch.ops.aten._assert_tensor_metadata.default(value, None, None, self.dtype)
                return torch.ops.aten._to_copy.default(value, dtype=torch.float32)

        for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                self.check(Model(dtype), (torch.tensor([[-2.0, 0.0, 3.0]]),))

    def test_metadata_assertions_reject_unproved_checks(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                torch.ops.aten._assert_tensor_metadata.default(x, dtype=x.dtype)
                return x + 1

        for key, expected, error, message in (
            ("dtype", torch.float64, ValueError, "failed for dtype"),
            ("size", [2, 3], NotImplementedError, "dynamic tensor metadata"),
            ("stride", [4, 1], ValueError, "failed for stride"),
            ("size", [3], ValueError, "rank"),
        ):
            with self.subTest(key=key, expected=expected):
                program = torch.export.export(
                    Model(), (torch.ones(2, 3),), dynamic_shapes=({0: torch.export.Dim("batch")},)
                )
                assertion = next(
                    node
                    for node in program.graph.nodes
                    if node.target == torch.ops.aten._assert_tensor_metadata.default
                )
                assertion.kwargs = {key: expected}
                program.graph_module.recompile()
                with self.assertRaisesRegex(error, message):
                    self.export(program, (torch.ones(2, 3),))


if __name__ == "__main__":
    unittest.main()

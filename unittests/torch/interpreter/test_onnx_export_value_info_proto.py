"""
Unit tests for ValueInfoProto support in yobx.torch.interpreter.to_onnx.
"""

import unittest
import numpy as np
import onnx
from yobx.ext_test_case import ExtTestCase, requires_torch, skipif_ci_windows
from yobx.reference import ExtendedReferenceEvaluator
from yobx.torch.interpreter import to_onnx


@requires_torch("2.4")
class TestTorchToOnnxValueInfoProto(ExtTestCase):
    """Tests that torch to_onnx accepts :class:`onnx.ValueInfoProto` as input descriptors."""

    @skipif_ci_windows("not yet supported on Windows")
    def test_single_input_value_info_proto(self):
        """ValueInfoProto replaces the torch tensor as input specification."""
        import torch

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int, n_targets: int):
                super().__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                return torch.relu(self.linear(x))

        model = Neuron(4, 2)
        vip = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["batch", 4])
        artifact = to_onnx(model, (vip,))
        onx = artifact.proto

        # The exported model must have a valid graph.
        self.assertIsNotNone(onx)
        self.assertGreater(len(onx.graph.input), 0)

        # Verify execution with a real input.
        x = torch.randn(3, 4)
        expected = model(x).detach().numpy()
        from yobx.torch.interpreter import match_input_parameters

        names = [i.name for i in onx.graph.input]
        pfeeds = match_input_parameters(model, names, (x,))
        nfeeds = {k: v.detach().numpy() for k, v in pfeeds.items()}
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, nfeeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not yet supported on Windows")
    def test_two_inputs_value_info_proto(self):
        """Two ValueInfoProto inputs with a shared batch dimension."""
        import torch

        class AddModel(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        model = AddModel()
        vip_x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["batch", 3])
        vip_y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, ["batch", 3])
        artifact = to_onnx(model, (vip_x, vip_y))
        onx = artifact.proto

        self.assertIsNotNone(onx)
        self.assertGreater(len(onx.graph.input), 0)

        x = np.random.randn(5, 3).astype(np.float32)
        y = np.random.randn(5, 3).astype(np.float32)
        expected = x + y

        # Build feed dict from input names.
        input_names = [i.name for i in onx.graph.input]
        feeds = {input_names[0]: x, input_names[1]: y}
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-6)

    @skipif_ci_windows("not yet supported on Windows")
    def test_static_shape_value_info_proto(self):
        """ValueInfoProto with fully static shape (no symbolic dims)."""
        import torch

        class LinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 5)

            def forward(self, x):
                return self.linear(x)

        model = LinearModel()
        # Fully static shape: (2, 3)
        vip = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 3])
        artifact = to_onnx(model, (vip,))
        onx = artifact.proto

        self.assertIsNotNone(onx)
        self.assertGreater(len(onx.graph.input), 0)

    @skipif_ci_windows("not yet supported on Windows")
    def test_dynamic_shapes_derived_from_value_info_proto(self):
        """Dynamic shapes are correctly derived from symbolic ValueInfoProto dims."""
        import torch

        class IdentityModel(torch.nn.Module):
            def forward(self, x):
                return x

        model = IdentityModel()
        # Both batch and seq are symbolic.
        vip = onnx.helper.make_tensor_value_info(
            "x", onnx.TensorProto.FLOAT, ["batch", "seq", 16]
        )
        artifact = to_onnx(model, (vip,))
        onx = artifact.proto

        self.assertIsNotNone(onx)
        # The single input placeholder.
        self.assertGreater(len(onx.graph.input), 0)

        # Run with two different batch/seq sizes to confirm shape is dynamic.
        for batch, seq in [(2, 4), (5, 7)]:
            x = np.random.randn(batch, seq, 16).astype(np.float32)
            feeds = {onx.graph.input[0].name: x}
            ref = ExtendedReferenceEvaluator(onx)
            got = ref.run(None, feeds)
            self.assertEqualArray(x, got[0], atol=1e-6)

    @skipif_ci_windows("not yet supported on Windows")
    def test_user_dynamic_shapes_not_overridden(self):
        """When dynamic_shapes is provided, it is not overridden by VIP-derived shapes."""
        import torch

        class IdentityModel(torch.nn.Module):
            def forward(self, x):
                return x

        model = IdentityModel()
        vip = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["batch", 4])
        # Explicitly pass dynamic_shapes; the VIP-derived shapes should not override it.
        artifact = to_onnx(model, (vip,), dynamic_shapes=({0: "batch"},))
        onx = artifact.proto

        self.assertIsNotNone(onx)
        self.assertGreater(len(onx.graph.input), 0)

    def test_value_info_proto_no_shape_raises(self):
        """ValueInfoProto without any shape information raises ValueError."""
        from yobx.torch.fake_tensor_helper import FakeTensorContext

        vip = onnx.ValueInfoProto()
        vip.name = "x"
        vip.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        # Intentionally leave the shape field unset.
        ctx = FakeTensorContext()
        with self.assertRaises(ValueError):
            ctx.value_info_proto_to_torch(vip)

    def test_value_info_proto_zero_dim_raises(self):
        """ValueInfoProto with dim_value=0 and no dim_param raises ValueError."""
        from yobx.torch.fake_tensor_helper import FakeTensorContext

        vip = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [0, 4])
        ctx = FakeTensorContext()
        with self.assertRaises(ValueError):
            ctx.value_info_proto_to_torch(vip)

    @skipif_ci_windows("not yet supported on Windows")
    def test_dict_args_value_info_proto(self):
        """ValueInfoProto objects inside a dict passed as args are converted correctly."""
        import torch
        from yobx.torch.interpreter.onnx_export import (
            _contains_value_info_proto,
            _replace_value_info_protos,
        )
        from yobx.torch.fake_tensor_helper import FakeTensorContext

        vip = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["batch", 4])
        d = {"x": vip, "extra": 1}

        self.assertTrue(_contains_value_info_proto(d))

        ctx = FakeTensorContext()
        converted, shapes = _replace_value_info_protos(d, ctx)

        # The VIP should have been converted to a torch tensor.
        self.assertIsInstance(converted["x"], torch.Tensor)
        # The non-VIP value should be left unchanged.
        self.assertEqual(converted["extra"], 1)
        # dynamic_shapes should have an entry for "x".
        self.assertIsNotNone(shapes)
        self.assertIsNotNone(shapes["x"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

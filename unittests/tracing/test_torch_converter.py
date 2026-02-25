import os
import tempfile
import unittest

import onnx
import torch
from onnx_pipe.tracing.torch_converter import convert_torch_to_onnx, get_leaf_modules


class _SimpleLinear(torch.nn.Module):
    """Single linear layer with no children – a leaf itself."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


class TestGetLeafModules(unittest.TestCase):
    def test_sequential_leaves(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)
        )
        leaves = get_leaf_modules(model)
        # Sequential has three direct children, each a leaf
        self.assertEqual(list(leaves.keys()), ["0", "1", "2"])
        self.assertIsInstance(leaves["0"], torch.nn.Linear)
        self.assertIsInstance(leaves["1"], torch.nn.ReLU)
        self.assertIsInstance(leaves["2"], torch.nn.Linear)

    def test_nested_leaves(self):
        model = _SimpleLinear()
        leaves = get_leaf_modules(model)
        # _SimpleLinear.fc (a Linear) is the only leaf
        self.assertIn("fc", leaves)
        self.assertIsInstance(leaves["fc"], torch.nn.Linear)

    def test_bare_module_is_leaf(self):
        # A module with no children should return itself under key ""
        linear = torch.nn.Linear(2, 2)
        leaves = get_leaf_modules(linear)
        self.assertIn("", leaves)
        self.assertIs(leaves[""], linear)


class TestConvertTorchToOnnx(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)
        )
        self.model.eval()
        self.dummy_input = torch.zeros(1, 4)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_returns_model_proto(self):
        result = convert_torch_to_onnx(self.model, (self.dummy_input,))
        self.assertIsInstance(result, onnx.ModelProto)

    def test_onnx_model_is_valid(self):
        result = convert_torch_to_onnx(self.model, (self.dummy_input,))
        # onnx.checker raises if the model is invalid
        onnx.checker.check_model(result)

    def test_saves_output_file(self):
        output_path = os.path.join(self.tmp_dir, "model.onnx")
        convert_torch_to_onnx(self.model, (self.dummy_input,), output_path=output_path)
        self.assertTrue(os.path.exists(output_path))

    def test_custom_input_output_names(self):
        result = convert_torch_to_onnx(
            self.model,
            (self.dummy_input,),
            input_names=["features"],
            output_names=["logits"],
        )
        input_names = [n.name for n in result.graph.input]
        output_names = [n.name for n in result.graph.output]
        self.assertIn("features", input_names)
        self.assertIn("logits", output_names)

    def test_dynamic_axes(self):
        result = convert_torch_to_onnx(
            self.model,
            (self.dummy_input,),
            input_names=["x"],
            dynamic_axes={"x": {0: "batch_size"}},
        )
        self.assertIsInstance(result, onnx.ModelProto)

    def test_opset_version(self):
        result = convert_torch_to_onnx(
            self.model, (self.dummy_input,), opset_version=14
        )
        opsets = {op.domain: op.version for op in result.opset_import}
        self.assertEqual(opsets.get(""), 14)

    def test_raises_on_non_module(self):
        with self.assertRaises(TypeError):
            convert_torch_to_onnx("not a module", (self.dummy_input,))

    def test_raises_on_empty_args(self):
        with self.assertRaises(ValueError):
            convert_torch_to_onnx(self.model, ())


if __name__ == "__main__":
    unittest.main()

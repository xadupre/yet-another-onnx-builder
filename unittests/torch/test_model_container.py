import os
import tempfile
import unittest
import numpy as np
import onnx
import onnx.helper as oh
from yobx.ext_test_case import ExtTestCase, requires_torch


def _make_simple_model_with_external(initializer_name: str, data: np.ndarray):
    """Helper to create a ModelProto that references an external initializer."""
    x_info = oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, list(data.shape))
    y_info = oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, list(data.shape))
    add_node = oh.make_node("Add", inputs=["x", initializer_name], outputs=["y"])

    # Create an external tensor
    init = onnx.TensorProto()
    init.data_type = onnx.TensorProto.FLOAT
    init.name = initializer_name
    for d in data.shape:
        init.dims.append(d)
    ext = init.external_data.add()
    ext.key = "location"
    ext.value = f"#{initializer_name}"

    graph = oh.make_graph([add_node], "test_graph", [x_info], [y_info], initializer=[init])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
    return model


@requires_torch("2.9")
class TestGetType(ExtTestCase):
    def setUp(self):
        from yobx.torch.model_container import _get_type

        self._get_type = _get_type

    def test_int_passthrough(self):
        self.assertEqual(self._get_type(1), 1)

    def test_float32(self):
        self.assertEqual(self._get_type("float32"), onnx.TensorProto.FLOAT)

    def test_float16(self):
        self.assertEqual(self._get_type("float16"), onnx.TensorProto.FLOAT16)

    def test_bfloat16(self):
        self.assertEqual(self._get_type("bfloat16"), onnx.TensorProto.BFLOAT16)

    def test_float64(self):
        self.assertEqual(self._get_type("float64"), onnx.TensorProto.DOUBLE)

    def test_int64(self):
        self.assertEqual(self._get_type("int64"), onnx.TensorProto.INT64)

    def test_int32(self):
        self.assertEqual(self._get_type("int32"), onnx.TensorProto.INT32)

    def test_bool(self):
        self.assertEqual(self._get_type("bool"), onnx.TensorProto.BOOL)

    def test_none(self):
        self.assertEqual(self._get_type(None), onnx.TensorProto.UNDEFINED)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            self._get_type("unknown_type")

    def test_unknown_no_exc(self):
        result = self._get_type("unknown_type", exc=False)
        self.assertEqual(result, "unknown_type")


@requires_torch("2.9")
class TestTorchModelContainer(ExtTestCase):
    def test_import(self):
        from yobx.torch.model_container import TorchModelContainer

        self.assertIsNotNone(TorchModelContainer)

    def test_init(self):
        from yobx.torch.model_container import TorchModelContainer

        container = TorchModelContainer()
        self.assertIsInstance(container._stats, dict)
        self.assertFalse(container.inline)
        self.assertIn("time_export_write_model", container._stats)

    def test_save_load_numpy(self):
        from yobx.torch.model_container import TorchModelContainer

        data = np.ones((3, 4), dtype=np.float32)
        model = _make_simple_model_with_external("weight", data)

        container = TorchModelContainer()
        container.model_proto = model
        container.large_initializers = {"#weight": data}

        with tempfile.TemporaryDirectory() as tmp:
            file_path = os.path.join(tmp, "model.onnx")
            saved = container.save(file_path)
            self.assertIsInstance(saved, onnx.ModelProto)
            self.assertTrue(os.path.exists(file_path))
            self.assertTrue(os.path.exists(file_path + ".data"))

    def test_save_torch_tensor(self):
        import torch
        from yobx.torch.model_container import TorchModelContainer

        data = torch.ones(3, 4, dtype=torch.float32)
        model = _make_simple_model_with_external("weight", data.numpy())

        container = TorchModelContainer()
        container.model_proto = model
        container.large_initializers = {"#weight": data}

        with tempfile.TemporaryDirectory() as tmp:
            file_path = os.path.join(tmp, "model.onnx")
            saved = container.save(file_path)
            self.assertIsInstance(saved, onnx.ModelProto)
            self.assertTrue(os.path.exists(file_path))

    def test_exported_from_yobx_torch(self):
        from yobx.torch import TorchModelContainer

        self.assertIsNotNone(TorchModelContainer)


if __name__ == "__main__":
    unittest.main(verbosity=2)


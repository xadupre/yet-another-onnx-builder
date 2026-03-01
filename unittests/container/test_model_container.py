import os
import tempfile
import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from yobx.ext_test_case import ExtTestCase, requires_torch

TFLOAT = onnx.TensorProto.FLOAT


def _make_simple_model_with_external(initializer_name: str, data: np.ndarray):
    """Helper to create a ModelProto that references an external initializer."""
    x_info = oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, list(data.shape))
    y_info = oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, list(data.shape))
    add_node = oh.make_node("Add", inputs=["x", initializer_name], outputs=["y"])

    # Create an external tensor
    init = onnx.TensorProto()
    init.data_type = onnx.TensorProto.FLOAT
    init.name = initializer_name
    init.data_location = onnx.TensorProto.EXTERNAL
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
        from yobx.container.model_container import _get_type

        self._get_type = _get_type

    def test_int_passthrough(self):
        self.assertEqual(self._get_type(1), 1)

    def test_float32(self):
        self.assertEqual(self._get_type(np.float32), onnx.TensorProto.FLOAT)

    def test_float16(self):
        self.assertEqual(self._get_type(np.float16), onnx.TensorProto.FLOAT16)

    def test_float64(self):
        self.assertEqual(self._get_type(np.float64), onnx.TensorProto.DOUBLE)

    def test_int64(self):
        self.assertEqual(self._get_type(np.int64), onnx.TensorProto.INT64)

    def test_int32(self):
        self.assertEqual(self._get_type(np.int32), onnx.TensorProto.INT32)

    def test_bool(self):
        self.assertEqual(self._get_type(np.bool_), onnx.TensorProto.BOOL)

    def test_none(self):
        self.assertEqual(self._get_type(None), onnx.TensorProto.UNDEFINED)

    def test_unknown_raises(self):
        with self.assertRaises(Exception):  # noqa: B017
            self._get_type("unknown_type")


@requires_torch("2.9")
class TestExtendedModelContainer(ExtTestCase):
    def _get_model(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [320, 1280])],
                [oh.make_tensor_value_info("Z", TFLOAT, [3, 5, 320, 640])],
                [
                    onh.from_array(np.random.rand(3, 5, 1280, 640).astype(np.float32), name="Y"),
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 320, 1280], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 1280, 640], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 320, 640], dtype=np.int64), name="shape3"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        return model

    def test_import(self):
        from yobx.container import ExtendedModelContainer

        self.assertIsNotNone(ExtendedModelContainer)

    def test_init(self):
        from yobx.container import ExtendedModelContainer

        container = ExtendedModelContainer()
        self.assertFalse(container.inline)

    def test_save_load_numpy(self):
        from yobx.container import ExtendedModelContainer

        data = np.ones((3, 4), dtype=np.float32)
        model = _make_simple_model_with_external("weight", data)

        container = ExtendedModelContainer()
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
        from yobx.container import ExtendedModelContainer

        data = torch.ones(3, 4, dtype=torch.float32)
        model = self._get_model()

        container = ExtendedModelContainer()
        container.model_proto = model
        container.large_initializers = {"#weight": data}

        with tempfile.TemporaryDirectory() as tmp:
            file_path = os.path.join(tmp, "model.onnx")
            saved = container.save(file_path)
            self.assertIsInstance(saved, onnx.ModelProto)
            self.assertTrue(os.path.exists(file_path))

    def test_save_then_load_all_in_one_file(self):
        """Round-trip: save a model with a numpy initializer and reload it."""
        from yobx.container import ExtendedModelContainer

        data = np.ones((3, 4), dtype=np.float32)
        model = _make_simple_model_with_external("weight", data)

        container = ExtendedModelContainer()
        container.model_proto = model
        container.large_initializers = {"#weight": data}

        with tempfile.TemporaryDirectory() as tmp:
            file_path = os.path.join(tmp, "model.onnx")
            container.save(file_path, all_tensors_to_one_file=True)
            self.assertTrue(os.path.exists(file_path))
            self.assertTrue(os.path.exists(file_path + ".data"))

            loaded = ExtendedModelContainer().load(file_path)
            self.assertEqual(len(loaded.model_proto.graph.initializer), 1)
            self.assertIsInstance(loaded.model_proto, onnx.ModelProto)
            self.assertEqual(loaded.model_proto.graph.node[0].op_type, "Add")
            self.assertGreater(len(loaded.large_initializers), 0)
            weight = next(iter(loaded.large_initializers.values()))
            self.assertEqual(weight.shape, data.shape)

    def test_save_then_load_one_file_per_tensor(self):
        """Round-trip: save with one file per tensor and reload it."""
        from yobx.container import ExtendedModelContainer

        data = np.ones((3, 4), dtype=np.float32)
        model = _make_simple_model_with_external("weight", data)

        container = ExtendedModelContainer()
        container.model_proto = model
        container.large_initializers = {"#weight": data}

        with tempfile.TemporaryDirectory() as tmp:
            file_path = os.path.join(tmp, "model.onnx")
            container.save(file_path, all_tensors_to_one_file=False)
            self.assertTrue(os.path.exists(file_path))
            # At least one per-tensor weight file should be written alongside
            weight_files = [f for f in os.listdir(tmp) if f != "model.onnx"]
            self.assertGreater(len(weight_files), 0)

            loaded = ExtendedModelContainer().load(file_path)
            self.assertIsInstance(loaded.model_proto, onnx.ModelProto)
            self.assertEqual(loaded.model_proto.graph.node[0].op_type, "Add")
            self.assertGreater(len(loaded.large_initializers), 0)
            weight = next(iter(loaded.large_initializers.values()))
            self.assertEqual(weight.shape, data.shape)

    def test_save_then_load_all_in_one_file_torch(self):
        """Round-trip: save a model with a torch.Tensor initializer and reload it."""
        import torch
        from yobx.container import ExtendedModelContainer

        data = torch.ones(3, 4, dtype=torch.float32)
        model = _make_simple_model_with_external("weight", data)

        container = ExtendedModelContainer()
        container.model_proto = model
        container.large_initializers = {"#weight": data}

        with tempfile.TemporaryDirectory() as tmp:
            file_path = os.path.join(tmp, "model.onnx")
            container.save(file_path, all_tensors_to_one_file=True)
            self.assertTrue(os.path.exists(file_path))
            self.assertTrue(os.path.exists(file_path + ".data"))

            loaded = ExtendedModelContainer().load(file_path)
            self.assertEqual(len(loaded.model_proto.graph.initializer), 1)
            self.assertIsInstance(loaded.model_proto, onnx.ModelProto)
            self.assertEqual(loaded.model_proto.graph.node[0].op_type, "Add")
            self.assertGreater(len(loaded.large_initializers), 0)
            weight = next(iter(loaded.large_initializers.values()))
            self.assertEqual(weight.shape, tuple(data.shape))

    def test_save_then_load_one_file_per_tensor_torch(self):
        """Round-trip: save a model with a torch.Tensor initializer, one file per tensor."""
        import torch
        from yobx.container import ExtendedModelContainer

        data = torch.ones(3, 4, dtype=torch.float32)
        model = _make_simple_model_with_external("weight", data)

        container = ExtendedModelContainer()
        container.model_proto = model
        container.large_initializers = {"#weight": data}

        with tempfile.TemporaryDirectory() as tmp:
            file_path = os.path.join(tmp, "model.onnx")
            container.save(file_path, all_tensors_to_one_file=False)
            self.assertTrue(os.path.exists(file_path))
            weight_files = [f for f in os.listdir(tmp) if f != "model.onnx"]
            self.assertGreater(len(weight_files), 0)

            loaded = ExtendedModelContainer().load(file_path)
            self.assertIsInstance(loaded.model_proto, onnx.ModelProto)
            self.assertEqual(loaded.model_proto.graph.node[0].op_type, "Add")
            self.assertGreater(len(loaded.large_initializers), 0)
            weight = next(iter(loaded.large_initializers.values()))
            self.assertEqual(weight.shape, tuple(data.shape))

    def test_exported_from_yobx_torch(self):
        from yobx.container import ExtendedModelContainer

        self.assertIsNotNone(ExtendedModelContainer)


if __name__ == "__main__":
    unittest.main(verbosity=2)

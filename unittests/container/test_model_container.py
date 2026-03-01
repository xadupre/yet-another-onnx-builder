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


def _make_external_init(name: str, data: np.ndarray, location: str) -> onnx.TensorProto:
    """Helper to create an external TensorProto referencing *location*."""
    init = onnx.TensorProto()
    init.data_type = onnx.TensorProto.FLOAT
    init.name = name
    init.data_location = onnx.TensorProto.EXTERNAL
    for d in data.shape:
        init.dims.append(d)
    ext = init.external_data.add()
    ext.key = "location"
    ext.value = location
    return init


class TestDeserializeGraph(ExtTestCase):
    """Tests for ExtendedModelContainer._deserialize_graph."""

    def _make_container(self, model: onnx.ModelProto, large_initializers=None):
        from yobx.container.model_container import ExtendedModelContainer

        c = ExtendedModelContainer()
        c.model_proto = model
        c.large_initializers = large_initializers or {}
        return c

    def test_simple_graph(self):
        """_deserialize_graph on a graph with inputs, output, and one node but no initializers."""
        import onnx_ir

        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Add", ["X", "Y"], ["Z"])],
                "g",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [2, 3]),
                    oh.make_tensor_value_info("Y", TFLOAT, [2, 3]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [2, 3])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        c = self._make_container(model)
        g = c._deserialize_graph(model.graph, [])
        self.assertIsInstance(g, onnx_ir.Graph)
        self.assertEqual(g.name, "g")
        self.assertEqual([v.name for v in g.inputs], ["X", "Y"])
        self.assertEqual([v.name for v in g.outputs], ["Z"])
        self.assertEqual(len(g.initializers), 0)
        self.assertEqual(len(list(g)), 1)

    def test_graph_with_inline_initializer(self):
        """_deserialize_graph deserialises a regular (non-external) initializer correctly."""
        import onnx_ir

        weight = onh.from_array(np.ones((2, 3), dtype=np.float32), name="W")
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Add", ["X", "W"], ["Z"])],
                "g",
                [oh.make_tensor_value_info("X", TFLOAT, [2, 3])],
                [oh.make_tensor_value_info("Z", TFLOAT, [2, 3])],
                initializer=[weight],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        c = self._make_container(model)
        g = c._deserialize_graph(model.graph, [])
        self.assertIsInstance(g, onnx_ir.Graph)
        self.assertIn("W", g.initializers)
        initializer_value = g.initializers["W"]
        self.assertIsNotNone(initializer_value.const_value)
        np.testing.assert_array_equal(
            initializer_value.const_value.numpy(), np.ones((2, 3), dtype=np.float32)
        )

    def test_graph_with_external_numpy_initializer(self):
        """_deserialize_graph loads an external initializer backed by a numpy array."""
        import onnx_ir

        data = np.ones((2, 3), dtype=np.float32)
        init = _make_external_init("W", data, "#W")
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Add", ["X", "W"], ["Z"])],
                "g",
                [oh.make_tensor_value_info("X", TFLOAT, [2, 3])],
                [oh.make_tensor_value_info("Z", TFLOAT, [2, 3])],
                initializer=[init],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        c = self._make_container(model, large_initializers={"#W": data})
        g = c._deserialize_graph(model.graph, [])
        self.assertIsInstance(g, onnx_ir.Graph)
        self.assertIn("W", g.initializers)
        np.testing.assert_array_equal(
            g.initializers["W"].const_value.numpy(), data
        )

    @requires_torch()
    def test_graph_with_external_torch_initializer(self):
        """_deserialize_graph loads an external initializer backed by a torch tensor."""
        import torch
        import onnx_ir

        data = torch.ones(2, 3, dtype=torch.float32)
        np_data = data.numpy()
        init = _make_external_init("W", np_data, "#W")
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Add", ["X", "W"], ["Z"])],
                "g",
                [oh.make_tensor_value_info("X", TFLOAT, [2, 3])],
                [oh.make_tensor_value_info("Z", TFLOAT, [2, 3])],
                initializer=[init],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        c = self._make_container(model, large_initializers={"#W": data})
        g = c._deserialize_graph(model.graph, [])
        self.assertIsInstance(g, onnx_ir.Graph)
        self.assertIn("W", g.initializers)
        np.testing.assert_array_equal(
            g.initializers["W"].const_value.numpy(), np_data
        )
class TestBuildStats(ExtTestCase):
    def test_len_empty(self):
        from yobx.container import BuildStats

        stats = BuildStats()
        self.assertEqual(len(stats), 0)

    def test_len_after_set(self):
        from yobx.container import BuildStats

        stats = BuildStats()
        stats["time_export_write_model"] = 1.0
        stats["time_export_tobytes"] = 0.5
        self.assertEqual(len(stats), 2)

    def test_getitem_default(self):
        from yobx.container import BuildStats

        stats = BuildStats()
        self.assertEqual(stats["time_export_write_model"], 0.0)

    def test_setitem_and_getitem(self):
        from yobx.container import BuildStats

        stats = BuildStats()
        stats["time_export_write_model"] += 0.5
        stats["time_export_tobytes"] += 0.1
        self.assertAlmostEqual(stats["time_export_write_model"], 0.5)
        self.assertAlmostEqual(stats["time_export_tobytes"], 0.1)

    def test_to_dict(self):
        from yobx.container import BuildStats

        stats = BuildStats()
        stats["time_export_write_model"] = 1.0
        d = stats.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("time_export_write_model", d)
        self.assertAlmostEqual(d["time_export_write_model"], 1.0)

    def test_to_dict_is_not_a_copy(self):
        from yobx.container import BuildStats

        stats = BuildStats()
        d = stats.to_dict()
        d["time_export_write_model"] = 99.0
        self.assertAlmostEqual(stats.to_dict()["time_export_write_model"], 99.0)

    def test_validate_valid_keys(self):
        from yobx.container import BuildStats

        stats = BuildStats()
        stats["time_export_write_model"] = 0.5
        stats["time_export_tobytes"] = 0.1
        # Should not raise.
        stats.validate()

    def test_validate_invalid_key_raises(self):
        from yobx.container import BuildStats

        stats = BuildStats()
        # __setitem__ does not enforce the prefix; validate() does.
        stats["unknown_key"] = 1.0
        with self.assertRaises(KeyError):
            stats.validate()

    def test_repr_empty(self):
        from yobx.container import BuildStats

        stats = BuildStats()
        r = repr(stats)
        self.assertIn("BuildStats", r)
        self.assertIn("{}", r)

    def test_repr_with_data(self):
        from yobx.container import BuildStats

        stats = BuildStats()
        stats["time_export_write_model"] = 0.5
        r = repr(stats)
        self.assertIn("BuildStats", r)
        self.assertIn("time_export_write_model", r)


if __name__ == "__main__":
    unittest.main(verbosity=2)

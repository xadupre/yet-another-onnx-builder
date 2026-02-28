import unittest
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from yobx.ext_test_case import ExtTestCase, requires_torch
from yobx.helpers.rt_helper import make_feeds


def _make_simple_model(input_names):
    """Create a minimal ONNX model with the given input names (all float32 scalars)."""
    inputs = [
        onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT, [1])
        for name in input_names
    ]
    outputs = [
        onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    ]
    node = onnx.helper.make_node("Identity", inputs=[input_names[0]], outputs=["output"])
    graph = onnx.helper.make_graph([node], "test_graph", inputs, outputs)
    model = onnx.helper.make_model(graph)
    model.ir_version = 8
    return model


class TestMakeFeeds(ExtTestCase):
    def test_make_feeds_list_names(self):
        """Proto is a plain list of input names."""
        names = ["x", "y"]
        inputs = [np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32)]
        feeds = make_feeds(names, inputs)
        self.assertIsInstance(feeds, dict)
        self.assertEqual(list(feeds.keys()), names)
        np.testing.assert_array_equal(feeds["x"], inputs[0])
        np.testing.assert_array_equal(feeds["y"], inputs[1])

    def test_make_feeds_onnx_model_proto(self):
        """Proto is an onnx.ModelProto."""
        model = _make_simple_model(["a", "b"])
        inputs = [np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32)]
        feeds = make_feeds(model, inputs)
        self.assertIsInstance(feeds, dict)
        self.assertIn("a", feeds)
        self.assertIn("b", feeds)

    def test_make_feeds_get_inputs(self):
        """Proto is an object with a get_inputs() method (e.g. InferenceSession)."""

        class FakeSession:
            class FakeInput:
                def __init__(self, name):
                    self.name = name

            def get_inputs(self):
                return [self.FakeInput("p"), self.FakeInput("q")]

        session = FakeSession()
        inputs = [np.array([3.0], dtype=np.float32), np.array([4.0], dtype=np.float32)]
        feeds = make_feeds(session, inputs)
        self.assertEqual(list(feeds.keys()), ["p", "q"])

    def test_make_feeds_input_names(self):
        """Proto is an object with an input_names attribute."""

        class FakeProto:
            input_names = ["u", "v"]

        proto = FakeProto()
        inputs = [np.array([5.0], dtype=np.float32), np.array([6.0], dtype=np.float32)]
        feeds = make_feeds(proto, inputs)
        self.assertEqual(list(feeds.keys()), ["u", "v"])

    def test_make_feeds_bool_conversion(self):
        """Python bool values are converted to np.bool_ arrays."""
        names = ["flag"]
        feeds = make_feeds(names, [True])
        self.assertIsInstance(feeds["flag"], np.ndarray)
        self.assertEqual(feeds["flag"].dtype, np.bool_)
        self.assertEqual(feeds["flag"].item(), True)

    def test_make_feeds_int_conversion(self):
        """Python int values are converted to np.int64 arrays."""
        names = ["idx"]
        feeds = make_feeds(names, [42])
        self.assertIsInstance(feeds["idx"], np.ndarray)
        self.assertEqual(feeds["idx"].dtype, np.int64)
        self.assertEqual(feeds["idx"].item(), 42)

    def test_make_feeds_float_conversion(self):
        """Python float values are converted to np.float32 arrays."""
        names = ["scale"]
        feeds = make_feeds(names, [3.14])
        self.assertIsInstance(feeds["scale"], np.ndarray)
        self.assertEqual(feeds["scale"].dtype, np.float32)

    def test_make_feeds_copy_numpy(self):
        """copy=True produces independent copies of numpy arrays."""
        names = ["x"]
        arr = np.array([1.0, 2.0], dtype=np.float32)
        feeds = make_feeds(names, [arr], copy=True)
        self.assertIsNot(feeds["x"], arr)
        np.testing.assert_array_equal(feeds["x"], arr)

    def test_make_feeds_assertion_too_few_names(self):
        """Fewer names than inputs raises AssertionError when using a plain list."""
        names = ["x"]
        inputs = [
            np.array([1.0], dtype=np.float32),
            np.array([2.0], dtype=np.float32),
        ]
        with self.assertRaises(AssertionError):
            make_feeds(names, inputs)

    @requires_torch()
    def test_make_feeds_use_numpy(self):
        """use_numpy=True converts torch tensors to numpy arrays."""
        import torch

        names = ["t"]
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
        feeds = make_feeds(names, [tensor], use_numpy=True)
        self.assertIsInstance(feeds["t"], np.ndarray)
        np.testing.assert_array_equal(feeds["t"], np.array([1.0, 2.0], dtype=np.float32))

    @requires_torch()
    def test_make_feeds_copy_torch(self):
        """copy=True calls .clone() on torch tensors."""
        import torch

        names = ["t"]
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
        feeds = make_feeds(names, [tensor], copy=True)
        self.assertIsNot(feeds["t"], tensor)
        np.testing.assert_array_equal(
            feeds["t"].numpy(), tensor.numpy()
        )

    @requires_torch()
    def test_make_feeds_is_modelbuilder_removes_position_ids(self):
        """is_modelbuilder=True removes position_ids from the inputs dict."""
        import torch

        names = ["input_ids"]
        inputs = {
            "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
            "position_ids": torch.tensor([[0, 1, 2]]),
        }
        feeds = make_feeds(names, inputs, is_modelbuilder=True)
        self.assertIn("input_ids", feeds)
        self.assertNotIn("position_ids", feeds)

    @requires_torch()
    def test_make_feeds_is_modelbuilder_invalid_position_ids(self):
        """is_modelbuilder=True with non-contiguous position_ids raises AssertionError."""
        import torch

        names = ["input_ids"]
        inputs = {
            "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
            "position_ids": torch.tensor([[5, 7, 9]]),
        }
        with self.assertRaises(AssertionError):
            make_feeds(names, inputs, is_modelbuilder=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

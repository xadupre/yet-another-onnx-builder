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
        onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT, [1]) for name in input_names
    ]
    outputs = [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])]
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
        np.testing.assert_array_equal(feeds["t"].numpy(), tensor.numpy())

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
class TestOnnxGenerate(ExtTestCase):
    """Tests for :func:`yobx.helpers.rt_helper.onnx_generate`."""

    VOCAB = 8
    BATCH = 1

    @classmethod
    def _make_fixed_logits_model(cls, winner_token: int) -> onnx.ModelProto:
        """
        Returns an ONNX model (no KV cache) whose logits always give the
        highest score to *winner_token*, regardless of the input sequence.
        """
        fixed_logits = np.zeros((cls.BATCH, 1, cls.VOCAB), dtype=np.float32)
        fixed_logits[0, 0, winner_token] = 10.0
        return onnx.helper.make_model(
            onnx.helper.make_graph(
                [
                    onnx.helper.make_node(
                        "Constant",
                        [],
                        ["logits"],
                        value=onnx.numpy_helper.from_array(fixed_logits),
                    )
                ],
                "tiny_lm_no_kv",
                [onnx.helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, None])],
                [
                    onnx.helper.make_tensor_value_info(
                        "logits", TensorProto.FLOAT, [1, 1, cls.VOCAB]
                    )
                ],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 18)],
            ir_version=9,
        )

    @classmethod
    def _make_kv_model(cls, winner_token: int) -> onnx.ModelProto:
        """
        Returns an ONNX model with 2-layer KV cache.  Logits always favour
        *winner_token*.  The present KV values are just the past KV values
        concatenated with a zero slice (so shapes update correctly).
        """
        HEADS = 2
        HEAD_DIM = 4
        LAYERS = 2
        fixed_logits = np.zeros((cls.BATCH, 1, cls.VOCAB), dtype=np.float32)
        fixed_logits[0, 0, winner_token] = 10.0
        new_kv = np.zeros((cls.BATCH, HEADS, 1, HEAD_DIM), dtype=np.float32)

        inputs = [
            onnx.helper.make_tensor_value_info("input_ids", TensorProto.INT64, [cls.BATCH, None]),
            onnx.helper.make_tensor_value_info(
                "attention_mask", TensorProto.INT64, [cls.BATCH, None]
            ),
        ]
        for i in range(LAYERS):
            inputs.append(
                onnx.helper.make_tensor_value_info(
                    f"past_key_values.{i}.key",
                    TensorProto.FLOAT,
                    [cls.BATCH, HEADS, None, HEAD_DIM],
                )
            )
            inputs.append(
                onnx.helper.make_tensor_value_info(
                    f"past_key_values.{i}.value",
                    TensorProto.FLOAT,
                    [cls.BATCH, HEADS, None, HEAD_DIM],
                )
            )

        outputs = [
            onnx.helper.make_tensor_value_info(
                "logits", TensorProto.FLOAT, [cls.BATCH, 1, cls.VOCAB]
            )
        ]
        for i in range(LAYERS):
            outputs.append(
                onnx.helper.make_tensor_value_info(
                    f"present_key_values.{i}.key",
                    TensorProto.FLOAT,
                    [cls.BATCH, HEADS, None, HEAD_DIM],
                )
            )
            outputs.append(
                onnx.helper.make_tensor_value_info(
                    f"present_key_values.{i}.value",
                    TensorProto.FLOAT,
                    [cls.BATCH, HEADS, None, HEAD_DIM],
                )
            )

        nodes = [
            onnx.helper.make_node(
                "Constant",
                [],
                ["logits"],
                value=onnx.numpy_helper.from_array(fixed_logits),
            )
        ]
        for i in range(LAYERS):
            nodes.append(
                onnx.helper.make_node(
                    "Concat",
                    [f"past_key_values.{i}.key", "new_kv"],
                    [f"present_key_values.{i}.key"],
                    axis=2,
                )
            )
            nodes.append(
                onnx.helper.make_node(
                    "Concat",
                    [f"past_key_values.{i}.value", "new_kv"],
                    [f"present_key_values.{i}.value"],
                    axis=2,
                )
            )

        inits = [onnx.numpy_helper.from_array(new_kv, name="new_kv")]
        return onnx.helper.make_model(
            onnx.helper.make_graph(nodes, "tiny_llm_with_kv", inputs, outputs, inits),
            opset_imports=[onnx.helper.make_opsetid("", 18)],
            ir_version=9,
        )

    def test_greedy_no_kv_stops_at_eos(self):
        """Without KV cache: generation stops when EOS token is produced."""
        from yobx.helpers.rt_helper import onnx_generate

        model = self._make_fixed_logits_model(winner_token=3)
        prompt = np.array([[1, 2]], dtype=np.int64)
        tokens = onnx_generate(model, prompt, max_new_tokens=10, eos_token_id=3)
        # Prompt + exactly one new token (which is EOS=3).
        self.assertEqual(tokens.shape, (1, 3))
        self.assertEqual(int(tokens[0, 2]), 3)

    def test_greedy_no_kv_max_tokens(self):
        """Without KV cache: exactly max_new_tokens tokens appended when no EOS."""
        from yobx.helpers.rt_helper import onnx_generate

        model = self._make_fixed_logits_model(winner_token=3)
        prompt = np.array([[1, 2]], dtype=np.int64)
        # Use a winner token that is NOT the EOS token.
        tokens = onnx_generate(model, prompt, max_new_tokens=4, eos_token_id=99)
        self.assertEqual(tokens.shape, (1, 6))
        self.assertTrue(np.all(tokens[0, 2:] == 3))

    def test_greedy_with_kv_stops_at_eos(self):
        """With KV cache: generation stops when EOS token is produced."""
        from yobx.helpers.rt_helper import onnx_generate

        model = self._make_kv_model(winner_token=5)
        prompt = np.array([[1, 2]], dtype=np.int64)
        attn = np.ones_like(prompt)
        tokens = onnx_generate(
            model, prompt, attention_mask=attn, max_new_tokens=10, eos_token_id=5
        )
        self.assertEqual(tokens.shape, (1, 3))
        self.assertEqual(int(tokens[0, 2]), 5)

    def test_greedy_with_kv_max_tokens(self):
        """With KV cache: max_new_tokens are generated when EOS is not reached."""
        from yobx.helpers.rt_helper import onnx_generate

        model = self._make_kv_model(winner_token=5)
        prompt = np.array([[1, 2]], dtype=np.int64)
        attn = np.ones_like(prompt)
        tokens = onnx_generate(
            model, prompt, attention_mask=attn, max_new_tokens=3, eos_token_id=99
        )
        self.assertEqual(tokens.shape, (1, 5))
        self.assertTrue(np.all(tokens[0, 2:] == 5))

    @requires_torch()
    def test_torch_input_ids(self):
        """Torch tensors are accepted as input_ids/attention_mask."""
        import torch
        from yobx.helpers.rt_helper import onnx_generate

        model = self._make_fixed_logits_model(winner_token=2)
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        attn = torch.ones_like(prompt)
        tokens = onnx_generate(
            model, prompt, attention_mask=attn, max_new_tokens=2, eos_token_id=99
        )
        self.assertIsInstance(tokens, np.ndarray)
        self.assertEqual(tokens.shape, (1, 5))


if __name__ == "__main__":
    unittest.main(verbosity=2)

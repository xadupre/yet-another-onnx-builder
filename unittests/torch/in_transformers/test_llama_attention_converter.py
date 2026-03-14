import unittest
import numpy as np
import torch
from yobx.ext_test_case import (
    ExtTestCase,
    requires_onnxruntime,
    requires_transformers,
)
from yobx.reference import ExtendedReferenceEvaluator


def _make_llama_attention(
    hidden_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
):
    """Creates a tiny LlamaAttention instance."""
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaAttention

    config = LlamaConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
    )
    return LlamaAttention(config, layer_idx=0)


@requires_transformers("4.40")
class TestLlamaAttentionConverter(ExtTestCase):
    """Unit tests for :func:`llama_attention_to_onnx`."""

    def _get_inputs(self, batch=2, seq=10, hidden_size=64, head_dim=16, torch_dtype=torch.float32):
        hs = torch.randn(batch, seq, hidden_size, dtype=torch_dtype)
        cos = torch.randn(batch, seq, head_dim, dtype=torch_dtype)
        sin = torch.randn(batch, seq, head_dim, dtype=torch_dtype)
        return hs, cos, sin

    def _torch_expected(self, attn, hs, cos, sin):
        with torch.no_grad():
            out, _ = attn(hs, position_embeddings=(cos, sin))
        return out.float().numpy()

    # ------------------------------------------------------------------ #
    # float32 tests                                                        #
    # ------------------------------------------------------------------ #

    def test_opset22_float32(self):
        """Standard ONNX ops path, float32 — ref evaluator and OnnxRuntime."""
        from yobx.torch.in_transformers.models import llama_attention_to_onnx

        attn = _make_llama_attention().eval()
        hs, cos, sin = self._get_inputs()
        expected = self._torch_expected(attn, hs, cos, sin)

        model = llama_attention_to_onnx(attn, (hs, cos, sin), target_opset=22)

        feeds = {
            "hidden_states": hs.numpy(),
            "cos": cos.numpy(),
            "sin": sin.numpy(),
        }

        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-4)

        import onnxruntime as ort

        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got_ort = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got_ort, atol=1e-4)

    # ------------------------------------------------------------------ #
    # float16 tests                                                        #
    # ------------------------------------------------------------------ #

    def test_opset22_float16(self):
        """Standard ONNX ops path, float16 — ref evaluator and OnnxRuntime."""
        from yobx.torch.in_transformers.models import llama_attention_to_onnx

        attn = _make_llama_attention().to(torch.float16).eval()
        hs, cos, sin = self._get_inputs(torch_dtype=torch.float16)
        expected = self._torch_expected(attn, hs, cos, sin)

        model = llama_attention_to_onnx(attn, (hs, cos, sin), target_opset=22)

        feeds = {
            "hidden_states": hs.numpy(),
            "cos": cos.numpy(),
            "sin": sin.numpy(),
        }
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, feeds)[0].astype(np.float32)
        self.assertEqualArray(expected, got, atol=5e-3)

        import onnxruntime as ort

        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got_ort = sess.run(None, feeds)[0].astype(np.float32)
        self.assertEqualArray(expected, got_ort, atol=5e-3)

    # ------------------------------------------------------------------ #
    # opset 24 (ONNX Attention op)                                         #
    # ------------------------------------------------------------------ #

    @requires_onnxruntime("1.23")
    def test_opset24_float32(self):
        """ONNX Attention op path (opset 24), float32."""
        from yobx.torch.in_transformers.models import llama_attention_to_onnx

        attn = _make_llama_attention().eval()
        hs, cos, sin = self._get_inputs()
        expected = self._torch_expected(attn, hs, cos, sin)

        model = llama_attention_to_onnx(attn, (hs, cos, sin), target_opset=24)

        feeds = {
            "hidden_states": hs.numpy(),
            "cos": cos.numpy(),
            "sin": sin.numpy(),
        }
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-4)

        import onnxruntime as ort

        try:
            sess = ort.InferenceSession(
                model.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            if "opset" in str(e).lower():
                raise unittest.SkipTest(f"onnxruntime too old for opset 24: {e}")
            raise
        got_ort = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got_ort, atol=1e-3)

    @requires_onnxruntime("1.23")
    def test_opset24_float16(self):
        """ONNX Attention op path (opset 24), float16."""
        from yobx.torch.in_transformers.models import llama_attention_to_onnx

        attn = _make_llama_attention().to(torch.float16).eval()
        hs, cos, sin = self._get_inputs(torch_dtype=torch.float16)
        expected = self._torch_expected(attn, hs, cos, sin)

        model = llama_attention_to_onnx(attn, (hs, cos, sin), target_opset=24)

        feeds = {
            "hidden_states": hs.numpy(),
            "cos": cos.numpy(),
            "sin": sin.numpy(),
        }
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, feeds)[0].astype(np.float32)
        self.assertEqualArray(expected, got, atol=5e-3)

        import onnxruntime as ort

        try:
            sess = ort.InferenceSession(
                model.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            if "opset" in str(e).lower():
                raise unittest.SkipTest(f"onnxruntime too old for opset 24: {e}")
            raise
        got_ort = sess.run(None, feeds)[0].astype(np.float32)
        self.assertEqualArray(expected, got_ort, atol=5e-3)

    # ------------------------------------------------------------------ #
    # com.microsoft MultiHeadAttention contrib ops                         #
    # ------------------------------------------------------------------ #

    @requires_onnxruntime("1.0")
    def test_com_microsoft_float32(self):
        """com.microsoft.MultiHeadAttention path, float32."""
        from yobx.torch.in_transformers.models import llama_attention_to_onnx

        attn = _make_llama_attention().eval()
        hs, cos, sin = self._get_inputs()
        expected = self._torch_expected(attn, hs, cos, sin)

        model = llama_attention_to_onnx(
            attn,
            (hs, cos, sin),
            target_opset={"": 22, "com.microsoft": 1},
        )

        # Verify that the model contains the MultiHeadAttention contrib op
        op_types = {n.op_type for n in model.graph.node}
        self.assertIn("MultiHeadAttention", op_types)

        feeds = {
            "hidden_states": hs.numpy(),
            "cos": cos.numpy(),
            "sin": sin.numpy(),
        }
        import onnxruntime as ort

        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-4)

    @requires_onnxruntime("1.0")
    def test_com_microsoft_float16(self):
        """com.microsoft.MultiHeadAttention path, float16."""
        from yobx.torch.in_transformers.models import llama_attention_to_onnx

        attn = _make_llama_attention().to(torch.float16).eval()
        hs, cos, sin = self._get_inputs(torch_dtype=torch.float16)
        expected = self._torch_expected(attn, hs, cos, sin)

        model = llama_attention_to_onnx(
            attn,
            (hs, cos, sin),
            target_opset={"": 22, "com.microsoft": 1},
        )

        op_types = {n.op_type for n in model.graph.node}
        self.assertIn("MultiHeadAttention", op_types)

        feeds = {
            "hidden_states": hs.numpy(),
            "cos": cos.numpy(),
            "sin": sin.numpy(),
        }
        import onnxruntime as ort

        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0].astype(np.float32)
        self.assertEqualArray(expected, got, atol=5e-3)

    # ------------------------------------------------------------------ #
    # MHA with equal heads (no GQA repeat)                                 #
    # ------------------------------------------------------------------ #

    @requires_onnxruntime("1.0")
    def test_com_microsoft_no_gqa_float32(self):
        """com.microsoft path when num_kv_heads == num_attention_heads."""
        from yobx.torch.in_transformers.models import llama_attention_to_onnx

        # No GQA: kv_heads == query heads
        attn = _make_llama_attention(num_attention_heads=4, num_key_value_heads=4).eval()
        hs, cos, sin = self._get_inputs()
        expected = self._torch_expected(attn, hs, cos, sin)

        model = llama_attention_to_onnx(
            attn,
            (hs, cos, sin),
            target_opset={"": 22, "com.microsoft": 1},
        )

        feeds = {
            "hidden_states": hs.numpy(),
            "cos": cos.numpy(),
            "sin": sin.numpy(),
        }
        import onnxruntime as ort

        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-4)

    # ------------------------------------------------------------------ #
    # Attention mask                                                        #
    # ------------------------------------------------------------------ #

    def test_opset22_with_attention_mask(self):
        """Standard ONNX ops path with an attention_mask input — ref evaluator and OnnxRuntime."""
        from yobx.torch.in_transformers.models import llama_attention_to_onnx

        attn = _make_llama_attention().eval()
        batch, seq = 2, 10
        hs, cos, sin = self._get_inputs(batch=batch, seq=seq)
        mask = torch.zeros(batch, 1, seq, seq, dtype=torch.float32)
        # Add a causal mask
        mask_tri = torch.full((seq, seq), float("-inf"))
        mask_tri = torch.triu(mask_tri, diagonal=1)
        mask[0, 0] = mask_tri
        mask[1, 0] = mask_tri

        with torch.no_grad():
            expected, _ = attn(hs, position_embeddings=(cos, sin), attention_mask=mask)
        expected = expected.float().numpy()

        model = llama_attention_to_onnx(attn, (hs, cos, sin, mask), target_opset=22)

        feeds = {
            "hidden_states": hs.numpy(),
            "cos": cos.numpy(),
            "sin": sin.numpy(),
            "attention_mask": mask.numpy(),
        }
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-4)

        import onnxruntime as ort

        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got_ort = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got_ort, atol=1e-4)

    # ------------------------------------------------------------------ #
    # Model structure tests                                                 #
    # ------------------------------------------------------------------ #

    def test_opset22_returns_model_proto(self):
        """Converter returns a valid ModelProto."""
        import onnx

        from yobx.torch.in_transformers.models import llama_attention_to_onnx

        attn = _make_llama_attention().eval()
        hs, cos, sin = self._get_inputs()
        model = llama_attention_to_onnx(attn, (hs, cos, sin), target_opset=22)

        self.assertIsInstance(model, onnx.ModelProto)
        # Inputs
        input_names = [i.name for i in model.graph.input]
        self.assertIn("hidden_states", input_names)
        self.assertIn("cos", input_names)
        self.assertIn("sin", input_names)
        self.assertEqual(len(model.graph.output), 1)

    def test_opset22_no_attention_op(self):
        """Opset 22 path does not use the Attention op."""
        from yobx.torch.in_transformers.models import llama_attention_to_onnx

        attn = _make_llama_attention().eval()
        hs, cos, sin = self._get_inputs()
        model = llama_attention_to_onnx(attn, (hs, cos, sin), target_opset=22)
        op_types = {n.op_type for n in model.graph.node}
        self.assertNotIn("Attention", op_types)

    def test_opset24_uses_attention_op(self):
        """Opset 24 path uses the ONNX Attention op."""
        from yobx.torch.in_transformers.models import llama_attention_to_onnx

        attn = _make_llama_attention().eval()
        hs, cos, sin = self._get_inputs()
        model = llama_attention_to_onnx(attn, (hs, cos, sin), target_opset=24)
        op_types = {n.op_type for n in model.graph.node}
        self.assertIn("Attention", op_types)


if __name__ == "__main__":
    unittest.main(verbosity=2)

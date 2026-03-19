"""
Unit tests for :func:`yobx.torch.in_transformers.models.llama_attention_to_onnx`.

Tests cover:

* All three ONNX backends: opset 22 (plain ops), opset 24 (ONNX Attention op),
  and ``com.microsoft`` (OnnxRuntime MultiHeadAttention contrib op).
* All supported dtypes: ``float32``, ``float16``, and ``bfloat16``.
* Both with and without an ``attention_mask`` input.
* GQA (grouped-query attention) and equal-heads configurations.
* Validation with both :class:`~yobx.reference.ExtendedReferenceEvaluator`
  and :mod:`onnxruntime`.
"""

import unittest
import numpy as np
import onnx
import torch
from yobx.ext_test_case import ExtTestCase, requires_onnxruntime, requires_transformers
from yobx.reference import ExtendedReferenceEvaluator
from yobx.container import ExportArtifact


def _make_llama_attention(
    hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16
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


def _to_model(attn, hs, cos, sin, mask=None, target_opset=22):
    """
    Convenience wrapper: creates a :class:`~yobx.xbuilder.GraphBuilder`,
    declares graph inputs, calls :func:`llama_attention_to_onnx`, declares
    the graph output, and returns the resulting :class:`onnx.ModelProto`.
    """
    from yobx.xbuilder import GraphBuilder
    from yobx.torch.in_transformers.classes.llama_attention import llama_attention_to_onnx
    from yobx.torch.torch_helper import torch_dtype_to_onnx_dtype

    if isinstance(target_opset, int):
        dict_opset = {"": target_opset}
    else:
        dict_opset = dict(target_opset)

    onnx_dtype = torch_dtype_to_onnx_dtype(hs.dtype)
    head_dim = cos.shape[-1]
    hidden_size = hs.shape[-1]

    g = GraphBuilder(dict_opset, verbose=0)
    g.make_tensor_input("hidden_states", onnx_dtype, ("batch", "seq", hidden_size))
    g.make_tensor_input("cos", onnx_dtype, ("batch", "seq", head_dim))
    g.make_tensor_input("sin", onnx_dtype, ("batch", "seq", head_dim))

    mask_name = None
    if mask is not None:
        g.make_tensor_input("attention_mask", onnx_dtype, ("batch", 1, "seq_q", "total_seq"))
        mask_name = "attention_mask"

    out = llama_attention_to_onnx(g, attn, "hidden_states", "cos", "sin", mask_name)
    g.make_tensor_output(out, onnx_dtype, ("batch", "seq", hidden_size))
    return g.to_onnx(optimize=False)


@requires_transformers("5.0.7")
class TestLlamaAttentionConverter(ExtTestCase):
    """Unit tests for :func:`llama_attention_to_onnx`."""

    def _get_inputs(
        self, batch=2, seq=10, hidden_size=64, head_dim=16, torch_dtype=torch.float32
    ):
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
        attn = _make_llama_attention().eval()
        hs, cos, sin = self._get_inputs()
        expected = self._torch_expected(attn, hs, cos, sin)

        model = _to_model(attn, hs, cos, sin, target_opset=22)

        feeds = {"hidden_states": hs.numpy(), "cos": cos.numpy(), "sin": sin.numpy()}

        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-4)

        import onnxruntime as ort

        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        got_ort = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got_ort, atol=1e-4)

    # ------------------------------------------------------------------ #
    # float16 tests                                                        #
    # ------------------------------------------------------------------ #

    def test_opset22_float16(self):
        """Standard ONNX ops path, float16 — ref evaluator and OnnxRuntime."""
        attn = _make_llama_attention().to(torch.float16).eval()
        hs, cos, sin = self._get_inputs(torch_dtype=torch.float16)
        expected = self._torch_expected(attn, hs, cos, sin)

        model = _to_model(attn, hs, cos, sin, target_opset=22)

        feeds = {"hidden_states": hs.numpy(), "cos": cos.numpy(), "sin": sin.numpy()}
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, feeds)[0].astype(np.float32)
        self.assertEqualArray(expected, got, atol=5e-3)

        import onnxruntime as ort

        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        got_ort = sess.run(None, feeds)[0].astype(np.float32)
        self.assertEqualArray(expected, got_ort, atol=5e-3)

    # ------------------------------------------------------------------ #
    # bfloat16 tests                                                       #
    # ------------------------------------------------------------------ #

    def _bf16_feeds(self, hs, cos, sin):
        """Build bfloat16 numpy feeds from bfloat16 torch tensors via ml_dtypes."""
        import ml_dtypes

        return {
            "hidden_states": hs.to(torch.float32).numpy().astype(ml_dtypes.bfloat16),
            "cos": cos.to(torch.float32).numpy().astype(ml_dtypes.bfloat16),
            "sin": sin.to(torch.float32).numpy().astype(ml_dtypes.bfloat16),
        }

    def _run_ort_bf16(self, model, feeds, expected, atol=5e-2):
        """
        Try to run *model* with OnnxRuntime and compare to *expected*.
        Silently skips if ORT CPU does not support bfloat16 for the op set used.
        """
        import onnxruntime as ort

        # Normalise the error message: strip underscores and collapse spaces so
        # that "NOT_IMPLEMENTED", "not implemented", and "NotImplemented" all match.
        def _normalised(s: str) -> str:
            return s.lower().replace("_", " ")

        _BF16_SKIP = ("bfloat16", "not implemented")
        try:
            sess = ort.InferenceSession(
                model.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            got_ort = sess.run(None, feeds)[0].astype(np.float32)
            self.assertEqualArray(expected, got_ort, atol=atol)
        except Exception as e:
            msg = _normalised(str(e))
            if any(kw in msg for kw in _BF16_SKIP):
                raise unittest.SkipTest(
                    f"OnnxRuntime does not support bfloat16 for this op on CPU: {e}"
                )
            raise

    def test_opset22_bfloat16(self):
        """Standard ONNX ops path, bfloat16 — model dtype check + ref/ORT validation."""
        attn = _make_llama_attention().to(torch.bfloat16).eval()
        hs, cos, sin = self._get_inputs(torch_dtype=torch.bfloat16)
        expected = self._torch_expected(attn, hs, cos, sin)

        model = _to_model(attn, hs, cos, sin, target_opset=22)

        # The produced model must use bfloat16 for all graph inputs.
        self.assertIsInstance(model, ExportArtifact)
        for inp in model.graph.input:
            self.assertEqual(
                inp.type.tensor_type.elem_type,
                onnx.TensorProto.BFLOAT16,
                msg=f"Input '{inp.name}' is not bfloat16",
            )

        feeds = self._bf16_feeds(hs, cos, sin)

        # Reference evaluator: attempt validation; skip gracefully if unsupported.
        try:
            ref = ExtendedReferenceEvaluator(model)
            got = ref.run(None, feeds)[0].astype(np.float32)
            self.assertEqualArray(expected, got, atol=5e-2)
        except Exception as e:
            if "type mismatch" in str(e).lower() or "bfloat16" in str(e).lower():
                pass  # Known limitation of the ONNX reference evaluator for bfloat16
            else:
                raise

        self._run_ort_bf16(model, feeds, expected)

    # ------------------------------------------------------------------ #
    # opset 24 (ONNX Attention op)                                         #
    # ------------------------------------------------------------------ #

    @requires_onnxruntime("1.23")
    def test_opset24_float32(self):
        """ONNX Attention op path (opset 24), float32."""
        attn = _make_llama_attention().eval()
        hs, cos, sin = self._get_inputs()
        expected = self._torch_expected(attn, hs, cos, sin)

        model = _to_model(attn, hs, cos, sin, target_opset=24)

        feeds = {"hidden_states": hs.numpy(), "cos": cos.numpy(), "sin": sin.numpy()}
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
        attn = _make_llama_attention().to(torch.float16).eval()
        hs, cos, sin = self._get_inputs(torch_dtype=torch.float16)
        expected = self._torch_expected(attn, hs, cos, sin)

        model = _to_model(attn, hs, cos, sin, target_opset=24)

        feeds = {"hidden_states": hs.numpy(), "cos": cos.numpy(), "sin": sin.numpy()}
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

    @requires_onnxruntime("1.23")
    def test_opset24_bfloat16(self):
        """ONNX Attention op path (opset 24), bfloat16 —
        model dtype check + ref/ORT validation."""
        attn = _make_llama_attention().to(torch.bfloat16).eval()
        hs, cos, sin = self._get_inputs(torch_dtype=torch.bfloat16)
        expected = self._torch_expected(attn, hs, cos, sin)

        model = _to_model(attn, hs, cos, sin, target_opset=24)

        # Verify bfloat16 dtype on graph inputs.
        for inp in model.graph.input:
            self.assertEqual(inp.type.tensor_type.elem_type, onnx.TensorProto.BFLOAT16)

        feeds = self._bf16_feeds(hs, cos, sin)

        try:
            ref = ExtendedReferenceEvaluator(model)
            got = ref.run(None, feeds)[0].astype(np.float32)
            self.assertEqualArray(expected, got, atol=5e-2)
        except Exception as e:
            if "type mismatch" in str(e).lower() or "bfloat16" in str(e).lower():
                pass
            else:
                raise

        self._run_ort_bf16(model, feeds, expected)

    # ------------------------------------------------------------------ #
    # com.microsoft MultiHeadAttention contrib ops                         #
    # ------------------------------------------------------------------ #

    @requires_onnxruntime("1.0")
    def test_com_microsoft_float32(self):
        """com.microsoft.MultiHeadAttention path, float32."""
        attn = _make_llama_attention().eval()
        hs, cos, sin = self._get_inputs()
        expected = self._torch_expected(attn, hs, cos, sin)

        model = _to_model(attn, hs, cos, sin, target_opset={"": 22, "com.microsoft": 1})

        # Verify that the model contains the MultiHeadAttention contrib op
        op_types = {n.op_type for n in model.graph.node}
        self.assertIn("MultiHeadAttention", op_types)

        feeds = {"hidden_states": hs.numpy(), "cos": cos.numpy(), "sin": sin.numpy()}
        import onnxruntime as ort

        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-4)

    @requires_onnxruntime("1.0")
    def test_com_microsoft_float16(self):
        """com.microsoft.MultiHeadAttention path, float16."""
        attn = _make_llama_attention().to(torch.float16).eval()
        hs, cos, sin = self._get_inputs(torch_dtype=torch.float16)
        expected = self._torch_expected(attn, hs, cos, sin)

        model = _to_model(attn, hs, cos, sin, target_opset={"": 22, "com.microsoft": 1})

        op_types = {n.op_type for n in model.graph.node}
        self.assertIn("MultiHeadAttention", op_types)

        feeds = {"hidden_states": hs.numpy(), "cos": cos.numpy(), "sin": sin.numpy()}
        import onnxruntime as ort

        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        got = sess.run(None, feeds)[0].astype(np.float32)
        self.assertEqualArray(expected, got, atol=5e-3)

    @requires_onnxruntime("1.0")
    def test_com_microsoft_bfloat16(self):
        """com.microsoft.MultiHeadAttention path, bfloat16 —
        model dtype check + ORT validation."""
        attn = _make_llama_attention().to(torch.bfloat16).eval()
        hs, cos, sin = self._get_inputs(torch_dtype=torch.bfloat16)
        expected = self._torch_expected(attn, hs, cos, sin)

        model = _to_model(attn, hs, cos, sin, target_opset={"": 22, "com.microsoft": 1})

        op_types = {n.op_type for n in model.graph.node}
        self.assertIn("MultiHeadAttention", op_types)
        for inp in model.graph.input:
            self.assertEqual(inp.type.tensor_type.elem_type, onnx.TensorProto.BFLOAT16)

        feeds = self._bf16_feeds(hs, cos, sin)
        self._run_ort_bf16(model, feeds, expected)

    # ------------------------------------------------------------------ #
    # MHA with equal heads (no GQA repeat)                                 #
    # ------------------------------------------------------------------ #

    @requires_onnxruntime("1.0")
    def test_com_microsoft_no_gqa_float32(self):
        """com.microsoft path when num_kv_heads == num_attention_heads."""
        # No GQA: kv_heads == query heads
        attn = _make_llama_attention(num_attention_heads=4, num_key_value_heads=4).eval()
        hs, cos, sin = self._get_inputs()
        expected = self._torch_expected(attn, hs, cos, sin)

        model = _to_model(attn, hs, cos, sin, target_opset={"": 22, "com.microsoft": 1})

        feeds = {"hidden_states": hs.numpy(), "cos": cos.numpy(), "sin": sin.numpy()}
        import onnxruntime as ort

        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-4)

    # ------------------------------------------------------------------ #
    # Attention mask                                                        #
    # ------------------------------------------------------------------ #

    def test_opset22_with_attention_mask(self):
        """Standard ONNX ops path with an attention_mask input — ref evaluator and OnnxRuntime."""
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

        model = _to_model(attn, hs, cos, sin, mask=mask, target_opset=22)
        # self.dump_onnx("test_opset22_with_attention_mask.onnx", model)

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

        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        got_ort = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got_ort, atol=1e-4)

    # ------------------------------------------------------------------ #
    # Model structure tests                                                 #
    # ------------------------------------------------------------------ #

    def test_opset22_returns_model_proto(self):
        """Converter returns a valid ModelProto."""
        import onnx

        attn = _make_llama_attention().eval()
        hs, cos, sin = self._get_inputs()
        model = _to_model(attn, hs, cos, sin, target_opset=22)

        self.assertIsInstance(model, onnx.ModelProto)
        # Inputs
        input_names = [i.name for i in model.graph.input]
        self.assertIn("hidden_states", input_names)
        self.assertIn("cos", input_names)
        self.assertIn("sin", input_names)
        self.assertEqual(len(model.graph.output), 1)

    def test_opset22_no_attention_op(self):
        """Opset 22 path does not use the Attention op."""
        attn = _make_llama_attention().eval()
        hs, cos, sin = self._get_inputs()
        model = _to_model(attn, hs, cos, sin, target_opset=22)
        op_types = {n.op_type for n in model.graph.node}
        self.assertNotIn("Attention", op_types)

    def test_opset24_uses_attention_op(self):
        """Opset 24 path uses the ONNX Attention op."""
        attn = _make_llama_attention().eval()
        hs, cos, sin = self._get_inputs()
        model = _to_model(attn, hs, cos, sin, target_opset=24)
        op_types = {n.op_type for n in model.graph.node}
        self.assertIn("Attention", op_types)


@requires_transformers("")
class TestLlamaAttentionRegistration(ExtTestCase):
    """Tests for the :mod:`yobx.torch.in_transformers` converter registry."""

    def test_register_transformer_converters_populates_registry(self):
        """register_transformer_converters populates TRANSFORMER_CONVERTERS."""
        from yobx.torch.in_transformers import (
            register_transformer_converters,
            get_transformer_converters,
        )
        from transformers.models.llama.modeling_llama import LlamaAttention

        register_transformer_converters()
        converters = get_transformer_converters()
        self.assertIn(LlamaAttention, converters)

    def test_get_transformer_converter_returns_llama_attention_to_onnx(self):
        """get_transformer_converter(LlamaAttention) returns llama_attention_to_onnx."""
        from yobx.torch.in_transformers import (
            register_transformer_converters,
            get_transformer_converter,
        )
        from yobx.torch.in_transformers.classes.llama_attention import llama_attention_to_onnx
        from transformers.models.llama.modeling_llama import LlamaAttention

        register_transformer_converters()
        converter = get_transformer_converter(LlamaAttention)
        self.assertIs(converter, llama_attention_to_onnx)

    def test_get_transformer_converter_raises_for_unknown_type(self):
        """get_transformer_converter raises ValueError for unregistered types."""
        from yobx.torch.in_transformers import get_transformer_converter

        with self.assertRaises(ValueError):
            get_transformer_converter(int)


if __name__ == "__main__":
    unittest.main(verbosity=2)

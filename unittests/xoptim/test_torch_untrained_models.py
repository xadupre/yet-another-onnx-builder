import collections
import unittest
import torch
from yobx.helpers import max_diff
from yobx.helpers.rt_helper import make_feeds
from yobx.torch.in_transformers.cache_helper import make_dynamic_cache
from yobx.torch.tiny_models import get_tiny_model
from yobx.torch import register_flattening_functions, apply_patches_for_model, to_onnx
from yobx.torch.torch_helper import torch_deepcopy
from yobx.ext_test_case import ExtTestCase, hide_stdout
from yobx.xbuilder import OptimizationOptions


class TestOptimizationUntrainedTorchModel(ExtTestCase):
    @hide_stdout()
    @unittest.skipIf(to_onnx is None, "not implement yet")
    def test_tiny_llm_to_onnx_24(self):
        import onnxruntime

        data = get_tiny_model("arnir0/Tiny-LLM")
        model, inputs, ds = data.model, data.export_inputs, data.dynamic_shapes

        b1 = data["inputs_batch1"]
        del inputs["position_ids"]
        del ds["position_ids"]
        del b1["position_ids"]

        filename = self.get_dump_file("test_tiny_llm_to_onnx_24.onnx")

        expected = model(**torch_deepcopy(b1))

        with (
            register_flattening_functions(patch_transformers=True),
            apply_patches_for_model(patch_transformers=True, model=model),
        ):
            onx = to_onnx(
                model,
                kwargs=inputs,
                dynamic_shapes=ds,
                filename=filename,
                verbose=1,
                large_model=True,
                options=OptimizationOptions(patterns="default"),
                target_opset=24,
            )

        outputs = [o.name for o in onx.model_proto.graph.output]
        self.assertEqual(
            ["output_0", "present_key_values_key_0", "present_key_values_value_0"], outputs
        )
        node_types = [n.op_type for n in onx.model_proto.graph.node]
        counter = collections.Counter(node_types)
        unique_ops = set(node_types)
        # self.assertNotIn("HalfRotaryEmbedding", unique_ops)
        # self.assertIn("RotaryEmbedding", unique_ops)
        self.assertIn("RMSNormalization", unique_ops)
        self.assertIn("CausalMaskMulAdd", unique_ops)
        self.assertIn("CausalMask", unique_ops)
        self.assertIn("Attention", unique_ops)
        self.assertNotIn("Squeeze", unique_ops)  # GQA
        self.assertInOr(("CosSinCache_p1", "CosSinCacheWithRange"), unique_ops)

        expected_counts = {
            "Add": 3,
            "And": 1,
            "Attention": 1,
            "Cast": 1,
            "CausalMask": 1,
            "CausalMaskMulAdd": 1,
            "Concat": 5,
            "CosSinCacheWithRange": 1,
            "Expand": 1,
            "Gather": 2,
            "HalfRotaryEmbedding": 2,
            "MatMul": 8,
            "Mul": 5,
            "Reshape": 3,
            "RMSNormalization": 3,
            "Shape": 5,
            "Sigmoid": 1,
            "Transpose": 2,
            "Unsqueeze": 6,
        }
        self.assertEqual(counter["Expand"], expected_counts["Expand"])
        self.assertEqual(counter["Transpose"], expected_counts["Transpose"])

        sess = onnxruntime.InferenceSession(filename, providers=["CPUExecutionProvider"])
        feeds = make_feeds(sess, b1, use_numpy=True)
        got = sess.run(None, feeds)
        diff = max_diff(expected, got)
        assert diff["abs"] <= 1e-5, f"diff={diff}"

        problem = dict(
            input_ids=torch.tensor([[24320]], dtype=torch.int64),
            attention_mask=torch.tensor([[1, 1, 1, 0]], dtype=torch.int64),
            past_key_values=make_dynamic_cache(
                [
                    torch.rand((1, 1, 3, 96), dtype=torch.float32),
                    torch.rand((1, 1, 3, 96), dtype=torch.float32),
                ]
            ),
        )

        expected = model(**torch_deepcopy(problem))
        sess = onnxruntime.InferenceSession(filename, providers=["CPUExecutionProvider"])
        feeds = make_feeds(sess, problem, use_numpy=True)
        got = sess.run(None, feeds)
        diff = max_diff(expected, got)
        assert diff["abs"] <= 1e-5, f"diff={diff}"

    @hide_stdout()
    @unittest.skipIf(to_onnx is None, "not implement yet")
    def test_tiny_llm_to_onnx_ort_22(self):
        import onnxruntime

        data = get_tiny_model("arnir0/Tiny-LLM")
        model, inputs, ds = data.model, data.export_inputs, data.dynamic_shapes
        b1 = data["inputs_batch1"]
        filename = self.get_dump_file("test_tiny_llm_to_onnx_ort_22.onnx")
        del inputs["position_ids"]
        del ds["position_ids"]
        del b1["position_ids"]

        expected = model(**torch_deepcopy(b1))

        with (
            register_flattening_functions(patch_transformers=True),
            apply_patches_for_model(patch_transformers=True, model=model),
        ):
            onx = to_onnx(
                model,
                kwargs=inputs,
                dynamic_shapes=ds,
                filename=filename,
                verbose=1,
                large_model=True,
                options=OptimizationOptions(patterns="default+onnxruntime"),
                target_opset=22,
            )

        outputs = [o.name for o in onx.model_proto.graph.output]
        self.assertEqual(
            ["output_0", "present_key_values_key_0", "present_key_values_value_0"], outputs
        )
        unique_ops = {n.op_type for n in onx.model_proto.graph.node}
        self.assertNotIn("HalfRotaryEmbedding", unique_ops)
        self.assertIn("RotaryEmbedding", unique_ops)
        self.assertIn("SimplifiedLayerNormalization", unique_ops)
        self.assertIn("SkipSimplifiedLayerNormalization", unique_ops)
        self.assertIn("QuickGelu", unique_ops)
        self.assertIn("CausalMaskMulAdd", unique_ops)
        self.assertIn("CausalMask", unique_ops)
        self.assertIn("GroupQueryAttention", unique_ops)
        self.assertInOr(("CosSinCache_p1", "CosSinCacheWithRange"), unique_ops)
        sess = onnxruntime.InferenceSession(filename, providers=["CPUExecutionProvider"])
        feeds = make_feeds(sess, b1, use_numpy=True)
        got = sess.run(None, feeds)
        diff = max_diff(expected, got)
        assert diff["abs"] <= 1e-5, f"diff={diff}"

        problem = dict(
            input_ids=torch.tensor([[24320]], dtype=torch.int64),
            attention_mask=torch.tensor([[1, 1, 1, 0]], dtype=torch.int64),
            past_key_values=make_dynamic_cache(
                [
                    torch.rand((1, 1, 3, 96), dtype=torch.float32),
                    torch.rand((1, 1, 3, 96), dtype=torch.float32),
                ]
            ),
        )

        expected = model(**torch_deepcopy(problem))
        sess = onnxruntime.InferenceSession(filename, providers=["CPUExecutionProvider"])
        feeds = make_feeds(sess, problem, use_numpy=True)
        got = sess.run(None, feeds)
        diff = max_diff(expected, got)
        assert diff["abs"] <= 1e-5, f"diff={diff}"


if __name__ == "__main__":
    unittest.main(verbosity=2)

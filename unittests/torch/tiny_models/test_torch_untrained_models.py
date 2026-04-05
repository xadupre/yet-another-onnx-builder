import collections
import unittest
import onnx
import torch
from yobx.helpers import max_diff
from yobx.helpers.rt_helper import make_feeds
from yobx.torch.in_transformers.cache_helper import make_dynamic_cache
from yobx.torch.tiny_models import get_tiny_model
from yobx.torch import (
    register_flattening_functions,
    apply_patches_for_model,
    to_onnx,
    ExportOptions,
)
from yobx.torch.torch_helper import torch_deepcopy
from yobx.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    ignore_warnings,
    requires_torch,
    requires_transformers,
    skipif_ci_windows,
)
from yobx.xbuilder import OptimizationOptions


class TestOptimizationUntrainedTorchModel(ExtTestCase):
    @classmethod
    def _get_missing_shapes(cls, model_proto):
        """
        Returns a list of ``(op_type, output_name)`` tuples for every node output
        in the main graph that has no type/shape entry in ``graph.value_info``.

        Inputs, graph outputs, and initializers already carry their own type
        information and are therefore excluded from the check.
        """
        known = set()
        known.update(v.name for v in model_proto.graph.input)
        known.update(v.name for v in model_proto.graph.output)
        known.update(v.name for v in model_proto.graph.value_info)
        known.update(v.name for v in model_proto.graph.initializer)
        missing = []
        for node in model_proto.graph.node:
            for out in node.output:
                if out and out not in known:
                    missing.append((node.op_type, out))
        return missing

    @classmethod
    def _chech_shape(cls, model_proto):
        for v in model_proto.graph.value_info:
            shape = tuple(d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim)
            for s in shape:
                if isinstance(s, int):
                    continue
                assert "batch//batch" not in s, f"Wrong dimension in {shape=}, name={v.name!r}"
                assert "batch//s61" not in s, f"Wrong dimension in {shape=}, name={v.name!r}"

    @hide_stdout()
    @requires_transformers("5.2")
    def test_tiny_llm_to_onnx_22_no_opt(self):
        import onnxruntime

        data = get_tiny_model("arnir0/Tiny-LLM")
        model, inputs, ds = data.model, data.export_inputs, data.dynamic_shapes
        b1 = data.inputs_batch1
        filename = self.get_dump_file("test_tiny_llm_to_onnx_22_no_opt.onnx")
        del inputs["position_ids"]
        del ds["position_ids"]
        del b1["position_ids"]

        expected_b1 = model(**torch_deepcopy(b1))

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
                optimize=False,
                target_opset=22,
            )

        sess = onnxruntime.InferenceSession(filename, providers=["CPUExecutionProvider"])
        feeds = make_feeds(sess, b1, use_numpy=True)
        got = sess.run(None, feeds)
        diff = max_diff(expected_b1, got)
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
        feeds = make_feeds(sess, problem, use_numpy=True)
        got = sess.run(None, feeds)
        diff = max_diff(expected, got)
        assert diff["abs"] <= 1e-5, f"diff={diff}"

        outputs = [o.name for o in onx.graph.output]
        self.assertEqual(
            ["output_0", "present_key_values_key_0", "present_key_values_value_0"], outputs
        )
        unique_ops = {n.op_type for n in onx.graph.node}
        self.assertNotIn("HalfRotaryEmbedding", unique_ops)
        self.assertNotIn("RotaryEmbedding", unique_ops)
        self.assertNotIn("SimplifiedLayerNormalization", unique_ops)
        self.assertNotIn("SkipSimplifiedLayerNormalization", unique_ops)
        self.assertNotIn("CausalMaskMulAdd", unique_ops)
        self.assertNotIn("CausalMask", unique_ops)
        self.assertNotIn("GroupQueryAttention", unique_ops)
        # self._chech_shape(onx.get_proto(include_weights=False))

    @hide_stdout()
    @requires_transformers("5.2")
    def test_tiny_llm_to_onnx_22_opt(self):
        import onnxruntime

        data = get_tiny_model("arnir0/Tiny-LLM")
        model, inputs, ds = data.model, data.export_inputs, data.dynamic_shapes
        b1 = data.inputs_batch1
        filename = self.get_dump_file("test_tiny_llm_to_onnx_22_opt.onnx")
        del inputs["position_ids"]
        del ds["position_ids"]
        del b1["position_ids"]

        expected_b1 = model(**torch_deepcopy(b1))

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
                target_opset=22,
            )

        sess = onnxruntime.InferenceSession(filename, providers=["CPUExecutionProvider"])
        feeds = make_feeds(sess, b1, use_numpy=True)
        got = sess.run(None, feeds)
        diff = max_diff(expected_b1, got)
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
        feeds = make_feeds(sess, problem, use_numpy=True)
        got = sess.run(None, feeds)
        diff = max_diff(expected, got)
        assert diff["abs"] <= 1e-5, f"diff={diff}"

        outputs = [o.name for o in onx.graph.output]
        self.assertEqual(
            ["output_0", "present_key_values_key_0", "present_key_values_value_0"], outputs
        )
        unique_ops = {n.op_type for n in onx.graph.node}
        self.assertIn("HalfRotaryEmbedding", unique_ops)
        self.assertNotIn("RotaryEmbedding", unique_ops)
        self.assertNotIn("SimplifiedLayerNormalization", unique_ops)
        self.assertNotIn("SkipSimplifiedLayerNormalization", unique_ops)
        self.assertIn("CausalMaskMulAdd", unique_ops)
        self.assertIn("CausalMask", unique_ops)
        self.assertNotIn("GroupQueryAttention", unique_ops)
        self.assertIn("LocalAttentionGQAsQ_to1", unique_ops)
        # not working yet
        # self._chech_shape(onx.get_proto(include_weights=False))

    @hide_stdout()
    @skipif_ci_windows("not available on windows")
    @requires_torch("2.10")
    @requires_transformers("5.2")
    @ignore_warnings(FutureWarning)
    def test_tiny_llm_to_onnx_24(self):
        import onnxruntime

        data = get_tiny_model("arnir0/Tiny-LLM")
        model, inputs, ds = data.model, data.export_inputs, data.dynamic_shapes

        b1 = data.inputs_batch1
        del inputs["position_ids"]
        del ds["position_ids"]
        del b1["position_ids"]
        # Ensure attention_mask is valid (non-zero) so the ONNX Attention op
        # produces results consistent with PyTorch's causal attention semantics.
        b1["attention_mask"] = torch.ones_like(b1["attention_mask"])

        filename = self.get_dump_file("test_tiny_llm_to_onnx_24.onnx")

        expected_b1 = model(**torch_deepcopy(b1))

        with (
            register_flattening_functions(patch_transformers=True),
            apply_patches_for_model(patch_transformers=True, patch_torch=True, model=model),
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

        sess = onnxruntime.InferenceSession(filename, providers=["CPUExecutionProvider"])
        feeds = make_feeds(sess, b1, use_numpy=True)
        got = sess.run(None, feeds)
        diff = max_diff(expected_b1, got)
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

        outputs = [o.name for o in onx.graph.output]
        self.assertEqual(
            ["output_0", "present_key_values_key_0", "present_key_values_value_0"], outputs
        )
        node_types = [n.op_type for n in onx.graph.node]
        counter = collections.Counter(node_types)
        unique_ops = set(node_types)
        self.assertNotIn("HalfRotaryEmbedding", unique_ops)
        self.assertIn("RotaryEmbedding", unique_ops)
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
            "Expand": 2,
            "Gather": 2,
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
        self._chech_shape(onx.get_proto(include_weights=False))

    @hide_stdout()
    @requires_transformers("5.2")
    def test_tiny_llm_to_onnx_ort_22(self):
        import onnxruntime

        data = get_tiny_model("arnir0/Tiny-LLM")
        model, inputs, ds = data.model, data.export_inputs, data.dynamic_shapes
        b1 = data.inputs_batch1
        filename = self.get_dump_file("test_tiny_llm_to_onnx_ort_22.onnx")
        del inputs["position_ids"]
        del ds["position_ids"]
        del b1["position_ids"]
        # Ensure attention_mask is valid (non-zero) so the ORT GroupQueryAttention op
        # produces results consistent with PyTorch's causal attention semantics.
        b1["attention_mask"] = torch.ones_like(b1["attention_mask"])

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

        outputs = [o.name for o in onx.graph.output]
        self.assertEqual(
            ["output_0", "present_key_values_key_0", "present_key_values_value_0"], outputs
        )
        unique_ops = {n.op_type for n in onx.graph.node}
        self.assertNotIn("HalfRotaryEmbedding", unique_ops)
        self.assertIn("RotaryEmbedding", unique_ops)
        self.assertIn("SimplifiedLayerNormalization", unique_ops)
        self.assertIn("SkipSimplifiedLayerNormalization", unique_ops)
        self.assertIn("QuickGelu", unique_ops)
        self.assertIn("CausalMaskMulAdd", unique_ops)
        self.assertIn("CausalMask", unique_ops)
        self.assertIn("GroupQueryAttention", unique_ops)
        self.assertInOr(("CosSinCache_p1", "CosSinCacheWithRange"), unique_ops)
        self._chech_shape(onx.get_proto(include_weights=False))

    def _export_tiny_llm(self, opset: int, patterns: str, return_builder: bool = False):
        """
        Export ``arnir0/Tiny-LLM`` and return the main ``ModelProto``.

        Inputs ``position_ids`` and their dynamic-shape entry are removed
        so that the export matches the form used in the rest of the test suite.
        """
        data = get_tiny_model("arnir0/Tiny-LLM")
        model, inputs, ds = data.model, data.export_inputs, data.dynamic_shapes
        del inputs["position_ids"]
        del ds["position_ids"]

        with (
            register_flattening_functions(patch_transformers=True),
            apply_patches_for_model(patch_transformers=True, patch_torch=True, model=model),
        ):
            onx = to_onnx(
                model,
                kwargs=inputs,
                dynamic_shapes=ds,
                verbose=0,
                options=OptimizationOptions(patterns=patterns),
                target_opset=opset,
                return_builder=return_builder,
            )
        # to_onnx returns ModelProto when large_model=False (the default)
        return onx

    @hide_stdout()
    @skipif_ci_windows("not available on windows")
    @requires_torch("2.10")
    @requires_transformers("5.2")
    @ignore_warnings(FutureWarning)
    def test_tiny_llm_shape_default_opset_22(self):
        """
        Checks that ``arnir0/Tiny-LLM`` is exported at opset 22 with type/shape
        information on every node output, and that ``patterns="default"``
        produces at least one local function (e.g. a fused attention function).
        """
        proto = self._export_tiny_llm(opset=22, patterns="default", return_builder=True)
        builder = proto.builder
        if builder.has_name("_onx_concat_sym_size_int_19::UnSq02"):
            self.assertTrue(builder.has_shape("_onx_concat_sym_size_int_19::UnSq02"))
            self.assertEqual((4,), builder.get_shape("_onx_concat_sym_size_int_19::UnSq02"))
            self.assertEqual(
                onnx.TensorProto.INT64, builder.get_type("_onx_concat_sym_size_int_19::UnSq02")
            )
        self.dump_onnx("test_tiny_llm_shape_default_opset_22.onnx", proto)
        missing = self._get_missing_shapes(proto)
        self.assertEqual(
            [],
            missing,
            f"Some node outputs are missing shape info at opset 22 / default: {missing}",
        )
        self.assertGreater(
            len(proto.functions),
            0,
            "default optimization should produce at least one local function at opset 22",
        )

    @hide_stdout()
    @skipif_ci_windows("not available on windows")
    @requires_torch("2.10")
    @requires_transformers("5.2")
    @ignore_warnings(FutureWarning)
    def test_tiny_llm_shape_default_opset_24(self):
        """
        Checks that ``arnir0/Tiny-LLM`` is exported at opset 24 with type/shape
        information on every node output, and that ``patterns="default"``
        produces at least one local function (e.g. a fused attention function).
        """
        proto = self._export_tiny_llm(opset=24, patterns="default")
        missing = self._get_missing_shapes(proto)
        self.assertEqual(
            [],
            missing,
            f"Some node outputs are missing shape info at opset 24 / default: {missing}",
        )
        self.assertGreater(
            len(proto.functions),
            0,
            "default optimization should produce at least one local function at opset 24",
        )

    @hide_stdout()
    @skipif_ci_windows("not available on windows")
    @requires_torch("2.10")
    @requires_transformers("5.2")
    @ignore_warnings(FutureWarning)
    def test_tiny_llm_shape_ort_opset_22(self):
        """
        Checks that ``arnir0/Tiny-LLM`` is exported at opset 22 with type/shape
        information on every node output, and that ``patterns="default+onnxruntime"``
        folds the local attention function into an ORT Attention operator
        (``Attention`` or ``GroupQueryAttention``).
        """
        proto = self._export_tiny_llm(opset=22, patterns="default+onnxruntime")
        missing = self._get_missing_shapes(proto)
        self.assertEqual(
            [],
            missing,
            f"Some node outputs are missing shape info at opset 22 / default+onnxruntime:"
            f" {missing}",
        )
        unique_ops = {n.op_type for n in proto.graph.node}
        self.assertInOr(
            ("Attention", "GroupQueryAttention"),
            unique_ops,
            "default+onnxruntime should produce an Attention op at opset 22",
        )

    @hide_stdout()
    @skipif_ci_windows("not available on windows")
    @requires_torch("2.10")
    @requires_transformers("5.2")
    @ignore_warnings(FutureWarning)
    def test_tiny_llm_shape_ort_opset_24(self):
        """
        Checks that ``arnir0/Tiny-LLM`` is exported at opset 24 with type/shape
        information on every node output, and that ``patterns="default+onnxruntime"``
        folds the local attention function into an ORT Attention operator
        (``Attention`` or ``GroupQueryAttention``).
        """
        proto = self._export_tiny_llm(opset=24, patterns="default+onnxruntime")
        missing = self._get_missing_shapes(proto)
        self.assertEqual(
            [],
            missing,
            f"Some node outputs are missing shape info at opset 24 / default+onnxruntime:"
            f" {missing}",
        )
        unique_ops = {n.op_type for n in proto.graph.node}
        self.assertInOr(
            ("Attention", "GroupQueryAttention"),
            unique_ops,
            "default+onnxruntime should produce an Attention op at opset 24",
        )

    @hide_stdout()
    @requires_transformers("5.2")
    def test_tiny_llm_tracing_to_onnx_22(self):
        import onnxruntime

        data = get_tiny_model("arnir0/Tiny-LLM")
        model, inputs, ds = data.model, data.export_inputs, data.dynamic_shapes
        b1 = data.inputs_batch1
        filename = self.get_dump_file("test_tiny_llm_tracing_to_onnx_22.onnx")
        # del inputs["position_ids"]
        # del ds["position_ids"]
        # del b1["position_ids"]
        print("***", ds)

        expected_b1 = model(**torch_deepcopy(b1))

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
                optimize=True,
                target_opset=22,
                export_options=ExportOptions(tracing=True),
            )

        sess = onnxruntime.InferenceSession(filename, providers=["CPUExecutionProvider"])
        feeds = make_feeds(sess, b1, use_numpy=True)
        got = sess.run(None, feeds)
        diff = max_diff(expected_b1, got)
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
        feeds = make_feeds(sess, problem, use_numpy=True)
        got = sess.run(None, feeds)
        diff = max_diff(expected, got)
        assert diff["abs"] <= 1e-5, f"diff={diff}"

        outputs = [o.name for o in onx.graph.output]
        self.assertEqual(
            ["output_0", "present_key_values_key_0", "present_key_values_value_0"], outputs
        )
        unique_ops = {n.op_type for n in onx.graph.node}
        self.assertNotIn("HalfRotaryEmbedding", unique_ops)
        self.assertNotIn("RotaryEmbedding", unique_ops)
        self.assertNotIn("SimplifiedLayerNormalization", unique_ops)
        self.assertNotIn("SkipSimplifiedLayerNormalization", unique_ops)
        self.assertNotIn("CausalMaskMulAdd", unique_ops)
        self.assertNotIn("CausalMask", unique_ops)
        self.assertNotIn("GroupQueryAttention", unique_ops)
        # self._chech_shape(onx.get_proto(include_weights=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest
import torch
from yobx.helpers import max_diff
from yobx.helpers.rt_helper import make_feeds
from yobx.torch.in_transformers.cache_helper import make_dynamic_cache
from yobx.torch.tiny_models import get_tiny_model
from yobx.torch import register_flattening_functions, apply_patches_for_model
from yobx.torch.torch_helper import torch_deepcopy
from yobx.ext_test_case import ExtTestCase, hide_stdout, requires_transformers


class TestOnnxExport(ExtTestCase):
    @unittest.skip("broken at decomposition")
    @hide_stdout()
    @requires_transformers("5.2")
    def test_toex_tiny_llm_to_onnx_22(self):
        import onnxruntime

        data = get_tiny_model("arnir0/Tiny-LLM")
        model, inputs, ds = data.model, data.export_inputs, data.dynamic_shapes
        b1 = data.inputs_batch1
        filename = self.get_dump_file("test_toex_tiny_llm_to_onnx_22.onnx")
        del inputs["position_ids"]
        del ds["position_ids"]
        del b1["position_ids"]

        expected_b1 = model(**torch_deepcopy(b1))
        print(self.string_type(inputs, with_shape=True))
        print(ds)

        with (
            register_flattening_functions(patch_transformers=True),
            apply_patches_for_model(patch_transformers=True, model=model),
        ):
            torch.onnx.export(
                model,
                (),
                filename,
                kwargs=inputs,
                dynamic_shapes=ds,
                verbose=1,
                opset_version=22,
                report=True,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)

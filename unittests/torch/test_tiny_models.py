import unittest
from yobx.ext_test_case import (
    ExtTestCase,
    requires_torch,
    requires_transformers,
)
from yobx.torch import get_tiny_model
from yobx.torch.torch_helper import torch_deepcopy


@requires_torch("2.0")
@requires_transformers("5.0")
class TestTinyModels(ExtTestCase):
    def test_tiny_llm(self):
        model_data = get_tiny_model("arnir0/Tiny-LLM")
        expected = model_data.model(**torch_deepcopy(model_data.export_inputs))
        text = self.string_type(expected, with_shape=True)
        self.assertEqual(
            "CausalLMOutputWithPast(logits:T1s2x3x32000,"
            "past_key_values:DynamicCache(key_cache=#1[T1s2x1x33x96], "
            "value_cache=#1[T1s2x1x33x96]))",
            text,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

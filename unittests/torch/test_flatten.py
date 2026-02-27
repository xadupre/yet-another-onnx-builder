import unittest
import torch
import transformers
from yobx.ext_test_case import ExtTestCase, requires_transformers
from yobx.helpers.cache_helper import make_dynamic_cache, make_static_cache
from yobx.torch.flatten import (
    flatten_dynamic_cache,
    flatten_with_keys_dynamic_cache,
    unflatten_dynamic_cache,
    flatten_static_cache,
    flatten_with_keys_static_cache,
    unflatten_static_cache,
    flatten_encoder_decoder_cache,
    flatten_with_keys_encoder_decoder_cache,
    unflatten_encoder_decoder_cache,
    flatten_base_model_output,
    flatten_with_keys_base_model_output,
    unflatten_base_model_output,
)
from transformers.modeling_outputs import BaseModelOutput


class TestFlatten(ExtTestCase):
    def _make_cache(self, n_layers=2, bsize=2, nheads=4, slen=3, dim=7):
        return make_dynamic_cache(
            [
                (torch.randn(bsize, nheads, slen, dim), torch.randn(bsize, nheads, slen, dim))
                for _ in range(n_layers)
            ]
        )

    def test_flatten_dynamic_cache(self):
        cache = self._make_cache()
        flat, context = flatten_dynamic_cache(cache)
        self.assertEqual(4, len(flat))
        self.assertEqual(["key_0", "value_0", "key_1", "value_1"], context)
        for t in flat:
            self.assertIsInstance(t, torch.Tensor)

    def test_flatten_with_keys_dynamic_cache(self):
        cache = self._make_cache()
        kv_pairs, context = flatten_with_keys_dynamic_cache(cache)
        self.assertEqual(4, len(kv_pairs))
        self.assertEqual(["key_0", "value_0", "key_1", "value_1"], context)
        for key_entry, val in kv_pairs:
            self.assertIsInstance(val, torch.Tensor)

    def test_unflatten_dynamic_cache(self):
        cache = self._make_cache()
        flat, context = flatten_dynamic_cache(cache)
        rebuilt = unflatten_dynamic_cache(flat, context)
        self.assertIsInstance(rebuilt, transformers.cache_utils.DynamicCache)
        from yobx.helpers import max_diff

        self.assertEqual(0, max_diff(cache, rebuilt)["abs"])

    def test_roundtrip_dynamic_cache(self):
        """Roundtrip via pytree flatten/unflatten."""
        cache = self._make_cache()
        flat, spec = torch.utils._pytree.tree_flatten(cache)
        cache2 = torch.utils._pytree.tree_unflatten(flat, spec)
        from yobx.helpers import max_diff

        self.assertEqual(0, max_diff(cache, cache2)["abs"])

    def test_flatten_static_cache(self):
        cache = make_static_cache(
            [(torch.rand((2, 4, 3, 7)), torch.rand((2, 4, 3, 7))) for _ in range(2)],
            max_cache_len=3,
        )
        flat, context = flatten_static_cache(cache)
        self.assertEqual(4, len(flat))
        self.assertEqual(["key_0", "value_0", "key_1", "value_1"], context)

    def test_unflatten_static_cache(self):
        cache = make_static_cache(
            [(torch.rand((2, 4, 3, 7)), torch.rand((2, 4, 3, 7))) for _ in range(2)],
            max_cache_len=3,
        )
        flat, context = flatten_static_cache(cache)
        rebuilt = unflatten_static_cache(flat, context)
        self.assertIsInstance(rebuilt, transformers.cache_utils.StaticCache)
        from yobx.helpers import max_diff

        self.assertEqual(0, max_diff(cache, rebuilt)["abs"])

    def test_flatten_encoder_decoder_cache(self):
        self_cache = self._make_cache()
        cross_cache = self._make_cache()
        ec_cache = transformers.cache_utils.EncoderDecoderCache(self_cache, cross_cache)
        flat, context = flatten_encoder_decoder_cache(ec_cache)
        self.assertIsInstance(flat, list)
        self.assertIsInstance(context, list)

    def test_unflatten_encoder_decoder_cache(self):
        self_cache = self._make_cache()
        cross_cache = self._make_cache()
        ec_cache = transformers.cache_utils.EncoderDecoderCache(self_cache, cross_cache)
        flat, context = flatten_encoder_decoder_cache(ec_cache)
        rebuilt = unflatten_encoder_decoder_cache(flat, context)
        self.assertIsInstance(rebuilt, transformers.cache_utils.EncoderDecoderCache)

    def test_flatten_base_model_output(self):
        output = BaseModelOutput(last_hidden_state=torch.randn(2, 3, 4))
        flat, context = flatten_base_model_output(output)
        self.assertIsInstance(flat, list)
        self.assertIsInstance(context, list)

    def test_unflatten_base_model_output(self):
        t = torch.randn(2, 3, 4)
        output = BaseModelOutput(last_hidden_state=t)
        flat, context = flatten_base_model_output(output)
        rebuilt = unflatten_base_model_output(flat, context)
        self.assertIsInstance(rebuilt, BaseModelOutput)
        self.assertEqualArray(t, rebuilt.last_hidden_state)

    @requires_transformers("4.57")
    def test_flatten_dynamic_cache_mixed_layers(self):
        cache = make_dynamic_cache(
            [
                (torch.rand((2, 4, 3, 7)), torch.rand((2, 4, 3, 7))),
                (torch.rand((2, 4, 3, 7)), torch.rand((2, 4, 3, 7))),
            ],
            cls_layers=[
                transformers.cache_utils.DynamicLayer,
                transformers.cache_utils.DynamicSlidingWindowLayer,
            ],
            cls_kwargs=[{}, {"sliding_window": 3}],
        )
        flat, context = flatten_dynamic_cache(cache)
        self.assertEqual(4, len(flat))
        rebuilt = unflatten_dynamic_cache(flat, context)
        self.assertIsInstance(rebuilt, transformers.cache_utils.DynamicCache)
        from yobx.helpers import max_diff

        self.assertEqual(0, max_diff(cache, rebuilt)["abs"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

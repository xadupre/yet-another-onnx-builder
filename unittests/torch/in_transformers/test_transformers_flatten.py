import unittest
import torch
import transformers
from transformers.modeling_outputs import BaseModelOutput
from yobx.ext_test_case import ExtTestCase, requires_transformers
from yobx.torch.flatten import register_class_flattening, unregister_class_flattening
from yobx.torch.in_transformers.cache_helper import make_dynamic_cache, make_static_cache
from yobx.torch.in_transformers.flatten_class import (
    flatten_dynamic_cache,
    flatten_with_keys_dynamic_cache,
    unflatten_dynamic_cache,
    flatten_static_cache,
    flatten_with_keys_static_cache,  #
    unflatten_static_cache,
    flatten_encoder_decoder_cache,
    flatten_with_keys_encoder_decoder_cache,  #
    unflatten_encoder_decoder_cache,
    flatten_with_keys_base_model_output,  #
    flatten_base_model_output,
    unflatten_base_model_output,
)


@requires_transformers("4.57")
class TestTransformersFlatten(ExtTestCase):
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
        for _key_entry, val in kv_pairs:
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

    def test_flatten_with_keys_static_cache(self):
        cache = make_static_cache(
            [(torch.rand((2, 4, 3, 7)), torch.rand((2, 4, 3, 7))) for _ in range(2)],
            max_cache_len=3,
        )
        kv_pairs, context = flatten_with_keys_static_cache(cache)
        self.assertEqual(4, len(kv_pairs))
        self.assertEqual(["key_0", "value_0", "key_1", "value_1"], context)
        for _key_entry, val in kv_pairs:
            self.assertIsInstance(val, torch.Tensor)

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

    def test_flatten_with_keys_encoder_decoder_cache(self):
        self_cache = self._make_cache()
        cross_cache = self._make_cache()
        ec_cache = transformers.cache_utils.EncoderDecoderCache(self_cache, cross_cache)
        kv_pairs, context = flatten_with_keys_encoder_decoder_cache(ec_cache)
        self.assertIsInstance(kv_pairs, list)
        self.assertIsInstance(context, list)
        self.assertEqual(2, len(kv_pairs))
        for _key_entry, val in kv_pairs:
            self.assertIsInstance(val, transformers.cache_utils.DynamicCache)

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

    def test_flatten_with_keys_base_model_output(self):
        t = torch.randn(2, 3, 4)
        output = BaseModelOutput(last_hidden_state=t)
        kv_pairs, context = flatten_with_keys_base_model_output(output)
        self.assertIsInstance(kv_pairs, list)
        self.assertIsInstance(context, list)
        for _key_entry, val in kv_pairs:
            self.assertIsInstance(val, torch.Tensor)

    def test_unflatten_base_model_output(self):
        t = torch.randn(2, 3, 4)
        output = BaseModelOutput(last_hidden_state=t)
        flat, context = flatten_base_model_output(output)
        rebuilt = unflatten_base_model_output(flat, context)
        self.assertIsInstance(rebuilt, BaseModelOutput)
        self.assertEqualArray(t, rebuilt.last_hidden_state)

    def _make_mixed_layer_cache(self, bsize=2, nheads=4, slen=3, dim=7):
        """Creates a DynamicCache with mixed DynamicLayer + DynamicSlidingWindowLayer layers."""
        from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer

        return make_dynamic_cache(
            [
                (torch.rand((bsize, nheads, slen, dim)), torch.rand((bsize, nheads, slen, dim))),
                (torch.rand((bsize, nheads, slen, dim)), torch.rand((bsize, nheads, slen, dim))),
            ],
            cls_layers=[DynamicLayer, DynamicSlidingWindowLayer],
            cls_kwargs=[{}, {"sliding_window": slen}],
        )

    @requires_transformers("4.57")
    def test_flatten_dynamic_cache_mixed_layers(self):
        cache = self._make_mixed_layer_cache()
        flat, context = flatten_dynamic_cache(cache)
        self.assertEqual(4, len(flat))
        # The context must not encode the sliding_window value so that the
        # tree spec remains stable across inputs with different sequence lengths.
        self.assertEqual(["key_D_0", "value_D_0", "key_W_1", "value_W_1"], context)
        rebuilt = unflatten_dynamic_cache(flat, context)
        self.assertIsInstance(rebuilt, transformers.cache_utils.DynamicCache)
        from yobx.helpers import max_diff

        self.assertEqual(0, max_diff(cache, rebuilt)["abs"])

    @requires_transformers("4.57")
    def test_flatten_dynamic_cache_mixed_layers_context_keys(self):
        """Flattening a mixed-layer cache encodes layer types in the context keys."""
        cache = self._make_mixed_layer_cache(slen=3)
        flat, context = flatten_dynamic_cache(cache)
        self.assertEqual(4, len(flat))
        # sliding_window is intentionally not encoded in context keys (data-independent spec)
        self.assertEqual(["key_D_0", "value_D_0", "key_W_1", "value_W_1"], context)

    @requires_transformers("4.57")
    def test_unflatten_dynamic_cache_mixed_layers_preserves_types(self):
        """Unflatten restores the correct layer types for a mixed-layer cache."""
        from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer

        cache = self._make_mixed_layer_cache()
        flat, context = flatten_dynamic_cache(cache)
        rebuilt = unflatten_dynamic_cache(flat, context)
        self.assertIsInstance(rebuilt, transformers.cache_utils.DynamicCache)
        self.assertIsInstance(rebuilt.layers[0], DynamicLayer)
        self.assertIsInstance(rebuilt.layers[1], DynamicSlidingWindowLayer)
        from yobx.helpers import max_diff

        self.assertEqual(0, max_diff(cache, rebuilt)["abs"])

    @requires_transformers("4.57")
    def test_flatten_with_keys_dynamic_cache_mixed_layers(self):
        """flatten_with_keys encodes layer types in context for mixed-layer caches."""
        cache = self._make_mixed_layer_cache(slen=3)
        kv_pairs, context = flatten_with_keys_dynamic_cache(cache)
        self.assertEqual(4, len(kv_pairs))
        # sliding_window is intentionally not encoded in context keys (data-independent spec)
        self.assertEqual(["key_D_0", "value_D_0", "key_W_1", "value_W_1"], context)
        for _key_entry, val in kv_pairs:
            self.assertIsInstance(val, torch.Tensor)

    @requires_transformers("4.57")
    def test_roundtrip_dynamic_cache_mixed_layers(self):
        """pytree tree_flatten/tree_unflatten roundtrip preserves mixed layer types."""
        from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer

        cache = self._make_mixed_layer_cache()
        flat, spec = torch.utils._pytree.tree_flatten(cache)
        cache2 = torch.utils._pytree.tree_unflatten(flat, spec)
        self.assertIsInstance(cache2, transformers.cache_utils.DynamicCache)
        self.assertIsInstance(cache2.layers[0], DynamicLayer)
        self.assertIsInstance(cache2.layers[1], DynamicSlidingWindowLayer)
        from yobx.helpers import max_diff

        self.assertEqual(0, max_diff(cache, cache2)["abs"])

    def test_flatten_dynamic_cache_mixed_layers_stable_context(self):
        """Verifies that the tree-spec context is the same for different sliding_window sizes.

        This is required for ``torch.export`` to accept inputs with varying sequence
        lengths without raising a tree-spec mismatch error.
        """
        cache3 = make_dynamic_cache(
            [(torch.rand((2, 4, 3, 7)), torch.rand((2, 4, 3, 7))) for _ in range(2)],
            cls_layers=[
                transformers.cache_utils.DynamicLayer,
                transformers.cache_utils.DynamicSlidingWindowLayer,
            ],
            cls_kwargs=[{}, {"sliding_window": 3}],
        )
        cache5 = make_dynamic_cache(
            [(torch.rand((2, 4, 5, 7)), torch.rand((2, 4, 5, 7))) for _ in range(2)],
            cls_layers=[
                transformers.cache_utils.DynamicLayer,
                transformers.cache_utils.DynamicSlidingWindowLayer,
            ],
            cls_kwargs=[{}, {"sliding_window": 5}],
        )
        _, ctx3 = flatten_dynamic_cache(cache3)
        _, ctx5 = flatten_dynamic_cache(cache5)
        self.assertEqual(ctx3, ctx5, "context must be the same regardless of sliding_window size")

    def test_register_class_flattening_with_f_check(self):
        """Test register_class_flattening verifies the registration using f_check."""
        already_registered = (
            transformers.cache_utils.DynamicCache in torch.utils._pytree.SUPPORTED_NODES
        )
        if already_registered:
            unregister_class_flattening(transformers.cache_utils.DynamicCache)
        try:

            def f_check():
                return self._make_cache()

            registered = register_class_flattening(
                transformers.cache_utils.DynamicCache,
                flatten_dynamic_cache,
                unflatten_dynamic_cache,
                flatten_with_keys_dynamic_cache,
                f_check=f_check,
            )
            self.assertTrue(registered)
        finally:
            unregister_class_flattening(transformers.cache_utils.DynamicCache)
            if already_registered:
                register_class_flattening(
                    transformers.cache_utils.DynamicCache,
                    flatten_dynamic_cache,
                    unflatten_dynamic_cache,
                    flatten_with_keys_dynamic_cache,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)

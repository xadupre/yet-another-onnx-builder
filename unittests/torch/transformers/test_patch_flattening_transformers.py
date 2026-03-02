import unittest
import torch
from transformers.modeling_outputs import BaseModelOutput
from yobx.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_torch,
    requires_transformers,
)
from yobx.helpers import flatten_object
from yobx.torch.transformers.cache_helper import (
    make_encoder_decoder_cache,
    make_dynamic_cache,
    make_dynamic_shapes_kv_cache,
    make_static_cache,
    flatten_unflatten_for_dynamic_shapes,
    CacheKeyValue,
)
from yobx.torch.torch_helper import torch_deepcopy
from yobx.torch.flatten import register_flattening_functions, replacement_before_exporting


@requires_transformers("4.57")
class TestPatchFlattening(ExtTestCase):
    @ignore_warnings(UserWarning)
    def test_encoder_decoder_cache_flatten(self):
        cache = make_encoder_decoder_cache(
            make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))]),
            make_dynamic_cache([(torch.rand((5, 5, 5)), torch.rand((5, 5, 5)))]),
        )
        with register_flattening_functions(patch_transformers=True):
            flat, _spec = torch.utils._pytree.tree_flatten(cache)
            self.assertEqual(
                "#4[T1s4x4x4,T1s4x4x4,T1s5x5x5,T1s5x5x5]",
                self.string_type(flat, with_shape=True),
            )
            cache2 = torch.utils._pytree.tree_unflatten(flat, _spec)
            self.assertEqual(
                self.string_type(cache, with_shape=True, with_min_max=True),
                self.string_type(cache2, with_shape=True, with_min_max=True),
            )

    @ignore_warnings(UserWarning)
    def test_encoder_decoder_cache_deepcopy(self):
        cache = make_encoder_decoder_cache(
            make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))]),
            make_dynamic_cache([(torch.rand((5, 5, 5)), torch.rand((5, 5, 5)))]),
        )
        with register_flattening_functions(patch_transformers=True):
            cache2 = torch_deepcopy([cache])
            self.assertEqualAny([cache], cache2)

    @ignore_warnings(UserWarning)
    def test_encoder_decoder_cache_export(self):
        class Model(torch.nn.Module):
            def forward(self, cache):
                att = CacheKeyValue(cache.self_attention_cache)
                return att.key_cache[0]

        cache1 = make_dynamic_cache(
            [(torch.randn(2, 4, 3, 7), torch.randn(2, 4, 3, 7)) for i in range(3)]
        )
        cache2 = make_dynamic_cache(
            [(torch.randn(2, 4, 3, 7), torch.randn(2, 4, 3, 7)) for i in range(3)]
        )

        cache = make_encoder_decoder_cache(cache1, cache2)
        model = Model()
        model(cache)
        DYN = torch.export.Dim.DYNAMIC
        ds = [
            make_dynamic_shapes_kv_cache(cache1, {0: DYN}),
            make_dynamic_shapes_kv_cache(cache2, {0: DYN}),
        ]

        with register_flattening_functions(patch_transformers=True):
            torch.export.export(model, (cache,), dynamic_shapes=(ds,))

    @ignore_warnings(UserWarning)
    def test_dynamic_cache_flatten(self):
        cache = make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))])
        with register_flattening_functions(patch_transformers=True):
            flat, _spec = torch.utils._pytree.tree_flatten(cache)
            self.assertEqual(
                "#2[T1s4x4x4,T1s4x4x4]",
                self.string_type(flat, with_shape=True),
            )
            cache2 = torch.utils._pytree.tree_unflatten(flat, _spec)
            self.assertEqual(
                self.string_type(cache, with_shape=True, with_min_max=True),
                self.string_type(cache2, with_shape=True, with_min_max=True),
            )

    @ignore_warnings(UserWarning)
    def test_dynamic_cache_export(self):
        class Model(torch.nn.Module):
            def forward(self, cache):
                cache = CacheKeyValue(cache)
                return cache.key_cache[0]

        cache = make_dynamic_cache(
            [(torch.randn(2, 4, 3, 7), torch.randn(2, 4, 3, 7)) for i in range(3)]
        )
        model = Model()
        model(cache)
        DYN = torch.export.Dim.DYNAMIC
        ds = make_dynamic_shapes_kv_cache(cache, {0: DYN})
        self.assertEqual(len(ds), 6)

        with register_flattening_functions(patch_transformers=True):
            flat, _spec = torch.utils._pytree.tree_flatten(cache)
            self.assertEqual(len(flat), len(ds))
            unflat = torch.utils._pytree.tree_unflatten(flat, _spec)
            if hasattr(unflat, "layers"):
                self.assertEqual(len(unflat.layers), 3)
            torch.export.export(model, (cache,), dynamic_shapes=(ds,))

    @ignore_warnings(UserWarning)
    def test_dynamic_cache_deepcopy(self):
        cache = make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))])
        with register_flattening_functions(patch_transformers=True):
            cache2 = torch_deepcopy([cache])
            self.assertEqualAny([cache], cache2)

    @ignore_warnings(UserWarning)
    def test_base_model_output_deepcopy(self):
        bo = BaseModelOutput(last_hidden_state=torch.rand((4, 4, 4)))
        self.assertEqual(bo.__class__.__name__, "BaseModelOutput")
        with register_flattening_functions(patch_transformers=True):
            bo2 = torch_deepcopy([bo])
            self.assertIsInstance(bo2, list)
            self.assertEqual(bo2[0].__class__.__name__, "BaseModelOutput")
            self.assertEqualAny([bo], bo2)

    @ignore_warnings(UserWarning)
    def test_base_model_output_string_type(self):
        bo = BaseModelOutput(last_hidden_state=torch.rand((4, 4, 4)))
        with register_flattening_functions(patch_transformers=True):
            self.assertEqual(
                "BaseModelOutput(last_hidden_state:T1s4x4x4)",
                self.string_type(bo, with_shape=True),
            )

    @ignore_warnings(UserWarning)
    def test_base_model_output_flatten(self):
        bo = BaseModelOutput(last_hidden_state=torch.rand((4, 4, 4)))
        with register_flattening_functions(patch_transformers=True):
            flat, _spec = torch.utils._pytree.tree_flatten(bo)
            self.assertEqual(
                "#1[T1s4x4x4]",
                self.string_type(flat, with_shape=True),
            )
            bo2 = torch.utils._pytree.tree_unflatten(flat, _spec)
            self.assertEqual(
                self.string_type(bo, with_shape=True, with_min_max=True),
                self.string_type(bo2, with_shape=True, with_min_max=True),
            )

    @ignore_warnings(UserWarning)
    def test_base_model_output_export(self):
        class Model(torch.nn.Module):
            def forward(self, cache):
                return cache.last_hidden_state[0]

        bo = BaseModelOutput(last_hidden_state=torch.rand((4, 4, 4)))
        model = Model()
        model(bo)
        DYN = torch.export.Dim.DYNAMIC
        ds = [{0: DYN}]

        with register_flattening_functions(patch_transformers=True):
            torch.export.export(model, (bo,), dynamic_shapes=(ds,))

    @ignore_warnings(UserWarning)
    def test_base_model_output_unflatten_flatten(self):
        bo = BaseModelOutput(last_hidden_state=torch.rand((4, 4, 4)))
        with register_flattening_functions(patch_transformers=True):
            _flat, _spec = torch.utils._pytree.tree_flatten(bo)
            unflat = flatten_unflatten_for_dynamic_shapes(bo, use_dict=True)
            self.assertIsInstance(unflat, list)
            self.assertEqual("#1[T1r3]", self.string_type(unflat))

    @ignore_warnings(UserWarning)
    @requires_torch("2.7.99")
    def test_static_cache(self):
        bo = make_static_cache(
            [
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
            ],
            max_cache_len=15,
        )
        self.assertEqual(bo.__class__.__name__, "StaticCache")
        bo2 = torch_deepcopy([bo])
        self.assertIsInstance(bo2, list)
        self.assertEqual(
            "StaticCache(key_cache=#3[T1s4x5x15x7,T1s4x5x15x7,T1s4x5x15x7], "
            "value_cache=#3[T1s4x5x15x7,T1s4x5x15x7,T1s4x5x15x7])",
            self.string_type(bo, with_shape=True),
        )

        with register_flattening_functions(patch_transformers=True):
            # internal function
            bo2 = torch_deepcopy([bo])
            self.assertIsInstance(bo2, list)
            self.assertEqual(bo2[0].__class__.__name__, "StaticCache")
            self.assertEqualAny([bo], bo2)
            self.assertEqual(
                "StaticCache(key_cache=#3[T1s4x5x15x7,T1s4x5x15x7,T1s4x5x15x7], "
                "value_cache=#3[T1s4x5x15x7,T1s4x5x15x7,T1s4x5x15x7])",
                self.string_type(bo, with_shape=True),
            )

            # serialization
            flat, _spec = torch.utils._pytree.tree_flatten(bo)
            self.assertEqual(
                "#6[T1s4x5x15x7,T1s4x5x15x7,T1s4x5x15x7,T1s4x5x15x7,T1s4x5x15x7,T1s4x5x15x7]",
                self.string_type(flat, with_shape=True),
            )
            bo2 = torch.utils._pytree.tree_unflatten(flat, _spec)
            self.assertEqual(
                self.string_type(bo, with_shape=True, with_min_max=True),
                self.string_type(bo2, with_shape=True, with_min_max=True),
            )

            # flatten_unflatten
            flat, _spec = torch.utils._pytree.tree_flatten(bo)
            unflat = flatten_unflatten_for_dynamic_shapes(bo, use_dict=True)
            self.assertIsInstance(unflat, list)
            self.assertEqual("#6[T1r4,T1r4,T1r4,T1r4,T1r4,T1r4]", self.string_type(unflat))

        # export
        class Model(torch.nn.Module):
            def forward(self, cache):
                cache = CacheKeyValue(cache)
                return cache.key_cache[0]

        model = Model()
        model(bo)
        DYN = torch.export.Dim.DYNAMIC
        ds = make_dynamic_shapes_kv_cache(bo, {0: DYN})

        with register_flattening_functions(patch_transformers=True):
            torch.export.export(model, (bo,), dynamic_shapes=(ds,))

    @ignore_warnings(UserWarning)
    @requires_transformers("4.99")
    def test_dynamic_cache_flatten_unflatten(self):
        values = [
            (torch.rand((2, 4, 4, 4)), torch.rand((2, 4, 4, 4))),
            (torch.rand((2, 4, 4, 4)), torch.rand((2, 4, 4, 4))),
        ]
        cache = make_dynamic_cache(values)
        flat_cache = flatten_object(cache)
        order_cache = flatten_object(values)
        with register_flattening_functions(patch_transformers=True):
            flat, _spec = torch.utils._pytree.tree_flatten(cache)
            cache2 = torch.utils._pytree.tree_unflatten(flat, _spec)
            self.assertEqualAny(flat_cache, flatten_object(cache2))
            self.assertEqualAny(order_cache, flatten_object(cache2))
            self.assertEqual(
                [type(ly) for ly in cache.layers], [type(ly) for ly in cache2.layers]
            )

    @ignore_warnings(UserWarning)
    @requires_transformers("4.99")
    def test_dynamic_cache_in_a_model_args(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, cache):
                acc = x.clone()
                for i, lay in enumerate(cache.layers):
                    acc = acc + lay.keys * (i + 1) - lay.values
                    cache.update(x * (i + 1), x * 2 * (i + 1), i)
                return acc, cache

        values = [
            (torch.rand((2, 4, 4, 4)), torch.rand((2, 4, 4, 4))),
            (torch.rand((2, 4, 4, 4)), torch.rand((2, 4, 4, 4))),
        ]
        cache = make_dynamic_cache(values)
        inputs = (torch.rand((2, 4, 1, 4)), cache)
        inputs_copied = torch_deepcopy(inputs)
        self.assertEqualAny(inputs, inputs_copied)
        model = Model()
        expected = model(*inputs)
        DYN = torch.export.Dim.DYNAMIC
        with register_flattening_functions(patch_transformers=True):
            ep = torch.export.export(
                model,
                torch_deepcopy(inputs_copied),
                dynamic_shapes=(
                    {0: DYN},
                    [{0: DYN, 2: DYN}, {0: DYN, 2: DYN}, {0: DYN, 2: DYN}, {0: DYN, 2: DYN}],
                ),
            )
            got = ep.module()(*inputs_copied)
            self.assertEqualAny(expected, got)

    @ignore_warnings(UserWarning)
    @requires_transformers("4.99")
    def test_dynamic_cache_in_a_model_kwargs(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, cache):
                acc = x.clone()
                for i, lay in enumerate(cache.layers):
                    acc = acc + lay.keys * (i + 1) - lay.values
                    cache.update(x * (i + 1), x * 2 * (i + 1), i)
                return acc, cache

        values = [
            (torch.rand((2, 4, 4, 4)), torch.rand((2, 4, 4, 4))),
            (torch.rand((2, 4, 4, 4)), torch.rand((2, 4, 4, 4))),
        ]
        cache = make_dynamic_cache(values)
        inputs = dict(x=torch.rand((2, 4, 1, 4)), cache=cache)
        inputs_copied = torch_deepcopy(inputs)
        self.assertEqualAny(inputs, inputs_copied)
        model = Model()
        expected = model(**inputs)
        DYN = torch.export.Dim.DYNAMIC
        with register_flattening_functions(patch_transformers=True):
            ep = torch.export.export(
                model,
                (),
                kwargs=torch_deepcopy(inputs_copied),
                dynamic_shapes=dict(
                    x={0: DYN},
                    cache=[
                        {0: DYN, 2: DYN},
                        {0: DYN, 2: DYN},
                        {0: DYN, 2: DYN},
                        {0: DYN, 2: DYN},
                    ],
                ),
            )
            got = ep.module()(**inputs_copied)
            self.assertEqualAny(expected, got)

    @ignore_warnings(UserWarning)
    def test_replacement_before_exporting_none(self):
        self.assertIsNone(replacement_before_exporting(None))

    @ignore_warnings(UserWarning)
    def test_replacement_before_exporting_scalars(self):
        self.assertEqual(1, replacement_before_exporting(1))
        self.assertEqual(1.5, replacement_before_exporting(1.5))

    @ignore_warnings(UserWarning)
    def test_replacement_before_exporting_tensor(self):
        t = torch.rand((3, 4))
        result = replacement_before_exporting(t)
        self.assertIs(t, result)

    @ignore_warnings(UserWarning)
    def test_replacement_before_exporting_dict(self):
        d = {"a": torch.rand((2, 3)), "b": 1}
        result = replacement_before_exporting(d)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), {"a", "b"})
        self.assertIs(result["a"], d["a"])
        self.assertEqual(result["b"], 1)

    @ignore_warnings(UserWarning)
    def test_replacement_before_exporting_tuple(self):
        t = (torch.rand((2, 3)), 1)
        result = replacement_before_exporting(t)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], t[0])
        self.assertEqual(result[1], 1)

    @ignore_warnings(UserWarning)
    def test_replacement_before_exporting_list(self):
        lst = [torch.rand((2, 3)), 1]
        result = replacement_before_exporting(lst)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], lst[0])
        self.assertEqual(result[1], 1)

    @ignore_warnings(UserWarning)
    def test_replacement_before_exporting_nested(self):
        inner = [torch.rand((2,)), torch.rand((3,))]
        outer = {"x": inner, "y": (1, 2)}
        result = replacement_before_exporting(outer)
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["x"], list)
        self.assertIsInstance(result["y"], tuple)
        self.assertIs(result["x"][0], inner[0])

    @ignore_warnings(UserWarning)
    def test_replacement_before_exporting_base_model_output(self):
        # BaseModelOutput is a dict subclass; type check prevents recursion
        bo = BaseModelOutput(last_hidden_state=torch.rand((4, 4, 4)))
        result = replacement_before_exporting(bo)
        self.assertIs(result, bo)

    @ignore_warnings(UserWarning)
    def test_replacement_before_exporting_base_model_output_in_tuple(self):
        bo = BaseModelOutput(last_hidden_state=torch.rand((4, 4, 4)))
        t = (bo,)
        result = replacement_before_exporting(t)
        self.assertIsInstance(result, tuple)
        self.assertIs(result[0], bo)


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest
import torch
import transformers
from yobx.ext_test_case import ExtTestCase, requires_transformers
from yobx.helpers import string_type, max_diff
from yobx.torch.in_transformers.cache_helper import (
    CacheKeyValue,
    make_dynamic_cache,
    make_static_cache,
)


class TestCacheHelpers(ExtTestCase):
    def test_string_type(self):
        DYN = torch.export.Dim.DYNAMIC
        self.assertEqual("DYNAMIC", string_type(DYN))
        AUTO = torch.export.Dim.AUTO
        self.assertEqual("AUTO", string_type(AUTO))
        self.assertEqual("#1[DYNAMIC]", string_type([DYN]))

        batch = torch.export.Dim("batch")
        dynamic_shapes = dict(
            input_ids={0: batch, 1: "seq"},
            attention_mask={0: batch, 1: "seq"},
            position_ids={0: batch, 1: "seq"},
            past_key_values=[[{0: batch, 2: "seq"}], [{0: batch, 2: "seq"}]],
        )
        self.assertEqual(
            "dict(input_ids:{0:Dim(batch),1:DYN(seq)},"
            "attention_mask:{0:Dim(batch),1:DYN(seq)},"
            "position_ids:{0:Dim(batch),1:DYN(seq)},"
            "past_key_values:#2[#1[{0:Dim(batch),2:DYN(seq)}],"
            "#1[{0:Dim(batch),2:DYN(seq)}]])",
            string_type(dynamic_shapes),
        )

    @requires_transformers("4.57")
    def test_replace_by(self):
        bsize, nheads, slen, dim = 2, 4, 3, 7

        past_key_values = make_dynamic_cache(
            [(torch.randn(bsize, nheads, slen, dim), torch.randn(bsize, nheads, slen, dim))]
        )
        self.assertEqual(
            "DynamicCache(key_cache=#1[T1s2x4x3x7], value_cache=#1[T1s2x4x3x7])",
            self.string_type(past_key_values, with_shape=True),
        )

    @requires_transformers("4.57")
    def test_make_static_cache(self):
        cache = make_static_cache(
            [
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
            ],
            max_cache_len=15,
        )
        text = self.string_type(cache, with_shape=True)
        self.assertEqual(
            "StaticCache(key_cache=#3[T1s4x5x15x7,T1s4x5x15x7,T1s4x5x15x7], "
            "value_cache=#3[T1s4x5x15x7,T1s4x5x15x7,T1s4x5x15x7])",
            text,
        )
        self.assertEqual(0, max_diff(cache, cache)["abs"])

    def test_simple_indices(self):
        class Model(torch.nn.Module):
            def forward(self, x, i, j):
                return x[i, j]

        inputs = (
            torch.rand((4, 4), dtype=torch.float32),
            torch.randint(0, 4, (4, 4, 4, 4), dtype=torch.int64),
            torch.randint(0, 4, (4, 4, 4, 4), dtype=torch.int64),
        )
        model = Model()
        expected = model(*inputs)
        self.assertEqual(expected.shape, (4, 4, 4, 4))
        DYN = torch.export.Dim.DYNAMIC
        sh = {0: DYN, 1: DYN, 2: DYN, 3: DYN}
        ep = torch.export.export(model, inputs, dynamic_shapes=({0: DYN, 1: DYN}, sh, sh))
        self.assertNotEmpty(ep)

    def test_cache_key_value_none(self):
        cv = CacheKeyValue(None)
        self.assertIsNone(cv.key_cache)
        self.assertIsNone(cv.value_cache)
        self.assertEqual(0, cv.n_layers)
        self.assertEqual(0, len(cv))
        self.assertEqual([], cv.aslist())

    def test_cache_key_value_from_list(self):
        bsize, nheads, slen, dim = 2, 4, 3, 7
        t1 = torch.randn(bsize, nheads, slen, dim)
        t2 = torch.randn(bsize, nheads, slen, dim)
        t3 = torch.randn(bsize, nheads, slen, dim)
        t4 = torch.randn(bsize, nheads, slen, dim)
        cv = CacheKeyValue([t1, t2, t3, t4])
        self.assertEqual(2, cv.n_layers)
        self.assertEqual(4, len(cv))
        lst = cv.aslist()
        self.assertEqual(4, len(lst))
        self.assertEqualArray(t1, lst[0])
        self.assertEqualArray(t2, lst[1])
        self.assertEqualArray(t3, lst[2])
        self.assertEqualArray(t4, lst[3])

    @requires_transformers("4.57")
    def test_cache_key_value_from_dynamic_cache(self):
        bsize, nheads, slen, dim = 2, 4, 3, 7
        t1 = torch.randn(bsize, nheads, slen, dim)
        t2 = torch.randn(bsize, nheads, slen, dim)
        t3 = torch.randn(bsize, nheads, slen, dim)
        t4 = torch.randn(bsize, nheads, slen, dim)
        cache = make_dynamic_cache([(t1, t2), (t3, t4)])
        cv = CacheKeyValue(cache)
        self.assertEqual(2, cv.n_layers)
        self.assertEqual(4, len(cv))
        lst = cv.aslist()
        self.assertEqual(4, len(lst))
        self.assertEqualArray(t1, lst[0])
        self.assertEqualArray(t2, lst[1])
        self.assertEqualArray(t3, lst[2])
        self.assertEqualArray(t4, lst[3])

    def test_cache_key_value_make_dynamic_cache(self):
        bsize, nheads, slen, dim = 2, 4, 3, 7
        t1 = torch.randn(bsize, nheads, slen, dim)
        t2 = torch.randn(bsize, nheads, slen, dim)
        t3 = torch.randn(bsize, nheads, slen, dim)
        t4 = torch.randn(bsize, nheads, slen, dim)
        cache = make_dynamic_cache([(t1, t2), (t3, t4)])
        cv = CacheKeyValue(cache)
        rebuilt = cv.make_dynamic_cache()
        self.assertIsInstance(rebuilt, transformers.cache_utils.DynamicCache)
        self.assertEqual(0, max_diff(cache, rebuilt)["abs"])

    @requires_transformers("4.57")
    def test_cache_key_value_cls_layers(self):
        bsize, nheads, slen, dim = 2, 4, 3, 7
        t1 = torch.randn(bsize, nheads, slen, dim)
        t2 = torch.randn(bsize, nheads, slen, dim)
        t3 = torch.randn(bsize, nheads, slen, dim)
        t4 = torch.randn(bsize, nheads, slen, dim)
        cache = make_dynamic_cache([(t1, t2), (t3, t4)])
        cv = CacheKeyValue(cache)
        # cls_layers is set when constructed from a cache with layers attribute,
        # or is None/passed-in when constructed from key_cache attribute
        if hasattr(cache, "layers"):
            self.assertIsInstance(cv.cls_layers, list)
            self.assertEqual(2, len(cv.cls_layers))
        self.assertEqual(2, cv.n_layers)
        self.assertEqual(4, len(cv))

    def test_max_diff_with_cache_key_value(self):
        bsize, nheads, slen, dim = 2, 4, 3, 7
        t1 = torch.randn(bsize, nheads, slen, dim)
        t2 = torch.randn(bsize, nheads, slen, dim)
        cache = make_dynamic_cache([(t1, t2)])
        cv = CacheKeyValue(cache)
        self.assertEqual(0, max_diff(cv, cv)["abs"])

    @requires_transformers("4.57")
    def test_max_diff_cache_key_value_vs_tuple(self):
        bsize, nheads, slen, dim = 2, 4, 3, 7
        t1 = torch.randn(bsize, nheads, slen, dim)
        t2 = torch.randn(bsize, nheads, slen, dim)
        cache = make_dynamic_cache([(t1, t2)])
        cv = CacheKeyValue(cache)
        res = max_diff(cv, ([t1], [t2]))
        self.assertEqual(0, res["abs"])

    @requires_transformers("4.57")
    def test_make_dynamic_cache_2_types(self):
        cache = make_dynamic_cache(
            [
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
            ],
            cls_layers=[
                transformers.cache_utils.DynamicLayer,
                transformers.cache_utils.DynamicSlidingWindowLayer,
            ],
        )
        text = self.string_type(cache, with_shape=True)
        self.assertEqual(
            "DynamicCache(DynamicLayer(T1s4x5x6x7, T1s4x5x6x7), "
            "DynamicSlidingWindowLayer(T1s4x5x6x7, T1s4x5x6x7))",
            text,
        )
        self.assertEqual(0, max_diff(cache, cache)["abs"])

    @requires_transformers("4.57")
    def test_make_dynamic_cache_2_types_kwargs(self):
        cache = make_dynamic_cache(
            [
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
            ],
            cls_layers=[
                transformers.cache_utils.DynamicLayer,
                transformers.cache_utils.DynamicSlidingWindowLayer,
            ],
            cls_kwargs=[{}, dict(sliding_window=12)],
        )
        text = self.string_type(cache, with_shape=True)
        self.assertEqual(
            "DynamicCache(DynamicLayer(T1s4x5x6x7, T1s4x5x6x7), "
            "DynamicSlidingWindowLayer(T1s4x5x6x7, T1s4x5x6x7))",
            text,
        )
        self.assertEqual(0, max_diff(cache, cache)["abs"])
        self.assertEqual(cache.layers[1].sliding_window, 12)


if __name__ == "__main__":
    unittest.main(verbosity=2)

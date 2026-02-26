import math
import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_torch, requires_transformers
from yobx.helpers import max_diff, string_diff


class TestMaxDiffNone(ExtTestCase):
    def test_both_none(self):
        res = max_diff(None, None)
        self.assertEqual(res["abs"], 0)
        self.assertEqual(res["rel"], 0)
        self.assertEqual(res["n"], 0)
        self.assertEqual(res["dnan"], 0)


class TestMaxDiffNdarray(ExtTestCase):
    def test_identical_arrays(self):
        a = np.array([1.0, 2.0, 3.0])
        res = max_diff(a, a.copy())
        self.assertEqual(res["abs"], 0.0)
        self.assertEqual(res["rel"], 0.0)
        self.assertEqual(res["dnan"], 0.0)

    def test_different_arrays(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 3.0])
        res = max_diff(a, b)
        self.assertEqual(res["abs"], 1.0)
        self.assertGreater(res["rel"], 0.0)
        self.assertEqual(res["n"], 2.0)

    def test_shape_mismatch(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0])
        res = max_diff(a, b)
        self.assertTrue(math.isinf(res["abs"]))

    def test_2d_shape_mismatch(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[1.0, 2.0]])
        res = max_diff(a, b)
        self.assertTrue(math.isinf(res["abs"]))

    def test_nan_identical(self):
        a = np.array([1.0, float("nan"), 3.0])
        res = max_diff(a, a.copy())
        self.assertEqual(res["dnan"], 0.0)

    def test_nan_different(self):
        a = np.array([1.0, float("nan"), 3.0])
        b = np.array([1.0, 2.0, 3.0])
        res = max_diff(a, b)
        self.assertGreater(res["dnan"], 0)

    def test_empty_array(self):
        a = np.array([]).reshape(0, 3)
        b = np.array([]).reshape(0, 3)
        res = max_diff(a, b)
        self.assertEqual(res["abs"], 0)
        self.assertEqual(res["n"], 0)

    def test_scalar_float_equal(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0])
        res = max_diff(a, b)
        self.assertEqual(res["abs"], 0.0)

    def test_complex64_identical(self):
        a = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        res = max_diff(a, a.copy())
        self.assertEqual(res["abs"], 0.0)

    def test_complex_vs_float(self):
        a = np.array([2 + 2j, 4 + 4j], dtype=np.complex64)
        b = np.array([1.0, 3.0], dtype=np.float32)
        res = max_diff(a, b)
        self.assertGreater(res["abs"], 0.0)

    def test_hist_bool(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.1])
        res = max_diff(a, b, hist=True)
        self.assertIn("rep", res)
        self.assertIsInstance(res["rep"], dict)

    def test_hist_list(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.5])
        res = max_diff(a, b, hist=[0, 0.1, 1.0])
        self.assertIn("rep", res)
        self.assertIn(">0", res["rep"])

    def test_begin_end(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 5.0])
        res = max_diff([a, b], [a, b], begin=0, end=1)
        self.assertEqual(res["abs"], 0)

    def test_argm_in_result(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 5.0])
        res = max_diff(a, b)
        self.assertIn("argm", res)
        self.assertEqual(res["argm"], (2,))

    def test_2d_identical(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        res = max_diff(a, a.copy())
        self.assertEqual(res["abs"], 0.0)

    def test_int64_array(self):
        a = np.array([1, 2, 3], dtype=np.int64)
        b = np.array([1, 2, 4], dtype=np.int64)
        res = max_diff(a, b)
        self.assertEqual(res["abs"], 1.0)


class TestMaxDiffListTuple(ExtTestCase):
    def test_list_identical(self):
        a = [np.array([1.0, 2.0]), np.array([3.0])]
        b = [np.array([1.0, 2.0]), np.array([3.0])]
        res = max_diff(a, b)
        self.assertEqual(res["abs"], 0)

    def test_list_different(self):
        a = [np.array([1.0]), np.array([2.0])]
        b = [np.array([1.0]), np.array([5.0])]
        res = max_diff(a, b)
        self.assertEqual(res["abs"], 3.0)

    def test_tuple_identical(self):
        a = (np.array([1.0, 2.0]), np.array([3.0]))
        b = (np.array([1.0, 2.0]), np.array([3.0]))
        res = max_diff(a, b)
        self.assertEqual(res["abs"], 0)

    def test_list_length_mismatch(self):
        a = [np.array([1.0]), np.array([2.0])]
        b = [np.array([1.0])]
        res = max_diff(a, b)
        self.assertTrue(math.isinf(res["abs"]))

    def test_list_single_element_vs_non_list(self):
        a = [np.array([1.0, 2.0])]
        b = np.array([1.0, 2.0])
        res = max_diff(a, b)
        self.assertEqual(res["abs"], 0)

    def test_list_vs_non_list(self):
        a = [np.array([1.0]), np.array([2.0])]
        b = np.array([1.0])
        res = max_diff(a, b)
        self.assertTrue(math.isinf(res["abs"]))

    def test_empty_lists(self):
        res = max_diff([], [])
        self.assertEqual(res["abs"], 0)
        self.assertEqual(res["n"], 0.0)

    def test_list_accumulates_max(self):
        a = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        b = [np.array([1.5, 2.0]), np.array([3.0, 6.0])]
        res = max_diff(a, b)
        self.assertEqual(res["abs"], 2.0)


class TestMaxDiffDict(ExtTestCase):
    def test_dict_identical(self):
        a = {"x": np.array([1.0, 2.0]), "y": np.array([3.0])}
        b = {"x": np.array([1.0, 2.0]), "y": np.array([3.0])}
        res = max_diff(a, b)
        self.assertEqual(res["abs"], 0)

    def test_dict_different_values(self):
        a = {"x": np.array([1.0]), "y": np.array([2.0])}
        b = {"x": np.array([1.0]), "y": np.array([5.0])}
        res = max_diff(a, b)
        self.assertEqual(res["abs"], 3.0)

    def test_dict_different_keys(self):
        a = {"x": np.array([1.0])}
        b = {"y": np.array([1.0])}
        res = max_diff(a, b)
        self.assertTrue(math.isinf(res["abs"]))

    def test_dict_different_size(self):
        a = {"x": np.array([1.0]), "y": np.array([2.0])}
        b = {"x": np.array([1.0])}
        res = max_diff(a, b)
        self.assertTrue(math.isinf(res["abs"]))

    def test_dict_vs_list(self):
        a = {"x": np.array([1.0]), "y": np.array([2.0])}
        b = [np.array([1.0]), np.array([2.0])]
        res = max_diff(a, b)
        self.assertEqual(res["abs"], 0)

    def test_dict_vs_non_list(self):
        a = {"x": np.array([1.0])}
        b = np.array([1.0])
        res = max_diff(a, b)
        self.assertTrue(math.isinf(res["abs"]))


class TestMaxDiffSkipNone(ExtTestCase):
    def test_skip_none_with_none_values(self):
        a = [np.array([1.0]), None]
        b = [np.array([1.0]), None]
        res = max_diff(a, b, skip_none=True)
        self.assertEqual(res["abs"], 0)
        self.assertEqual(res["n"], 1.0)

    def test_skip_none_false_raises(self):
        with self.assertRaises(Exception):  # noqa: B017
            max_diff(None, np.array([1.0]))


@requires_torch("2.9")
class TestMaxDiffTorch(ExtTestCase):
    def test_identical_tensors(self):
        import torch

        a = torch.tensor([1.0, 2.0, 3.0])
        res = max_diff(a, a.clone())
        self.assertEqual(res["abs"], 0.0)
        self.assertEqual(res["dnan"], 0.0)

    def test_different_tensors(self):
        import torch

        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([1.0, 3.0])
        res = max_diff(a, b)
        self.assertEqual(res["abs"], 1.0)

    def test_shape_mismatch(self):
        import torch

        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([1.0])
        res = max_diff(a, b)
        self.assertTrue(math.isinf(res["abs"]))

    def test_nan_identical(self):
        import torch

        a = torch.tensor([1.0, float("nan"), 3.0])
        res = max_diff(a, a.clone())
        self.assertEqual(res["dnan"], 0.0)

    def test_nan_different(self):
        import torch

        a = torch.tensor([1.0, float("nan"), 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        res = max_diff(a, b)
        self.assertGreater(res["dnan"], 0)

    def test_empty_tensor(self):
        import torch

        a = torch.empty(0, 3)
        b = torch.empty(0, 3)
        res = max_diff(a, b)
        self.assertEqual(res["abs"], 0.0)
        self.assertEqual(res["n"], 0.0)

    def test_hist_bool(self):
        import torch

        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([1.0, 2.1])
        res = max_diff(a, b, hist=True)
        self.assertIn("rep", res)
        self.assertIsInstance(res["rep"], dict)

    def test_hist_list(self):
        import torch

        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([1.0, 2.5])
        res = max_diff(a, b, hist=[0, 0.1, 1.0])
        self.assertIn("rep", res)

    def test_int_vs_scalar_tensor(self):
        import torch

        res = max_diff(5, torch.tensor(5))
        self.assertEqual(res["abs"], 0)

    def test_int_vs_non_scalar_tensor(self):
        import torch

        res = max_diff(5, torch.tensor([5, 6]))
        self.assertTrue(math.isinf(res["abs"]))

    def test_complex64_tensors(self):
        import torch

        a = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64)
        res = max_diff(a, a.clone())
        self.assertEqual(res["abs"], 0.0)

    def test_list_of_tensors(self):
        import torch

        a = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
        b = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
        res = max_diff(a, b)
        self.assertEqual(res["abs"], 0)

    def test_list_of_tensors_different(self):
        import torch

        a = [torch.tensor([1.0]), torch.tensor([2.0])]
        b = [torch.tensor([1.0]), torch.tensor([5.0])]
        res = max_diff(a, b)
        self.assertEqual(res["abs"], 3.0)

    def test_unique_tensor_in_list(self):
        import torch

        a = torch.tensor([1.0, 2.0])
        b = [torch.tensor([1.0, 2.0])]
        res = max_diff(a, b)
        self.assertEqual(res["abs"], 0.0)

    def test_begin_end_tensors(self):
        import torch

        a = [torch.tensor([1.0]), torch.tensor([2.0])]
        b = [torch.tensor([1.0]), torch.tensor([9.0])]
        res = max_diff(a, b, begin=0, end=1)
        self.assertEqual(res["abs"], 0)

    def test_dev_field_same_device(self):
        import torch

        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([1.0, 2.0])
        res = max_diff(a, b)
        self.assertIn("dev", res)
        self.assertEqual(res["dev"], 0)

    def test_argm_field(self):
        import torch

        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 7.0])
        res = max_diff(a, b)
        self.assertIn("argm", res)
        self.assertEqual(res["argm"], (2,))

    def test_list_hist_accumulated(self):
        import torch

        a = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
        b = [torch.tensor([1.0, 2.1]), torch.tensor([3.0])]
        res = max_diff(a, b, hist=True)
        self.assertIn("rep", res)
        self.assertIn(">0.0", res["rep"])


@requires_transformers("4.50")
class TestMaxDiffDynamicCache(ExtTestCase):
    def test_dynamic_cache_vs_dynamic_cache(self):
        import torch
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        key = torch.rand(1, 4, 2, 8)
        value = torch.rand(1, 4, 2, 8)
        cache.update(key, value, layer_idx=0)
        res = max_diff(cache, cache)
        self.assertEqual(res["abs"], 0.0)

    def test_dynamic_cache_vs_tuple(self):
        import torch
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        key = torch.rand(1, 4, 2, 8)
        value = torch.rand(1, 4, 2, 8)
        cache.update(key, value, layer_idx=0)
        res = max_diff(cache, ([key], [value]))
        self.assertEqual(res["abs"], 0.0)

    def test_static_cache_vs_static_cache(self):
        import torch
        from transformers import GPT2Config
        from transformers.cache_utils import StaticCache

        cfg = GPT2Config(n_head=4, n_embd=32)
        cache = StaticCache(config=cfg, max_cache_len=16)
        key = torch.rand(1, 4, 2, 8)
        value = torch.rand(1, 4, 2, 8)
        cache.update(key, value, layer_idx=0)
        res = max_diff(cache, cache)
        self.assertEqual(res["abs"], 0.0)


class TestStringDiff(ExtTestCase):
    @requires_torch("2.9")
    def test_identical_arrays(self):
        a = np.array([1.0, 2.0, 3.0])
        diff = max_diff(a, a.copy())
        s = string_diff(diff)
        self.assertIn("abs=0", s)
        self.assertIn("rel=0", s)

    @requires_torch("2.9")
    def test_different_arrays(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 3.0])
        diff = max_diff(a, b)
        s = string_diff(diff)
        self.assertIn("abs=", s)
        self.assertIn("rel=", s)
        self.assertIn("n=", s)

    @requires_torch("2.9")
    def test_with_nan(self):
        a = np.array([1.0, float("nan"), 3.0])
        b = np.array([1.0, 2.0, 3.0])
        diff = max_diff(a, b)
        s = string_diff(diff)
        self.assertIn("dnan=", s)

    @requires_torch("2.9")
    def test_js_format(self):
        import json

        a = np.array([1.0, 2.0])
        b = np.array([1.0, 3.0])
        diff = max_diff(a, b)
        s = string_diff(diff, js=True)
        obj = json.loads(s)
        self.assertIn("abs", obj)
        self.assertIn("rel", obj)

    @requires_torch("2.9")
    def test_js_format_with_kwargs(self):
        import json

        a = np.array([1.0, 2.0])
        b = np.array([1.0, 3.0])
        diff = max_diff(a, b, hist=True)
        s = string_diff(diff, js=True, tag="test")
        obj = json.loads(s)
        self.assertEqual(obj["tag"], "test")

    @requires_torch("2.9")
    def test_with_hist(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.1])
        diff = max_diff(a, b, hist=True)
        s = string_diff(diff)
        self.assertIn("abs=", s)

    @requires_torch("2.9")
    def test_virtual_tensor_string_type(self):
        from yobx.helpers import string_type

        class VirtualTensor:
            def __init__(self, name, dtype, shape):
                self.name = name
                self.dtype = dtype
                self.shape = shape

        vt = VirtualTensor("x", "float32", (2, 3))
        s = string_type(vt)
        self.assertIn("VirtualTensor", s)
        self.assertIn("name='x'", s)
        self.assertIn("float32", s)

    @requires_torch("2.9")
    def test_with_argm(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 5.0])
        diff = max_diff(a, b)
        s = string_diff(diff)
        self.assertIn("amax=", s)

    @requires_torch("2.9")
    def test_with_dev_field(self):
        import torch

        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([1.0, 2.0])
        diff = max_diff(a, b)
        s = string_diff(diff)
        self.assertIn("dev=", s)

    @requires_torch("2.9")
    def test_js_with_rep(self):
        import json

        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.1])
        diff = max_diff(a, b, hist=True)
        s = string_diff(diff, js=True, ratio=True)
        obj = json.loads(s)
        self.assertIn("mean", obj)


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase
from yobx.helpers import string_type

try:
    import torch as _torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import transformers as _transformers  # noqa: F401

    HAS_TRANSFORMERS = HAS_TORCH
except ImportError:
    HAS_TRANSFORMERS = False


class TestStringType(ExtTestCase):
    def test_none(self):
        self.assertEqual(string_type(None), "None")

    def test_empty_tuple(self):
        self.assertEqual(string_type(()), "()")

    def test_single_element_tuple(self):
        self.assertEqual(string_type((None,)), "(None,)")

    def test_tuple(self):
        self.assertEqual(string_type((None, None)), "(None,None)")

    def test_large_tuple(self):
        large = tuple(None for _ in range(25))
        self.assertIn("#25(None,...)", string_type(large))

    def test_empty_list(self):
        self.assertEqual(string_type([]), "#0[]")

    def test_list(self):
        self.assertEqual(string_type([None, None]), "#2[None,None]")

    def test_large_list(self):
        large = [None for _ in range(25)]
        self.assertIn("#25[None,...]", string_type(large))

    def test_empty_dict(self):
        self.assertEqual(string_type({}), "{}")

    def test_ndarray_no_shape(self):
        arr = np.array([1.0, 2.0, 3.0])
        s = string_type(arr)
        self.assertIn("A", s)
        self.assertIn("r1", s)

    def test_ndarray_with_shape(self):
        arr = np.array([1.0, 2.0, 3.0])
        s = string_type(arr, with_shape=True)
        self.assertIn("s3", s)

    def test_ndarray_2d_with_shape(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        s = string_type(arr, with_shape=True)
        self.assertIn("s2x2", s)

    def test_ndarray_int64_with_shape(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        s = string_type(arr, with_shape=True)
        self.assertIn("A7s3", s)

    def test_ndarray_with_min_max(self):
        arr = np.array([1.0, 2.0, 3.0])
        s = string_type(arr, with_shape=True, with_min_max=True)
        self.assertIn("[1.0,3.0", s)

    def test_ndarray_empty_with_min_max(self):
        arr = np.array([]).reshape(0, 3)
        s = string_type(arr, with_shape=True, with_min_max=True)
        self.assertIn("[empty]", s)

    def test_ndarray_scalar_with_min_max(self):
        arr = np.array(5.0)
        s = string_type(arr, with_min_max=True)
        self.assertIn("=5.0", s)

    def test_ndarray_with_nans(self):
        arr = np.array([1.0, float("nan"), 3.0])
        s = string_type(arr, with_shape=True, with_min_max=True)
        self.assertIn("N1nans", s)

    def test_ndarray_all_nans(self):
        arr = np.array([float("nan"), float("nan")])
        s = string_type(arr, with_shape=True, with_min_max=True)
        self.assertIn("N2nans", s)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_bool(self):
        self.assertEqual(string_type(True), "bool")

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_bool_with_min_max(self):
        s = string_type(True, with_min_max=True)
        self.assertIn("bool=", s)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_int(self):
        self.assertEqual(string_type(1), "int")

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_int_with_min_max(self):
        self.assertEqual(string_type(5, with_min_max=True), "int=5")

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_float(self):
        self.assertEqual(string_type(1.0), "float")

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_float_with_min_max(self):
        s = string_type(3.14, with_min_max=True)
        self.assertIn("float=", s)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_str(self):
        self.assertEqual(string_type("hello"), "str")

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_slice(self):
        self.assertEqual(string_type(slice(1, 5)), "slice")

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_torch_tensor(self):
        import torch

        t = torch.rand(3, 4)
        s = string_type(t)
        self.assertIn("T", s)
        self.assertIn("r2", s)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_torch_tensor_with_shape(self):
        import torch

        t = torch.rand(3, 4)
        s = string_type(t, with_shape=True)
        self.assertIn("s3x4", s)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_torch_tensor_with_min_max(self):
        import torch

        t = torch.tensor([1.0, 2.0, 3.0])
        s = string_type(t, with_shape=True, with_min_max=True)
        self.assertIn("1.0", s)
        self.assertIn("3.0", s)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_dict_with_tensors(self):
        import torch

        d = {"x": torch.rand(2, 3)}
        s = string_type(d, with_shape=True)
        self.assertIn("x:", s)
        self.assertIn("s2x3", s)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_tuple_with_min_max_ints(self):
        large = tuple(range(25))
        s = string_type(large, with_min_max=True)
        self.assertIn("#25(", s)
        self.assertIn("[0,24", s)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_list_with_min_max_ints(self):
        large = list(range(25))
        s = string_type(large, with_min_max=True)
        self.assertIn("#25[", s)
        self.assertIn("[0,24", s)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_set_small(self):
        s = string_type({1, 2, 3})
        self.assertIn("{", s)
        self.assertIn("}", s)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_set_large_with_min_max(self):
        large_set = set(range(15))
        s = string_type(large_set, with_min_max=True)
        self.assertIn("#15", s)

    @unittest.skipUnless(HAS_TRANSFORMERS, "transformers or torch not installed")
    def test_dynamic_cache(self):
        import torch
        from transformers.cache_utils import DynamicCache

        dc = DynamicCache()
        key = torch.rand(1, 4, 2, 8)
        value = torch.rand(1, 4, 2, 8)
        dc.update(key, value, layer_idx=0)
        s = string_type(dc)
        self.assertIn("DynamicCache", s)
        self.assertIn("key_cache=", s)
        self.assertIn("value_cache=", s)

    @unittest.skipUnless(HAS_TRANSFORMERS, "transformers or torch not installed")
    def test_dynamic_cache_with_shape(self):
        import torch
        from transformers.cache_utils import DynamicCache

        dc = DynamicCache()
        key = torch.rand(1, 4, 2, 8)
        value = torch.rand(1, 4, 2, 8)
        dc.update(key, value, layer_idx=0)
        s = string_type(dc, with_shape=True)
        self.assertIn("DynamicCache", s)
        self.assertIn("1x4x2x8", s)

    @unittest.skipUnless(HAS_TRANSFORMERS, "transformers or torch not installed")
    def test_dynamic_layer(self):
        import torch
        from transformers.cache_utils import DynamicCache

        dc = DynamicCache()
        key = torch.rand(1, 4, 2, 8)
        value = torch.rand(1, 4, 2, 8)
        dc.update(key, value, layer_idx=0)
        dl = dc.layers[0]
        s = string_type(dl)
        self.assertIn("DynamicLayer", s)
        self.assertIn("keys=", s)
        self.assertIn("values=", s)

    @unittest.skipUnless(HAS_TRANSFORMERS, "transformers or torch not installed")
    def test_dynamic_layer_with_shape(self):
        import torch
        from transformers.cache_utils import DynamicCache

        dc = DynamicCache()
        key = torch.rand(1, 4, 2, 8)
        value = torch.rand(1, 4, 2, 8)
        dc.update(key, value, layer_idx=0)
        dl = dc.layers[0]
        s = string_type(dl, with_shape=True)
        self.assertIn("DynamicLayer", s)
        self.assertIn("1x4x2x8", s)

    @unittest.skipUnless(HAS_TRANSFORMERS, "transformers or torch not installed")
    def test_static_cache(self):
        import torch
        from transformers import GPT2Config
        from transformers.cache_utils import StaticCache

        cfg = GPT2Config(n_head=4, n_embd=32)
        sc = StaticCache(config=cfg, max_cache_len=16)
        key = torch.rand(1, 4, 2, 8)
        value = torch.rand(1, 4, 2, 8)
        sc.update(key, value, layer_idx=0)
        s = string_type(sc)
        self.assertIn("StaticCache", s)
        self.assertIn("key_cache=", s)
        self.assertIn("value_cache=", s)

    @unittest.skipUnless(HAS_TRANSFORMERS, "transformers or torch not installed")
    def test_static_cache_with_shape(self):
        import torch
        from transformers import GPT2Config
        from transformers.cache_utils import StaticCache

        cfg = GPT2Config(n_head=4, n_embd=32)
        # max_cache_len=16 pre-allocates tensors of length 16 regardless of input seq len
        sc = StaticCache(config=cfg, max_cache_len=16)
        key = torch.rand(1, 4, 2, 8)
        value = torch.rand(1, 4, 2, 8)
        sc.update(key, value, layer_idx=0)
        s = string_type(sc, with_shape=True)
        self.assertIn("StaticCache", s)
        # StaticCache pre-allocates to max_cache_len, so seq dim is 16 not 2
        self.assertIn("1x4x16x8", s)

    @unittest.skipUnless(HAS_TRANSFORMERS, "transformers or torch not installed")
    def test_static_layer(self):
        import torch
        from transformers import GPT2Config
        from transformers.cache_utils import StaticCache

        cfg = GPT2Config(n_head=4, n_embd=32)
        sc = StaticCache(config=cfg, max_cache_len=16)
        key = torch.rand(1, 4, 2, 8)
        value = torch.rand(1, 4, 2, 8)
        sc.update(key, value, layer_idx=0)
        sl = sc.layers[0]
        s = string_type(sl)
        self.assertIn("StaticLayer", s)
        self.assertIn("keys=", s)
        self.assertIn("values=", s)

    @unittest.skipUnless(HAS_TRANSFORMERS, "transformers or torch not installed")
    def test_static_layer_with_shape(self):
        import torch
        from transformers import GPT2Config
        from transformers.cache_utils import StaticCache

        cfg = GPT2Config(n_head=4, n_embd=32)
        sc = StaticCache(config=cfg, max_cache_len=16)
        key = torch.rand(1, 4, 2, 8)
        value = torch.rand(1, 4, 2, 8)
        sc.update(key, value, layer_idx=0)
        sl = sc.layers[0]
        s = string_type(sl, with_shape=True)
        self.assertIn("StaticLayer", s)
        # StaticLayer.keys has shape [batch, heads, max_cache_len, head_dim];
        # list() iterates over batch dim, exposing [heads, max_cache_len, head_dim]
        self.assertIn("4x16x8", s)

    @unittest.skipUnless(HAS_TRANSFORMERS, "transformers or torch not installed")
    def test_encoder_decoder_cache(self):
        import torch
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache

        self_cache = DynamicCache()
        cross_cache = DynamicCache()
        key = torch.rand(1, 4, 2, 8)
        value = torch.rand(1, 4, 2, 8)
        self_cache.update(key, value, layer_idx=0)
        cross_cache.update(torch.rand(1, 4, 3, 8), torch.rand(1, 4, 3, 8), layer_idx=0)
        edc = EncoderDecoderCache(self_cache, cross_cache)
        s = string_type(edc)
        self.assertIn("EncoderDecoderCache", s)
        self.assertIn("self_attention_cache=", s)
        self.assertIn("cross_attention_cache=", s)

    @unittest.skipUnless(HAS_TRANSFORMERS, "transformers or torch not installed")
    def test_encoder_decoder_cache_with_shape(self):
        import torch
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache

        self_cache = DynamicCache()
        cross_cache = DynamicCache()
        key = torch.rand(1, 4, 2, 8)
        value = torch.rand(1, 4, 2, 8)
        self_cache.update(key, value, layer_idx=0)
        cross_cache.update(torch.rand(1, 4, 3, 8), torch.rand(1, 4, 3, 8), layer_idx=0)
        edc = EncoderDecoderCache(self_cache, cross_cache)
        s = string_type(edc, with_shape=True)
        self.assertIn("EncoderDecoderCache", s)
        # self-attention uses seq len 2, cross-attention uses seq len 3
        self.assertIn("1x4x2x8", s)
        self.assertIn("1x4x3x8", s)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_string_tensor_no_shape(self):
        import torch
        from yobx.helpers.helper import _string_tensor

        t = torch.rand(3, 4)
        s = _string_tensor(t, "T", with_shape=False, with_device=False, verbose=0)
        self.assertIn("T", s)
        self.assertIn("r2", s)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_string_tensor_with_shape(self):
        import torch
        from yobx.helpers.helper import _string_tensor

        t = torch.rand(3, 4)
        s = _string_tensor(t, "T", with_shape=True, with_device=False, verbose=0)
        self.assertIn("T", s)
        self.assertIn("s3x4", s)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_string_tensor_with_device_cpu(self):
        import torch
        from yobx.helpers.helper import _string_tensor

        t = torch.rand(3, 4)
        s = _string_tensor(t, "T", with_shape=False, with_device=True, verbose=0)
        self.assertIn("C", s)
        self.assertIn("r2", s)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_string_tensor_custom_cls(self):
        import torch
        from yobx.helpers.helper import _string_tensor

        t = torch.rand(2, 5)
        s = _string_tensor(t, "F", with_shape=True, with_device=False, verbose=0)
        self.assertIn("F", s)
        self.assertIn("s2x5", s)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_string_tensor_verbose(self):
        import torch
        from yobx.helpers.helper import _string_tensor

        t = torch.rand(3, 4)
        s = _string_tensor(t, "T", with_shape=False, with_device=False, verbose=1)
        self.assertIn("r2", s)


if __name__ == "__main__":
    unittest.main(verbosity=2)

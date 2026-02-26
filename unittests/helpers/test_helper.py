import inspect
import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, hide_stdout, requires_torch, requires_transformers
from yobx.helpers import string_type, string_sig, string_signature
from yobx.helpers.helper import flatten_object


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

    @requires_torch("2.9")
    def test_bool(self):
        self.assertEqual(string_type(True), "bool")

    @requires_torch("2.9")
    def test_bool_with_min_max(self):
        s = string_type(True, with_min_max=True)
        self.assertIn("bool=", s)

    @requires_torch("2.9")
    def test_int(self):
        self.assertEqual(string_type(1), "int")

    @requires_torch("2.9")
    def test_int_with_min_max(self):
        self.assertEqual(string_type(5, with_min_max=True), "int=5")

    @requires_torch("2.9")
    def test_float(self):
        self.assertEqual(string_type(1.0), "float")

    @requires_torch("2.9")
    def test_float_with_min_max(self):
        s = string_type(3.14, with_min_max=True)
        self.assertIn("float=", s)

    @requires_torch("2.9")
    def test_str(self):
        self.assertEqual(string_type("hello"), "str")

    @requires_torch("2.9")
    def test_slice(self):
        self.assertEqual(string_type(slice(1, 5)), "slice")

    @requires_torch("2.9")
    def test_torch_tensor(self):
        import torch

        t = torch.rand(3, 4)
        s = string_type(t)
        self.assertIn("T", s)
        self.assertIn("r2", s)

    @requires_torch("2.9")
    def test_torch_tensor_with_shape(self):
        import torch

        t = torch.rand(3, 4)
        s = string_type(t, with_shape=True)
        self.assertIn("s3x4", s)

    @requires_torch("2.9")
    def test_torch_tensor_with_min_max(self):
        import torch

        t = torch.tensor([1.0, 2.0, 3.0])
        s = string_type(t, with_shape=True, with_min_max=True)
        self.assertIn("1.0", s)
        self.assertIn("3.0", s)

    @requires_torch("2.9")
    def test_dict_with_tensors(self):
        import torch

        d = {"x": torch.rand(2, 3)}
        s = string_type(d, with_shape=True)
        self.assertIn("x:", s)
        self.assertIn("s2x3", s)

    @requires_torch("2.9")
    def test_tuple_with_min_max_ints(self):
        large = tuple(range(25))
        s = string_type(large, with_min_max=True)
        self.assertIn("#25(", s)
        self.assertIn("[0,24", s)

    @requires_torch("2.9")
    def test_list_with_min_max_ints(self):
        large = list(range(25))
        s = string_type(large, with_min_max=True)
        self.assertIn("#25[", s)
        self.assertIn("[0,24", s)

    @requires_torch("2.9")
    def test_set_small(self):
        s = string_type({1, 2, 3})
        self.assertIn("{", s)
        self.assertIn("}", s)

    @requires_torch("2.9")
    def test_set_large_with_min_max(self):
        large_set = set(range(15))
        s = string_type(large_set, with_min_max=True)
        self.assertIn("#15", s)

    @requires_transformers("4.50")
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

    @requires_transformers("4.50")
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

    @requires_transformers("4.50")
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

    @requires_transformers("4.50")
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

    @requires_transformers("4.50")
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

    @requires_transformers("4.50")
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

    @requires_transformers("4.50")
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

    @requires_transformers("4.50")
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

    @requires_transformers("4.50")
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

    @requires_transformers("4.50")
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

    @requires_torch("2.9")
    def test_dim(self):
        import torch

        d = torch.export.Dim("batch", min=2, max=10)
        s = string_type(d)
        self.assertEqual(s, "Dim(batch)")

    @requires_torch("2.9")
    def test_derived_dim(self):
        import torch

        d = torch.export.Dim("batch", min=2, max=10)
        dd = d * 2
        s = string_type(dd)
        self.assertEqual(s, "DerivedDim")

    @requires_torch("2.9")
    def test_dim_hint_dynamic(self):
        import torch

        s = string_type(torch.export.Dim.DYNAMIC)
        self.assertEqual(s, "DYNAMIC")

    @requires_torch("2.9")
    def test_dim_hint_auto(self):
        import torch

        s = string_type(torch.export.Dim.AUTO)
        self.assertEqual(s, "AUTO")

    @requires_torch("2.9")
    def test_dataclass(self):
        from dataclasses import dataclass

        @dataclass
        class MyData:
            x: int
            y: float

        obj = MyData(x=1, y=2.0)
        s = string_type(obj)
        self.assertIn("MyData", s)
        self.assertIn("x:int", s)
        self.assertIn("y:float", s)

    @requires_torch("2.9")
    def test_dataclass_with_tensor(self):
        import torch
        from dataclasses import dataclass

        @dataclass
        class TensorData:
            t: object

        obj = TensorData(t=torch.rand(2, 3))
        s = string_type(obj, with_shape=True)
        self.assertIn("TensorData", s)
        self.assertIn("s2x3", s)

    @requires_torch("2.9")
    def test_torch_tensor_scalar_with_min_max(self):
        import torch

        t = torch.tensor(3.14)
        s = string_type(t, with_min_max=True)
        self.assertIn("=", s)

    @requires_torch("2.9")
    def test_torch_tensor_empty_with_min_max(self):
        import torch

        t = torch.empty(0, 3)
        s = string_type(t, with_shape=True, with_min_max=True)
        self.assertIn("[empty]", s)

    @requires_torch("2.9")
    def test_torch_tensor_with_nans_with_min_max(self):
        import torch

        t = torch.tensor([1.0, float("nan"), 3.0])
        s = string_type(t, with_shape=True, with_min_max=True)
        self.assertIn("N1nans", s)

    @requires_torch("2.9")
    def test_torch_tensor_float16(self):
        import torch

        t = torch.rand(2, 4, dtype=torch.float16)
        s = string_type(t, with_shape=True)
        self.assertIn("s2x4", s)

    @requires_torch("2.9")
    def test_torch_tensor_int64(self):
        import torch

        t = torch.arange(6, dtype=torch.int64).reshape(2, 3)
        s = string_type(t, with_shape=True)
        self.assertIn("T7s2x3", s)

    @requires_torch("2.9")
    def test_with_min_max_int(self):
        s = string_type(42, with_min_max=True)
        self.assertEqual(s, "int=42")

    @requires_torch("2.9")
    def test_with_min_max_float(self):
        s = string_type(1.5, with_min_max=True)
        self.assertEqual(s, "float=1.5")

    @requires_torch("2.9")
    def test_with_min_max_bool_false(self):
        s = string_type(False, with_min_max=True)
        self.assertIn("bool=", s)

    def test_string_tensor_no_shape(self):
        import torch
        from yobx.helpers.helper import _string_tensor

        t = torch.rand(3, 4)
        s = _string_tensor(t, "T", with_shape=False, with_device=False, verbose=0)
        self.assertIn("T", s)
        self.assertIn("r2", s)

    @requires_torch("2.9")
    def test_string_tensor_with_shape(self):
        import torch
        from yobx.helpers.helper import _string_tensor

        t = torch.rand(3, 4)
        s = _string_tensor(t, "T", with_shape=True, with_device=False, verbose=0)
        self.assertIn("T", s)
        self.assertIn("s3x4", s)

    @requires_torch("2.9")
    def test_string_tensor_with_device_cpu(self):
        import torch
        from yobx.helpers.helper import _string_tensor

        t = torch.rand(3, 4)
        s = _string_tensor(t, "T", with_shape=False, with_device=True, verbose=0)
        self.assertIn("C", s)
        self.assertIn("r2", s)

    @requires_torch("2.9")
    def test_string_tensor_custom_cls(self):
        import torch
        from yobx.helpers.helper import _string_tensor

        t = torch.rand(2, 5)
        s = _string_tensor(t, "F", with_shape=True, with_device=False, verbose=0)
        self.assertIn("F", s)
        self.assertIn("s2x5", s)

    @hide_stdout()
    @requires_torch("2.9")
    def test_string_tensor_verbose(self):
        import torch
        from yobx.helpers.helper import _string_tensor

        t = torch.rand(3, 4)
        s = _string_tensor(t, "T", with_shape=False, with_device=False, verbose=1)
        self.assertIn("r2", s)

    @hide_stdout()
    def test_string_type_verbose_none(self):
        s = string_type(None, verbose=1)
        self.assertEqual(s, "None")

    @hide_stdout()
    @requires_torch("2.9")
    def test_string_type_verbose_tuple(self):
        s = string_type((1, 2, 3), verbose=1)
        self.assertIn("int", s)

    @hide_stdout()
    @requires_torch("2.9")
    def test_string_type_verbose_list(self):
        s = string_type([1, 2, 3], verbose=1)
        self.assertIn("int", s)

    @hide_stdout()
    @requires_torch("2.9")
    def test_string_type_verbose_dict(self):
        s = string_type({"a": 1}, verbose=1)
        self.assertIn("a:", s)

    @hide_stdout()
    def test_string_type_verbose_ndarray(self):
        arr = np.array([1.0, 2.0, 3.0])
        s = string_type(arr, with_shape=True, verbose=1)
        self.assertIn("s3", s)


class TestStringSignature(ExtTestCase):
    def test_simple_function(self):
        def foo(a, b):
            pass

        sig = inspect.signature(foo)
        s = string_signature(sig)
        self.assertIn("__call__", s)
        self.assertIn("a", s)
        self.assertIn("b", s)

    def test_function_with_annotation(self):
        def foo(a: int, b: str) -> float:
            pass

        sig = inspect.signature(foo)
        s = string_signature(sig)
        self.assertIn("__call__", s)
        self.assertIn("int", s)
        self.assertIn("str", s)
        self.assertIn("float", s)

    def test_function_with_default(self):
        def foo(a, b=5):
            pass

        sig = inspect.signature(foo)
        s = string_signature(sig)
        self.assertIn("b = 5", s)

    def test_function_no_return_annotation(self):
        def foo(x):
            pass

        sig = inspect.signature(foo)
        s = string_signature(sig)
        self.assertIn("__call__", s)
        self.assertNotIn("->", s)

    def test_function_with_return_annotation(self):
        def foo(x) -> int:
            pass

        sig = inspect.signature(foo)
        s = string_signature(sig)
        self.assertIn("-> <class 'int'>", s)


class TestStringSig(ExtTestCase):
    def test_function_no_kwargs(self):
        def foo(a, b=2):
            pass

        s = string_sig(foo, {})
        self.assertEqual(s, "foo()")

    def test_function_kwargs_differ_from_default(self):
        def foo(a=1, b=2):
            pass

        s = string_sig(foo, {"b": 99})
        self.assertIn("b=99", s)
        self.assertNotIn("a=", s)

    def test_function_kwargs_same_as_default(self):
        def foo(a=1, b=2):
            pass

        s = string_sig(foo, {"a": 1, "b": 2})
        self.assertEqual(s, "foo()")

    def test_function_no_default_in_kwargs(self):
        def foo(a, b=2):
            pass

        s = string_sig(foo, {"a": 10})
        self.assertIn("a=10", s)

    def test_object_with_init(self):
        class MyObj:
            def __init__(self, x=1, y=2):
                self.x = x
                self.y = y

        obj = MyObj(x=1, y=99)
        s = string_sig(obj)
        self.assertIn("MyObj", s)
        self.assertIn("y=99", s)
        self.assertNotIn("x=", s)

    def test_object_all_defaults(self):
        class MyObj:
            def __init__(self, x=1, y=2):
                self.x = x
                self.y = y

        obj = MyObj()
        s = string_sig(obj)
        self.assertEqual(s, "MyObj()")


class TestFlattenObject(ExtTestCase):
    def test_none(self):
        self.assertIsNone(flatten_object(None))

    def test_empty_list(self):
        self.assertEqual(flatten_object([]), [])

    def test_empty_tuple(self):
        self.assertEqual(flatten_object(()), ())

    def test_list_of_primitives(self):
        self.assertEqual(flatten_object([1, 2.0, "a"]), [1, 2.0, "a"])

    def test_tuple_of_primitives(self):
        self.assertEqual(flatten_object((1, 2, 3)), (1, 2, 3))

    def test_nested_list(self):
        self.assertEqual(flatten_object([[1, 2], [3, 4]]), [1, 2, 3, 4])

    def test_nested_tuple(self):
        self.assertEqual(flatten_object(((1, 2), (3, 4))), (1, 2, 3, 4))

    def test_list_with_none(self):
        self.assertEqual(flatten_object([None, 1, None]), [None, 1, None])

    def test_numpy_array_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = flatten_object(arr)
        self.assertIs(result, arr)

    def test_list_with_numpy_array(self):
        arr = np.array([1.0, 2.0])
        result = flatten_object([arr, 1])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], arr)
        self.assertEqual(result[1], 1)

    def test_dict_keep_keys(self):
        arr1 = np.array([1.0])
        arr2 = np.array([2.0])
        result = flatten_object({"a": arr1, "b": arr2}, drop_keys=False)
        # dict items are flattened as (key, value) pairs
        self.assertIn("a", result)
        self.assertIn("b", result)
        self.assertIn(arr1, result)
        self.assertIn(arr2, result)

    def test_dict_drop_keys(self):
        arr1 = np.array([1.0])
        arr2 = np.array([2.0])
        result = flatten_object({"a": arr1, "b": arr2}, drop_keys=True)
        # only values are kept, no string keys
        self.assertNotIn("a", result)
        self.assertNotIn("b", result)
        self.assertIn(arr1, result)
        self.assertIn(arr2, result)

    def test_object_with_to_tuple(self):
        class MyObj:
            def to_tuple(self):
                return (1, 2.0, "x")

        result = flatten_object(MyObj())
        self.assertEqual(result, (1, 2.0, "x"))

    def test_object_with_shape(self):
        arr = np.zeros((2, 3))
        result = flatten_object(arr)
        self.assertIs(result, arr)

    @requires_torch("2.9")
    def test_unsupported_type_raises(self):
        with self.assertRaises(TypeError):
            flatten_object(object())

    @requires_torch("2.9")
    def test_torch_tensor_passthrough(self):
        import torch

        t = torch.rand(2, 3)
        result = flatten_object(t)
        self.assertIs(result, t)

    @requires_torch("2.9")
    def test_list_of_torch_tensors(self):
        import torch

        t1 = torch.rand(2, 3)
        t2 = torch.rand(4, 5)
        result = flatten_object([t1, t2])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], t1)
        self.assertIs(result[1], t2)

    @requires_torch("2.9")
    def test_nested_list_of_torch_tensors(self):
        import torch

        t1 = torch.rand(2)
        t2 = torch.rand(3)
        result = flatten_object([[t1], [t2]])
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], t1)
        self.assertIs(result[1], t2)

    @requires_transformers("4.50")
    def test_dynamic_cache(self):
        import torch
        from transformers.cache_utils import DynamicCache

        dc = DynamicCache()
        key = torch.rand(1, 4, 2, 8)
        value = torch.rand(1, 4, 2, 8)
        dc.update(key, value, layer_idx=0)
        result = flatten_object(dc)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    @requires_transformers("4.50")
    def test_encoder_decoder_cache(self):
        import torch
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache

        self_cache = DynamicCache()
        cross_cache = DynamicCache()
        self_cache.update(torch.rand(1, 4, 2, 8), torch.rand(1, 4, 2, 8), layer_idx=0)
        cross_cache.update(torch.rand(1, 4, 3, 8), torch.rand(1, 4, 3, 8), layer_idx=0)
        edc = EncoderDecoderCache(self_cache, cross_cache)
        result = flatten_object(edc)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase
from yobx.helpers import string_type

try:
    import torch as _torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


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

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_dim(self):
        import torch

        d = torch.export.Dim("batch", min=2, max=10)
        s = string_type(d)
        self.assertEqual(s, "Dim(batch)")

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_derived_dim(self):
        import torch

        d = torch.export.Dim("batch", min=2, max=10)
        dd = d * 2
        s = string_type(dd)
        self.assertEqual(s, "DerivedDim")

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_dim_hint_dynamic(self):
        import torch

        s = string_type(torch.export.Dim.DYNAMIC)
        self.assertEqual(s, "DYNAMIC")

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_dim_hint_auto(self):
        import torch

        s = string_type(torch.export.Dim.AUTO)
        self.assertEqual(s, "AUTO")

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
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

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
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

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_torch_tensor_scalar_with_min_max(self):
        import torch

        t = torch.tensor(3.14)
        s = string_type(t, with_min_max=True)
        self.assertIn("=", s)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_torch_tensor_empty_with_min_max(self):
        import torch

        t = torch.empty(0, 3)
        s = string_type(t, with_shape=True, with_min_max=True)
        self.assertIn("[empty]", s)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_torch_tensor_with_nans_with_min_max(self):
        import torch

        t = torch.tensor([1.0, float("nan"), 3.0])
        s = string_type(t, with_shape=True, with_min_max=True)
        self.assertIn("N1nans", s)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_torch_tensor_float16(self):
        import torch

        t = torch.rand(2, 4, dtype=torch.float16)
        s = string_type(t, with_shape=True)
        self.assertIn("s2x4", s)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_torch_tensor_int64(self):
        import torch

        t = torch.arange(6, dtype=torch.int64).reshape(2, 3)
        s = string_type(t, with_shape=True)
        self.assertIn("T7s2x3", s)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_with_min_max_int(self):
        s = string_type(42, with_min_max=True)
        self.assertEqual(s, "int=42")

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_with_min_max_float(self):
        s = string_type(1.5, with_min_max=True)
        self.assertEqual(s, "float=1.5")

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_with_min_max_bool_false(self):
        s = string_type(False, with_min_max=True)
        self.assertIn("bool=", s)


if __name__ == "__main__":
    unittest.main(verbosity=2)

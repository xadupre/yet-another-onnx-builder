import unittest

import torch

from onnx_pipe.tracing.fake_tensor import (
    FakeTensorContext,
    make_fake,
    make_fake_with_dynamic_dimensions,
)


class TestFakeTensorContext(unittest.TestCase):
    def setUp(self):
        self.ctx = FakeTensorContext()

    def test_init_default(self):
        ctx = FakeTensorContext()
        self.assertIsNotNone(ctx.fake_mode)
        self.assertIsInstance(ctx._candidates, list)
        self.assertTrue(all(p >= 13 for p in ctx._candidates))

    def test_init_with_custom_fake_mode(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        ctx = FakeTensorContext(fake_mode=fake_mode)
        self.assertIs(ctx.fake_mode, fake_mode)

    def test_first_primes_returns_primes(self):
        primes = FakeTensorContext._first_primes(50)
        # All values should be prime and >= 13
        for p in primes:
            self.assertGreaterEqual(p, 13)
            self.assertTrue(all(p % i != 0 for i in range(2, p)))

    def test_first_primes_not_empty(self):
        primes = FakeTensorContext._first_primes()
        self.assertGreater(len(primes), 0)

    def test_from_tensor_returns_fake(self):
        from torch._subclasses.fake_tensor import FakeTensor

        t = torch.rand(2, 3)
        fake = self.ctx.from_tensor(t)
        self.assertIsInstance(fake, FakeTensor)
        self.assertEqual(len(fake.shape), 2)

    def test_make_fake_none(self):
        result = self.ctx.make_fake(None)
        self.assertIsNone(result)

    def test_make_fake_tensor(self):
        from torch._subclasses.fake_tensor import FakeTensor

        t = torch.rand(4, 5)
        result = self.ctx.make_fake(t)
        self.assertIsInstance(result, FakeTensor)

    def test_make_fake_list(self):
        from torch._subclasses.fake_tensor import FakeTensor

        tensors = [torch.rand(2, 3), torch.rand(4, 5)]
        result = self.ctx.make_fake(tensors)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for r in result:
            self.assertIsInstance(r, FakeTensor)

    def test_make_fake_tuple(self):
        from torch._subclasses.fake_tensor import FakeTensor

        tensors = (torch.rand(2, 3), torch.rand(4, 5))
        result = self.ctx.make_fake(tensors)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        for r in result:
            self.assertIsInstance(r, FakeTensor)

    def test_make_fake_dict(self):
        from torch._subclasses.fake_tensor import FakeTensor

        tensors = {"a": torch.rand(2, 3), "b": torch.rand(4, 5)}
        result = self.ctx.make_fake(tensors)
        self.assertIsInstance(result, dict)
        self.assertIn("a", result)
        self.assertIn("b", result)
        for r in result.values():
            self.assertIsInstance(r, FakeTensor)

    def test_make_fake_unsupported_type_raises(self):
        with self.assertRaises(TypeError):
            self.ctx.make_fake("not a tensor")

    def test_make_fake_with_dynamic_dimensions_none(self):
        result = self.ctx.make_fake_with_dynamic_dimensions(None, None)
        self.assertIsNone(result)

    def test_make_fake_with_dynamic_dimensions_tensor(self):
        from torch._subclasses.fake_tensor import FakeTensor

        t = torch.rand(2, 3, 4)
        result = self.ctx.make_fake_with_dynamic_dimensions(t, {0: "batch", 2: "seq"})
        self.assertIsInstance(result, FakeTensor)

    def test_make_fake_with_dynamic_dimensions_tuple(self):
        from torch._subclasses.fake_tensor import FakeTensor

        tensors = (torch.rand(2, 3), torch.rand(2, 5))
        result = self.ctx.make_fake_with_dynamic_dimensions(
            tensors,
            ({0: "batch"}, {0: "batch"}),
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        for r in result:
            self.assertIsInstance(r, FakeTensor)

    def test_make_fake_with_dynamic_dimensions_shared_dim_name(self):
        """Two tensors sharing the same dynamic dimension name should share the same symbolic dim."""
        t1 = torch.rand(4, 3)
        t2 = torch.rand(4, 5)
        ctx = FakeTensorContext()
        ft1 = ctx.make_fake_with_dynamic_dimensions(t1, {0: "batch"})
        ft2 = ctx.make_fake_with_dynamic_dimensions(t2, {0: "batch"})
        # Both batch dims should resolve to the same symbolic value
        self.assertEqual(ft1.shape[0], ft2.shape[0])

    def test_make_fake_with_dynamic_dimensions_scalar(self):
        result = self.ctx.make_fake_with_dynamic_dimensions(42, None)
        self.assertEqual(result, 42)

    def test_make_fake_with_dynamic_dimensions_dict(self):
        from torch._subclasses.fake_tensor import FakeTensor

        data = {"x": torch.rand(2, 3), "y": torch.rand(2, 5)}
        result = self.ctx.make_fake_with_dynamic_dimensions(
            data,
            {"x": {0: "batch"}, "y": {0: "batch"}},
        )
        self.assertIsInstance(result, dict)
        for v in result.values():
            self.assertIsInstance(v, FakeTensor)


class TestModuleLevelMakeFake(unittest.TestCase):
    def test_make_fake_none(self):
        result, ctx = make_fake(None)
        self.assertIsNone(result)
        self.assertIsNone(ctx)

    def test_make_fake_tensor(self):
        from torch._subclasses.fake_tensor import FakeTensor

        t = torch.rand(3, 4)
        result, ctx = make_fake(t)
        self.assertIsInstance(result, FakeTensor)
        self.assertIsInstance(ctx, FakeTensorContext)

    def test_make_fake_reuses_context(self):
        ctx = FakeTensorContext()
        t = torch.rand(3, 4)
        result, returned_ctx = make_fake(t, context=ctx)
        self.assertIs(returned_ctx, ctx)

    def test_make_fake_dict(self):
        from torch._subclasses.fake_tensor import FakeTensor

        data = {"a": torch.rand(2, 3), "b": torch.rand(4, 5)}
        result, ctx = make_fake(data)
        self.assertIsInstance(ctx, FakeTensorContext)
        for v in result.values():
            self.assertIsInstance(v, FakeTensor)


class TestModuleLevelMakeFakeWithDynamicDimensions(unittest.TestCase):
    def test_make_fake_with_dynamic_dimensions_none(self):
        result, ctx = make_fake_with_dynamic_dimensions(None, None)
        self.assertIsNone(result)
        self.assertIsNone(ctx)

    def test_make_fake_with_dynamic_dimensions_tensor(self):
        from torch._subclasses.fake_tensor import FakeTensor

        t = torch.rand(2, 3, 4)
        result, ctx = make_fake_with_dynamic_dimensions(t, {0: "batch", 2: "seq"})
        self.assertIsInstance(result, FakeTensor)
        self.assertIsInstance(ctx, FakeTensorContext)

    def test_make_fake_with_dynamic_dimensions_reuses_context(self):
        ctx = FakeTensorContext()
        t = torch.rand(2, 3)
        result, returned_ctx = make_fake_with_dynamic_dimensions(
            t, {0: "batch"}, context=ctx
        )
        self.assertIs(returned_ctx, ctx)

    def test_make_fake_with_dynamic_dimensions_tuple(self):
        from torch._subclasses.fake_tensor import FakeTensor

        tensors = (torch.rand(2, 3), torch.rand(2, 5))
        result, ctx = make_fake_with_dynamic_dimensions(
            tensors,
            ({0: "batch"}, {0: "batch"}),
        )
        self.assertIsInstance(result, tuple)
        for r in result:
            self.assertIsInstance(r, FakeTensor)


if __name__ == "__main__":
    unittest.main()

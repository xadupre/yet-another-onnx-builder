import unittest
from yobx.ext_test_case import ExtTestCase, ignore_warnings, skipif_ci_windows
from yobx.reference import ExtendedReferenceEvaluator
from yobx.torch import ExportOptions
from yobx.torch.interpreter import to_onnx


class TestOnnxExportControlFlow(ExtTestCase):
    @ignore_warnings((UserWarning, FutureWarning))
    @skipif_ci_windows("while_loop does not work")
    def test_while_loop_simple_backward(self):
        import torch

        class Simple(torch.nn.Module):
            def forward(self, ci, a, b):
                def cond_fn(i, x, y):
                    return i > 0

                def body_fn(i, x, y):
                    return i - 1, x + y, y - x

                return torch._higher_order_ops.while_loop(cond_fn, body_fn, [ci, a, b])

        example_inputs = torch.tensor(1), torch.randn(2, 3), torch.randn(2, 3)
        ep = torch.export.export(Simple(), example_inputs)
        expected = Simple()(*example_inputs)
        self.assertEqualAny(expected, ep.module()(*example_inputs))

    @ignore_warnings((UserWarning, FutureWarning))
    @skipif_ci_windows("while_loop does not work")
    def test_while_loop_simple_forward(self):
        import torch

        class Simple(torch.nn.Module):
            def forward(self, ci, a, b):
                def cond_fn(i, x, y):
                    return i < x.size(0)

                def body_fn(i, x, y):
                    return i + 1, x + y, y - x

                return torch._higher_order_ops.while_loop(cond_fn, body_fn, [ci, a, b])

        example_inputs = torch.tensor(0), torch.randn(2, 3), torch.randn(2, 3)
        ep = torch.export.export(Simple(), example_inputs)
        expected = Simple()(*example_inputs)
        self.assertEqualAny(expected, ep.module()(*example_inputs))

    @ignore_warnings((UserWarning, FutureWarning))
    @skipif_ci_windows("while_loop does not work")
    def test_while_loop_simple_forward2(self):
        import torch

        class Simple(torch.nn.Module):
            def forward(self, ci, a, b):
                def cond_fn(i, x, y):
                    return i < x.size(0)

                def body_fn(i, x, y):
                    return i + 1, x + y, y - x

                return torch.ops.higher_order.while_loop(cond_fn, body_fn, [ci, a, b], [])

        example_inputs = torch.tensor(0), torch.randn(2, 3), torch.randn(2, 3)
        ep = torch.export.export(Simple(), example_inputs)
        expected = Simple()(*example_inputs)
        self.assertEqualAny(expected, ep.module()(*example_inputs))

    @ignore_warnings((UserWarning, FutureWarning))
    @skipif_ci_windows("while_loop does not work")
    def test_while_repeat_interleave(self):
        import torch

        def cond(i, repeats, x, out):
            return i < repeats.size(0)

        def body(i, repeats, x, out):
            torch._check(i.item() >= 0)
            torch._check(i.item() < x.size(0))
            y = x[i : i + 1, :]
            new_shape = (2, -1)
            y = y.expand(new_shape)
            return i + 1, repeats.clone(), x.clone(), torch.cat([out, y.clone()], dim=0)

        class Model(torch.nn.Module):
            def forward(self, repeats, x):
                i = torch.tensor(0, dtype=torch.int64)
                return torch.ops.higher_order.while_loop(
                    cond, body, (i, repeats, x, torch.empty((0, x.shape[1]), dtype=x.dtype)), []
                )

        inputs = (
            torch.tensor([1, 2, 3], dtype=torch.int64),
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32),
        )
        model = Model()
        expected = model(*inputs)
        self.assertEqual(expected[-1].shape, (6, 3))
        # This while_loop does not work.
        # ep = torch.export.export(model, inputs, strict=True)
        # print(ep)

    @ignore_warnings((UserWarning, FutureWarning))
    @skipif_ci_windows("while_loop does not work")
    def test_while_loop_onnx_export_dec(self):
        import torch

        class Simple(torch.nn.Module):
            def forward(self, ci, a, b):
                def cond_fn(i, x, y):
                    return i > 0

                def body_fn(i, x, y):
                    return i - 1, x + y, y - x

                return torch._higher_order_ops.while_loop(cond_fn, body_fn, [ci, a, b])

        example_inputs = torch.tensor(2), torch.randn(2, 3), torch.randn(2, 3)
        model = Simple()
        expected = model(*example_inputs)

        for optimize in [False, True]:
            with self.subTest(optimize=optimize):
                onx = to_onnx(
                    model,
                    example_inputs,
                    optimize=optimize,
                    export_options=ExportOptions(strict=False),
                )
                self.dump_onnx(f"test_while_loop_onnx_export_dec_{optimize}.onnx", onx)

                ref = ExtendedReferenceEvaluator(onx)
                feeds = {
                    "ci": example_inputs[0].detach().numpy(),
                    "a": example_inputs[1].detach().numpy(),
                    "b": example_inputs[2].detach().numpy(),
                }
                got = ref.run(None, feeds)
                for e, g in zip(expected, got):
                    self.assertEqualArray(e, g, atol=1e-5)

    @ignore_warnings((UserWarning, FutureWarning))
    @skipif_ci_windows("while_loop does not work")
    def test_while_loop_onnx_export_inc(self):
        import torch

        class Simple(torch.nn.Module):
            def forward(self, ci, a, b):
                def cond_fn(i, x, y):
                    return i < x.size(0)

                def body_fn(i, x, y):
                    return i + 1, x + y, y - x

                return torch.ops.higher_order.while_loop(cond_fn, body_fn, [ci, a, b], [])

        example_inputs = torch.tensor(0), torch.randn(2, 3), torch.randn(2, 3)
        model = Simple()
        expected = model(*example_inputs)

        for optimize in [False, True]:
            with self.subTest(optimize=optimize):
                onx = to_onnx(
                    model,
                    example_inputs,
                    optimize=optimize,
                    export_options=ExportOptions(strict=False),
                )
                self.dump_onnx(f"test_while_loop_onnx_export_inc_{optimize}.onnx", onx)

                ref = ExtendedReferenceEvaluator(onx)
                feeds = {
                    "ci": example_inputs[0].detach().numpy(),
                    "a": example_inputs[1].detach().numpy(),
                    "b": example_inputs[2].detach().numpy(),
                }
                got = ref.run(None, feeds)
                for e, g in zip(expected, got):
                    self.assertEqualArray(e, g, atol=1e-5)

    @ignore_warnings((UserWarning, FutureWarning))
    @skipif_ci_windows("while_loop does not work")
    def test_while_loop_onnx_export_dec_new_tracing(self):
        import torch
        from yobx.torch.export_options import TracingMode

        class Simple(torch.nn.Module):
            def forward(self, ci, a, b):
                def cond_fn(i, x, y):
                    return i > 0

                def body_fn(i, x, y):
                    return i - 1, x + y, y - x

                return torch._higher_order_ops.while_loop(cond_fn, body_fn, [ci, a, b])

        example_inputs = torch.tensor(2), torch.randn(2, 3), torch.randn(2, 3)
        model = Simple()
        expected = model(*example_inputs)

        for optimize in [False, True]:
            with self.subTest(optimize=optimize):
                onx = to_onnx(
                    model,
                    example_inputs,
                    optimize=optimize,
                    export_options=ExportOptions(tracing=TracingMode.NEW_TRACING),
                )
                self.dump_onnx(
                    f"test_while_loop_onnx_export_dec_new_tracing_{optimize}.onnx", onx
                )

                ref = ExtendedReferenceEvaluator(onx)
                feeds = {
                    "ci": example_inputs[0].detach().numpy(),
                    "a": example_inputs[1].detach().numpy(),
                    "b": example_inputs[2].detach().numpy(),
                }
                got = ref.run(None, feeds)
                for e, g in zip(expected, got):
                    self.assertEqualArray(e, g, atol=1e-5)

    @ignore_warnings((UserWarning, FutureWarning))
    @skipif_ci_windows("while_loop does not work")
    def test_while_loop_onnx_export_inc_new_tracing(self):
        import torch
        from yobx.torch.export_options import TracingMode

        class Simple(torch.nn.Module):
            def forward(self, ci, a, b):
                def cond_fn(i, x, y):
                    return i < x.size(0)

                def body_fn(i, x, y):
                    return i + 1, x + y, y - x

                return torch.ops.higher_order.while_loop(cond_fn, body_fn, [ci, a, b], [])

        example_inputs = torch.tensor(0), torch.randn(2, 3), torch.randn(2, 3)
        model = Simple()
        expected = model(*example_inputs)

        for optimize in [False, True]:
            with self.subTest(optimize=optimize):
                onx = to_onnx(
                    model,
                    example_inputs,
                    optimize=optimize,
                    export_options=ExportOptions(tracing=TracingMode.NEW_TRACING),
                )
                self.dump_onnx(
                    f"test_while_loop_onnx_export_inc_new_tracing_{optimize}.onnx", onx
                )

                ref = ExtendedReferenceEvaluator(onx)
                feeds = {
                    "ci": example_inputs[0].detach().numpy(),
                    "a": example_inputs[1].detach().numpy(),
                    "b": example_inputs[2].detach().numpy(),
                }
                got = ref.run(None, feeds)
                for e, g in zip(expected, got):
                    self.assertEqualArray(e, g, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest
from yobx.ext_test_case import ExtTestCase, requires_torch, requires_cuda
from yobx.torch import to_onnx
from yobx.torch import apply_patches_for_model


class TestOnnxExportDevice(ExtTestCase):
    @requires_torch("2.12")
    def test_export_dynamic_shapes_kwargs(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, **kwargs):
                return kwargs["x"] + kwargs["y"]

        x, y = torch.randn(2, 3), torch.randn(2, 3)
        model = Model()
        model(x=x, y=y)
        ds = {
            "kwargs": {"x": {0: torch.export.Dim("batch")}, "y": {0: torch.export.Dim("batch")}}
        }
        with apply_patches_for_model(patch_torch=True, model=model):
            ep = torch.export.export(Model(), tuple(), kwargs={"x": x, "y": y}, dynamic_shapes=ds)
        self.assertNotEmpty(ep)

    def test_export_devices(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        x, y = torch.randn(2, 3), torch.randn(2, 3)
        ds = ({0: "batch"}, {0: "batch"})
        Model()(x=x, y=y)
        onx = to_onnx(Model(), (x, y), dynamic_shapes=ds, return_builder=True)
        self.assertNotEmpty(onx.builder._known_devices)
        self.assertEqual(set(onx.builder._known_devices.values()), {-1})

    @requires_cuda()
    def test_export_devices_cuda(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        x, y = torch.randn(2, 3).cuda(), torch.randn(2, 3).cuda()
        ds = ({0: "batch"}, {0: "batch"})
        Model()(x=x, y=y)
        onx = to_onnx(Model(), (x, y), dynamic_shapes=ds, return_builder=True)
        self.assertNotEmpty(onx.builder._known_devices)
        self.assertEqual(set(onx.builder._known_devices.values()), {0})


if __name__ == "__main__":
    unittest.main(verbosity=2)

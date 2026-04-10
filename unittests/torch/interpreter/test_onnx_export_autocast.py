"""Tests that verify yobx can export models that use ``torch.autocast``."""

import unittest
from yobx.ext_test_case import ExtTestCase, ignore_warnings, requires_torch, skipif_ci_windows
from yobx.torch.interpreter import to_onnx


@requires_torch("2.4")
class TestOnnxExportAutocast(ExtTestCase):
    """Tests for exporting models that use ``torch.autocast``."""

    @skipif_ci_windows("torch dynamo not supported on windows")
    @ignore_warnings((UserWarning, DeprecationWarning, FutureWarning))
    def test_autocast_disabled(self):
        """Exports a model whose forward uses ``torch.autocast(enabled=False)``."""
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                with torch.autocast(device_type="cpu", dtype=torch.float32, enabled=False):
                    return torch.mm(x, x.T)

        model = Model()
        x = torch.randn(3, 4)
        inputs = (x,)
        expected = model(x)
        onx = to_onnx(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    @skipif_ci_windows("torch dynamo not supported on windows")
    @ignore_warnings((UserWarning, DeprecationWarning, FutureWarning))
    def test_autocast_bfloat16_enabled(self):
        """Exports a model whose forward uses ``torch.autocast(enabled=True)`` with bfloat16.

        This exercises the ``aten_wrap_with_autocast`` path where ``enabled=True``
        and verifies that the resulting ONNX model runs without error and
        produces numerically close results.
        """
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    return torch.mm(x, x.T)

        model = Model()
        x = torch.randn(3, 4)
        inputs = (x,)
        expected = model(x).float()
        onx = to_onnx(model, inputs)
        # The ONNX output is bfloat16; cast to float32 for comparison.
        import numpy as np
        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)
        got_f32 = got[0].astype(np.float32)
        self.assertEqualArray(expected.detach().numpy(), got_f32, atol=1e-1)

    @skipif_ci_windows("torch dynamo not supported on windows")
    @ignore_warnings((UserWarning, DeprecationWarning, FutureWarning))
    def test_autocast_linear_bfloat16(self):
        """Exports a model with a Linear layer under ``torch.autocast(bfloat16)``."""
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 3)

            def forward(self, x):
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    return self.linear(x)

        model = Model()
        x = torch.randn(2, 4)
        inputs = (x,)
        expected = model(x).float()
        onx = to_onnx(model, inputs)
        import numpy as np
        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)
        got_f32 = got[0].astype(np.float32)
        self.assertEqualArray(expected.detach().numpy(), got_f32, atol=1e-1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

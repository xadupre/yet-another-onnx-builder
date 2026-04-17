"""Tests that verify CustomTracer works for aten ops through to_onnx(..., tracing=True)."""

import unittest
import torch
from yobx.ext_test_case import ExtTestCase, requires_torch
from yobx.torch.torch_helper import torch_deepcopy
from yobx.torch import ExportOptions
from yobx.torch.interpreter import to_onnx


@requires_torch("2.0")
class TestOnnxExportAtenTracing(ExtTestCase):
    """Tests that aten ops are exported correctly when tracing=True."""

    def _to_onnx_tracing(self, model, inputs, dynamic_shapes=None):
        """Exports *model* with the tracing path and returns the ONNX proto."""
        return to_onnx(
            model,
            inputs,
            export_options=ExportOptions(tracing=True),
            dynamic_shapes=dynamic_shapes,
        )

    # ------------------------------------------------------------------
    # Basic arithmetic ops
    # ------------------------------------------------------------------

    def test_aten_add_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        model = Model()
        inputs = (torch.rand(3, 4), torch.rand(3, 4))
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_sub_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x - y

        model = Model()
        inputs = (torch.rand(3, 4), torch.rand(3, 4))
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_mul_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x * y

        model = Model()
        inputs = (torch.rand(3, 4), torch.rand(3, 4))
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_div_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x / y

        model = Model()
        inputs = (torch.rand(3, 4) + 0.5, torch.rand(3, 4) + 0.5)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_add_scalar_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 3.0

        model = Model()
        inputs = (torch.rand(3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_mul_scalar_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x * 2.5

        model = Model()
        inputs = (torch.rand(3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    # ------------------------------------------------------------------
    # Unary ops
    # ------------------------------------------------------------------

    def test_aten_abs_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.abs(x)

        model = Model()
        inputs = (torch.rand(3, 4) - 0.5,)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_neg_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return -x

        model = Model()
        inputs = (torch.rand(3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_relu_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)

        model = Model()
        inputs = (torch.rand(3, 4) - 0.5,)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_exp_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.exp(x)

        model = Model()
        inputs = (torch.rand(3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_log_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.log(x)

        model = Model()
        inputs = (torch.rand(3, 4) + 0.5,)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_sqrt_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.sqrt(x)

        model = Model()
        inputs = (torch.rand(3, 4) + 0.1,)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_sin_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x)

        model = Model()
        inputs = (torch.rand(3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_cos_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.cos(x)

        model = Model()
        inputs = (torch.rand(3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_tanh_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.tanh(x)

        model = Model()
        inputs = (torch.rand(3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_sigmoid_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.sigmoid(x)

        model = Model()
        inputs = (torch.rand(3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    # ------------------------------------------------------------------
    # Reduction ops
    # ------------------------------------------------------------------

    def test_aten_sum_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.sum()

        model = Model()
        inputs = (torch.rand(3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_sum_dim_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.sum(dim=1, keepdim=True)

        model = Model()
        inputs = (torch.rand(3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_mean_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.mean()

        model = Model()
        inputs = (torch.rand(3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_mean_dim_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.mean(dim=0)

        model = Model()
        inputs = (torch.rand(3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_max_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.max()

        model = Model()
        inputs = (torch.rand(3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_min_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.min()

        model = Model()
        inputs = (torch.rand(3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    # ------------------------------------------------------------------
    # Matrix / linear algebra ops
    # ------------------------------------------------------------------

    def test_aten_matmul_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return torch.matmul(x, y)

        model = Model()
        inputs = (torch.rand(3, 5), torch.rand(5, 4))
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_mm_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return torch.mm(x, y)

        model = Model()
        inputs = (torch.rand(3, 5), torch.rand(5, 4))
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_bmm_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return torch.bmm(x, y)

        model = Model()
        inputs = (torch.rand(2, 3, 5), torch.rand(2, 5, 4))
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_linear_tracing(self):
        """Tests that nn.Linear (addmm internally) works via tracing."""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 3, bias=True)

            def forward(self, x):
                return self.linear(x)

        model = Model()
        model.eval()
        inputs = (torch.rand(4, 5),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_linear_no_bias_tracing(self):
        """Tests that nn.Linear without bias works via tracing."""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 3, bias=False)

            def forward(self, x):
                return self.linear(x)

        model = Model()
        model.eval()
        inputs = (torch.rand(4, 5),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    # ------------------------------------------------------------------
    # Shape / layout ops
    # ------------------------------------------------------------------

    def test_aten_reshape_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.reshape(2, 6)

        model = Model()
        inputs = (torch.rand(3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_transpose_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.transpose(0, 1)

        model = Model()
        inputs = (torch.rand(3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_permute_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.permute(0, 2, 1)

        model = Model()
        inputs = (torch.rand(2, 3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_squeeze_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.squeeze(0)

        model = Model()
        inputs = (torch.rand(1, 3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_unsqueeze_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.unsqueeze(0)

        model = Model()
        inputs = (torch.rand(3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_flatten_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.flatten()

        model = Model()
        inputs = (torch.rand(2, 3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_cat_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return torch.cat([x, y], dim=0)

        model = Model()
        inputs = (torch.rand(2, 4), torch.rand(3, 4))
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    # ------------------------------------------------------------------
    # Comparison / selection ops
    # ------------------------------------------------------------------

    def test_aten_clamp_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.clamp(x, min=0.2, max=0.8)

        model = Model()
        inputs = (torch.rand(3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_where_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, cond, x, y):
                return torch.where(cond, x, y)

        model = Model()
        cond = torch.tensor([[True, False, True, False], [False, True, False, True]])
        x = torch.rand(2, 4)
        y = torch.rand(2, 4)
        inputs = (cond, x, y)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    # ------------------------------------------------------------------
    # Misc ops
    # ------------------------------------------------------------------

    def test_aten_clone_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.clone()

        model = Model()
        inputs = (torch.rand(3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_cast_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.to(torch.float64)

        model = Model()
        inputs = (torch.rand(3, 4).to(torch.float32),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-6)

    def test_aten_pow_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.pow(x, 2.0)

        model = Model()
        inputs = (torch.rand(3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_softmax_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.softmax(dim=-1)

        model = Model()
        inputs = (torch.rand(3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_layer_norm_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.layer_norm(x, [x.shape[-1]])

        model = Model()
        inputs = (torch.rand(2, 3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=2e-4)

    def test_aten_split_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                parts = torch.split(x, 2, dim=1)
                return parts[0] + parts[1]

        model = Model()
        inputs = (torch.rand(3, 4),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    def test_aten_stack_tracing(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return torch.stack([x, y], dim=0)

        model = Model()
        inputs = (torch.rand(3, 4), torch.rand(3, 4))
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    # ------------------------------------------------------------------
    # Composite model: verify tracing handles multiple ops in sequence
    # ------------------------------------------------------------------

    def test_multi_op_model_tracing(self):
        """Tests a model that combines several aten ops via tracing."""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(4, 5))
                self.bias = torch.nn.Parameter(torch.rand(4))

            def forward(self, x):
                y = torch.relu(x @ self.weight.t() + self.bias)
                return y / y.sum(dim=-1, keepdim=True)

        model = Model()
        model.eval()
        inputs = (torch.rand(2, 5),)
        expected = model(*torch_deepcopy(inputs))
        onx = self._to_onnx_tracing(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

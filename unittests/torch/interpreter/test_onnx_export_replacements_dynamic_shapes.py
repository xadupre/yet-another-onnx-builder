import unittest
from yobx.ext_test_case import (
    ExtTestCase,
    requires_torch,
)
from yobx.torch.interpreter.onnx_export import _replacements_dynamic_shapes


class TestReplacementsDynamicShapes(ExtTestCase):
    @requires_torch("2.2")
    def test_no_input_names(self):
        """Model with two named params; dynamic shapes use the model's own param names."""
        import torch

        class MyModel(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        model = MyModel()
        x = torch.rand(3, 4)
        y = torch.rand(3, 4)
        DYN = torch.export.Dim.DYNAMIC
        dynamic_shapes = {"x": {0: DYN}, "y": {0: DYN}}
        result = _replacements_dynamic_shapes(
            model,
            (x, y),
            dict_dynamic_shapes=dynamic_shapes,
        )
        self.assertEqual(result, {"x": {0: DYN}, "y": {0: DYN}})

    @requires_torch("2.2")
    def test_with_input_names_renames_key(self):
        """input_names maps a user-supplied name to the model's param name."""
        import torch

        class MyModel(torch.nn.Module):
            def forward(self, x):
                return x + 1

        model = MyModel()
        x = torch.rand(3, 4)
        DYN = torch.export.Dim.DYNAMIC
        # Dynamic shape is keyed by the user-supplied name "my_x"
        dynamic_shapes = {"my_x": {0: DYN}}
        result = _replacements_dynamic_shapes(
            model,
            (x,),
            dict_dynamic_shapes=dynamic_shapes,
            input_names=["my_x"],
        )
        # "my_x" should be mapped to the model param name "x"
        self.assertEqual(result, {"x": {0: DYN}})

    @requires_torch("2.2")
    def test_var_positional_args(self):
        """Model using *args; shapes are packed into a tuple under the *args name."""
        import torch

        class MyModel(torch.nn.Module):
            def forward(self, *args):
                return args[0] + args[1]

        model = MyModel()
        x = torch.rand(3, 4)
        y = torch.rand(3, 4)
        DYN = torch.export.Dim.DYNAMIC
        dynamic_shapes = {"input0": {0: DYN}, "input1": {0: DYN}}
        result = _replacements_dynamic_shapes(
            model,
            (x, y),
            dict_dynamic_shapes=dynamic_shapes,
            input_names=["input0", "input1"],
        )
        # Shapes are packed as a tuple under the VAR_POSITIONAL parameter name "args"
        self.assertEqual(result, {"args": ({0: DYN}, {0: DYN})})

    @requires_torch("2.2")
    def test_param_not_in_dynamic_shapes_not_in_result(self):
        """A param that doesn't appear in dict_dynamic_shapes is absent from the result."""
        import torch

        class MyModel(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        model = MyModel()
        x = torch.rand(3, 4)
        y = torch.rand(3, 4)
        DYN = torch.export.Dim.DYNAMIC
        # Only x has a dynamic shape; y is static
        dynamic_shapes = {"x": {0: DYN}}
        result = _replacements_dynamic_shapes(
            model,
            (x, y),
            dict_dynamic_shapes=dynamic_shapes,
        )
        self.assertEqual(result, {"x": {0: DYN}})
        self.assertNotIn("y", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)

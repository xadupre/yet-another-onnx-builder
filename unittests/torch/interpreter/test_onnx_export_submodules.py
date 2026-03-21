import unittest
from onnx.checker import check_model
from yobx.ext_test_case import (
    ExtTestCase,
    skipif_ci_windows,
    requires_torch,
    requires_transformers,
    ignore_warnings,
)
from yobx.xbuilder import FunctionOptions
from yobx.torch.interpreter import to_onnx
from yobx.reference import ExtendedReferenceEvaluator


class TestOnnxExportSubModules(ExtTestCase):

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    @ignore_warnings(FutureWarning)
    def test_submodule_local_functions_simple(self):
        import torch

        class SubNeuron2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                z = self.linear(x)
                return torch.sigmoid(z)

        class Neuron2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.neuron = SubNeuron2(n_dims, n_targets)

            def forward(self, x):
                z = self.neuron(x)
                return torch.relu(z)

        model = Neuron2()
        inputs = (torch.randn(1, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions=True,
            optimize=False,
            verbose=0,
            inline=False,
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 2)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    @ignore_warnings(FutureWarning)
    def test_submodule_local_functions_double(self):
        import torch

        class SubNeuron2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear1 = torch.nn.Linear(n_dims, n_targets)
                self.linear2 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                z1 = self.linear1(x)
                z2 = self.linear2(x)
                return torch.sigmoid(z1) + torch.sigmoid(z2)

        class Neuron2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.neuron = SubNeuron2(n_dims, n_targets)

            def forward(self, x):
                z = self.neuron(x)
                return torch.relu(z)

        model = Neuron2()
        inputs = (torch.randn(1, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions=True,
            optimize=False,
            verbose=0,
            function_options=FunctionOptions(merge_allowed=True, external_threshold=0),
            inline=False,
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 2)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    @ignore_warnings(FutureWarning)
    def test_submodule_local_functions_two_outputs(self):
        import torch

        class SubNeuron2Outputs(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()

            def forward(self, x):
                return (torch.sigmoid(x), torch.sigmoid(x * x))

        class Neuron2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.neuron = SubNeuron2Outputs(n_dims, n_targets)

            def forward(self, x):
                z, z1 = self.neuron(x)
                return torch.relu(z) + z1

        model = Neuron2()
        inputs = (torch.randn(1, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions=True,
            optimize=False,
            verbose=0,
            inline=False,
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 1)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    @ignore_warnings(FutureWarning)
    def test_submodule_local_functions_more_depth_static(self):
        import torch

        class Level1(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear1 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                z1 = self.linear1(x)
                ones = torch.ones(z1.shape, dtype=z1.dtype, device=z1.device)
                ones[0, 0] = 0
                return torch.sigmoid(z1 + ones)

        class Level2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevela = Level1(n_dims, n_targets)
                self.linear2 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                y1 = self.linear2(x)
                y2 = self.sublevela(x)
                ones = torch.ones(y2.shape, dtype=y2.dtype, device=y2.device)
                ones[0, 0] = 0
                return torch.sigmoid(y1 + ones) + y2

        class Level3(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevelb = Level2(n_dims, n_targets)
                self.linear3 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                w1 = self.linear3(x)
                w2 = self.sublevelb(x)
                ones = torch.ones(w2.shape, dtype=w2.dtype, device=w2.device)
                ones[0, 0] = 0
                return torch.sigmoid(w1 + ones) + w2

        model = Level3()
        inputs = (torch.randn(1, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        filename = self.get_dump_file("test_submodule_local_functions_more_depth_static.onnx")
        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions=True,
            optimize=False,
            verbose=0,
            function_options=FunctionOptions(merge_allowed=True, external_threshold=0),
            inline=False,
            filename=filename,
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 3)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    @ignore_warnings(FutureWarning)
    def test_submodule_local_functions_more_depth_dynamic1(self):
        import torch

        class Level1(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear1 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                z1 = self.linear1(x)
                ones = torch.ones(z1.shape, dtype=z1.dtype, device=z1.device)
                ones[0, 0] = 0
                return torch.sigmoid(z1 + ones)

        class Level2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevela = Level1(n_dims, n_targets)
                self.linear2 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                y1 = self.linear2(x)
                y2 = self.sublevela(x)
                ones = torch.ones(y2.shape, dtype=y2.dtype, device=y2.device)
                ones[0, 0] = 0
                return torch.sigmoid(y1 + ones) + y2

        model = Level2()
        inputs = (torch.randn(2, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        filename = self.get_dump_file("test_submodule_local_functions_more_depth_dynamic1.onnx")
        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions=True,
            optimize=False,
            verbose=0,
            function_options=FunctionOptions(merge_allowed=True, external_threshold=0),
            inline=False,
            filename=filename,
            dynamic_shapes=({0: "batch"},),
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 2)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    @ignore_warnings(FutureWarning)
    def test_submodule_local_functions_more_depth_dynamic2(self):
        import torch

        class Level1(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear1 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                z1 = self.linear1(x)
                ones = torch.ones(z1.shape, dtype=z1.dtype, device=z1.device)
                ones[0, 0] = 0
                return torch.sigmoid(z1 + ones)

        class Level2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevela = Level1(n_dims, n_targets)
                self.linear2 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                y1 = self.linear2(x)
                y2 = self.sublevela(x)
                ones = torch.ones(y2.shape, dtype=y2.dtype, device=y2.device)
                ones[0, 0] = 0
                return torch.sigmoid(y1 + ones) + y2

        class Level3(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevelb = Level2(n_dims, n_targets)
                self.linear3 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                w1 = self.linear3(x)
                w2 = self.sublevelb(x)
                ones = torch.ones(w2.shape, dtype=w2.dtype, device=w2.device)
                ones[0, 0] = 0
                return torch.sigmoid(w1 + ones) + w2

        model = Level3()
        inputs = (torch.randn(2, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        filename = self.get_dump_file("test_submodule_local_functions_more_depth_dynamic2.onnx")
        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions=True,
            optimize=False,
            verbose=0,
            function_options=FunctionOptions(merge_allowed=True, external_threshold=0),
            inline=False,
            filename=filename,
            dynamic_shapes=({0: "batch"},),
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 3)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    @ignore_warnings(FutureWarning)
    def test_submodule_local_functions_more_depth_dynamic2_preserve2(self):
        import torch

        class Level1(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear1 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                z1 = self.linear1(x)
                ones = torch.ones(z1.shape, dtype=z1.dtype, device=z1.device)
                ones[0, 0] = 0
                return torch.sigmoid(z1 + ones)

        class Level2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevela = Level1(n_dims, n_targets)
                self.linear2 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                y1 = self.linear2(x)
                y2 = self.sublevela(x)
                ones = torch.ones(y2.shape, dtype=y2.dtype, device=y2.device)
                ones[0, 0] = 0
                return torch.sigmoid(y1 + ones) + y2

        class Level3(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevelb = Level2(n_dims, n_targets)
                self.linear3 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                w1 = self.linear3(x)
                w2 = self.sublevelb(x)
                ones = torch.ones(w2.shape, dtype=w2.dtype, device=w2.device)
                ones[0, 0] = 0
                return torch.sigmoid(w1 + ones) + w2

        model = Level3()
        inputs = (torch.randn(2, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        filename = self.get_dump_file(
            "test_submodule_local_functions_more_depth_dynamic2_preserve2.onnx"
        )
        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions={Level2},
            optimize=False,
            verbose=0,
            function_options=FunctionOptions(merge_allowed=True, external_threshold=0),
            inline=False,
            filename=filename,
            dynamic_shapes=({0: "batch"},),
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 1)
        self.assertEqual(["<locals>_Level2"], [f.name for f in onx.functions])
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    @ignore_warnings(FutureWarning)
    def test_submodule_local_functions_more_depth_dynamic2_preserve1(self):
        import torch

        class Level1(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear1 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                z1 = self.linear1(x)
                ones = torch.ones(z1.shape, dtype=z1.dtype, device=z1.device)
                ones[0, 0] = 0
                return torch.sigmoid(z1 + ones)

        class Level2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevela = Level1(n_dims, n_targets)
                self.linear2 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                y1 = self.linear2(x)
                y2 = self.sublevela(x)
                ones = torch.ones(y2.shape, dtype=y2.dtype, device=y2.device)
                ones[0, 0] = 0
                return torch.sigmoid(y1 + ones) + y2

        class Level3(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevelb = Level2(n_dims, n_targets)
                self.linear3 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                w1 = self.linear3(x)
                w2 = self.sublevelb(x)
                ones = torch.ones(w2.shape, dtype=w2.dtype, device=w2.device)
                ones[0, 0] = 0
                return torch.sigmoid(w1 + ones) + w2

        model = Level3()
        inputs = (torch.randn(2, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        filename = self.get_dump_file(
            "test_submodule_local_functions_more_depth_dynamic2_preserve1.onnx"
        )
        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions={Level1},
            optimize=False,
            verbose=0,
            function_options=FunctionOptions(merge_allowed=True, external_threshold=0),
            inline=False,
            filename=filename,
            dynamic_shapes=({0: "batch"},),
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 1)
        self.assertEqual(["<locals>_Level1"], [f.name for f in onx.functions])
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    @ignore_warnings(FutureWarning)
    def test_submodule_local_functions_more_depth_dynamic1_2io(self):
        import torch

        class Level1(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear1 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                z1 = self.linear1(x)
                ones = torch.ones(z1.shape, dtype=z1.dtype, device=z1.device)
                ones[0, 0] = 0
                return z1, torch.sigmoid(z1 + ones)

        class Level2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevela = Level1(n_dims, n_targets)
                self.linear2 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                y1 = self.linear2(x)
                y2, y3 = self.sublevela(x)
                ones = torch.ones(y2.shape, dtype=y2.dtype, device=y2.device)
                ones[0, 0] = 0
                return torch.sigmoid(y1 + y2 + ones) + y3

        model = Level2()
        inputs = (torch.randn(2, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        filename = self.get_dump_file(
            "test_submodule_local_functions_more_depth_dynamic1_2io.onnx"
        )
        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions=True,
            optimize=False,
            verbose=0,
            function_options=FunctionOptions(merge_allowed=True, external_threshold=0),
            inline=False,
            filename=filename,
            dynamic_shapes=({0: "batch"},),
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 2)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    @ignore_warnings(FutureWarning)
    def test_submodule_local_functions_more_depth_dynamic2_preserve2_2io1(self):
        import torch

        class Level1(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear1 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                z1 = self.linear1(x)
                ones = torch.ones(z1.shape, dtype=z1.dtype, device=z1.device)
                ones[0, 0] = 0
                return z1, torch.sigmoid(z1 + ones)

        class Level2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevela = Level1(n_dims, n_targets)
                self.linear2 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                y1 = self.linear2(x)
                y2, y3 = self.sublevela(x)
                ones = torch.ones(y2.shape, dtype=y2.dtype, device=y2.device)
                ones[0, 0] = 0
                return torch.sigmoid(y1 + ones + y2) + y3

        class Level3(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevelb = Level2(n_dims, n_targets)
                self.linear3 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                w1 = self.linear3(x)
                w2 = self.sublevelb(x)
                ones = torch.ones(w2.shape, dtype=w2.dtype, device=w2.device)
                ones[0, 0] = 0
                return torch.sigmoid(w1 + ones) + w2

        model = Level3()
        inputs = (torch.randn(2, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        filename = self.get_dump_file(
            "test_submodule_local_functions_more_depth_dynamic2_preserve2_2io1.onnx"
        )
        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions={Level2},
            optimize=False,
            verbose=0,
            function_options=FunctionOptions(merge_allowed=True, external_threshold=0),
            inline=False,
            filename=filename,
            dynamic_shapes=({0: "batch"},),
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 1)
        self.assertEqual(["<locals>_Level2"], [f.name for f in onx.functions])
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    @ignore_warnings(FutureWarning)
    def test_submodule_local_functions_shapes(self):
        import torch

        class Level1(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear1 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                z1 = self.linear1(x)
                ones = torch.ones(z1.shape, dtype=z1.dtype, device=z1.device)
                ones[0, 0] = 0
                return z1.shape[0], torch.sigmoid(z1 + ones)

        class Level2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevela = Level1(n_dims, n_targets)
                self.linear2 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                y1 = self.linear2(x)
                batch_dim, y3 = self.sublevela(x)
                ones = torch.ones((batch_dim, y3.shape[1]), dtype=y3.dtype, device=y3.device)
                ones[0, 0] = 0
                return torch.sigmoid(y1 + ones) + y3

        class Level3(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevelb = Level2(n_dims, n_targets)
                self.linear3 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                w1 = self.linear3(x)
                w2 = self.sublevelb(x)
                ones = torch.ones(w2.shape, dtype=w2.dtype, device=w2.device)
                ones[0, 0] = 0
                return torch.sigmoid(w1 + ones) + w2

        model = Level3()
        inputs = (torch.randn(2, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        filename = self.get_dump_file("test_submodule_local_functions_shapes.onnx")
        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions={Level1},
            optimize=True,
            verbose=0,
            function_options=FunctionOptions(merge_allowed=True, external_threshold=0),
            inline=False,
            filename=filename,
            dynamic_shapes=({0: "batch"},),
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 1)
        self.assertEqual(["<locals>_Level1"], [f.name for f in onx.functions])
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    @requires_transformers("5.2")
    @ignore_warnings(FutureWarning)
    def test_submodule_local_functions_tiny_llm(self):
        """
        Tests that export_modules_as_functions=True works for Tiny-LLM
        (arnir0/Tiny-LLM, a LlamaForCausalLM model) where some submodule
        outputs include SymInt values mixed with tensors.
        """
        from yobx.torch.tiny_models import get_tiny_model
        from yobx.torch import apply_patches_for_model, register_flattening_functions

        model_id = "arnir0/Tiny-LLM"
        data = get_tiny_model(model_id)
        model, inputs, ds = data.model, data.export_inputs, data.dynamic_shapes
        filename1 = self.get_dump_file("test_submodule_local_functions_tiny_llm.1.onnx")
        filename2 = self.get_dump_file("test_submodule_local_functions_tiny_llm.2.onnx")

        with (
            apply_patches_for_model(patch_torch=True, patch_transformers=True, model=model),
            register_flattening_functions(patch_transformers=True),
        ):
            to_onnx(
                model,
                (),
                kwargs=inputs,
                dynamic_shapes=ds,
                inline=False,
                optimize=False,
                verbose=0,
                filename=filename1,
            )
            onx = to_onnx(
                model,
                (),
                kwargs=inputs,
                dynamic_shapes=ds,
                export_modules_as_functions=True,
                inline=False,
                optimize=False,
                verbose=0,
                filename=filename2,
            )
        check_model(onx)
        self.assertGreater(len(onx.functions), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

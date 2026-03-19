import unittest
from typing import List
from yobx.ext_test_case import ExtTestCase, ignore_warnings, requires_torch, skipif_ci_windows
from yobx.container import ExportArtifact
from yobx.torch.interpreter import ForceDispatcher


class TestForceDispatcher(ExtTestCase):
    """Tests for :class:`ForceDispatcher`."""

    # ------------------------------------------------------------------
    # _convert_into_type
    # ------------------------------------------------------------------

    def test_convert_into_type_float(self):
        result = ForceDispatcher._convert_into_type(float)
        self.assertIs(result, float)

    def test_convert_into_type_int(self):
        result = ForceDispatcher._convert_into_type(int)
        self.assertIs(result, int)

    def test_convert_into_type_bool(self):
        result = ForceDispatcher._convert_into_type(bool)
        self.assertIs(result, bool)

    def test_convert_into_type_list_int(self):
        result = ForceDispatcher._convert_into_type(List[int])
        self.assertIsNotNone(result)
        converted = result([1.5, 2.7, 3.9])
        self.assertEqual(converted, [1, 2, 3])

    def test_convert_into_type_list_float(self):
        result = ForceDispatcher._convert_into_type(List[float])
        self.assertIsNotNone(result)
        converted = result([1, 2, 3])
        self.assertEqual(converted, [1.0, 2.0, 3.0])

    def test_convert_into_type_list_bool(self):
        result = ForceDispatcher._convert_into_type(List[bool])
        self.assertIsNotNone(result)
        converted = result([0, 1, 0])
        self.assertEqual(converted, [False, True, False])

    def test_convert_into_type_unsupported_raises(self):
        self.assertRaise(lambda: ForceDispatcher._convert_into_type(str), RuntimeError)

    def test_convert_into_type_none_raises(self):
        self.assertRaise(lambda: ForceDispatcher._convert_into_type(None), AssertionError)

    # ------------------------------------------------------------------
    # _process_signature
    # ------------------------------------------------------------------

    def test_process_signature_positional_only(self):
        def my_func(x, y):
            pass

        dispatcher = ForceDispatcher()
        args, kwargs = dispatcher._process_signature(my_func)
        self.assertEqual(args, ["x", "y"])
        self.assertEqual(kwargs, [])

    def test_process_signature_with_annotated_defaults(self):
        def my_func(x, alpha: float = 1.0, beta: float = 0.0):
            pass

        dispatcher = ForceDispatcher()
        args, kwargs = dispatcher._process_signature(my_func)
        self.assertEqual(args, ["x"])
        self.assertEqual(len(kwargs), 2)
        self.assertEqual(kwargs[0][0], "alpha")
        self.assertAlmostEqual(kwargs[0][1], 1.0)
        self.assertEqual(kwargs[1][0], "beta")
        self.assertAlmostEqual(kwargs[1][1], 0.0)

    def test_process_signature_with_list_annotation(self):
        def my_func(x, dims: List[int] = None):  # noqa: RUF013
            pass

        dispatcher = ForceDispatcher()
        args, kwargs = dispatcher._process_signature(my_func)
        self.assertEqual(args, ["x"])
        self.assertEqual(len(kwargs), 1)
        self.assertEqual(kwargs[0][0], "dims")

    def test_process_signature_int_annotation(self):
        def my_func(x, num_groups: int = 1):
            pass

        dispatcher = ForceDispatcher()
        args, kwargs = dispatcher._process_signature(my_func)
        self.assertEqual(args, ["x"])
        self.assertEqual(len(kwargs), 1)
        name, default, converter = kwargs[0]
        self.assertEqual(name, "num_groups")
        self.assertEqual(default, 1)
        self.assertIs(converter, int)

    # ------------------------------------------------------------------
    # _process_signatures
    # ------------------------------------------------------------------

    def test_process_signatures_multiple(self):
        def func_a(x, y):
            pass

        def func_b(x, scale: float = 1.0):
            pass

        dispatcher = ForceDispatcher(signatures={"func_a": func_a, "func_b": func_b})
        self.assertIn("func_a", dispatcher.sigs_)
        self.assertIn("func_b", dispatcher.sigs_)
        args_a, _kwargs_a = dispatcher.sigs_["func_a"]
        self.assertEqual(args_a, ["x", "y"])
        args_b, kwargs_b = dispatcher.sigs_["func_b"]
        self.assertEqual(args_b, ["x"])
        self.assertEqual(len(kwargs_b), 1)

    def test_process_signatures_empty(self):
        dispatcher = ForceDispatcher()
        self.assertEqual(dispatcher.sigs_, {})

    # ------------------------------------------------------------------
    # __init__ defaults and custom parameters
    # ------------------------------------------------------------------

    def test_init_defaults(self):
        dispatcher = ForceDispatcher()
        self.assertEqual(dispatcher.domain, "aten.lib")
        self.assertEqual(dispatcher.version, 1)
        self.assertFalse(dispatcher.strict)
        self.assertFalse(dispatcher.only_registered)
        self.assertEqual(dispatcher.signatures, {})
        self.assertEqual(dispatcher.registered_functions, {})

    def test_init_custom_params(self):
        dispatcher = ForceDispatcher(
            domain="my.domain", version=3, strict=True, only_registered=True
        )
        self.assertEqual(dispatcher.domain, "my.domain")
        self.assertEqual(dispatcher.version, 3)
        self.assertTrue(dispatcher.strict)
        self.assertTrue(dispatcher.only_registered)

    # ------------------------------------------------------------------
    # fallback: return fct unchanged when already found
    # ------------------------------------------------------------------

    def test_fallback_returns_existing_function(self):
        """When a converter function is already found, fallback returns it unchanged."""

        def my_converter():
            pass

        dispatcher = ForceDispatcher()
        result = dispatcher.fallback("some_name", my_converter, [], {}, None)
        self.assertIs(result, my_converter)

    def test_fallback_returns_wrapper_when_no_function(self):
        """When no converter is found, fallback returns a callable wrapper."""
        dispatcher = ForceDispatcher()
        result = dispatcher.fallback("aten_relu", None, [], {}, None)
        self.assertTrue(callable(result))

    # ------------------------------------------------------------------
    # Integration tests: ForceDispatcher used through to_onnx
    # ------------------------------------------------------------------

    @skipif_ci_windows("dynamo not working on Windows")
    @requires_torch("2.5", "module 'torch.library' has no attribute 'infer_schema'")
    @ignore_warnings((UserWarning, FutureWarning))
    def test_force_dispatcher_basic(self):
        """ForceDispatcher without signatures produces ONNX nodes in a custom domain."""
        import torch

        from yobx.torch.interpreter import to_onnx

        def twice_fn(x: torch.Tensor) -> torch.Tensor:
            return x + x

        def twice_meta(x):
            return torch.empty_like(x)

        schema_str = torch.library.infer_schema(twice_fn, mutates_args=())
        twice = torch.library.CustomOpDef("testlib_fd1", "twice", schema_str, twice_fn)
        twice.register_kernel(None)(twice_fn)
        twice._abstract_fn = twice_meta

        class DummyModel(torch.nn.Module):
            def forward(self, x):
                return torch.ops.testlib_fd1.twice(x) * x

        x = torch.rand(5, 3)
        model = DummyModel()
        dispatcher = ForceDispatcher(domain="testlib_fd1", version=1)
        onx = to_onnx(model, (x,), dispatcher=dispatcher)
        self.assertIsInstance(onx, ExportArtifact)
        domains = {n.domain for n in onx.graph.node}
        self.assertIn("testlib_fd1", domains)

    @skipif_ci_windows("dynamo not working on Windows")
    @requires_torch("2.5", "module 'torch.library' has no attribute 'infer_schema'")
    @ignore_warnings((UserWarning, FutureWarning))
    def test_force_dispatcher_with_signature(self):
        """ForceDispatcher with a signature maps scalar kwargs correctly."""
        import torch

        from yobx.torch.interpreter import to_onnx

        def scaled_fn(x: torch.Tensor) -> torch.Tensor:
            return x * 2.0

        def scaled_meta(x):
            return torch.empty_like(x)

        schema_str = torch.library.infer_schema(scaled_fn, mutates_args=())
        scaled = torch.library.CustomOpDef("testlib_fd2", "scaled", schema_str, scaled_fn)
        scaled.register_kernel(None)(scaled_fn)
        scaled._abstract_fn = scaled_meta

        class DummyModel(torch.nn.Module):
            def forward(self, x):
                return torch.ops.testlib_fd2.scaled(x)

        x = torch.rand(5, 3)
        model = DummyModel()

        def sig_fn(x):
            pass

        dispatcher = ForceDispatcher(
            signatures={"testlib_fd2_scaled_default": sig_fn}, domain="testlib_fd2", version=1
        )
        onx = to_onnx(model, (x,), dispatcher=dispatcher)
        self.assertIsInstance(onx, ExportArtifact)
        domains = {n.domain for n in onx.graph.node}
        self.assertIn("testlib_fd2", domains)

    @skipif_ci_windows("dynamo not working on Windows")
    @requires_torch("2.5", "module 'torch.library' has no attribute 'infer_schema'")
    @ignore_warnings((UserWarning, FutureWarning))
    def test_force_dispatcher_only_registered_raises(self):
        """ForceDispatcher with only_registered=True raises when no signature is found."""
        import torch

        from yobx.torch.interpreter import to_onnx

        def twice_fn3(x: torch.Tensor) -> torch.Tensor:
            return x + x

        def twice_meta3(x):
            return torch.empty_like(x)

        schema_str = torch.library.infer_schema(twice_fn3, mutates_args=())
        twice3 = torch.library.CustomOpDef("testlib_fd3", "twice3", schema_str, twice_fn3)
        twice3.register_kernel(None)(twice_fn3)
        twice3._abstract_fn = twice_meta3

        class DummyModel(torch.nn.Module):
            def forward(self, x):
                return torch.ops.testlib_fd3.twice3(x) * x

        x = torch.rand(5, 3)
        model = DummyModel()

        # No signatures provided but only_registered=True: should raise
        dispatcher = ForceDispatcher(
            signatures={}, domain="testlib_fd3", version=1, only_registered=True
        )
        self.assertRaise(lambda: to_onnx(model, (x,), dispatcher=dispatcher), AssertionError)

    @skipif_ci_windows("dynamo not working on Windows")
    @requires_torch("2.5", "module 'torch.library' has no attribute 'infer_schema'")
    @ignore_warnings((UserWarning, FutureWarning))
    def test_force_dispatcher_op_type_in_onnx(self):
        """Verify the custom op type name appears in the exported ONNX model."""
        import torch

        from yobx.torch.interpreter import to_onnx

        def negate_fn(x: torch.Tensor) -> torch.Tensor:
            return -x

        def negate_meta(x):
            return torch.empty_like(x)

        schema_str = torch.library.infer_schema(negate_fn, mutates_args=())
        negate = torch.library.CustomOpDef("testlib_fd4", "negate", schema_str, negate_fn)
        negate.register_kernel(None)(negate_fn)
        negate._abstract_fn = negate_meta

        class DummyModel(torch.nn.Module):
            def forward(self, x):
                return torch.ops.testlib_fd4.negate(x)

        x = torch.rand(5, 3)
        model = DummyModel()
        dispatcher = ForceDispatcher(domain="testlib_fd4", version=1)
        onx = to_onnx(model, (x,), dispatcher=dispatcher)
        self.assertIsInstance(onx, ExportArtifact)
        op_types = [(n.domain, n.op_type) for n in onx.graph.node]
        # The op_type should contain the function name
        custom_nodes = [(d, t) for d, t in op_types if d == "testlib_fd4"]
        self.assertGreater(len(custom_nodes), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

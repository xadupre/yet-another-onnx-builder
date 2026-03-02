import unittest
from yobx.ext_test_case import ExtTestCase, requires_torch


@requires_torch("2.0")
class TestTorchSymIntToStr(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from yobx.xbuilder import GraphBuilder

        cls.torch = torch
        cls.builder = GraphBuilder(18, ir_version=9)

    def test_str_input(self):
        """A plain string is returned as-is."""
        result = self.builder._torch_sym_int_to_str("batch")
        self.assertEqual(result, "batch")

    def test_integer_input(self):
        """An integer-convertible value is returned as int."""
        result = self.builder._torch_sym_int_to_str(5)
        self.assertEqual(result, 5)
        self.assertIsInstance(result, int)

    def test_sym_int_node_str(self):
        """A SymInt whose .node is a string returns that string."""
        sym = self.torch.SymInt("s0")
        result = self.builder._torch_sym_int_to_str(sym)
        self.assertEqual(result, "s0")

    def test_sym_int_node_symnode(self):
        """A SymInt whose .node is a SymNode returns the expression string."""
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        import torch.fx.experimental.sym_node as sym_node_module

        node = sym_node_module.SymNode("s0", ShapeEnv(), int, 0)
        sym = self.torch.SymInt(node)
        result = self.builder._torch_sym_int_to_str(sym)
        self.assertIsInstance(result, str)
        self.assertIn("s0", result)

    def test_unknown_raises(self):
        """An object that cannot be converted to int raises AssertionError."""

        class _Unconvertible:
            def __repr__(self):
                return "_Unconvertible()"

            def __int__(self):
                raise TypeError("cannot convert")

        with self.assertRaises(AssertionError):
            self.builder._torch_sym_int_to_str(_Unconvertible())


@requires_torch("2.0")
class TestTorchSymInt(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from yobx.xbuilder import GraphBuilder

        cls.torch = torch
        cls.builder = GraphBuilder(18, ir_version=9)

    def test_sym_int_str_node(self):
        """A SymInt with a string node returns that string."""
        sym = self.torch.SymInt("s0")
        result = self.builder._torch_sym_int(sym)
        self.assertEqual(result, "s0")

    def test_sym_int_symnode(self):
        """A SymInt backed by a SymNode returns the expression string."""
        import sympy
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        import torch.fx.experimental.sym_node as sym_node_module

        node = sym_node_module.SymNode(sympy.Symbol("s0"), ShapeEnv(), int, 0)  # hint=0
        sym = self.torch.SymInt(node)
        result = self.builder._torch_sym_int(sym)
        self.assertIsInstance(result, str)
        self.assertIn("s0", result)


class TestWrapDimNameAsString(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from yobx.xbuilder import GraphBuilder

        cls.torch = torch
        cls.builder = GraphBuilder(18, ir_version=9)
        cls.WrapDim = GraphBuilder.WrapDim

    def test_str_name(self):
        """A plain string name is returned as-is."""
        wd = self.WrapDim("batch")
        self.assertEqual(wd.name_as_string, "batch")

    def test_torch_dim_name(self):
        """A torch.export.Dim object returns its __name__ attribute."""
        dim = self.torch.export.Dim("seq")
        wd = self.WrapDim(dim)
        self.assertEqual(wd.name_as_string, "seq")

    def test_unknown_type_raises(self):
        """An unknown type raises AssertionError."""

        class _Unknown:
            pass

        wd = self.WrapDim(_Unknown())
        with self.assertRaises(AssertionError):
            _ = wd.name_as_string


if __name__ == "__main__":
    unittest.main(verbosity=2)

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
class TestImproveConstraints(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        from yobx.xbuilder import GraphBuilder

        cls.GraphBuilder = GraphBuilder

    def test_improve_constraints_deduces_equivalence(self):
        """_improve_constraints should deduce that seq_length == s70 from
        the constraint s52+seq_length == s52+s70."""
        gr = self.GraphBuilder(18, ir_version=9)
        gr.add_to_constraints("s52+seq_length", "s52+s70")
        gr._improve_constraints()
        constraints = gr.get_registered_constraints()
        # The method should have deduced that seq_length and s70 are equivalent
        self.assertIn("seq_length", constraints)
        self.assertIn("s70", constraints["seq_length"])
        self.assertIn("s70", constraints)
        self.assertIn("seq_length", constraints["s70"])

    def test_improve_constraints_adds_renamed_expressions(self):
        """_improve_constraints should also link the renamed expressions."""
        gr = self.GraphBuilder(18, ir_version=9)
        gr.add_to_constraints("s52+seq_length", "s52+s70")
        gr._improve_constraints()
        constraints = gr.get_registered_constraints()
        # s52+s70 and s52+seq_length should be linked to each other
        self.assertIn("s52+seq_length", constraints)
        self.assertIn("s52+s70", constraints["s52+seq_length"])
        self.assertIn("s52+s70", constraints)
        self.assertIn("s52+seq_length", constraints["s52+s70"])

    def test_improve_constraints_integer_values_skipped(self):
        """_improve_constraints should skip constraints containing integers."""
        gr = self.GraphBuilder(18, ir_version=9)
        gr.add_to_constraints("batch", 4)
        gr._improve_constraints()
        constraints = gr.get_registered_constraints()
        # Integer constraint should remain unchanged
        self.assertIn("batch", constraints)
        self.assertIn(4, constraints["batch"])

    def test_improve_constraints_skips_equal_expressions(self):
        """_improve_constraints should not add constraints when simplification
        does not yield exactly two terms."""
        gr = self.GraphBuilder(18, ir_version=9)
        # e*2 == e+e simplifies to {} (equal expressions), so no new constraints
        gr.add_to_constraints("e*2", "e+e")
        gr._improve_constraints()
        constraints = gr.get_registered_constraints()
        # No new equivalences should be deduced
        self.assertNotIn("e", constraints)


@requires_torch()
class TestTorchSymIntTorch(ExtTestCase):
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
        from yobx.xbuilder import GraphBuilder

        cls.builder = GraphBuilder(18, ir_version=9)
        cls.WrapDim = GraphBuilder.WrapDim

    def test_str_name(self):
        """A plain string name is returned as-is."""
        wd = self.WrapDim("batch")
        self.assertEqual(wd.name_as_string, "batch")

    def test_unknown_type_raises(self):
        """An unknown type raises AssertionError."""

        class _Unknown:
            pass

        wd = self.WrapDim(_Unknown())
        with self.assertRaises(AssertionError):
            _ = wd.name_as_string


@requires_torch()
class TestWrapDimNameAsStringTorch(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from yobx.xbuilder import GraphBuilder

        cls.torch = torch
        cls.builder = GraphBuilder(18, ir_version=9)
        cls.WrapDim = GraphBuilder.WrapDim

    def test_torch_dim_name(self):
        """A torch.export.Dim object returns its __name__ attribute."""
        dim = self.torch.export.Dim("seq")
        wd = self.WrapDim(dim)
        self.assertEqual(wd.name_as_string, "seq")


if __name__ == "__main__":
    unittest.main(verbosity=2)

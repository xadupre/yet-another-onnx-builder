"""
Tests for the opset query API (``main_opset``, ``has_opset``, ``get_opset``,
``set_opset``) on :class:`~yobx.builder.light.OnnxGraph`.

This validates the API described in ``docs/design/sklearn/expected_api.rst``
under the "Opset API" section.
"""

import unittest
from yobx.builder.light import OnnxGraph, start
from yobx.ext_test_case import ExtTestCase


class TestLightOpsetApi(ExtTestCase):
    """Validates opset API on :class:`~yobx.builder.light.OnnxGraph`."""

    # ------------------------------------------------------------------
    # main_opset
    # ------------------------------------------------------------------

    def test_main_opset_explicit(self):
        g = OnnxGraph(opset=18)
        self.assertEqual(g.main_opset, 18)

    def test_main_opset_from_dict(self):
        g = OnnxGraph(opsets={"": 20, "ai.onnx.ml": 3})
        self.assertEqual(g.main_opset, 20)

    def test_main_opset_start_helper(self):
        g = start(opset=17)
        self.assertEqual(g.main_opset, 17)

    # ------------------------------------------------------------------
    # has_opset
    # ------------------------------------------------------------------

    def test_has_opset_main_domain_explicit(self):
        g = OnnxGraph(opset=18)
        self.assertEqual(g.has_opset(""), 18)

    def test_has_opset_secondary_domain_present(self):
        g = OnnxGraph(opsets={"": 18, "ai.onnx.ml": 3})
        self.assertEqual(g.has_opset("ai.onnx.ml"), 3)

    def test_has_opset_missing_domain_returns_zero(self):
        g = OnnxGraph(opset=18)
        self.assertEqual(g.has_opset("nonexistent.domain"), 0)

    def test_has_opset_no_opsets_dict(self):
        g = OnnxGraph(opset=18)
        self.assertEqual(g.has_opset("ai.onnx.ml"), 0)

    # ------------------------------------------------------------------
    # get_opset
    # ------------------------------------------------------------------

    def test_get_opset_main_domain(self):
        g = OnnxGraph(opset=18)
        self.assertEqual(g.get_opset(""), 18)

    def test_get_opset_secondary_domain(self):
        g = OnnxGraph(opsets={"": 18, "ai.onnx.ml": 3})
        self.assertEqual(g.get_opset("ai.onnx.ml"), 3)

    def test_get_opset_missing_exc_true_raises(self):
        g = OnnxGraph(opset=18)
        with self.assertRaises(AssertionError):
            g.get_opset("unknown.domain", exc=True)

    def test_get_opset_missing_exc_false_returns_zero(self):
        g = OnnxGraph(opset=18)
        result = g.get_opset("unknown.domain", exc=False)
        self.assertEqual(0, result)

    # ------------------------------------------------------------------
    # set_opset
    # ------------------------------------------------------------------

    def test_set_opset_new_domain(self):
        g = OnnxGraph(opset=18)
        g.set_opset("ai.onnx.ml", 3)
        self.assertEqual(g.get_opset("ai.onnx.ml"), 3)

    def test_set_opset_new_domain_no_prior_opsets(self):
        g = OnnxGraph(opset=18)
        self.assertIsNone(g.opsets)
        g.set_opset("com.example", 2)
        self.assertEqual(g.get_opset("com.example"), 2)

    def test_set_opset_existing_same_version_noop(self):
        g = OnnxGraph(opsets={"": 18, "ai.onnx.ml": 3})
        g.set_opset("ai.onnx.ml", 3)
        self.assertEqual(g.get_opset("ai.onnx.ml"), 3)

    def test_set_opset_version_mismatch_raises(self):
        g = OnnxGraph(opsets={"": 18, "ai.onnx.ml": 3})
        with self.assertRaises(AssertionError):
            g.set_opset("ai.onnx.ml", 5)

    def test_set_opset_default_version(self):
        g = OnnxGraph(opset=18)
        g.set_opset("com.example")
        self.assertEqual(g.get_opset("com.example"), 1)

    def test_set_opset_main_domain(self):
        g = OnnxGraph(opset=18)
        g.set_opset("", 18)  # same version — no-op
        self.assertEqual(g.main_opset, 18)

    def test_set_opset_main_domain_mismatch_raises(self):
        g = OnnxGraph(opset=18)
        with self.assertRaises(AssertionError):
            g.set_opset("", 20)

    def test_set_opset_reflected_in_to_onnx(self):
        """set_opset domain must appear in the exported model's opset_imports."""
        import onnx
        from onnx import TensorProto

        g = OnnxGraph(opset=18)
        g.set_opset("ai.onnx.ml", 3)
        inp = g.make_input("X", TensorProto.FLOAT, [None, 4])
        node = g.make_node("Identity", inp, domain="")
        g.make_output(node.output[0], TensorProto.FLOAT, [None, 4])
        model = g.to_onnx()
        domains = {opset.domain: opset.version for opset in model.opset_import}
        self.assertIn("ai.onnx.ml", domains)
        self.assertEqual(domains["ai.onnx.ml"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

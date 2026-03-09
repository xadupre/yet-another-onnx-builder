"""
Tests for the opset query API (``main_opset``, ``get_opset``, ``set_opset``)
on :class:`~yobx.xbuilder.GraphBuilder`.

This validates the API described in ``docs/design/sklearn/expected_api.rst``
under the "Opset API" section.
"""

import unittest
from yobx.ext_test_case import ExtTestCase
from yobx.xbuilder import GraphBuilder


class TestGraphBuilderOpsetApi(ExtTestCase):
    """Validates opset API on :class:`~yobx.xbuilder.GraphBuilder`."""

    # ------------------------------------------------------------------
    # main_opset
    # ------------------------------------------------------------------

    def test_main_opset_int_init(self):
        g = GraphBuilder(18, ir_version=10)
        self.assertEqual(g.main_opset, 18)

    def test_main_opset_dict_init(self):
        g = GraphBuilder({"": 20, "ai.onnx.ml": 3}, ir_version=10)
        self.assertEqual(g.main_opset, 20)

    # ------------------------------------------------------------------
    # get_opset
    # ------------------------------------------------------------------

    def test_get_opset_main_domain(self):
        g = GraphBuilder(18, ir_version=10)
        self.assertEqual(g.get_opset(""), 18)

    def test_get_opset_secondary_domain(self):
        g = GraphBuilder({"": 18, "ai.onnx.ml": 3}, ir_version=10)
        self.assertEqual(g.get_opset("ai.onnx.ml"), 3)

    def test_get_opset_missing_exc_true(self):
        g = GraphBuilder(18, ir_version=10)
        with self.assertRaises(AssertionError):
            g.get_opset("unknown.domain", exc=True)

    def test_get_opset_missing_exc_false(self):
        g = GraphBuilder(18, ir_version=10)
        result = g.get_opset("unknown.domain", exc=False)
        self.assertEqual(0, result)

    # ------------------------------------------------------------------
    # set_opset
    # ------------------------------------------------------------------

    def test_set_opset_new(self):
        g = GraphBuilder(18, ir_version=10)
        g.set_opset("ai.onnx.ml", 3)
        self.assertEqual(g.get_opset("ai.onnx.ml"), 3)

    def test_set_opset_existing_same_version_noop(self):
        g = GraphBuilder({"": 18, "ai.onnx.ml": 3}, ir_version=10)
        # Should not raise; version matches
        g.set_opset("ai.onnx.ml", 3)
        self.assertEqual(g.get_opset("ai.onnx.ml"), 3)

    def test_set_opset_version_mismatch_raises(self):
        g = GraphBuilder({"": 18, "ai.onnx.ml": 3}, ir_version=10)
        with self.assertRaises(AssertionError):
            g.set_opset("ai.onnx.ml", 5)

    def test_set_opset_default_version(self):
        g = GraphBuilder(18, ir_version=10)
        g.set_opset("com.example")
        self.assertEqual(g.get_opset("com.example"), 1)

    # ------------------------------------------------------------------
    # add_domain (deprecated alias for set_opset)
    # ------------------------------------------------------------------

    def test_add_domain_new(self):
        g = GraphBuilder(18, ir_version=10)
        g.add_domain("ai.onnx.ml", 3)
        self.assertEqual(g.get_opset("ai.onnx.ml"), 3)

    def test_add_domain_existing_same_version_noop(self):
        g = GraphBuilder({"": 18, "ai.onnx.ml": 3}, ir_version=10)
        # Should not raise; version matches
        g.add_domain("ai.onnx.ml", 3)
        self.assertEqual(g.get_opset("ai.onnx.ml"), 3)

    def test_add_domain_version_mismatch_raises(self):
        g = GraphBuilder({"": 18, "ai.onnx.ml": 3}, ir_version=10)
        with self.assertRaises(AssertionError):
            g.add_domain("ai.onnx.ml", 5)

    def test_add_domain_default_version(self):
        g = GraphBuilder(18, ir_version=10)
        g.add_domain("com.example")
        self.assertEqual(g.get_opset("com.example"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

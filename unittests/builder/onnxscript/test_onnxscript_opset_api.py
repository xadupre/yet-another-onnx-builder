"""
Tests for the opset query API (``main_opset``, ``get_opset``, ``add_domain``)
on :class:`~yobx.builder.onnxscript.OnnxScriptGraphBuilder`.

This validates the API described in ``docs/design/sklearn/expected_api.rst``
under the "Opset API" section.
"""

import unittest
from yobx.ext_test_case import ExtTestCase, requires_onnxscript


@requires_onnxscript()
class TestOnnxScriptGraphBuilderOpsetApi(ExtTestCase):
    """Validates opset API on :class:`~yobx.builder.onnxscript.OnnxScriptGraphBuilder`."""

    def _make_builder(self, opset=18):
        from yobx.builder.onnxscript import OnnxScriptGraphBuilder

        return OnnxScriptGraphBuilder(opset)

    # ------------------------------------------------------------------
    # main_opset
    # ------------------------------------------------------------------

    def test_main_opset_int_init(self):
        from yobx.builder.onnxscript import OnnxScriptGraphBuilder

        g = OnnxScriptGraphBuilder(18)
        self.assertEqual(g.main_opset, 18)

    def test_main_opset_dict_init(self):
        from yobx.builder.onnxscript import OnnxScriptGraphBuilder

        g = OnnxScriptGraphBuilder({"": 20, "ai.onnx.ml": 3})
        self.assertEqual(g.main_opset, 20)

    # ------------------------------------------------------------------
    # get_opset
    # ------------------------------------------------------------------

    def test_get_opset_main_domain(self):
        g = self._make_builder(18)
        self.assertEqual(g.get_opset(""), 18)

    def test_get_opset_secondary_domain(self):
        from yobx.builder.onnxscript import OnnxScriptGraphBuilder

        g = OnnxScriptGraphBuilder({"": 18, "ai.onnx.ml": 3})
        self.assertEqual(g.get_opset("ai.onnx.ml"), 3)

    def test_get_opset_missing_exc_true(self):
        g = self._make_builder()
        with self.assertRaises(AssertionError):
            g.get_opset("unknown.domain", exc=True)

    def test_get_opset_missing_exc_false(self):
        g = self._make_builder()
        result = g.get_opset("unknown.domain", exc=False)
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # add_domain
    # ------------------------------------------------------------------

    def test_add_domain_new(self):
        g = self._make_builder()
        g.add_domain("ai.onnx.ml", 3)
        self.assertEqual(g.get_opset("ai.onnx.ml"), 3)

    def test_add_domain_new_reflected_in_graph(self):
        """add_domain must also update the underlying ir.Graph opset_imports."""
        g = self._make_builder()
        g.add_domain("ai.onnx.ml", 3)
        self.assertIn("ai.onnx.ml", g._graph.opset_imports)
        self.assertEqual(g._graph.opset_imports["ai.onnx.ml"], 3)

    def test_add_domain_existing_same_version_noop(self):
        from yobx.builder.onnxscript import OnnxScriptGraphBuilder

        g = OnnxScriptGraphBuilder({"": 18, "ai.onnx.ml": 3})
        g.add_domain("ai.onnx.ml", 3)
        self.assertEqual(g.get_opset("ai.onnx.ml"), 3)

    def test_add_domain_version_mismatch_raises(self):
        from yobx.builder.onnxscript import OnnxScriptGraphBuilder

        g = OnnxScriptGraphBuilder({"": 18, "ai.onnx.ml": 3})
        with self.assertRaises(AssertionError):
            g.add_domain("ai.onnx.ml", 5)

    def test_add_domain_default_version(self):
        g = self._make_builder()
        g.add_domain("com.example")
        self.assertEqual(g.get_opset("com.example"), 1)

    def test_add_domain_exported_model_contains_domain(self):
        """add_domain is reflected in the exported ONNX ModelProto."""
        import onnx

        g = self._make_builder()
        g.add_domain("ai.onnx.ml", 3)
        g.make_tensor_input("X", onnx.TensorProto.FLOAT, (None, 4))
        g.make_node("Relu", ["X"], ["Y"])
        g.make_tensor_output("Y", onnx.TensorProto.FLOAT)
        proto = g.to_onnx()
        domains = {oi.domain for oi in proto.opset_import}
        self.assertIn("ai.onnx.ml", domains)


if __name__ == "__main__":
    unittest.main(verbosity=2)

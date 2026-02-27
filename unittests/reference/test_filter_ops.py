import unittest
import onnx.helper as oh
from onnx.reference.op_run import OpRun
from yobx.ext_test_case import ExtTestCase
from yobx.reference import ExtendedReferenceEvaluator

TFLOAT = 1  # onnx.TensorProto.FLOAT


class TestFilterOps(ExtTestCase):
    def test_non_versioned_ops_pass_through(self):
        """Non-versioned ops (no '_<int>' suffix) should pass through unchanged."""

        class NonVersionedOp(OpRun):
            op_domain = "test.domain"
            op_schema = None

        result = ExtendedReferenceEvaluator.filter_ops(None, [NonVersionedOp], {})
        self.assertIn(NonVersionedOp, result)

    def test_versioned_op_exceeds_opset_is_not_renamed(self):
        """A versioned op whose version exceeds the opset version is not renamed.

        The original versioned class is retained as-is, but no un-versioned alias
        is created for it because it was not compatible with the declared opset.
        """

        class MyOp_2(OpRun):
            op_domain = "test.domain"
            op_schema = None

        # opset version 1 < op version 2 → no un-versioned "MyOp" alias is created
        result = ExtendedReferenceEvaluator.filter_ops(None, [MyOp_2], {"test.domain": 1})
        result_names = [cl.__name__ for cl in result]
        self.assertNotIn("MyOp", result_names)
        # The original versioned class is kept as-is (not silently dropped)
        self.assertIn("MyOp_2", result_names)

    def test_versioned_op_within_opset_is_included(self):
        """A versioned op whose version is within the opset version should be included."""

        class MyOp_1(OpRun):
            op_domain = "test.domain"
            op_schema = None

        # opset version 2 >= op version 1 → op should be included as "MyOp"
        result = ExtendedReferenceEvaluator.filter_ops(None, [MyOp_1], {"test.domain": 2})
        result_names = [cl.__name__ for cl in result]
        self.assertIn("MyOp", result_names)

    def test_highest_compatible_version_is_selected(self):
        """When multiple versions are compatible, the highest version should be selected."""

        class MyOp_1(OpRun):
            op_domain = "test.domain"
            op_schema = None

        class MyOp_2(OpRun):
            op_domain = "test.domain"
            op_schema = None

        # Both versions are within opset 3; MyOp_2 should win
        result = ExtendedReferenceEvaluator.filter_ops(None, [MyOp_1, MyOp_2], {"test.domain": 3})
        result_names = [cl.__name__ for cl in result]
        self.assertIn("MyOp", result_names)
        self.assertNotIn("MyOp_1", result_names)
        self.assertNotIn("MyOp_2", result_names)
        myop_cls = next(cl for cl in result if cl.__name__ == "MyOp")
        self.assertTrue(issubclass(myop_cls, MyOp_2))

    def test_no_opsets_keeps_highest_version(self):
        """With opsets=None and a plain proto, all versioned ops are kept (highest wins)."""

        class MyOp_1(OpRun):
            op_domain = "test.domain"
            op_schema = None

        class MyOp_2(OpRun):
            op_domain = "test.domain"
            op_schema = None

        result = ExtendedReferenceEvaluator.filter_ops(None, [MyOp_1, MyOp_2], None)
        result_names = [cl.__name__ for cl in result]
        self.assertIn("MyOp", result_names)
        myop_cls = next(cl for cl in result if cl.__name__ == "MyOp")
        self.assertTrue(issubclass(myop_cls, MyOp_2))

    def test_mixed_versioned_and_non_versioned(self):
        """Non-versioned ops are kept as-is; versioned ops are merged by highest version."""

        class OtherOp(OpRun):
            op_domain = "test.domain"
            op_schema = None

        class MyOp_1(OpRun):
            op_domain = "test.domain"
            op_schema = None

        class MyOp_2(OpRun):
            op_domain = "test.domain"
            op_schema = None

        result = ExtendedReferenceEvaluator.filter_ops(
            None, [OtherOp, MyOp_1, MyOp_2], {"test.domain": 2}
        )
        result_names = [cl.__name__ for cl in result]
        self.assertIn("OtherOp", result_names)
        self.assertIn("MyOp", result_names)
        self.assertNotIn("MyOp_1", result_names)
        self.assertNotIn("MyOp_2", result_names)

    def test_opsets_from_model_proto(self):
        """When proto is a ModelProto and opsets=None, opsets are inferred from the model."""

        class MyOp_1(OpRun):
            op_domain = "test.domain"
            op_schema = None

        class MyOp_2(OpRun):
            op_domain = "test.domain"
            op_schema = None

        # Build a minimal model with opset version 1 for the domain
        model = oh.make_model(
            oh.make_graph([], "g", [], []),
            opset_imports=[
                oh.make_opsetid("", 18),
                oh.make_opsetid("test.domain", 1),
            ],
        )

        # With domain opset=1, MyOp_2 (version 2) should be excluded
        result = ExtendedReferenceEvaluator.filter_ops(model, [MyOp_1, MyOp_2], None)
        result_names = [cl.__name__ for cl in result]
        self.assertIn("MyOp", result_names)
        myop_cls = next(cl for cl in result if cl.__name__ == "MyOp")
        self.assertTrue(issubclass(myop_cls, MyOp_1))


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
Unit tests for yobx.sklearn.tree decision tree converters.
"""

import unittest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnDecisionTree(ExtTestCase):
    def test_decision_tree_classifier_binary(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,))

        # Check graph structure
        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertLess(ml_opsets["ai.onnx.ml"], 5)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsembleClassifier", op_types)

        # Check outputs
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = dt.predict(X)
        expected_proba = dt.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_decision_tree_classifier_multiclass(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 2, 2])
        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,))

        # Check graph structure
        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertLess(ml_opsets["ai.onnx.ml"], 5)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsembleClassifier", op_types)

        # Check outputs
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = dt.predict(X)
        expected_proba = dt.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_decision_tree_regressor(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        dt = DecisionTreeRegressor(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,))

        # Check graph structure
        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertLess(ml_opsets["ai.onnx.ml"], 5)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsembleRegressor", op_types)

        # Check outputs
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected_predictions = dt.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected_predictions, predictions, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_predictions, ort_results[0], atol=1e-5)

    def test_pipeline_decision_tree_classifier(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("clf", DecisionTreeClassifier(random_state=0))]
        )
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        # Check graph contains nodes from both steps
        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertLess(ml_opsets["ai.onnx.ml"], 5)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("Div", op_types)
        self.assertIn("TreeEnsembleClassifier", op_types)

        # Check outputs
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = pipe.predict(X)
        expected_proba = pipe.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_decision_tree_classifier_binary_v5(self):
        """TreeEnsemble (ai.onnx.ml opset 5) - binary classification."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertEqual(ml_opsets["ai.onnx.ml"], 5)

        # Should use the new unified operator, not the legacy one
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsemble", op_types)
        self.assertNotIn("TreeEnsembleClassifier", op_types)

        # Numerical correctness
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = dt.predict(X)
        expected_proba = dt.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_decision_tree_classifier_multiclass_v5(self):
        """TreeEnsemble (ai.onnx.ml opset 5) - multi-class classification."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 2, 2])
        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertEqual(ml_opsets["ai.onnx.ml"], 5)

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsemble", op_types)
        self.assertNotIn("TreeEnsembleClassifier", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = dt.predict(X)
        expected_proba = dt.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_decision_tree_regressor_v5(self):
        """TreeEnsemble (ai.onnx.ml opset 5) - regression."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        dt = DecisionTreeRegressor(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertEqual(ml_opsets["ai.onnx.ml"], 5)

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsemble", op_types)
        self.assertNotIn("TreeEnsembleRegressor", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected_predictions = dt.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected_predictions, predictions, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_predictions, ort_results[0], atol=1e-5)

    def test_decision_tree_legacy_opset_unchanged(self):
        """Passing an integer target_opset still emits legacy operators."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y_cls = np.array([0, 0, 1, 1])
        y_reg = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)

        dtc = DecisionTreeClassifier(random_state=0)
        dtc.fit(X, y_cls)
        onx_c = to_onnx(dtc, (X,), target_opset=20)
        ml_opsets_c = {op.domain: op.version for op in onx_c.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets_c)
        self.assertLess(ml_opsets_c["ai.onnx.ml"], 5)
        op_types_c = [n.op_type for n in onx_c.graph.node]
        self.assertIn("TreeEnsembleClassifier", op_types_c)
        self.assertNotIn("TreeEnsemble", op_types_c)

        dtr = DecisionTreeRegressor(random_state=0)
        dtr.fit(X, y_reg)
        onx_r = to_onnx(dtr, (X,), target_opset=20)
        ml_opsets_r = {op.domain: op.version for op in onx_r.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets_r)
        self.assertLess(ml_opsets_r["ai.onnx.ml"], 5)
        op_types_r = [n.op_type for n in onx_r.graph.node]
        self.assertIn("TreeEnsembleRegressor", op_types_r)
        self.assertNotIn("TreeEnsemble", op_types_r)

    def test_pipeline_decision_tree_classifier_v5(self):
        """Pipeline with TreeEnsemble (ai.onnx.ml opset 5)."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("clf", DecisionTreeClassifier(random_state=0))]
        )
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertEqual(ml_opsets["ai.onnx.ml"], 5)

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsemble", op_types)
        self.assertNotIn("TreeEnsembleClassifier", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = pipe.predict(X)
        expected_proba = pipe.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_decision_tree_classifier_float32(self):
        """DecisionTreeClassifier works with float32 input."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 2, 2])
        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,))

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertLess(ml_opsets["ai.onnx.ml"], 5)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = dt.predict(X)
        expected_proba = dt.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_decision_tree_classifier_float64(self):
        """DecisionTreeClassifier works with float64 input."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float64)
        y = np.array([0, 0, 1, 1, 2, 2])
        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,))
        self.print_onnx(onx)

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertLess(ml_opsets["ai.onnx.ml"], 5)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = dt.predict(X)
        expected_proba = dt.predict_proba(X)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba.astype(np.float64), atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1].astype(np.float64), atol=1e-5)

    def test_decision_tree_regressor_float32(self):
        """DecisionTreeRegressor works with float32 input."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        dt = DecisionTreeRegressor(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,))

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertLess(ml_opsets["ai.onnx.ml"], 5)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected_predictions = dt.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected_predictions, predictions, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_predictions, ort_results[0], atol=1e-5)

    def test_decision_tree_regressor_float64(self):
        """DecisionTreeRegressor works with float64 input."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
        y = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
        dt = DecisionTreeRegressor(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,))

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertLess(ml_opsets["ai.onnx.ml"], 5)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected_predictions = dt.predict(X).reshape(-1, 1)
        self.assertEqualArray(expected_predictions, predictions, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_predictions, ort_results[0], atol=1e-5)

    def test_decision_tree_classifier_float32_v5(self):
        """DecisionTreeClassifier, float32 input, ai.onnx.ml opset 5."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 2, 2])
        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertEqual(ml_opsets["ai.onnx.ml"], 5)

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsemble", op_types)
        self.assertNotIn("TreeEnsembleClassifier", op_types)

        # nodes_splits tensor must be float32 when input is float32
        for node in onx.graph.node:
            if node.op_type == "TreeEnsemble":
                for attr in node.attribute:
                    if attr.name == "nodes_splits":
                        self.assertEqual(attr.t.data_type, 1)  # TensorProto.FLOAT

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqual(proba.dtype, np.float32)
        self.assertEqualArray(dt.predict(X), label)
        self.assertEqualArray(dt.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_decision_tree_classifier_float64_v5(self):
        """DecisionTreeClassifier, float64 input, ai.onnx.ml opset 5."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float64)
        y = np.array([0, 0, 1, 1, 2, 2])
        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertEqual(ml_opsets["ai.onnx.ml"], 5)

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsemble", op_types)
        self.assertNotIn("TreeEnsembleClassifier", op_types)

        # nodes_splits tensor must be float64 when input is float64
        for node in onx.graph.node:
            if node.op_type == "TreeEnsemble":
                for attr in node.attribute:
                    if attr.name == "nodes_splits":
                        self.assertEqual(attr.t.data_type, 11)  # TensorProto.DOUBLE

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqual(proba.dtype, np.float64)
        self.assertEqualArray(dt.predict(X), label)
        self.assertEqualArray(dt.predict_proba(X), proba, atol=1e-5)

    def test_decision_tree_regressor_float32_v5(self):
        """DecisionTreeRegressor, float32 input, ai.onnx.ml opset 5."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        dt = DecisionTreeRegressor(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertEqual(ml_opsets["ai.onnx.ml"], 5)

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsemble", op_types)
        self.assertNotIn("TreeEnsembleRegressor", op_types)

        # nodes_splits tensor must be float32 when input is float32
        for node in onx.graph.node:
            if node.op_type == "TreeEnsemble":
                for attr in node.attribute:
                    if attr.name == "nodes_splits":
                        self.assertEqual(attr.t.data_type, 1)  # TensorProto.FLOAT

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        self.assertEqual(predictions.dtype, np.float32)
        self.assertEqualArray(
            dt.predict(X).astype(np.float32).reshape(-1, 1), predictions, atol=1e-5
        )

    def test_decision_tree_regressor_float64_v5(self):
        """DecisionTreeRegressor, float64 input, ai.onnx.ml opset 5."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
        y = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
        dt = DecisionTreeRegressor(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertEqual(ml_opsets["ai.onnx.ml"], 5)

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsemble", op_types)
        self.assertNotIn("TreeEnsembleRegressor", op_types)

        # nodes_splits tensor must be float64 when input is float64
        for node in onx.graph.node:
            if node.op_type == "TreeEnsemble":
                for attr in node.attribute:
                    if attr.name == "nodes_splits":
                        self.assertEqual(attr.t.data_type, 11)  # TensorProto.DOUBLE

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        self.assertEqual(predictions.dtype, np.float64)
        self.assertEqualArray(dt.predict(X).reshape(-1, 1), predictions, atol=1e-5)

    def test_decision_tree_classifier_float32_opset3(self):
        """DecisionTreeClassifier, float32 input, ai.onnx.ml opset 3 (legacy)."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 2, 2])
        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,), target_opset={"": 20, "ai.onnx.ml": 3})

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertEqual(ml_opsets["ai.onnx.ml"], 3)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsembleClassifier", op_types)
        self.assertNotIn("TreeEnsemble", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(dt.predict(X), label)
        self.assertEqualArray(dt.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_decision_tree_classifier_float64_opset3(self):
        """DecisionTreeClassifier, float64 input, ai.onnx.ml opset 3 (legacy)."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float64)
        y = np.array([0, 0, 1, 1, 2, 2])
        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,), target_opset={"": 20, "ai.onnx.ml": 3})

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertEqual(ml_opsets["ai.onnx.ml"], 3)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsembleClassifier", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(dt.predict(X), label)
        self.assertEqualArray(dt.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_decision_tree_regressor_float32_opset3(self):
        """DecisionTreeRegressor, float32 input, ai.onnx.ml opset 3 (legacy)."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        dt = DecisionTreeRegressor(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,), target_opset={"": 20, "ai.onnx.ml": 3})

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertEqual(ml_opsets["ai.onnx.ml"], 3)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsembleRegressor", op_types)
        self.assertNotIn("TreeEnsemble", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        self.assertEqualArray(
            dt.predict(X).astype(np.float32).reshape(-1, 1), predictions, atol=1e-5
        )

    def test_decision_tree_regressor_float64_opset3(self):
        """DecisionTreeRegressor, float64 input, ai.onnx.ml opset 3 (legacy)."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
        y = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
        dt = DecisionTreeRegressor(random_state=0)
        dt.fit(X, y)

        onx = to_onnx(dt, (X,), target_opset={"": 20, "ai.onnx.ml": 3})

        ml_opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertIn("ai.onnx.ml", ml_opsets)
        self.assertEqual(ml_opsets["ai.onnx.ml"], 3)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("TreeEnsembleRegressor", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        self.assertEqualArray(
            dt.predict(X).astype(np.float64).reshape(-1, 1), predictions, atol=1e-5
        )

    # ------------------------------------------------------------------ #
    # Random tensor discrepancy tests                                      #
    # ------------------------------------------------------------------ #

    def test_decision_tree_classifier_random_float32(self):
        """DecisionTreeClassifier: no discrepancy on real random float32 tensors."""
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((100, 5)).astype(np.float32)
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
        X_test = rng.standard_normal((50, 5)).astype(np.float32)

        dt = DecisionTreeClassifier(max_depth=5, random_state=0)
        dt.fit(X_train, y_train)

        for opset_kw in [{}, {"target_opset": {"": 20, "ai.onnx.ml": 5}}]:
            onx = to_onnx(dt, (X_train,), **opset_kw)

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": X_test})
            label, proba = results[0], results[1]

            self.assertEqualArray(dt.predict(X_test), label)
            self.assertEqualArray(
                dt.predict_proba(X_test).astype(np.float32), proba, atol=1e-5
            )

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": X_test})
            self.assertEqualArray(dt.predict(X_test), ort_results[0])
            self.assertEqualArray(
                dt.predict_proba(X_test).astype(np.float32), ort_results[1], atol=1e-5
            )

    def test_decision_tree_classifier_random_float64(self):
        """DecisionTreeClassifier: no discrepancy on real random float64 tensors."""
        rng = np.random.default_rng(1)
        X_train = rng.standard_normal((100, 5))
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
        X_test = rng.standard_normal((50, 5))

        dt = DecisionTreeClassifier(max_depth=5, random_state=0)
        dt.fit(X_train, y_train)

        for opset_kw in [{}, {"target_opset": {"": 20, "ai.onnx.ml": 5}}]:
            onx = to_onnx(dt, (X_train,), **opset_kw)

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": X_test})
            label, proba = results[0], results[1]

            expected_proba_raw = dt.predict_proba(X_test)
            self.assertEqualArray(dt.predict(X_test), label)
            self.assertEqualArray(expected_proba_raw.astype(proba.dtype), proba, atol=1e-5)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": X_test})
            self.assertEqualArray(dt.predict(X_test), ort_results[0])
            self.assertEqualArray(
                expected_proba_raw.astype(ort_results[1].dtype), ort_results[1], atol=1e-5
            )

    def test_decision_tree_regressor_random_float32(self):
        """DecisionTreeRegressor: no discrepancy on real random float32 tensors."""
        rng = np.random.default_rng(2)
        X_train = rng.standard_normal((100, 5)).astype(np.float32)
        y_train = (X_train[:, 0] * 2 + X_train[:, 1]).astype(np.float32)
        X_test = rng.standard_normal((50, 5)).astype(np.float32)

        dt = DecisionTreeRegressor(max_depth=5, random_state=0)
        dt.fit(X_train, y_train)

        for opset_kw in [{}, {"target_opset": {"": 20, "ai.onnx.ml": 5}}]:
            onx = to_onnx(dt, (X_train,), **opset_kw)

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": X_test})
            predictions = results[0]

            expected = dt.predict(X_test).astype(np.float32).reshape(-1, 1)
            self.assertEqualArray(expected, predictions, atol=1e-5)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": X_test})
            self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_decision_tree_regressor_random_float64(self):
        """DecisionTreeRegressor: no discrepancy on real random float64 tensors."""
        rng = np.random.default_rng(3)
        X_train = rng.standard_normal((100, 5))
        y_train = X_train[:, 0] * 2 + X_train[:, 1]
        X_test = rng.standard_normal((50, 5))

        dt = DecisionTreeRegressor(max_depth=5, random_state=0)
        dt.fit(X_train, y_train)

        for opset_kw in [{}, {"target_opset": {"": 20, "ai.onnx.ml": 5}}]:
            onx = to_onnx(dt, (X_train,), **opset_kw)

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": X_test})
            predictions = results[0]

            expected = dt.predict(X_test).reshape(-1, 1)
            self.assertEqualArray(expected, predictions, atol=1e-5)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": X_test})
            self.assertEqualArray(expected, ort_results[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

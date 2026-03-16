import unittest
import numpy as np
import onnxruntime
import onnx
import onnx.helper as oh
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split
from yobx import DEFAULT_TARGET_OPSET as TARGET_OPSET
from yobx.ext_test_case import ExtTestCase
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx, ConvertOptions


class TestSklearnEnsembleConverters(ExtTestCase):
    def test_model_random_forest_classifier(self):
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        model_onnx = to_onnx(
            model,
            [
                oh.make_tensor_value_info(
                    "input", onnx.TensorProto.FLOAT, [None, X_train.shape[1]]
                )
            ],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        feeds = {model_onnx.graph.input[0].name: X_test}
        sess = onnxruntime.InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        ort_out = sess.run(None, feeds)
        ref_out = ExtendedReferenceEvaluator(model_onnx).run(None, feeds)
        expected_labels = model.predict(X_test)
        expected_proba = model.predict_proba(X_test).astype(np.float32)
        np.testing.assert_array_equal(ort_out[0], expected_labels)
        np.testing.assert_allclose(ort_out[1], expected_proba, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(ref_out[0], expected_labels)
        np.testing.assert_allclose(ref_out[1], expected_proba, rtol=1e-5, atol=1e-5)

    def test_model_random_forest_regressor(self):
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        model_onnx = to_onnx(
            model,
            [
                oh.make_tensor_value_info(
                    "input", onnx.TensorProto.FLOAT, [None, X_train.shape[1]]
                )
            ],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        feeds = {model_onnx.graph.input[0].name: X_test}
        sess = onnxruntime.InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        ort_out = sess.run(None, feeds)[0].flatten()
        ref_out = ExtendedReferenceEvaluator(model_onnx).run(None, feeds)[0].flatten()
        expected = model.predict(X_test).astype(np.float32)
        np.testing.assert_allclose(ort_out, expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ref_out, expected, rtol=1e-5, atol=1e-5)

    def test_model_gradient_boosting_classifier(self):
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=42)
        model.fit(X_train, y_train)
        model_onnx = to_onnx(
            model,
            [
                oh.make_tensor_value_info(
                    "input", onnx.TensorProto.FLOAT, [None, X_train.shape[1]]
                )
            ],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        feeds = {model_onnx.graph.input[0].name: X_test}
        sess = onnxruntime.InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        ort_out = sess.run(None, feeds)
        ref_out = ExtendedReferenceEvaluator(model_onnx).run(None, feeds)
        expected_labels = model.predict(X_test)
        expected_proba = model.predict_proba(X_test).astype(np.float32)
        np.testing.assert_array_equal(ort_out[0], expected_labels)
        np.testing.assert_allclose(ort_out[1], expected_proba, rtol=1e-5, atol=1e-4)
        np.testing.assert_array_equal(ref_out[0], expected_labels)
        np.testing.assert_allclose(ref_out[1], expected_proba, rtol=1e-5, atol=1e-4)

    def test_model_gradient_boosting_regressor(self):
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = GradientBoostingRegressor(n_estimators=10, max_depth=2, random_state=42)
        model.fit(X_train, y_train)
        model_onnx = to_onnx(
            model,
            [
                oh.make_tensor_value_info(
                    "input", onnx.TensorProto.FLOAT, [None, X_train.shape[1]]
                )
            ],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        feeds = {model_onnx.graph.input[0].name: X_test}
        sess = onnxruntime.InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        ort_out = sess.run(None, feeds)[0].flatten()
        ref_out = ExtendedReferenceEvaluator(model_onnx).run(None, feeds)[0].flatten()
        expected = model.predict(X_test).astype(np.float32)
        np.testing.assert_allclose(ort_out, expected, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(ref_out, expected, rtol=1e-5, atol=1e-4)

    def test_model_random_forest_classifier_decision_path(self):
        """Check extra decision_path output for RandomForestClassifier."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        model_onnx = to_onnx(
            model,
            [
                oh.make_tensor_value_info(
                    "input", onnx.TensorProto.FLOAT, [None, X_train.shape[1]]
                )
            ],
            target_opset=TARGET_OPSET,
            convert_options=ConvertOptions(decision_path=True),
        )
        self.assertTrue(model_onnx is not None)
        self.assertEqual(len(model_onnx.graph.output), 3)
        feeds = {model_onnx.graph.input[0].name: X_test}
        sess = onnxruntime.InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        ort_out = sess.run(None, feeds)
        ref_out = ExtendedReferenceEvaluator(model_onnx).run(None, feeds)
        # Verify labels and probabilities
        expected_labels = model.predict(X_test)
        expected_proba = model.predict_proba(X_test).astype(np.float32)
        np.testing.assert_array_equal(ort_out[0], expected_labels)
        np.testing.assert_allclose(ort_out[1], expected_proba, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(ref_out[0], expected_labels)
        np.testing.assert_allclose(ref_out[1], expected_proba, rtol=1e-5, atol=1e-5)
        # Verify the extra decision_path output matches between ORT and the reference evaluator
        np.testing.assert_array_equal(ort_out[2], ref_out[2])
        # decision_path for an ensemble has shape (n_samples, n_estimators)
        self.assertEqual(ort_out[2].ndim, 2)
        self.assertEqual(ort_out[2].shape[0], X_test.shape[0])
        self.assertEqual(ort_out[2].shape[1], model.n_estimators)
        # The decision_path output should be an integer array of non-negative indices/indicators
        self.assertTrue(np.issubdtype(ort_out[2].dtype, np.integer))
        self.assertTrue(np.issubdtype(ref_out[2].dtype, np.integer))
        self.assertTrue(np.all(ort_out[2] >= 0))
        self.assertTrue(np.all(ref_out[2] >= 0))

    def test_model_random_forest_regressor_decision_path(self):
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        model_onnx = to_onnx(
            model,
            [
                oh.make_tensor_value_info(
                    "input", onnx.TensorProto.FLOAT, [None, X_train.shape[1]]
                )
            ],
            target_opset=TARGET_OPSET,
            convert_options=ConvertOptions(decision_path=True),
        )
        self.assertTrue(model_onnx is not None)
        self.assertEqual(len(model_onnx.graph.output), 2)
        feeds = {model_onnx.graph.input[0].name: X_test}
        sess = onnxruntime.InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        ort_out = sess.run(None, feeds)
        ref_out = ExtendedReferenceEvaluator(model_onnx).run(None, feeds)
        # Verify predictions
        expected = model.predict(X_test).astype(np.float32)
        np.testing.assert_allclose(ort_out[0].flatten(), expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ref_out[0].flatten(), expected, rtol=1e-5, atol=1e-5)
        # Verify the extra decision_path output matches between ORT and the reference evaluator
        np.testing.assert_array_equal(ort_out[1], ref_out[1])
        # decision_path for an ensemble has shape (n_samples, n_estimators)
        self.assertEqual(ort_out[1].ndim, 2)
        self.assertEqual(ort_out[1].shape[0], X_test.shape[0])
        self.assertEqual(ort_out[1].shape[1], model.n_estimators)

    def test_model_extra_trees_classifier_decision_path(self):
        """Check extra decision_path output for ExtraTreesClassifier."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = ExtraTreesClassifier(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        model_onnx = to_onnx(
            model,
            [
                oh.make_tensor_value_info(
                    "input", onnx.TensorProto.FLOAT, [None, X_train.shape[1]]
                )
            ],
            target_opset=TARGET_OPSET,
            convert_options=ConvertOptions(decision_path=True),
        )
        self.assertTrue(model_onnx is not None)
        self.assertEqual(len(model_onnx.graph.output), 3)
        feeds = {model_onnx.graph.input[0].name: X_test}
        sess = onnxruntime.InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        ort_out = sess.run(None, feeds)
        ref_out = ExtendedReferenceEvaluator(model_onnx).run(None, feeds)
        # Verify labels and probabilities
        expected_labels = model.predict(X_test)
        expected_proba = model.predict_proba(X_test).astype(np.float32)
        np.testing.assert_array_equal(ort_out[0], expected_labels)
        np.testing.assert_allclose(ort_out[1], expected_proba, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(ref_out[0], expected_labels)
        np.testing.assert_allclose(ref_out[1], expected_proba, rtol=1e-5, atol=1e-5)
        # Verify the extra decision_path output matches between ORT and the reference evaluator
        np.testing.assert_array_equal(ort_out[2], ref_out[2])
        self.assertEqual(ort_out[2].ndim, 2)
        self.assertEqual(ort_out[2].shape[0], X_test.shape[0])
        self.assertEqual(ort_out[2].shape[1], model.n_estimators)

    def test_model_extra_trees_regressor_decision_path(self):
        """Check extra decision_path output for ExtraTreesRegressor."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = ExtraTreesRegressor(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        model_onnx = to_onnx(
            model,
            [
                oh.make_tensor_value_info(
                    "input", onnx.TensorProto.FLOAT, [None, X_train.shape[1]]
                )
            ],
            target_opset=TARGET_OPSET,
            convert_options=ConvertOptions(decision_path=True),
        )
        self.assertTrue(model_onnx is not None)
        self.assertEqual(len(model_onnx.graph.output), 2)
        feeds = {model_onnx.graph.input[0].name: X_test}
        sess = onnxruntime.InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        ort_out = sess.run(None, feeds)
        ref_out = ExtendedReferenceEvaluator(model_onnx).run(None, feeds)
        # Verify predictions
        expected = model.predict(X_test).astype(np.float32)
        np.testing.assert_allclose(ort_out[0].flatten(), expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ref_out[0].flatten(), expected, rtol=1e-5, atol=1e-5)
        # Verify the extra decision_path output matches between ORT and the reference evaluator
        np.testing.assert_array_equal(ort_out[1], ref_out[1])
        self.assertEqual(ort_out[1].ndim, 2)
        self.assertEqual(ort_out[1].shape[0], X_test.shape[0])
        self.assertEqual(ort_out[1].shape[1], model.n_estimators)


if __name__ == "__main__":
    unittest.main()

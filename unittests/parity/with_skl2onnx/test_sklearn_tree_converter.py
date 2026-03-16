import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnxruntime
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from yobx import DEFAULT_TARGET_OPSET as TARGET_OPSET
from yobx.sklearn import to_onnx, ConvertOptions
from yobx.ext_test_case import ExtTestCase
from yobx.reference import ExtendedReferenceEvaluator


class TestSklearnTreeConverters(ExtTestCase):
    def test_model_decision_tree_classifier(self):
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
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

    def test_model_decision_tree_regressor(self):
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = DecisionTreeRegressor(max_depth=4, random_state=42)
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

    def test_model_decision_tree_multiclass(self):
        X, y = make_classification(
            n_samples=300,
            n_features=5,
            n_classes=3,
            n_informative=3,
            n_redundant=0,
            random_state=42,
        )
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
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

    def test_model_decision_tree_classifier_decision_path(self):
        """Check extra decision_path output for DecisionTreeClassifier."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
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
        # decision_path is a 2-D array of binary strings, shape (n_samples, 1) for a single tree
        self.assertEqual(ort_out[2].ndim, 2)
        self.assertEqual(ort_out[2].shape[0], X_test.shape[0])
        # Verify the path strings match sklearn's decision_path sparse matrix
        sklearn_dp = model.decision_path(X_test).toarray()
        yobx_dp = np.array([[int(c) for c in s[0]] for s in ort_out[2]], dtype=np.int8)
        np.testing.assert_array_equal(yobx_dp, sklearn_dp)

    def test_model_decision_tree_classifier_decision_leaf(self):
        """Check extra decision_leaf output for DecisionTreeClassifier."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        model_onnx = to_onnx(
            model,
            [
                oh.make_tensor_value_info(
                    "input", onnx.TensorProto.FLOAT, [None, X_train.shape[1]]
                )
            ],
            target_opset=TARGET_OPSET,
            convert_options=ConvertOptions(decision_leaf=True),
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
        # Verify the extra decision_leaf output matches between ORT and the reference evaluator
        np.testing.assert_array_equal(ort_out[2], ref_out[2])
        # decision_leaf contains the leaf node index for each sample, shape (n_samples, 1)
        self.assertEqual(ort_out[2].ndim, 2)
        self.assertEqual(ort_out[2].shape[0], X_test.shape[0])
        expected_leaves = model.apply(X_test).reshape(-1, 1)
        np.testing.assert_array_equal(ort_out[2], expected_leaves)

    def test_model_decision_tree_regressor_decision_path(self):
        """Check extra decision_path output for DecisionTreeRegressor."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = DecisionTreeRegressor(max_depth=4, random_state=42)
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
        # Verify path strings match sklearn's decision_path
        sklearn_dp = model.decision_path(X_test).toarray()
        yobx_dp = np.array([[int(c) for c in s[0]] for s in ort_out[1]], dtype=np.int8)
        np.testing.assert_array_equal(yobx_dp, sklearn_dp)

    def test_model_decision_tree_regressor_decision_leaf(self):
        """Check extra decision_leaf output for DecisionTreeRegressor."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = DecisionTreeRegressor(max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        model_onnx = to_onnx(
            model,
            [
                oh.make_tensor_value_info(
                    "input", onnx.TensorProto.FLOAT, [None, X_train.shape[1]]
                )
            ],
            target_opset=TARGET_OPSET,
            convert_options=ConvertOptions(decision_leaf=True),
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
        # Verify the extra decision_leaf output matches between ORT and the reference evaluator
        np.testing.assert_array_equal(ort_out[1], ref_out[1])
        # decision_leaf contains the leaf node index for each sample, shape (n_samples, 1)
        self.assertEqual(ort_out[1].ndim, 2)
        self.assertEqual(ort_out[1].shape[0], X_test.shape[0])
        expected_leaves = model.apply(X_test).reshape(-1, 1)
        np.testing.assert_array_equal(ort_out[1], expected_leaves)

    def test_model_extra_tree_classifier_decision_path(self):
        """Check extra decision_path output for ExtraTreeClassifier."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = ExtraTreeClassifier(max_depth=4, random_state=42)
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
        # Verify path strings match sklearn's decision_path
        sklearn_dp = model.decision_path(X_test).toarray()
        yobx_dp = np.array([[int(c) for c in s[0]] for s in ort_out[2]], dtype=np.int8)
        np.testing.assert_array_equal(yobx_dp, sklearn_dp)

    def test_model_extra_tree_regressor_decision_path(self):
        """Check extra decision_path output for ExtraTreeRegressor."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        X = X.astype(np.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        model = ExtraTreeRegressor(max_depth=4, random_state=42)
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
        # Verify path strings match sklearn's decision_path
        sklearn_dp = model.decision_path(X_test).toarray()
        yobx_dp = np.array([[int(c) for c in s[0]] for s in ort_out[1]], dtype=np.int8)
        np.testing.assert_array_equal(yobx_dp, sklearn_dp)


if __name__ == "__main__":
    unittest.main()

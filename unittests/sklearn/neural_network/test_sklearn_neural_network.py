"""
Unit tests for yobx.sklearn.neural_network converters (MLP).
"""

import unittest
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnNeuralNetworkConverters(ExtTestCase):
    def test_mlp_classifier_binary(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=500, random_state=0)
        mlp.fit(X, y)

        onx = to_onnx(mlp, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertTrue(
            "MatMul" in op_types or "Gemm" in op_types, f"Expected MatMul or Gemm in {op_types}"
        )
        self.assertIn("Sigmoid", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = mlp.predict(X)
        expected_proba = mlp.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_mlp_classifier_multiclass(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 2, 2])
        mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=500, random_state=0)
        mlp.fit(X, y)

        onx = to_onnx(mlp, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertTrue(
            "MatMul" in op_types or "Gemm" in op_types, f"Expected MatMul or Gemm in {op_types}"
        )
        self.assertIn("Softmax", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = mlp.predict(X)
        expected_proba = mlp.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_mlp_classifier_multiple_hidden_layers(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 2, 2])
        mlp = MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=500, random_state=0)
        mlp.fit(X, y)

        onx = to_onnx(mlp, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = mlp.predict(X)
        expected_proba = mlp.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_mlp_classifier_tanh_activation(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        mlp = MLPClassifier(
            hidden_layer_sizes=(4,), activation="tanh", max_iter=500, random_state=0
        )
        mlp.fit(X, y)

        onx = to_onnx(mlp, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Tanh", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = mlp.predict(X)
        expected_proba = mlp.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_mlp_regressor(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        mlp = MLPRegressor(hidden_layer_sizes=(4,), max_iter=500, random_state=0)
        mlp.fit(X, y)

        onx = to_onnx(mlp, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertTrue(
            "MatMul" in op_types or "Gemm" in op_types, f"Expected MatMul or Gemm in {op_types}"
        )

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected_predictions = mlp.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected_predictions, predictions, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_predictions, ort_results[0], atol=1e-5)

    def test_mlp_classifier_float32(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 2, 2])
        mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=500, random_state=0)
        mlp.fit(X, y)

        onx = to_onnx(mlp, (X,))

        self.assertEqual(onx.proto.graph.input[0].type.tensor_type.elem_type, 1)  # FLOAT

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = mlp.predict(X)
        expected_proba = mlp.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_mlp_classifier_float64(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float64)
        y = np.array([0, 0, 1, 1, 2, 2])
        mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=500, random_state=0)
        mlp.fit(X, y)

        onx = to_onnx(mlp, (X,))

        self.assertEqual(onx.proto.graph.input[0].type.tensor_type.elem_type, 11)  # DOUBLE

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = mlp.predict(X)
        expected_proba = mlp.predict_proba(X).astype(np.float64)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_mlp_regressor_float32(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        mlp = MLPRegressor(hidden_layer_sizes=(4,), max_iter=500, random_state=0)
        mlp.fit(X, y)

        onx = to_onnx(mlp, (X,))

        self.assertEqual(onx.proto.graph.input[0].type.tensor_type.elem_type, 1)  # FLOAT

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected_predictions = mlp.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected_predictions, predictions, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_predictions, ort_results[0], atol=1e-5)

    def test_mlp_regressor_float64(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
        y = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
        mlp = MLPRegressor(hidden_layer_sizes=(4,), max_iter=500, random_state=0)
        mlp.fit(X, y)

        onx = to_onnx(mlp, (X,))

        self.assertEqual(onx.proto.graph.input[0].type.tensor_type.elem_type, 11)  # DOUBLE

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        expected_predictions = mlp.predict(X).astype(np.float64).reshape(-1, 1)
        self.assertEqualArray(expected_predictions, predictions, atol=1e-10)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_predictions, ort_results[0], atol=1e-10)

    def test_pipeline_mlp_classifier(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(4,), max_iter=500, random_state=0)),
            ]
        )
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Sub", op_types)
        self.assertTrue(
            "MatMul" in op_types or "Gemm" in op_types, f"Expected MatMul or Gemm in {op_types}"
        )

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


if __name__ == "__main__":
    unittest.main(verbosity=2)

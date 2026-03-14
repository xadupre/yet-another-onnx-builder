"""
Unit tests for yobx.sklearn.preprocessing.OneHotEncoder converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestOneHotEncoder(ExtTestCase):
    def test_one_hot_encoder_basic(self):
        from sklearn.preprocessing import OneHotEncoder
        from yobx.sklearn import to_onnx

        X = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 4.0]], dtype=np.float32)
        enc = OneHotEncoder(handle_unknown="ignore")
        enc.fit(X)

        onx = to_onnx(enc, (X,))

        # Check graph contains Equal and Cast nodes
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Equal", op_types)
        self.assertIn("Cast", op_types)
        self.assertIn("Concat", op_types)

        # Check numerical output
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = enc.transform(X).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_one_hot_encoder_single_feature(self):
        from sklearn.preprocessing import OneHotEncoder
        from yobx.sklearn import to_onnx

        X = np.array([[1.0], [2.0], [3.0], [1.0]], dtype=np.float32)
        enc = OneHotEncoder(handle_unknown="ignore")
        enc.fit(X)

        onx = to_onnx(enc, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = enc.transform(X).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_one_hot_encoder_drop_first(self):
        from sklearn.preprocessing import OneHotEncoder
        from yobx.sklearn import to_onnx

        X = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 4.0]], dtype=np.float32)
        enc = OneHotEncoder(drop="first", handle_unknown="ignore")
        enc.fit(X)

        onx = to_onnx(enc, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Equal", op_types)
        self.assertIn("Cast", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = enc.transform(X).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_one_hot_encoder_drop_if_binary(self):
        from sklearn.preprocessing import OneHotEncoder
        from yobx.sklearn import to_onnx

        # Binary feature (2 categories) and multi-category feature
        X = np.array([[0.0, 1.0], [1.0, 2.0], [0.0, 3.0], [1.0, 1.0]], dtype=np.float32)
        enc = OneHotEncoder(drop="if_binary", handle_unknown="ignore")
        enc.fit(X)

        onx = to_onnx(enc, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = enc.transform(X).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_one_hot_encoder_unknown_categories_ignored(self):
        from sklearn.preprocessing import OneHotEncoder
        from yobx.sklearn import to_onnx

        X_train = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        X_test = np.array([[5.0, 2.0], [1.0, 6.0]], dtype=np.float32)

        enc = OneHotEncoder(handle_unknown="ignore")
        enc.fit(X_train)

        onx = to_onnx(enc, (X_train,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_test})[0]
        expected = enc.transform(X_test).toarray().astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_test})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_one_hot_encoder_in_pipeline(self):
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from yobx.sklearn import to_onnx

        X = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        y = np.array([0, 1, 0, 1])
        pipe = Pipeline(
            [("enc", OneHotEncoder(handle_unknown="ignore")), ("clf", LogisticRegression())]
        )
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Equal", op_types)
        self.assertIn("Cast", op_types)

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

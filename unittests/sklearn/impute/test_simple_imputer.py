import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestSimpleImputer(ExtTestCase):
    def test_simple_imputer_mean(self):
        from sklearn.impute import SimpleImputer
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [np.nan, 3], [7, 6]], dtype=np.float32)
        imp = SimpleImputer(strategy="mean")
        imp.fit(X)

        onx = to_onnx(imp, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("IsNaN", op_types)
        self.assertIn("Where", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = imp.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_simple_imputer_median(self):
        from sklearn.impute import SimpleImputer
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [np.nan, 3], [7, 6], [4, np.nan]], dtype=np.float32)
        imp = SimpleImputer(strategy="median")
        imp.fit(X)

        onx = to_onnx(imp, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = imp.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_simple_imputer_most_frequent(self):
        from sklearn.impute import SimpleImputer
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [np.nan, 2], [1, 6]], dtype=np.float32)
        imp = SimpleImputer(strategy="most_frequent")
        imp.fit(X)

        onx = to_onnx(imp, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = imp.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_simple_imputer_constant(self):
        from sklearn.impute import SimpleImputer
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [np.nan, 3], [7, np.nan]], dtype=np.float32)
        imp = SimpleImputer(strategy="constant", fill_value=0.0)
        imp.fit(X)

        onx = to_onnx(imp, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = imp.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_simple_imputer_numeric_missing_value(self):
        from sklearn.impute import SimpleImputer
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [-1, 3], [7, -1]], dtype=np.float32)
        imp = SimpleImputer(missing_values=-1, strategy="mean")
        imp.fit(X)

        onx = to_onnx(imp, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Equal", op_types)
        self.assertIn("Where", op_types)
        self.assertNotIn("IsNaN", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = imp.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_simple_imputer_add_indicator_raises(self):
        from sklearn.impute import SimpleImputer
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [np.nan, 3], [7, 6]], dtype=np.float32)
        imp = SimpleImputer(add_indicator=True)
        imp.fit(X)

        with self.assertRaises(NotImplementedError):
            to_onnx(imp, (X,))

    def test_pipeline_simple_imputer_logistic_regression(self):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        X = np.array([[1, 2], [np.nan, 3], [7, 6], [4, np.nan]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        pipe = Pipeline([("imp", SimpleImputer(strategy="mean")), ("clf", LogisticRegression())])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("IsNaN", op_types)
        self.assertIn("Where", op_types)
        self.assertIn("Gemm", op_types)

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

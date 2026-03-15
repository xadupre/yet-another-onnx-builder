"""
Unit tests for yobx.sklearn.preprocessing.OrdinalEncoder converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestOrdinalEncoder(ExtTestCase):
    def _make_data(self, dtype=np.float32):
        X = np.array(
            [[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [1.0, 2.0], [2.0, 1.0]],
            dtype=dtype,
        )
        return X

    def _run_test(self, enc, X_train, X_test, dtype):
        from yobx.sklearn import to_onnx

        onx = to_onnx(enc, (X_train,))

        expected = enc.transform(X_test).astype(dtype)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_test})[0]
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_test})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_ordinal_encoder_float32(self):
        """Basic multi-feature ordinal encoding with float32 inputs."""
        from sklearn.preprocessing import OrdinalEncoder

        X = self._make_data(np.float32)
        enc = OrdinalEncoder()
        enc.fit(X)
        self._run_test(enc, X, X, np.float32)

    def test_ordinal_encoder_float64(self):
        """Basic multi-feature ordinal encoding with float64 inputs."""
        from sklearn.preprocessing import OrdinalEncoder

        X = self._make_data(np.float64)
        enc = OrdinalEncoder()
        enc.fit(X)
        self._run_test(enc, X, X, np.float64)

    def test_ordinal_encoder_single_feature_float32(self):
        """Single feature with float32."""
        from sklearn.preprocessing import OrdinalEncoder

        X = np.array([[1.0], [2.0], [3.0], [1.0]], dtype=np.float32)
        enc = OrdinalEncoder()
        enc.fit(X)
        self._run_test(enc, X, X, np.float32)

    def test_ordinal_encoder_single_feature_float64(self):
        """Single feature with float64."""
        from sklearn.preprocessing import OrdinalEncoder

        X = np.array([[1.0], [2.0], [3.0], [1.0]], dtype=np.float64)
        enc = OrdinalEncoder()
        enc.fit(X)
        self._run_test(enc, X, X, np.float64)

    def test_ordinal_encoder_unknown_use_encoded_value_float32(self):
        """Unknown categories mapped to unknown_value=-1 with float32."""
        from sklearn.preprocessing import OrdinalEncoder

        X_train = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]], dtype=np.float32)
        X_test = np.array([[1.0, 0.0], [99.0, 1.0], [2.0, 99.0]], dtype=np.float32)

        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(X_train)
        self._run_test(enc, X_train, X_test, np.float32)

    def test_ordinal_encoder_unknown_use_encoded_value_float64(self):
        """Unknown categories mapped to unknown_value=-1 with float64."""
        from sklearn.preprocessing import OrdinalEncoder

        X_train = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]], dtype=np.float64)
        X_test = np.array([[1.0, 0.0], [99.0, 1.0], [2.0, 99.0]], dtype=np.float64)

        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(X_train)
        self._run_test(enc, X_train, X_test, np.float64)

    def test_ordinal_encoder_graph_contains_equal_argmax(self):
        """Check graph structure: Equal and ArgMax nodes must appear."""
        from sklearn.preprocessing import OrdinalEncoder
        from yobx.sklearn import to_onnx

        X = self._make_data(np.float32)
        enc = OrdinalEncoder()
        enc.fit(X)

        onx = to_onnx(enc, (X,))
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Equal", op_types)
        self.assertIn("ArgMax", op_types)
        self.assertIn("Concat", op_types)

    def test_ordinal_encoder_in_pipeline(self):
        """OrdinalEncoder in a sklearn Pipeline followed by LinearRegression."""
        from sklearn.preprocessing import OrdinalEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LinearRegression
        from yobx.sklearn import to_onnx

        X = self._make_data(np.float32)
        y = np.array([1.0, 2.0, 3.0, 1.0, 2.0], dtype=np.float32)
        pipe = Pipeline([("enc", OrdinalEncoder()), ("reg", LinearRegression())])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))
        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]

        expected = pipe.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, ort_result, atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

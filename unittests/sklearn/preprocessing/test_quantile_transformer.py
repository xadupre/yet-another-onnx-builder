"""
Unit tests for yobx.sklearn.preprocessing.QuantileTransformer converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestQuantileTransformer(ExtTestCase):
    def _rng(self):
        return np.random.default_rng(0)

    def test_quantile_transformer_uniform(self):
        from sklearn.preprocessing import QuantileTransformer
        from yobx.sklearn import to_onnx

        rng = self._rng()
        X = rng.standard_normal((40, 3)).astype(np.float32)
        qt = QuantileTransformer(n_quantiles=20, output_distribution="uniform", random_state=0)
        qt.fit(X)

        X_test = X[:8]
        onx = to_onnx(qt, (X_test,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_test})[0]
        expected = qt.transform(X_test).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_test})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_quantile_transformer_normal(self):
        from sklearn.preprocessing import QuantileTransformer
        from yobx.sklearn import to_onnx

        rng = self._rng()
        X = rng.standard_normal((40, 3)).astype(np.float32)
        qt = QuantileTransformer(n_quantiles=20, output_distribution="normal", random_state=0)
        qt.fit(X)

        X_test = X[:8]
        onx = to_onnx(qt, (X_test,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_test})[0]
        expected = qt.transform(X_test).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-3)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_test})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-3)

    def test_quantile_transformer_boundary_values(self):
        """Values at or beyond the fitted range map to 0/1 (uniform) or clipped (normal)."""
        from sklearn.preprocessing import QuantileTransformer
        from yobx.sklearn import to_onnx

        rng = self._rng()
        X_train = rng.standard_normal((40, 2)).astype(np.float32)
        qt = QuantileTransformer(n_quantiles=10, output_distribution="uniform", random_state=0)
        qt.fit(X_train)

        # Include values outside the training range.
        X_test = np.vstack([
            X_train[:3],
            np.array([[-100.0, 100.0]], dtype=np.float32),
        ])
        onx = to_onnx(qt, (X_test,))

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X_test})[0]
        expected = qt.transform(X_test).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_quantile_transformer_large_quantiles(self):
        """Default n_quantiles=1000 converter."""
        from sklearn.preprocessing import QuantileTransformer
        from yobx.sklearn import to_onnx

        rng = self._rng()
        X = rng.standard_normal((200, 4)).astype(np.float32)
        qt = QuantileTransformer(random_state=0)  # default n_quantiles=1000
        qt.fit(X)

        X_test = X[:5]
        onx = to_onnx(qt, (X_test,))

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X_test})[0]
        expected = qt.transform(X_test).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_pipeline_quantile_transformer_logistic_regression(self):
        from sklearn.preprocessing import QuantileTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        rng = self._rng()
        X = rng.standard_normal((40, 3)).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        pipe = Pipeline([
            ("qt", QuantileTransformer(n_quantiles=20, random_state=0)),
            ("clf", LogisticRegression()),
        ])
        pipe.fit(X, y)

        X_test = X[:5]
        onx = to_onnx(pipe, (X_test,))

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_test})
        label_ort, proba_ort = ort_results[0], ort_results[1]

        expected_label = pipe.predict(X_test)
        expected_proba = pipe.predict_proba(X_test).astype(np.float32)
        self.assertEqualArray(expected_label, label_ort)
        self.assertEqualArray(expected_proba, proba_ort, atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
Unit tests for the RANSACRegressor converter.
"""

import unittest
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnRANSACRegressor(ExtTestCase):
    _X = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [0, 1], [9, 10]], dtype=np.float32
    )
    _y = np.array([1.5, 2.5, 3.5, 4.5, 2.0, 3.0, 1.0, 5.0], dtype=np.float32)

    def _check_ransac(self, estimator, X, y, atol=1e-4, min_samples=None):
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            yd = y.astype(dtype)
            kwargs = {} if min_samples is None else {"min_samples": min_samples}
            reg = RANSACRegressor(estimator=estimator, random_state=0, **kwargs)
            reg.fit(Xd, yd)
            onx = to_onnx(reg, (Xd,))

            output_names = [o.name for o in onx.graph.output]
            self.assertEqual(len(output_names), 1, f"Expected 1 output, got {output_names}")

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            pred = results[0].ravel()

            expected_pred = reg.predict(Xd).astype(dtype)
            self.assertEqualArray(expected_pred, pred, atol=atol)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected_pred, ort_results[0].ravel(), atol=atol)

    def test_ransac_default(self):
        """RANSACRegressor with default LinearRegression base estimator."""
        self._check_ransac(None, self._X, self._y)

    def test_ransac_linear_regression(self):
        """RANSACRegressor with explicit LinearRegression base estimator."""
        self._check_ransac(LinearRegression(), self._X, self._y)

    def test_ransac_ridge(self):
        """RANSACRegressor with Ridge base estimator."""
        self._check_ransac(Ridge(), self._X, self._y, min_samples=3)

    def test_ransac_decision_tree(self):
        """RANSACRegressor with DecisionTreeRegressor base estimator."""
        self._check_ransac(
            DecisionTreeRegressor(max_depth=3, random_state=0), self._X, self._y, min_samples=3
        )

    def test_ransac_in_pipeline(self):
        """RANSACRegressor as last step in a Pipeline."""
        Xd = self._X.astype(np.float32)
        reg = RANSACRegressor(estimator=LinearRegression(), random_state=0)
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", reg)])
        pipe.fit(Xd, self._y)

        onx = to_onnx(pipe, (Xd,))
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": Xd})
        expected_pred = pipe.predict(Xd).astype(np.float32)
        self.assertEqualArray(expected_pred, results[0].ravel(), atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": Xd})
        self.assertEqualArray(expected_pred, ort_results[0].ravel(), atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

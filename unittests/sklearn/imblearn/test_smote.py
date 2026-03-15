"""
Unit tests for yobx.sklearn.imblearn.SMOTE converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn, requires_imblearn
from yobx.reference import ExtendedReferenceEvaluator


def _make_imbalanced_data(dtype=np.float32, n_samples=100, n_features=4):
    """Create a simple imbalanced binary classification dataset."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_samples, n_features)).astype(dtype)
    y = np.array([0] * (n_samples - 20) + [1] * 20)
    return X, y


@requires_sklearn("1.4")
@requires_imblearn("0.10")
class TestSMOTE(ExtTestCase):
    def test_smote_float32(self):
        """SMOTE converter passes float32 input through unchanged."""
        from imblearn.over_sampling import SMOTE
        from yobx.sklearn import to_onnx

        X, y = _make_imbalanced_data(np.float32)
        estimator = SMOTE(random_state=0)
        estimator.fit_resample(X, y)

        onx = to_onnx(estimator, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        self.assertEqualArray(X, result)

    def test_smote_float64(self):
        """SMOTE converter passes float64 input through unchanged."""
        from imblearn.over_sampling import SMOTE
        from yobx.sklearn import to_onnx

        X, y = _make_imbalanced_data(np.float64)
        estimator = SMOTE(random_state=0)
        estimator.fit_resample(X, y)

        onx = to_onnx(estimator, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        self.assertEqualArray(X, result)

    def test_smote_ort_float32(self):
        """SMOTE ONNX model runs correctly in OnnxRuntime with float32."""
        from imblearn.over_sampling import SMOTE
        from yobx.sklearn import to_onnx

        X, y = _make_imbalanced_data(np.float32)
        estimator = SMOTE(random_state=0)
        estimator.fit_resample(X, y)

        onx = to_onnx(estimator, (X,))

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(X, result)

    def test_smote_ort_float64(self):
        """SMOTE ONNX model runs correctly in OnnxRuntime with float64."""
        from imblearn.over_sampling import SMOTE
        from yobx.sklearn import to_onnx

        X, y = _make_imbalanced_data(np.float64)
        estimator = SMOTE(random_state=0)
        estimator.fit_resample(X, y)

        onx = to_onnx(estimator, (X,))

        sess = self.check_ort(onx)
        result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(X, result)

    def test_smote_pipeline_float32(self):
        """SMOTE in an imblearn Pipeline with StandardScaler, float32."""
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        from sklearn.preprocessing import StandardScaler
        from yobx.sklearn import to_onnx

        X, y = _make_imbalanced_data(np.float32)
        pipe = ImbPipeline([
            ("smote", SMOTE(random_state=0)),
            ("scaler", StandardScaler()),
        ])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        expected = pipe.named_steps["scaler"].transform(X).astype(np.float32)
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_smote_pipeline_float64(self):
        """SMOTE in an imblearn Pipeline with StandardScaler, float64."""
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        from sklearn.preprocessing import StandardScaler
        from yobx.sklearn import to_onnx

        X, y = _make_imbalanced_data(np.float64)
        pipe = ImbPipeline([
            ("smote", SMOTE(random_state=0)),
            ("scaler", StandardScaler()),
        ])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        expected = pipe.named_steps["scaler"].transform(X).astype(np.float64)
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        self.assertEqualArray(expected, result, atol=1e-10)

    def test_smote_pipeline_classifier_float32(self):
        """SMOTE in an imblearn Pipeline with LogisticRegression, float32."""
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        from sklearn.linear_model import LogisticRegression
        from yobx.sklearn import to_onnx

        X, y = _make_imbalanced_data(np.float32)
        pipe = ImbPipeline([
            ("smote", SMOTE(random_state=0)),
            ("clf", LogisticRegression(random_state=0, max_iter=200)),
        ])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        sess = self.check_ort(onx)
        labels, _probas = sess.run(None, {"X": X})
        expected_labels = pipe.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, labels)


if __name__ == "__main__":
    unittest.main(verbosity=2)

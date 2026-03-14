"""
Unit tests for sklearn SVM converters:
  LinearSVC, LinearSVR, SVC, NuSVC, SVR, NuSVR.
"""

import unittest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR, NuSVC, NuSVR, SVC, SVR
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnLinearSVC(ExtTestCase):

    _X_bin = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [0, 1], [9, 10], [2, 3], [4, 5]], dtype=np.float32
    )
    _y_bin = np.array([0, 0, 1, 1, 0, 1, 0, 1])

    _X_multi = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [6, 7], [0, 1], [8, 9]], dtype=np.float32
    )
    _y_multi = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    def _check_linear_svc(self, X, y, atol=1e-4):
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            clf = LinearSVC(random_state=42, max_iter=5000)
            clf.fit(Xd, y)
            onx = to_onnx(clf, (Xd,))

            output_names = [o.name for o in onx.graph.output]
            self.assertEqual(len(output_names), 1, f"Expected 1 output, got {output_names}")

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            expected = clf.predict(Xd)
            self.assertEqualArray(expected, ort_results[0])

    def test_linear_svc_binary(self):
        self._check_linear_svc(self._X_bin, self._y_bin)

    def test_linear_svc_multiclass(self):
        self._check_linear_svc(self._X_multi, self._y_multi)

    def test_linear_svc_in_pipeline(self):
        Xd = self._X_bin.astype(np.float32)
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("clf", LinearSVC(random_state=42, max_iter=5000))]
        )
        pipe.fit(Xd, self._y_bin)
        onx = to_onnx(pipe, (Xd,))
        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": Xd})
        self.assertEqualArray(pipe.predict(Xd), ort_results[0])


@requires_sklearn("1.4")
class TestSklearnLinearSVR(ExtTestCase):

    _X = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [0, 1], [9, 10], [2, 3], [4, 5]], dtype=np.float32
    )
    _y = np.array([1.0, 2.0, 3.0, 4.0, 0.5, 5.0, 1.5, 2.5])

    def _check_linear_svr(self, X, y, atol=1e-4):
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            reg = LinearSVR(random_state=42, max_iter=5000)
            reg.fit(Xd, y)
            onx = to_onnx(reg, (Xd,))

            output_names = [o.name for o in onx.graph.output]
            self.assertEqual(len(output_names), 1, f"Expected 1 output, got {output_names}")

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            expected = reg.predict(Xd).reshape(-1, 1).astype(dtype)
            self.assertEqualArray(expected, ort_results[0], atol=atol)

    def test_linear_svr(self):
        self._check_linear_svr(self._X, self._y)

    def test_linear_svr_in_pipeline(self):
        Xd = self._X.astype(np.float32)
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("reg", LinearSVR(random_state=42, max_iter=5000))]
        )
        pipe.fit(Xd, self._y)
        onx = to_onnx(pipe, (Xd,))
        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": Xd})
        expected = pipe.predict(Xd).reshape(-1, 1).astype(np.float32)
        self.assertEqualArray(expected, ort_results[0], atol=1e-4)


@requires_sklearn("1.4")
class TestSklearnSVC(ExtTestCase):

    _X_bin, _y_bin = make_classification(n_samples=40, n_features=4, n_classes=2, random_state=42)
    _X_bin = _X_bin.astype(np.float32)

    _X_multi, _y_multi = make_classification(
        n_samples=60, n_features=4, n_classes=3, n_informative=3, n_redundant=0, random_state=42
    )
    _X_multi = _X_multi.astype(np.float32)

    def _check_svc(self, X, y, probability=False, kernel="rbf", atol=1e-4):
        clf = SVC(kernel=kernel, probability=probability, random_state=42)
        clf.fit(X, y)
        onx = to_onnx(clf, (X,))

        expected_n_outputs = 2 if probability else 1
        output_names = [o.name for o in onx.graph.output]
        self.assertEqual(
            len(output_names), expected_n_outputs, f"Expected {expected_n_outputs} outputs"
        )

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})

        self.assertEqualArray(clf.predict(X), ort_results[0])

        if probability:
            expected_proba = clf.predict_proba(X).astype(np.float32)
            self.assertEqualArray(expected_proba, ort_results[1], atol=atol)

    def test_svc_binary_no_proba(self):
        self._check_svc(self._X_bin, self._y_bin, probability=False)

    def test_svc_binary_with_proba(self):
        self._check_svc(self._X_bin, self._y_bin, probability=True)

    def test_svc_multiclass_no_proba(self):
        self._check_svc(self._X_multi, self._y_multi, probability=False)

    def test_svc_multiclass_with_proba(self):
        self._check_svc(self._X_multi, self._y_multi, probability=True)

    def test_svc_linear_kernel(self):
        self._check_svc(self._X_bin, self._y_bin, kernel="linear")

    def test_svc_poly_kernel(self):
        self._check_svc(self._X_bin, self._y_bin, kernel="poly")

    def test_svc_sigmoid_kernel(self):
        self._check_svc(self._X_bin, self._y_bin, kernel="sigmoid")

    def test_svc_in_pipeline(self):
        clf = SVC(kernel="rbf", probability=True, random_state=42)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(self._X_bin, self._y_bin)
        onx = to_onnx(pipe, (self._X_bin,))
        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": self._X_bin})
        self.assertEqualArray(pipe.predict(self._X_bin), ort_results[0])


@requires_sklearn("1.4")
class TestSklearnNuSVC(ExtTestCase):

    _X_bin, _y_bin = make_classification(n_samples=40, n_features=4, n_classes=2, random_state=42)
    _X_bin = _X_bin.astype(np.float32)

    _X_multi, _y_multi = make_classification(
        n_samples=60, n_features=4, n_classes=3, n_informative=3, n_redundant=0, random_state=42
    )
    _X_multi = _X_multi.astype(np.float32)

    def _check_nusvc(self, X, y, probability=False, atol=1e-4):
        clf = NuSVC(probability=probability, random_state=42)
        clf.fit(X, y)
        onx = to_onnx(clf, (X,))

        expected_n_outputs = 2 if probability else 1
        output_names = [o.name for o in onx.graph.output]
        self.assertEqual(len(output_names), expected_n_outputs)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(clf.predict(X), ort_results[0])

        if probability:
            expected_proba = clf.predict_proba(X).astype(np.float32)
            self.assertEqualArray(expected_proba, ort_results[1], atol=atol)

    def test_nusvc_binary_no_proba(self):
        self._check_nusvc(self._X_bin, self._y_bin, probability=False)

    def test_nusvc_binary_with_proba(self):
        self._check_nusvc(self._X_bin, self._y_bin, probability=True)

    def test_nusvc_multiclass_no_proba(self):
        self._check_nusvc(self._X_multi, self._y_multi, probability=False)

    def test_nusvc_multiclass_with_proba(self):
        self._check_nusvc(self._X_multi, self._y_multi, probability=True)


@requires_sklearn("1.4")
class TestSklearnSVR(ExtTestCase):

    _X, _y = make_regression(n_samples=30, n_features=4, random_state=42)
    _X = _X.astype(np.float32)

    def _check_svr(self, X, y, kernel="rbf", atol=1e-4):
        reg = SVR(kernel=kernel)
        reg.fit(X, y)
        onx = to_onnx(reg, (X,))

        output_names = [o.name for o in onx.graph.output]
        self.assertEqual(len(output_names), 1)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        expected = reg.predict(X).astype(np.float32)
        self.assertEqualArray(expected, ort_results[0].flatten(), atol=atol)

    def test_svr_rbf(self):
        self._check_svr(self._X, self._y, kernel="rbf")

    def test_svr_linear(self):
        self._check_svr(self._X, self._y, kernel="linear")

    def test_svr_poly(self):
        self._check_svr(self._X, self._y, kernel="poly")

    def test_svr_in_pipeline(self):
        reg = SVR(kernel="rbf")
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", reg)])
        pipe.fit(self._X, self._y)
        onx = to_onnx(pipe, (self._X,))
        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": self._X})
        expected = pipe.predict(self._X).astype(np.float32)
        self.assertEqualArray(expected, ort_results[0].flatten(), atol=1e-4)


@requires_sklearn("1.4")
class TestSklearnNuSVR(ExtTestCase):

    _X, _y = make_regression(n_samples=30, n_features=4, random_state=42)
    _X = _X.astype(np.float32)

    def _check_nusvr(self, X, y, kernel="rbf", atol=1e-4):
        reg = NuSVR(kernel=kernel)
        reg.fit(X, y)
        onx = to_onnx(reg, (X,))

        output_names = [o.name for o in onx.graph.output]
        self.assertEqual(len(output_names), 1)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        expected = reg.predict(X).astype(np.float32)
        self.assertEqualArray(expected, ort_results[0].flatten(), atol=atol)

    def test_nusvr_rbf(self):
        self._check_nusvr(self._X, self._y, kernel="rbf")

    def test_nusvr_linear(self):
        self._check_nusvr(self._X, self._y, kernel="linear")


if __name__ == "__main__":
    unittest.main(verbosity=2)

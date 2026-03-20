"""
Unit tests for sklearn Naive Bayes converters.
"""

import unittest
import numpy as np
from sklearn.naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.8")
class TestSklearnNaiveBayes(ExtTestCase):

    _X_bin = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 1], [9, 10]], dtype=np.float32)
    _y_bin = np.array([0, 0, 1, 1, 0, 1])

    _X_multi = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
    _y_multi = np.array([0, 0, 1, 1, 2, 2])

    def _check_classifier(self, estimator, X, y, atol=1e-5):
        """Fit, convert to ONNX, and compare outputs against sklearn for float32/64."""
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            estimator.fit(Xd, y)
            onx = to_onnx(estimator, (Xd,))

            output_names = [o.name for o in onx.proto.graph.output]
            self.assertEqual(len(output_names), 2, f"Expected 2 outputs, got {output_names}")

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            label, proba = results[0], results[1]

            expected_label = estimator.predict(Xd)
            expected_proba = estimator.predict_proba(Xd).astype(dtype)

            self.assertEqualArray(expected_label, label)
            self.assertEqualArray(expected_proba, proba, atol=atol)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected_label, ort_results[0])
            self.assertEqualArray(expected_proba, ort_results[1], atol=atol)

    def _check_classifier_int(self, estimator, X, y, atol=1e-5):
        """Fit, convert to ONNX, and compare outputs against sklearn for integer inputs."""
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            estimator.fit(X, y)
            onx = to_onnx(estimator, (Xd,))

            output_names = [o.name for o in onx.proto.graph.output]
            self.assertEqual(len(output_names), 2, f"Expected 2 outputs, got {output_names}")

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            label, proba = results[0], results[1]

            expected_label = estimator.predict(X)
            expected_proba = estimator.predict_proba(X).astype(dtype)

            self.assertEqualArray(expected_label, label)
            self.assertEqualArray(expected_proba, proba, atol=atol)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected_label, ort_results[0])
            self.assertEqualArray(expected_proba, ort_results[1], atol=atol)

    # ── GaussianNB ────────────────────────────────────────────────────────────

    def test_gaussian_nb_binary(self):
        self._check_classifier(GaussianNB(), self._X_bin, self._y_bin)

    def test_gaussian_nb_multiclass(self):
        self._check_classifier(GaussianNB(), self._X_multi, self._y_multi)

    def test_gaussian_nb_in_pipeline(self):
        for dtype in (np.float32, np.float64):
            Xd = self._X_multi.astype(dtype)
            y = self._y_multi
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())])
            pipe.fit(Xd, y)
            onx = to_onnx(pipe, (Xd,))

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            self.assertEqualArray(pipe.predict(Xd), results[0])

            sess = self.check_ort(onx)
            self.assertEqualArray(pipe.predict(Xd), sess.run(None, {"X": Xd})[0])

    # ── MultinomialNB ─────────────────────────────────────────────────────────

    def test_multinomial_nb_binary(self):
        X = np.abs(self._X_bin)
        self._check_classifier(MultinomialNB(), X, self._y_bin)

    def test_multinomial_nb_multiclass(self):
        X = np.abs(self._X_multi)
        self._check_classifier(MultinomialNB(), X, self._y_multi)

    def test_multinomial_nb_in_pipeline(self):
        rng = np.random.default_rng(7)
        X = rng.uniform(0.5, 5.0, size=(30, 4)).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10 + [2] * 10)
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            pipe = Pipeline([("scaler", MinMaxScaler()), ("clf", MultinomialNB())])
            pipe.fit(Xd, y)
            onx = to_onnx(pipe, (Xd,))

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            self.assertEqualArray(pipe.predict(Xd), results[0])

            sess = self.check_ort(onx)
            self.assertEqualArray(pipe.predict(Xd), sess.run(None, {"X": Xd})[0])

    # ── BernoulliNB ───────────────────────────────────────────────────────────

    def test_bernoulli_nb_binary(self):
        self._check_classifier(BernoulliNB(), self._X_bin, self._y_bin)

    def test_bernoulli_nb_multiclass(self):
        self._check_classifier(BernoulliNB(), self._X_multi, self._y_multi)

    def test_bernoulli_nb_no_binarize(self):
        """BernoulliNB with binarize=None (input already binary)."""
        rng = np.random.default_rng(0)
        X = rng.integers(0, 2, size=(30, 5)).astype(np.float32)
        y = rng.integers(0, 3, size=30)
        self._check_classifier(BernoulliNB(binarize=None), X, y)

    def test_bernoulli_nb_custom_binarize(self):
        """BernoulliNB with a non-default binarize threshold."""
        self._check_classifier(BernoulliNB(binarize=3.0), self._X_bin, self._y_bin)

    def test_bernoulli_nb_in_pipeline(self):
        for dtype in (np.float32, np.float64):
            Xd = self._X_multi.astype(dtype)
            y = self._y_multi
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", BernoulliNB(binarize=None))])
            pipe.fit(Xd, y)
            onx = to_onnx(pipe, (Xd,))

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            self.assertEqualArray(pipe.predict(Xd), results[0])

            sess = self.check_ort(onx)
            self.assertEqualArray(pipe.predict(Xd), sess.run(None, {"X": Xd})[0])

    # ── ComplementNB ──────────────────────────────────────────────────────────

    def test_complement_nb_binary(self):
        X = np.abs(self._X_bin)
        self._check_classifier(ComplementNB(), X, self._y_bin)

    def test_complement_nb_multiclass(self):
        X = np.abs(self._X_multi)
        self._check_classifier(ComplementNB(), X, self._y_multi)

    def test_complement_nb_in_pipeline(self):
        rng = np.random.default_rng(7)
        X = rng.uniform(0.5, 5.0, size=(30, 4)).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10 + [2] * 10)
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            pipe = Pipeline([("scaler", MinMaxScaler()), ("clf", ComplementNB())])
            pipe.fit(Xd, y)
            onx = to_onnx(pipe, (Xd,))

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            self.assertEqualArray(pipe.predict(Xd), results[0])

            sess = self.check_ort(onx)
            self.assertEqualArray(pipe.predict(Xd), sess.run(None, {"X": Xd})[0])

    # ── CategoricalNB ─────────────────────────────────────────────────────────

    _X_cat_bin = np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 1], [0, 0]], dtype=np.int64)
    _y_cat_bin = np.array([0, 0, 1, 1, 1, 0])

    _X_cat_multi = np.array(
        [[0, 1], [1, 0], [2, 1], [0, 0], [1, 1], [2, 0], [0, 1], [1, 1], [2, 0]], dtype=np.int64
    )
    _y_cat_multi = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    def test_categorical_nb_binary(self):
        self._check_classifier_int(CategoricalNB(), self._X_cat_bin, self._y_cat_bin)

    def test_categorical_nb_multiclass(self):
        self._check_classifier_int(CategoricalNB(), self._X_cat_multi, self._y_cat_multi)

    def test_categorical_nb_mixed_categories(self):
        """CategoricalNB with different numbers of categories per feature."""
        rng = np.random.default_rng(3)
        X = np.column_stack(
            [
                rng.integers(0, 4, size=30),  # 4 categories
                rng.integers(0, 2, size=30),  # 2 categories
                rng.integers(0, 3, size=30),  # 3 categories
            ]
        ).astype(np.int64)
        y = rng.integers(0, 3, size=30)
        self._check_classifier_int(CategoricalNB(), X, y)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
Unit tests for every possible combination of
:func:`yobx.sklearn.sklearn_helper.get_output_names` and
:func:`yobx.sklearn.sklearn_helper.get_n_expected_outputs`.

The comment at line 53 of ``sklearn_helper.py`` requested tests that cover
every code path of ``get_output_names``.
"""

import unittest
import numpy as np

from yobx.ext_test_case import ExtTestCase, requires_sklearn


@requires_sklearn("1.4")
class TestGetOutputNames(ExtTestCase):
    """Tests for get_output_names and get_n_expected_outputs."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.standard_normal((30, 4)).astype(np.float32)
        self.y_binary = np.array([0, 1] * 15)
        self.y_multiclass = np.array([0, 1, 2] * 10)

    # ------------------------------------------------------------------ helpers

    def _fit(self, cls, *args, **kwargs):
        est = cls(*args, **kwargs)
        est.fit(self.X, self.y_binary)
        return est

    def _fit_unsupervised(self, cls, *args, **kwargs):
        est = cls(*args, **kwargs)
        est.fit(self.X)
        return est

    # ------------------------------------------------------------------ classifiers

    def test_classifier_with_predict_proba(self):
        from sklearn.linear_model import LogisticRegression
        from yobx.sklearn.sklearn_helper import get_output_names, get_n_expected_outputs

        est = self._fit(LogisticRegression)
        self.assertEqual(list(get_output_names(est)), ["label", "probabilities"])
        self.assertEqual(get_n_expected_outputs(est), 2)

    def test_classifier_without_predict_proba(self):
        from sklearn.linear_model import SGDClassifier
        from yobx.sklearn.sklearn_helper import get_output_names, get_n_expected_outputs

        est = self._fit(SGDClassifier, loss="hinge")
        self.assertEqual(list(get_output_names(est)), ["label"])
        self.assertEqual(get_n_expected_outputs(est), 1)

    def test_random_forest_classifier(self):
        from sklearn.ensemble import RandomForestClassifier
        from yobx.sklearn.sklearn_helper import get_output_names, get_n_expected_outputs

        est = self._fit(RandomForestClassifier, n_estimators=2, random_state=0)
        self.assertEqual(list(get_output_names(est)), ["label", "probabilities"])
        self.assertEqual(get_n_expected_outputs(est), 2)

    # ------------------------------------------------------------------ regressors

    def test_regressor(self):
        from sklearn.linear_model import LinearRegression
        from yobx.sklearn.sklearn_helper import get_output_names, get_n_expected_outputs

        est = LinearRegression()
        est.fit(self.X, self.y_binary.astype(float))
        self.assertEqual(list(get_output_names(est)), ["predictions"])
        self.assertEqual(get_n_expected_outputs(est), 1)

    def test_random_forest_regressor(self):
        from sklearn.ensemble import RandomForestRegressor
        from yobx.sklearn.sklearn_helper import get_output_names, get_n_expected_outputs

        est = RandomForestRegressor(n_estimators=2, random_state=0)
        est.fit(self.X, self.y_binary.astype(float))
        self.assertEqual(list(get_output_names(est)), ["predictions"])
        self.assertEqual(get_n_expected_outputs(est), 1)

    # ------------------------------------------------------------------ clustering

    def test_kmeans(self):
        from sklearn.cluster import KMeans
        from yobx.sklearn.sklearn_helper import get_output_names, get_n_expected_outputs

        est = self._fit_unsupervised(KMeans, n_clusters=2, n_init=10, random_state=0)
        self.assertEqual(list(get_output_names(est)), ["label", "distances"])
        self.assertEqual(get_n_expected_outputs(est), 2)

    def test_birch(self):
        from sklearn.cluster import Birch
        from yobx.sklearn.sklearn_helper import get_output_names, get_n_expected_outputs

        est = self._fit_unsupervised(Birch, n_clusters=2)
        self.assertEqual(list(get_output_names(est)), ["label", "distances"])
        self.assertEqual(get_n_expected_outputs(est), 2)

    def test_feature_agglomeration(self):
        """FeatureAgglomeration is a ClusterMixin but should act as a transformer."""
        from sklearn.cluster import FeatureAgglomeration
        from yobx.sklearn.sklearn_helper import get_output_names, get_n_expected_outputs

        est = self._fit_unsupervised(FeatureAgglomeration, n_clusters=2)
        names = list(get_output_names(est))
        # Feature names come from get_feature_names_out; the prefix is derived
        # from the estimator class name.
        self.assertEqual(len(names), 1)
        self.assertTrue(all("featureagglomeration" in n for n in names))
        self.assertEqual(get_n_expected_outputs(est), 1)

    # ------------------------------------------------------------------ transformers

    def test_standard_scaler(self):
        from sklearn.preprocessing import StandardScaler
        from yobx.sklearn.sklearn_helper import get_output_names, get_n_expected_outputs

        est = self._fit_unsupervised(StandardScaler)
        names = list(get_output_names(est))
        self.assertEqual(len(names), 1)
        self.assertEqual(get_n_expected_outputs(est), 1)

    def test_variance_threshold(self):
        from sklearn.feature_selection import VarianceThreshold
        from yobx.sklearn.sklearn_helper import get_output_names, get_n_expected_outputs

        est = self._fit_unsupervised(VarianceThreshold)
        names = list(get_output_names(est))
        self.assertEqual(len(names), 1)
        self.assertEqual(get_n_expected_outputs(est), 1)

    # ------------------------------------------------------------------ outlier detectors

    def test_one_class_svm(self):
        from sklearn.svm import OneClassSVM
        from yobx.sklearn.sklearn_helper import get_output_names, get_n_expected_outputs

        est = self._fit_unsupervised(OneClassSVM)
        self.assertEqual(list(get_output_names(est)), ["label", "scores"])
        self.assertEqual(get_n_expected_outputs(est), 2)

    def test_isolation_forest(self):
        from sklearn.ensemble import IsolationForest
        from yobx.sklearn.sklearn_helper import get_output_names, get_n_expected_outputs

        est = self._fit_unsupervised(IsolationForest, random_state=0)
        self.assertEqual(list(get_output_names(est)), ["label", "scores"])
        self.assertEqual(get_n_expected_outputs(est), 2)

    # ------------------------------------------------------------------ mixture models

    def test_gaussian_mixture(self):
        from sklearn.mixture import GaussianMixture
        from yobx.sklearn.sklearn_helper import get_output_names, get_n_expected_outputs

        est = self._fit_unsupervised(GaussianMixture, n_components=2, random_state=0)
        self.assertEqual(list(get_output_names(est)), ["label", "probabilities"])
        self.assertEqual(get_n_expected_outputs(est), 2)

    # ------------------------------------------------------------------ pipelines

    def test_pipeline_cluster_last_step(self):
        """Pipeline ending with a clustering model → 2 outputs."""
        from sklearn.cluster import KMeans
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from yobx.sklearn.sklearn_helper import get_output_names

        pipe = Pipeline([("ss", StandardScaler()), ("km", KMeans(n_clusters=2, n_init=10))])
        pipe.fit(self.X)
        self.assertEqual(list(get_output_names(pipe)), ["label", "distances"])

    def test_pipeline_feature_agglomeration_last_step(self):
        """Pipeline ending with FeatureAgglomeration → 1 output (feature names)."""
        from sklearn.cluster import FeatureAgglomeration
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from yobx.sklearn.sklearn_helper import get_output_names

        pipe = Pipeline([("ss", StandardScaler()), ("fa", FeatureAgglomeration(n_clusters=2))])
        pipe.fit(self.X)
        names = list(get_output_names(pipe))
        self.assertEqual(len(names), 1)
        self.assertTrue(all("featureagglomeration" in n for n in names))

    def test_pipeline_classifier_last_step(self):
        """Pipeline ending with a classifier → 2 outputs."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from yobx.sklearn.sklearn_helper import get_output_names

        pipe = Pipeline([("ss", StandardScaler()), ("clf", LogisticRegression())])
        pipe.fit(self.X, self.y_binary)
        self.assertEqual(list(get_output_names(pipe)), ["label", "probabilities"])

    def test_pipeline_outlier_last_step(self):
        """Pipeline ending with an outlier detector → 2 outputs."""
        from sklearn.ensemble import IsolationForest
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from yobx.sklearn.sklearn_helper import get_output_names

        pipe = Pipeline([("ss", StandardScaler()), ("iso", IsolationForest(random_state=0))])
        pipe.fit(self.X)
        self.assertEqual(list(get_output_names(pipe)), ["label", "scores"])

    def test_pipeline_transformer_last_step(self):
        """Pipeline ending with a plain transformer → feature names (collapsed to 1)."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from yobx.sklearn.sklearn_helper import get_output_names

        pipe = Pipeline([("ss1", StandardScaler()), ("ss2", StandardScaler())])
        pipe.fit(self.X)
        names = list(get_output_names(pipe))
        self.assertEqual(len(names), 1)

    def test_pipeline_regressor_last_step(self):
        """Pipeline ending with a regressor → 1 output."""
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from yobx.sklearn.sklearn_helper import get_output_names

        pipe = Pipeline([("ss", StandardScaler()), ("reg", LinearRegression())])
        pipe.fit(self.X, self.y_binary.astype(float))
        self.assertEqual(list(get_output_names(pipe)), ["predictions"])

    # ------------------------------------------------------------------ helpers

    def test_should_use_feature_names_selector(self):
        """SelectorMixin → _should_use_feature_names returns True."""
        from sklearn.feature_selection import VarianceThreshold
        from yobx.sklearn.sklearn_helper import _should_use_feature_names

        est = self._fit_unsupervised(VarianceThreshold)
        self.assertTrue(_should_use_feature_names(est))

    def test_should_use_feature_names_feature_agglomeration(self):
        """FeatureAgglomeration → _should_use_feature_names returns True."""
        from sklearn.cluster import FeatureAgglomeration
        from yobx.sklearn.sklearn_helper import _should_use_feature_names

        est = self._fit_unsupervised(FeatureAgglomeration, n_clusters=2)
        self.assertTrue(_should_use_feature_names(est))

    def test_should_use_feature_names_transformer(self):
        """Plain transformer → _should_use_feature_names returns True."""
        from sklearn.preprocessing import StandardScaler
        from yobx.sklearn.sklearn_helper import _should_use_feature_names

        est = self._fit_unsupervised(StandardScaler)
        self.assertTrue(_should_use_feature_names(est))

    def test_should_use_feature_names_cluster(self):
        from sklearn.cluster import KMeans
        from yobx.sklearn.sklearn_helper import _should_use_feature_names

        est = self._fit_unsupervised(KMeans, n_clusters=2, n_init=10, random_state=0)
        self.assertFalse(_should_use_feature_names(est))

    def test_should_use_feature_names_classifier(self):
        """Classifier → _should_use_feature_names returns False."""
        from sklearn.linear_model import LogisticRegression
        from yobx.sklearn.sklearn_helper import _should_use_feature_names

        est = self._fit(LogisticRegression)
        self.assertFalse(_should_use_feature_names(est))

    # ------------------------------------------------------------------ consistency

    def test_consistency_all_estimators(self):
        """get_n_expected_outputs must equal len(get_output_names) for every estimator type."""
        from sklearn.cluster import Birch, FeatureAgglomeration, KMeans
        from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
        from sklearn.mixture import GaussianMixture
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import OneClassSVM
        from yobx.sklearn.sklearn_helper import get_n_expected_outputs, get_output_names

        estimators = [
            # classifiers
            self._fit(LogisticRegression),
            self._fit(SGDClassifier, loss="hinge"),
            self._fit(RandomForestClassifier, n_estimators=2, random_state=0),
            # regressors
            LinearRegression().fit(self.X, self.y_binary.astype(float)),
            RandomForestRegressor(n_estimators=2, random_state=0).fit(
                self.X, self.y_binary.astype(float)
            ),
            # clustering
            self._fit_unsupervised(KMeans, n_clusters=2, n_init=10, random_state=0),
            self._fit_unsupervised(Birch, n_clusters=2),
            self._fit_unsupervised(FeatureAgglomeration, n_clusters=2),
            # transformers
            self._fit_unsupervised(StandardScaler),
            self._fit_unsupervised(VarianceThreshold),
            # outlier detectors
            self._fit_unsupervised(OneClassSVM),
            self._fit_unsupervised(IsolationForest, random_state=0),
            # mixture models
            self._fit_unsupervised(GaussianMixture, n_components=2, random_state=0),
            # pipelines
            Pipeline([("ss", StandardScaler()), ("km", KMeans(n_clusters=2, n_init=10))]).fit(
                self.X
            ),
            Pipeline(
                [("ss", StandardScaler()), ("fa", FeatureAgglomeration(n_clusters=2))]
            ).fit(self.X),
            Pipeline([("ss", StandardScaler()), ("clf", LogisticRegression())]).fit(
                self.X, self.y_binary
            ),
            Pipeline(
                [("ss", StandardScaler()), ("iso", IsolationForest(random_state=0))]
            ).fit(self.X),
            Pipeline([("ss", StandardScaler()), ("reg", LinearRegression())]).fit(
                self.X, self.y_binary.astype(float)
            ),
            Pipeline([("ss1", StandardScaler()), ("ss2", StandardScaler())]).fit(self.X),
        ]

        for est in estimators:
            n = get_n_expected_outputs(est)
            names = list(get_output_names(est))
            self.assertEqual(
                n,
                len(names),
                msg=(
                    f"get_n_expected_outputs={n} != len(get_output_names)={len(names)} "
                    f"for {type(est).__name__}"
                ),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

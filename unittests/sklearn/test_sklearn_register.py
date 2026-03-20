import os
import unittest
import yobx.sklearn as yskl
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.sklearn import register_sklearn_converters
from yobx.sklearn.register import get_sklearn_estimator_coverage


class TestSklearnRegister(ExtTestCase):
    def test_list_of_tests(self):
        loader = unittest.TestLoader()
        suite = loader.discover(os.path.dirname(__file__ or "."))

        def iter_tests(suite):
            for test in suite:
                if isinstance(test, unittest.TestSuite):
                    yield from iter_tests(test)
                else:
                    yield test

        subfolders = set()
        for test in iter_tests(suite):
            spl = test.__class__.__module__.split(".")
            if len(spl) >= 2:
                subfolders.add(spl[-2])
        self.assertNotEmpty(subfolders)

        folder = os.path.dirname(yskl.__file__)

        subs = {
            f
            for f in os.listdir(folder)
            if "." not in f and "__" not in f and not f.startswith("test_")
        }
        self.assertNotEmpty(subs)
        for f in {
            "_imblearn",
            "_xgboost",
            "_lightgbm",
            "_category_encoders",
            "_sksurv",
            "_statsmodels",
        }:
            if f in subfolders:
                subfolders.add(f[1:])
        not_here = subs - subfolders & subs

        self.assertEmpty(
            not_here, msg=lambda: f"not_here={not_here}\nsubs={subs}\nsubfolders={subfolders}"
        )

    def test_get_sklearn_estimator_coverage(self):
        register_sklearn_converters()
        cov = get_sklearn_estimator_coverage()
        self.assertIsInstance(cov, list)
        self.assertGreater(len(cov), 0)
        rst = get_sklearn_estimator_coverage(rst=True)
        self.assertIsInstance(rst, str)
        self.assertIn("**Coverage**", rst)
        self.assertRaise(lambda: get_sklearn_estimator_coverage(libraries=("nolib",)), ValueError)

    @requires_sklearn("1.8")
    def test_get_sklearn_estimator_coverage_per_lib(self):
        register_sklearn_converters()
        boundaries = {
            "sklearn": (100, 300),
            "imblearn": (1, 20),
            "xgboost": (5, 5),
            "lightgbm": (3, 3),
            "category_encoders": (10, 20),
            "sksurv": (10, 30),
        }
        for lib, (mine, maxe) in boundaries.items():
            with self.subTest(lib=lib):
                res = get_sklearn_estimator_coverage(libraries=lib, rst=False)
                self.assertIsInstance(res, list)
                self.assertGreaterOrEqual(len(res), mine)
                self.assertLessOrEqual(len(res), maxe)
                modules = {r["module"] for r in res}
                self.assertTrue(
                    all(m.startswith(lib) for m in modules),
                    lambda: f"lib={lib!r}, modules={modules}",
                )
                rst = get_sklearn_estimator_coverage(libraries=lib, rst=True)
                n_lines = len(rst.split("\n"))
                self.assertLess(n_lines, (len(res) + 2) * 6 + 2)
                if lib == "sklearn":
                    self.assertIn("sklearn.linear_model.GammaRegressor", rst)
                    self.assertIn("sklearn.ensemble.HistGradientBoostingClassifier", rst)
                elif lib == "lightgbm":
                    self.assertIn("lightgbm.LGBMRanker", rst)
                elif lib == "xgboost":
                    self.assertIn("xgboost.XGBRanker", rst)


if __name__ == "__main__":
    unittest.main(verbosity=2)

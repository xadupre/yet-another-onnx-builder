import os
import unittest
import yobx.sklearn as yskl
from yobx.ext_test_case import ExtTestCase


class TestList(ExtTestCase):
    def test_list_of_tests(self):
        loader = unittest.TestLoader()
        suite = loader.discover(".")

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

        folder = os.path.dirname(yskl.__file__)

        subs = {
            f
            for f in os.listdir(folder)
            if "." not in f and "__" not in f and not f.startswith("test_")
        }
        for f in {"_xgboost", "_lightgbm", "_category_encoders"}:
            if f in subfolders:
                subfolders.add(f[1:])
        not_here = subs - subfolders & subs

        self.assertEmpty(not_here)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
Unit tests for yobx.sklearn converters.
"""

import unittest
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from yobx.ext_test_case import ExtTestCase, requires_sklearn, hide_stdout
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx, TraceableMixin


@requires_sklearn("1.4")
class TestSklearnTraceable(ExtTestCase):
    def test_numpy_traceable(self):
        class TraceableNumpyTransformer(BaseEstimator, TransformerMixin, TraceableMixin):
            def __init__(self):
                BaseEstimator.__init__(self)

            def fit(self, X):
                self.mean_ = np.abs(X).mean(axis=0, keepdims=True)
                return self

            def transform(self, X):
                return np.log(np.abs(X) / self.mean_ + 1)

        X = np.array([[1, 2], [3, 4]], dtype=np.float32)
        est = TraceableNumpyTransformer().fit(X)
        expected = est.transform(X)
        onx = to_onnx(est, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        self.assertEqualArray(expected, result, atol=1e-5)

    @hide_stdout()
    def test_dataframe_traceable(self):
        import pandas

        class TraceableDataFrameTransformer(BaseEstimator, TransformerMixin, TraceableMixin):
            def __init__(self):
                BaseEstimator.__init__(self)

            def fit(self, df):
                self.mean_ = df.mean(axis=0)
                return self

            def transform(self, df):
                return df + 1

        df = pandas.DataFrame([dict(a=1, b=2), dict(a=3.5, b=5.6)])
        est = TraceableDataFrameTransformer().fit(df)
        expected = est.transform(df)
        onx = to_onnx(est, (df,), verbose=1)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"a": df[["a"]].values, "b": df[["b"]].values})
        self.assertEqualArray(expected[["a"]].values, result[0], atol=1e-5)
        self.assertEqualArray(expected[["b"]].values, result[1], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

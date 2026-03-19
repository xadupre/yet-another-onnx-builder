"""
Unit tests for yobx.sklearn Pipeline converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnPipeline(ExtTestCase):
    def test_pipeline_step_names_are_prefixed(self):
        """Tensor names produced during each step conversion should carry the step prefix."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)
        y = np.array([0, 1, 0, 1])
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        # Every intermediate tensor name produced during "scaler" step conversion
        # should include the step prefix.
        all_names = [n.name for n in onx.proto.graph.node]
        all_names += [i.name for i in onx.proto.graph.initializer]
        all_names += [v.name for v in onx.proto.graph.value_info]
        scaler_names = [n for n in all_names if "scaler" in n]
        self.assertTrue(
            len(scaler_names) > 0,
            msg="Expected at least one name containing 'scaler' but found none",
        )

    def test_pipeline_produces_correct_predictions(self):
        """Pipeline converter should produce numerically correct ONNX output."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)
        y = np.array([0, 1, 0, 1])
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        expected = pipe.predict(X)
        self.assertEqualArray(expected, results[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)

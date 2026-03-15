"""
Unit tests for yobx.sklearn.cluster.FeatureAgglomeration converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestFeatureAgglomeration(ExtTestCase):
    def test_feature_agglomeration_transform_float32(self):
        from sklearn.cluster import FeatureAgglomeration
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 8)).astype(np.float32)
        model = FeatureAgglomeration(n_clusters=3)
        model.fit(X)

        onx = to_onnx(model, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("MatMul", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        (result,) = ref.run(None, {"X": X})

        expected = model.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        (ort_result,) = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_feature_agglomeration_transform_float64(self):
        from sklearn.cluster import FeatureAgglomeration
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((30, 8)).astype(np.float64)
        model = FeatureAgglomeration(n_clusters=3)
        model.fit(X)

        onx = to_onnx(model, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        (result,) = ref.run(None, {"X": X})

        expected = model.transform(X)
        self.assertEqualArray(expected, result, atol=1e-10)

        sess = self.check_ort(onx)
        (ort_result,) = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_result, atol=1e-10)

    def test_feature_agglomeration_dtypes(self):
        from sklearn.cluster import FeatureAgglomeration
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X32 = rng.standard_normal((20, 6)).astype(np.float32)
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                X = X32.astype(dtype)
                model = FeatureAgglomeration(n_clusters=2)
                model.fit(X)
                onx = to_onnx(model, (X,))

                ref = ExtendedReferenceEvaluator(onx)
                (result,) = ref.run(None, {"X": X})

                expected = model.transform(X).astype(dtype)
                self.assertEqualArray(expected, result, atol=1e-5)

                sess = self.check_ort(onx)
                (ort_result,) = sess.run(None, {"X": X})
                self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_feature_agglomeration_max_pooling(self):
        from sklearn.cluster import FeatureAgglomeration
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(3)
        X = rng.standard_normal((20, 6)).astype(np.float32)
        model = FeatureAgglomeration(n_clusters=3, pooling_func=np.max)
        model.fit(X)

        onx = to_onnx(model, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("ReduceMax", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        (result,) = ref.run(None, {"X": X})

        expected = model.transform(X)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        (ort_result,) = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_feature_agglomeration_min_pooling(self):
        from sklearn.cluster import FeatureAgglomeration
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(4)
        X = rng.standard_normal((20, 6)).astype(np.float32)
        model = FeatureAgglomeration(n_clusters=3, pooling_func=np.min)
        model.fit(X)

        onx = to_onnx(model, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("ReduceMin", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        (result,) = ref.run(None, {"X": X})

        expected = model.transform(X)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        (ort_result,) = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_feature_agglomeration_max_min_dtypes(self):
        from sklearn.cluster import FeatureAgglomeration
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(5)
        X32 = rng.standard_normal((20, 6)).astype(np.float32)
        for pf_name, pf in [("max", np.max), ("min", np.min)]:
            for dtype in (np.float32, np.float64):
                with self.subTest(pooling=pf_name, dtype=dtype):
                    X = X32.astype(dtype)
                    model = FeatureAgglomeration(n_clusters=3, pooling_func=pf)
                    model.fit(X)
                    onx = to_onnx(model, (X,))

                    ref = ExtendedReferenceEvaluator(onx)
                    (result,) = ref.run(None, {"X": X})

                    expected = model.transform(X)
                    self.assertEqualArray(expected, result, atol=1e-5)

                    sess = self.check_ort(onx)
                    (ort_result,) = sess.run(None, {"X": X})
                    self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_feature_agglomeration_pipeline(self):
        from sklearn.cluster import FeatureAgglomeration
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(6)
        X = rng.standard_normal((30, 8)).astype(np.float32)
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("fa", FeatureAgglomeration(n_clusters=3))]
        )
        pipe.fit(X)

        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        (result,) = ref.run(None, {"X": X})

        expected = pipe.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        (ort_result,) = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_feature_agglomeration_unsupported_pooling_func(self):
        from sklearn.cluster import FeatureAgglomeration
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(7)
        X = rng.standard_normal((20, 6)).astype(np.float32)
        model = FeatureAgglomeration(n_clusters=3, pooling_func=np.median)
        model.fit(X)

        with self.assertRaises(NotImplementedError):
            to_onnx(model, (X,))


if __name__ == "__main__":
    unittest.main(verbosity=2)

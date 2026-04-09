"""
Unit tests for yobx.sklearn converters.
"""

import unittest
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from yobx.ext_test_case import ExtTestCase, requires_sklearn, requires_pandas, hide_stdout
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx
from yobx.typing import ConvertOptionsProtocol


@requires_sklearn("1.4")
class TestSklearnBaseConverters(ExtTestCase):
    def test_standard_scaler(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        ss = StandardScaler()
        ss.fit(X)

        onx = to_onnx(ss, (X,))

        # Check graph structure
        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("Div", op_types)

        # Check numerical output
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = ss.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_logistic_regression_binary(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        ss = StandardScaler()
        X_scaled = ss.fit_transform(X).astype(np.float32)
        lr = LogisticRegression()
        lr.fit(X_scaled, y)

        onx = to_onnx(lr, (X,))

        # Check graph structure
        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Gemm", op_types)

        # Check outputs
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_scaled})
        label, proba = results[0], results[1]

        expected_label = lr.predict(X_scaled)
        expected_proba = lr.predict_proba(X_scaled).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X_scaled})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_logistic_regression_multiclass(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 2, 2])
        lr = LogisticRegression(max_iter=200)
        lr.fit(X, y)

        onx = to_onnx(lr, (X,))

        # Check graph structure
        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Gemm", op_types)
        self.assertIn("Softmax", op_types)

        # Check outputs
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = lr.predict(X)
        expected_proba = lr.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_pipeline_standard_scaler_logistic_regression(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        # Check graph contains nodes from both steps
        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("Div", op_types)
        self.assertIn("Gemm", op_types)

        # Check outputs
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = pipe.predict(X)
        expected_proba = pipe.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    @requires_sklearn("1.4")
    def test_pipeline_standard_scaler_only(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        pipe = Pipeline([("scaler", StandardScaler())])
        pipe.fit(X)

        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pipe.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    @requires_sklearn("1.4")
    def test_pipeline_multiclass(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 2, 2])
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = pipe.predict(X)
        expected_proba = pipe.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_custom_estimator_with_extra_converters(self):
        class ScaleByConstant(TransformerMixin, BaseEstimator):
            """Custom transformer that multiplies inputs by a constant."""

            def __init__(self, scale=2.0):
                self.scale = scale

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X * self.scale

        def convert_scale_by_constant(g, sts, outputs, estimator, X, name="scale"):
            import numpy as np

            scale = np.array([estimator.scale], dtype=np.float32)
            res = g.op.Mul(X, scale, name=name, outputs=outputs)
            if not sts:
                g.set_type(res, g.get_type(X))
                g.set_shape(res, g.get_shape(X))
                if g.has_device(X):
                    g.set_device(res, g.get_device(X))
            return res

        X = np.array([[1, 2], [3, 4]], dtype=np.float32)
        est = ScaleByConstant(scale=3.0)
        est.fit(X)

        onx = to_onnx(est, (X,), extra_converters={ScaleByConstant: convert_scale_by_constant})

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Mul", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = est.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_extra_converters_overrides_builtin(self):
        """extra_converters entries take priority over built-in converters."""
        called = []

        def custom_scaler_converter(g, sts, outputs, estimator, X, name="scaler"):
            called.append(True)
            res = g.op.Identity(X, name=name, outputs=outputs)
            if not sts:
                g.set_type(res, g.get_type(X))
                g.set_shape(res, g.get_shape(X))
                if g.has_device(X):
                    g.set_device(res, g.get_device(X))
            return res

        X = np.array([[1, 2], [3, 4]], dtype=np.float32)
        ss = StandardScaler()
        ss.fit(X)

        onx = to_onnx(ss, (X,), extra_converters={StandardScaler: custom_scaler_converter})

        self.assertTrue(called, "custom converter was not called")
        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Identity", op_types)
        self.assertNotIn("Sub", op_types)

    def test_estimator_without_transform_or_predict_raises(self):
        from sklearn.exceptions import NotFittedError

        class NoOpEstimator(BaseEstimator):
            def fit(self, X, y=None):
                return self

        estimator = NoOpEstimator().fit(np.zeros((4, 2), dtype=np.float32))
        X = np.zeros((4, 2), dtype=np.float32)
        with self.assertRaises(NotFittedError) as cm:
            to_onnx(estimator, (X,))
        self.assertIn("transform", str(cm.exception))
        self.assertIn("predict", str(cm.exception))

    def test_random_forest_classifier_binary(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        rf = RandomForestClassifier(n_estimators=5, random_state=0)
        rf.fit(X, y)

        onx = to_onnx(rf, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("TreeEnsemble", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(rf.predict(X), label)
        self.assertEqualArray(rf.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_random_forest_classifier_multiclass(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 2, 2])
        rf = RandomForestClassifier(n_estimators=5, random_state=0)
        rf.fit(X, y)

        onx = to_onnx(rf, (X,), target_opset=18)

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("TreeEnsembleClassifier", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(rf.predict(X), label)
        self.assertEqualArray(rf.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_random_forest_regressor(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        rf = RandomForestRegressor(n_estimators=5, random_state=0)
        rf.fit(X, y)

        onx = to_onnx(rf, (X,), target_opset=18)

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("TreeEnsembleRegressor", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        self.assertEqualArray(
            rf.predict(X).astype(np.float32).reshape(-1, 1), predictions, atol=1e-5
        )

    def test_random_forest_classifier_binary_v5(self):
        """TreeEnsemble (ai.onnx.ml opset 5) - binary classification."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        rf = RandomForestClassifier(n_estimators=5, random_state=0)
        rf.fit(X, y)

        onx = to_onnx(rf, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("TreeEnsemble", op_types)
        self.assertNotIn("TreeEnsembleClassifier", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(rf.predict(X), label)
        self.assertEqualArray(rf.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_random_forest_classifier_multiclass_v5(self):
        """TreeEnsemble (ai.onnx.ml opset 5) - multi-class classification."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 2, 2])
        rf = RandomForestClassifier(n_estimators=5, random_state=0)
        rf.fit(X, y)

        onx = to_onnx(rf, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("TreeEnsemble", op_types)
        self.assertNotIn("TreeEnsembleClassifier", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(rf.predict(X), label)
        self.assertEqualArray(rf.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_random_forest_regressor_v5(self):
        """TreeEnsemble (ai.onnx.ml opset 5) - regression."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        rf = RandomForestRegressor(n_estimators=5, random_state=0)
        rf.fit(X, y)

        onx = to_onnx(rf, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("TreeEnsemble", op_types)
        self.assertNotIn("TreeEnsembleRegressor", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        predictions = results[0]

        self.assertEqualArray(
            rf.predict(X).astype(np.float32).reshape(-1, 1), predictions, atol=1e-5
        )

    def test_pipeline_random_forest_classifier(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=5, random_state=0)),
            ]
        )
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("TreeEnsemble", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(pipe.predict(X), label)
        self.assertEqualArray(pipe.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_standard_scaler_large_model(self):
        """to_onnx with large_model=True returns an ExportArtifact with container set."""
        from yobx.container import ExportArtifact, ExtendedModelContainer

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        ss = StandardScaler()
        ss.fit(X)

        artifact = to_onnx(ss, (X,), large_model=True)
        self.assertIsInstance(artifact, ExportArtifact)
        self.assertIsInstance(artifact.container, ExtendedModelContainer)

    def test_logistic_regression_large_model(self):
        """to_onnx with large_model=True returns an ExportArtifact with container set."""
        from yobx.container import ExportArtifact, ExtendedModelContainer

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32)
        lr = LogisticRegression()
        lr.fit(X_scaled, y)

        artifact = to_onnx(lr, (X,), large_model=True)
        self.assertIsInstance(artifact, ExportArtifact)
        self.assertIsInstance(artifact.container, ExtendedModelContainer)

    def test_custom_converter_with_convert_options(self):
        """A custom converter can use g.convert_options.has() to emit optional outputs."""
        from yobx.helpers.onnx_helper import tensor_dtype_to_np_dtype

        # ── custom estimator ──────────────────────────────────────────────────
        class ClipTransformer(TransformerMixin, BaseEstimator):
            def __init__(self, clip_min=0.0, clip_max=1.0):
                self.clip_min = clip_min
                self.clip_max = clip_max

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return np.clip(X, self.clip_min, self.clip_max)

        # ── custom convert-options class ──────────────────────────────────────
        class ClipOptions(ConvertOptionsProtocol):
            def __init__(self, clip_mask=False):
                self.clip_mask = clip_mask

            def available_options(self):
                return ["clip_mask"]

            def has(self, option_name, piece, name=None):
                if option_name == "clip_mask":
                    return bool(self.clip_mask) and hasattr(piece, "clip_min")
                return False

        # ── converter function ────────────────────────────────────────────────
        def convert_clip_transformer(g, sts, outputs, estimator, X, name="clip"):
            itype = g.get_type(X)
            dtype = tensor_dtype_to_np_dtype(itype)
            low = np.array(estimator.clip_min, dtype=dtype)
            high = np.array(estimator.clip_max, dtype=dtype)

            _clipped = g.op.Clip(X, low, high, name=name, outputs=outputs[:1])

            if g.convert_options.has("clip_mask", estimator, name):
                below = g.op.Less(X, low, name=f"{name}_below")
                above = g.op.Greater(X, high, name=f"{name}_above")
                g.op.Or(below, above, name=f"{name}_mask", outputs=outputs[1:2])

            return outputs[0] if len(outputs) == 1 else tuple(outputs)

        X = np.array([[-1.0, 0.5], [0.3, 2.0], [-0.8, 1.5]], dtype=np.float32)
        transformer = ClipTransformer(clip_min=-0.5, clip_max=1.0).fit(X)
        extra = {ClipTransformer: convert_clip_transformer}

        # Without convert_options: single output
        onx_plain = to_onnx(transformer, (X,), extra_converters=extra)
        self.assertEqual(1, len(onx_plain.proto.graph.output))

        ref_plain = ExtendedReferenceEvaluator(onx_plain)
        (clipped_onnx,) = ref_plain.run(None, {"X": X})
        self.assertEqualArray(
            transformer.transform(X).astype(np.float32), clipped_onnx, atol=1e-6
        )

        # With clip_mask=True: two outputs
        onx_mask = to_onnx(
            transformer, (X,), extra_converters=extra, convert_options=ClipOptions(clip_mask=True)
        )
        self.assertEqual(2, len(onx_mask.proto.graph.output))

        ref_mask = ExtendedReferenceEvaluator(onx_mask)
        clipped_onnx2, mask_onnx = ref_mask.run(None, {"X": X})
        self.assertEqualArray(
            transformer.transform(X).astype(np.float32), clipped_onnx2, atol=1e-6
        )
        expected_mask = (X < transformer.clip_min) | (X > transformer.clip_max)
        self.assertEqualArray(expected_mask, mask_onnx)

        # The mask ops should appear in the graph
        op_types = [n.op_type for n in onx_mask.proto.graph.node]
        self.assertIn("Less", op_types)
        self.assertIn("Greater", op_types)
        self.assertIn("Or", op_types)


@requires_sklearn("1.4")
class TestGetSklearnEstimatorCoverage(ExtTestCase):
    def setUp(self):
        from yobx.sklearn import register_sklearn_converters

        register_sklearn_converters()

    def test_returns_list_of_dicts(self):
        from yobx.sklearn.register import get_sklearn_estimator_coverage

        rows = get_sklearn_estimator_coverage()
        self.assertIsInstance(rows, list)
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertIn("name", row)
            self.assertIn("cls", row)
            self.assertIn("module", row)
            self.assertIn("yobx", row)
            self.assertIn("predictable", row)

    def test_known_yobx_converters_marked_true(self):
        from yobx.sklearn.register import get_sklearn_estimator_coverage

        rows = get_sklearn_estimator_coverage()
        by_name = {r["name"]: r for r in rows}
        self.assertTrue(by_name["LinearRegression"]["yobx"])
        self.assertTrue(by_name["StandardScaler"]["yobx"])

    def test_transformers_included(self):
        from yobx.sklearn.register import get_sklearn_estimator_coverage

        rows = get_sklearn_estimator_coverage()
        names = {r["name"] for r in rows}
        # Standard trainable transforms must be present
        self.assertIn("StandardScaler", names)
        self.assertIn("PCA", names)
        self.assertIn("MinMaxScaler", names)

    def test_yobx_registered_non_type_filter_classes_included(self):
        from yobx.sklearn.register import get_sklearn_estimator_coverage

        rows = get_sklearn_estimator_coverage()
        names = {r["name"] for r in rows}
        # Pipeline is registered in yobx but has no _estimator_type
        self.assertIn("Pipeline", names)
        self.assertTrue(next(r for r in rows if r["name"] == "Pipeline")["yobx"])

    def test_skl2onnx_field_is_bool_or_none(self):
        from yobx.sklearn.register import get_sklearn_estimator_coverage

        rows = get_sklearn_estimator_coverage()
        for row in rows:
            self.assertIn(row["predictable"], (True, False, None))

    def test_sorted_by_name(self):
        from yobx.sklearn.register import get_sklearn_estimator_coverage

        rows = get_sklearn_estimator_coverage()
        names = [r["name"] for r in rows]
        self.assertEqual(names, sorted(names))


@requires_sklearn("1.4")
class TestSklearnToOnnxValueInfoProto(ExtTestCase):
    """Tests that to_onnx accepts :class:`onnx.ValueInfoProto` as input descriptors."""

    def test_standard_scaler_value_info_proto(self):
        """ValueInfoProto replaces the numpy array as input specification."""
        import onnx

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        ss = StandardScaler()
        ss.fit(X)

        vip = onnx.helper.make_tensor_value_info("features", onnx.TensorProto.FLOAT, ["N", 2])
        onx = to_onnx(ss, (vip,))

        # The ONNX graph input must use the name from the ValueInfoProto.
        self.assertEqual(onx.proto.graph.input[0].name, "features")
        # Shape dim 0 should be symbolic ("N"), dim 1 should be static 2.
        shape = onx.proto.graph.input[0].type.tensor_type.shape
        self.assertEqual(shape.dim[0].dim_param, "N")
        self.assertEqual(shape.dim[1].dim_value, 2)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"features": X})[0]
        self.assertEqualArray(ss.transform(X).astype(np.float32), result, atol=1e-5)

    def test_standard_scaler_value_info_proto_with_input_names_override(self):
        """input_names overrides the name embedded in a ValueInfoProto."""
        import onnx

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        ss = StandardScaler()
        ss.fit(X)

        vip = onnx.helper.make_tensor_value_info("features", onnx.TensorProto.FLOAT, ["N", 2])
        onx = to_onnx(ss, (vip,), input_names=["my_input"])

        self.assertEqual(onx.proto.graph.input[0].name, "my_input")

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"my_input": X})[0]
        self.assertEqualArray(ss.transform(X).astype(np.float32), result, atol=1e-5)

    def test_pipeline_value_info_proto(self):
        """ValueInfoProto works with a Pipeline (scaler + regressor)."""
        import onnx
        from sklearn.linear_model import LinearRegression

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
        pipe.fit(X, y)

        vip = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None, 2])
        onx = to_onnx(pipe, (vip,))

        self.assertEqual(onx.proto.graph.input[0].name, "X")

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pipe.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, result, atol=1e-5)

    @hide_stdout()
    def test_verbosity(self):
        """ValueInfoProto works with a Pipeline (scaler + regressor)."""
        import onnx
        from sklearn.linear_model import LinearRegression

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
        pipe.fit(X, y)

        vip = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None, 2])
        onx = to_onnx(pipe, (vip,), verbose=1)

        self.assertEqual(onx.proto.graph.input[0].name, "X")

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pipe.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, result, atol=1e-5)


class TestSklearnToOnnxTupleSpec(ExtTestCase):
    """Tests that to_onnx accepts ``(name, dtype, shape)`` tuples as input descriptors."""

    def test_standard_scaler_tuple_spec(self):
        """(name, dtype, shape) tuple replaces the numpy array as input specification."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        ss = StandardScaler()
        ss.fit(X)

        onx = to_onnx(ss, (("features", np.float32, ("N", 2)),))

        # The ONNX graph input must use the name from the tuple.
        self.assertEqual(onx.proto.graph.input[0].name, "features")
        # Shape dim 0 should be symbolic ("N"), dim 1 should be static 2.
        shape = onx.proto.graph.input[0].type.tensor_type.shape
        self.assertEqual(shape.dim[0].dim_param, "N")
        self.assertEqual(shape.dim[1].dim_value, 2)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"features": X})[0]
        self.assertEqualArray(ss.transform(X).astype(np.float32), result, atol=1e-5)

    def test_standard_scaler_tuple_spec_with_input_names_override(self):
        """input_names overrides the name embedded in a (name, dtype, shape) tuple."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        ss = StandardScaler()
        ss.fit(X)

        onx = to_onnx(ss, (("features", np.float32, ("N", 2)),), input_names=["my_input"])

        self.assertEqual(onx.proto.graph.input[0].name, "my_input")

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"my_input": X})[0]
        self.assertEqualArray(ss.transform(X).astype(np.float32), result, atol=1e-5)

    def test_pipeline_tuple_spec(self):
        """(name, dtype, shape) tuple works with a Pipeline (scaler + regressor)."""
        from sklearn.linear_model import LinearRegression

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (("X", np.float32, ("N", 2)),))

        self.assertEqual(onx.proto.graph.input[0].name, "X")

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pipe.predict(X).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_tuple_spec_with_static_first_dim(self):
        """(name, dtype, shape) tuple with all static dims is supported."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        ss = StandardScaler()
        ss.fit(X)

        onx = to_onnx(ss, (("X", np.float32, (4, 2)),))

        shape = onx.proto.graph.input[0].type.tensor_type.shape
        self.assertEqual(shape.dim[0].dim_value, 4)
        self.assertEqual(shape.dim[1].dim_value, 2)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        self.assertEqualArray(ss.transform(X).astype(np.float32), result, atol=1e-5)


@requires_sklearn("1.4")
@requires_pandas()
class TestSklearnToOnnxDataFrame(ExtTestCase):
    """Tests that to_onnx exposes each DataFrame column as a separate ONNX input."""

    def test_standard_scaler_dataframe(self):
        """Each column of a float32 DataFrame becomes a separate 1-D ONNX graph input."""
        import pandas as pd

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        df = pd.DataFrame(X, columns=["a", "b"])
        ss = StandardScaler()
        ss.fit(df)

        onx = to_onnx(ss, (df,))

        # ONNX graph should have two 1-D float32 inputs named after the columns.
        graph_inputs = {inp.name: inp for inp in onx.proto.graph.input}
        self.assertIn("a", graph_inputs)
        self.assertIn("b", graph_inputs)
        for col in ("a", "b"):
            inp = graph_inputs[col]
            self.assertEqual(inp.type.tensor_type.elem_type, 1)  # FLOAT = 1
            shape = inp.type.tensor_type.shape
            self.assertEqual(len(shape.dim), 2)  # 1-D column
            self.assertEqual(shape.dim[1].dim_value, 1)

        # Numerical output should match sklearn when feeding per-column 1-D arrays.
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"a": X[:, 0], "b": X[:, 1]})[0]
        expected = ss.transform(df).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_standard_scaler_dataframe_float64(self):
        """DataFrame with float64 dtype exposes 1-D float64 ONNX inputs per column."""
        import pandas as pd

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
        df = pd.DataFrame(X, columns=["a", "b"])
        ss = StandardScaler()
        ss.fit(df)

        onx = to_onnx(ss, (df,))

        graph_inputs = {inp.name: inp for inp in onx.proto.graph.input}
        self.assertIn("a", graph_inputs)
        self.assertIn("b", graph_inputs)
        for col in ("a", "b"):
            self.assertEqual(graph_inputs[col].type.tensor_type.elem_type, 11)  # DOUBLE = 11

    def test_pipeline_dataframe(self):
        """DataFrame works as input to to_onnx for a Pipeline (scaler + LR)."""
        import pandas as pd

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        df = pd.DataFrame(X, columns=["a", "b"])
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        pipe.fit(df, y)

        onx = to_onnx(pipe, (df,))

        # ONNX graph should have per-column inputs.
        graph_inputs = {inp.name: inp for inp in onx.proto.graph.input}
        self.assertIn("a", graph_inputs)
        self.assertIn("b", graph_inputs)

        ref = ExtendedReferenceEvaluator(onx)
        label, _ = ref.run(None, {"a": X[:, 0], "b": X[:, 1]})
        expected = pipe.predict(df)
        self.assertEqualArray(expected, label)


@requires_sklearn("1.4")
class TestSklearnConvertersBasicInvocation(ExtTestCase):
    """Verify that every registered sklearn converter can be invoked successfully.

    Each converter is called directly with output names pre-computed from
    :func:`~yobx.sklearn.sklearn_helper.get_output_names`, mirroring exactly
    what :func:`~yobx.sklearn.to_onnx` does internally.  This catches
    registration bugs (e.g. a newly added converter that returns ``None`` or
    has a mis-named output) that would not be caught by individual estimator
    tests.
    """

    # Estimator classes that require non-trivial setup (extra estimators,
    # supervised feature selectors, 1-D inputs, etc.) and are intentionally
    # excluded from this smoke test.
    _SKIP = frozenset(
        {
            # Container meta-estimators
            "Pipeline",
            "ColumnTransformer",
            "FeatureUnion",
            # Meta-estimators that wrap other estimators
            "GridSearchCV",
            "HalvingGridSearchCV",
            "HalvingRandomSearchCV",
            "RandomizedSearchCV",
            "StackingClassifier",
            "StackingRegressor",
            "VotingClassifier",
            "VotingRegressor",
            "MultiOutputClassifier",
            "MultiOutputRegressor",
            "ClassifierChain",
            "RegressorChain",
            "OneVsOneClassifier",
            "OneVsRestClassifier",
            "OutputCodeClassifier",
            "CalibratedClassifierCV",
            "TransformedTargetRegressor",
            "TunedThresholdClassifierCV",
            # Feature selectors that need supervised fit(X, y)
            "SelectFdr",
            "SelectFpr",
            "SelectFwe",
            "SelectKBest",
            "SelectPercentile",
            "SelectFromModel",
            "RFE",
            "RFECV",
            # Estimators with special input requirements
            "RANSACRegressor",
            "IsotonicRegression",  # 1-D input
            "BernoulliRBM",  # different input semantics
            "KernelCenterer",  # expects square kernel matrix
            "GaussianRandomProjection",  # fails with tiny dataset
            "PatchExtractor",  # needs image input
            "CountVectorizer",  # needs text input
            "TfidfVectorizer",  # needs text input
            # Estimators needing supervised fit for dimensionality reduction
            "NeighborhoodComponentsAnalysis",
            "CCA",
            "PLSSVD",
            # Multi-task estimators (require multi-column y)
            "MultiTaskLasso",
            "MultiTaskElasticNet",
            "MultiTaskLassoCV",
            "MultiTaskElasticNetCV",
            # NMF default solver not supported
            "NMF",
            "MiniBatchNMF",
            # LocalOutlierFactor needs novelty=True which changes semantics
            "LocalOutlierFactor",
            # Supervised feature selection / imputation needing y or special X
            "MissingIndicator",
            # KernelPCA with n_components=None (default) triggers a converter bug
            # unrelated to this test
            "KernelPCA",
            # STRING
            "FeatureHasher",
        }
    )

    def setUp(self):
        from yobx.sklearn import register_sklearn_converters

        register_sklearn_converters()

        # Small float32 dataset suitable for most estimators.
        rng = np.random.default_rng(0)
        self._X = np.abs(rng.standard_normal((20, 4)).astype(np.float32)) + 1
        self._y_bin = (rng.random(20) > 0.5).astype(np.int64)
        self._y_reg = np.abs(rng.standard_normal(20).astype(np.float32)) + 1

    def _make_graph_builder(self):
        from yobx.xbuilder import GraphBuilder

        return GraphBuilder({"": 20, "ai.onnx.ml": 5})

    def _fit_estimator(self, cls):
        """Return a fitted instance of *cls* or raise ``SkipTest``."""
        from sklearn.base import ClusterMixin, OutlierMixin, is_classifier, is_regressor

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = cls()
            try:
                if is_classifier(est):
                    est.fit(self._X, self._y_bin)
                elif is_regressor(est):
                    est.fit(self._X, self._y_reg)
                elif isinstance(est, (ClusterMixin, OutlierMixin)):
                    est.fit(self._X)
                else:
                    est.fit(self._X)
            except Exception as exc:
                self.skipTest(f"Cannot fit {cls.__name__}: {exc}")
        return est

    def test_all_sklearn_converters_basic_invocation(self):
        """Every registered sklearn-library converter must return a non-None result."""
        from sklearn.utils import all_estimators
        from yobx.sklearn.register import get_sklearn_converters
        from yobx.sklearn.sklearn_helper import get_output_names

        converters = get_sklearn_converters()

        tested = 0
        for _name, cls in sorted(all_estimators(), key=lambda x: x[0]):
            if cls not in converters:
                continue
            if cls.__name__ in self._SKIP:
                continue

            with self.subTest(estimator=cls.__name__):
                est = self._fit_estimator(cls)

                g = self._make_graph_builder()
                inp = g.make_tensor_input("X", 1, (None, self._X.shape[1]))

                # Pre-compute output names exactly as to_onnx() does internally.
                output_names = get_output_names(est, g.convert_options)
                outputs = [g.unique_name(n) for n in output_names] if output_names else None

                fct = converters[cls]
                result = fct(g, {}, outputs, est, inp, name="test")

                self.assertIsNotNone(result, msg=f"Converter for {cls.__name__} returned None")
                tested += 1

        self.assertGreater(tested, 0, "No converters were tested")

    def test_to_onnx_with_no_known_output_mixin(self):
        """to_onnx works for a custom estimator with NoKnownOutputMixin."""
        from sklearn.base import TransformerMixin
        from yobx.sklearn import NoKnownOutputMixin
        from yobx.sklearn import to_onnx

        class _DoubleTransformer(BaseEstimator, TransformerMixin, NoKnownOutputMixin):
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X * 2.0

        def _double_converter(g, sts, outputs, estimator, X, name="double"):
            import numpy as np

            two = np.array([2.0], dtype=np.float32)
            res = g.op.Mul(X, two, name=name, outputs=outputs)
            if not sts:
                g.set_type(res, g.get_type(X))
                g.set_shape(res, g.get_shape(X))
            return res

        X = np.array([[1, 2], [3, 4]], dtype=np.float32)
        est = _DoubleTransformer().fit(X)
        onx = to_onnx(est, (X,), extra_converters={_DoubleTransformer: _double_converter})

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        self.assertEqualArray(X * 2.0, result, atol=1e-5)


@requires_sklearn("1.4")
class TestSklearnToOnnxReturnOptimizeReport(ExtTestCase):
    """Tests for *return_optimize_report* parameter in yobx.sklearn.to_onnx."""

    def test_default_report_is_none(self):
        """Report is None by default (return_optimize_report=False)."""
        from yobx.container import ExportArtifact

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        ss = StandardScaler()
        ss.fit(X)

        artifact = to_onnx(ss, (X,))
        self.assertIsInstance(artifact, ExportArtifact)
        self.assertIsNone(artifact.report)

    def test_report_populated_when_true(self):
        """Report is an ExportReport with stats when return_optimize_report=True."""
        from yobx.container import ExportArtifact, ExportReport

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        ss = StandardScaler()
        ss.fit(X)

        artifact = to_onnx(ss, (X,), return_optimize_report=True)
        self.assertIsInstance(artifact, ExportArtifact)
        self.assertIsNotNone(artifact.report)
        self.assertIsInstance(artifact.report, ExportReport)
        self.assertIsInstance(artifact.report.stats, list)
        self.assertGreater(len(artifact.report.stats), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

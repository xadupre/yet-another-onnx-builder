import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.typing import GraphBuilderExtendedProtocol
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnUsingSklearnOnnx(ExtTestCase):
    def test_mlp_with_sklearn_onnx(self):
        from typing import Dict, List, Tuple
        import onnx
        from sklearn.neural_network import MLPClassifier
        from yobx.xbuilder import FunctionOptions
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import DoubleTensorType, FloatTensorType

        def to_skl2onnx_input_type(elem_type: int, n_features: int):
            if elem_type == onnx.TensorProto.FLOAT:
                return FloatTensorType([None, n_features])
            if elem_type == onnx.TensorProto.DOUBLE:
                return DoubleTensorType([None, n_features])
            raise NotImplementedError(
                f"Input elem_type {elem_type} is not supported. "
                "Only FLOAT (1) and DOUBLE (11) are supported by the skl2onnx MLP converter."
            )

        def convert_sklearn_mlp_classifier(
            g: GraphBuilderExtendedProtocol,
            sts: Dict,
            outputs: List[str],
            estimator: MLPClassifier,
            X: str,
            name: str = "mlp_classifier",
        ) -> Tuple[str, str]:
            assert isinstance(
                estimator, MLPClassifier
            ), f"Unexpected type {type(estimator)} for estimator."
            assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

            itype = g.get_type(X)
            n_features = estimator.coefs_[0].shape[0]

            onx = convert_sklearn(
                estimator,
                initial_types=[("X", to_skl2onnx_input_type(itype, n_features))],
                options={"zipmap": False},
                target_opset=g.main_opset,
            )
            # sklearn chooses the lowest opset equivalent to the current ones
            # given the nodes it contains. We need to overwrite that.
            del onx.opset_import[:]
            d = onx.opset_import.add()
            d.domain = ""
            d.version = g.main_opset
            builder = g.__class__(onx)

            f_options = FunctionOptions(
                export_as_function=True,
                name=g.unique_function_name("MLPClassifier"),
                domain="sklean_onnx_functions",
                move_initializer_to_constant=True,
            )
            g.make_local_function(builder, f_options)
            return g.make_node(f_options.name, [X], outputs, domain=f_options.domain, name=name)

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        mlp = MLPClassifier(
            hidden_layer_sizes=(4,), activation="relu", random_state=0, max_iter=2000
        )
        mlp.fit(X, y)

        onx = to_onnx(mlp, (X,), extra_converters={MLPClassifier: convert_sklearn_mlp_classifier})

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("MatMul", op_types)
        self.assertIn("Sigmoid", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = mlp.predict(X)
        expected_proba = mlp.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_wrap_skl2onnx_converter_mlp(self):
        """Test wrap_skl2onnx_converter factory with MLPClassifier."""
        from sklearn.neural_network import MLPClassifier
        from skl2onnx._supported_operators import sklearn_operator_name_map
        from skl2onnx.common._registration import get_converter
        from yobx.sklearn import wrap_skl2onnx_converter

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        mlp = MLPClassifier(
            hidden_layer_sizes=(4,), activation="relu", random_state=0, max_iter=2000
        )
        mlp.fit(X, y)

        skl2onnx_fn = get_converter(sklearn_operator_name_map[MLPClassifier])
        converter = wrap_skl2onnx_converter(skl2onnx_fn)
        onx = to_onnx(mlp, (X,), extra_converters={MLPClassifier: converter})

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("MatMul", op_types)
        self.assertIn("Sigmoid", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = mlp.predict(X)
        expected_proba = mlp.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_wrap_skl2onnx_converter_linear_regression(self):
        """Test wrap_skl2onnx_converter factory with LinearRegression."""
        from sklearn.linear_model import LinearRegression
        from skl2onnx._supported_operators import sklearn_operator_name_map
        from skl2onnx.common._registration import get_converter
        from yobx.sklearn import wrap_skl2onnx_converter

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        reg = LinearRegression().fit(X, y)

        skl2onnx_fn = get_converter(sklearn_operator_name_map[LinearRegression])
        converter = wrap_skl2onnx_converter(skl2onnx_fn)
        onx = to_onnx(reg, (X,), extra_converters={LinearRegression: converter})

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        (pred,) = results

        expected = reg.predict(X).astype(np.float32)
        self.assertEqualArray(expected, pred.ravel(), atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

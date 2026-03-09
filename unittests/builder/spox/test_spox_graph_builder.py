"""
Unit tests for :class:`~yobx.builder.spox.SpoxGraphBuilder`.

The tests validate:

* Opset management (``main_opset``, ``get_opset``, ``set_opset``, ``has_opset``)
* Protocol conformance (:class:`~yobx.typing.GraphBuilderExtendedProtocol`)
* Core builder API (``make_tensor_input``, ``make_initializer``, ``make_node``,
  ``make_tensor_output``, ``to_onnx``)
* ``g.op.*`` dispatch (variadic, single-input, attribute-bearing ops)
* Type/shape side-channel methods
* Sklearn :class:`~sklearn.pipeline.Pipeline` conversion end-to-end
"""

import unittest
import numpy as np
from onnx import TensorProto

from yobx.ext_test_case import ExtTestCase, requires_sklearn, requires_spox
from yobx.typing import GraphBuilderExtendedProtocol

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_builder(opset=18):
    from yobx.builder.spox import SpoxGraphBuilder

    return SpoxGraphBuilder(opset)


# ---------------------------------------------------------------------------
# Protocol / import tests
# ---------------------------------------------------------------------------


@requires_spox()
class TestSpoxGraphBuilderImport(ExtTestCase):
    """SpoxGraphBuilder is importable and satisfies GraphBuilderExtendedProtocol."""

    def test_import(self):
        from yobx.builder.spox import SpoxGraphBuilder

        self.assertIsNotNone(SpoxGraphBuilder)

    def test_is_instance_extended_protocol(self):
        g = _make_builder(18)
        self.assertIsInstance(g, GraphBuilderExtendedProtocol)


# ---------------------------------------------------------------------------
# Opset API
# ---------------------------------------------------------------------------


@requires_spox()
class TestSpoxGraphBuilderOpsetApi(ExtTestCase):
    """Validates opset management API."""

    def test_main_opset_int(self):
        g = _make_builder(18)
        self.assertEqual(g.main_opset, 18)

    def test_main_opset_dict(self):
        from yobx.builder.spox import SpoxGraphBuilder

        g = SpoxGraphBuilder({"": 20, "ai.onnx.ml": 3})
        self.assertEqual(g.main_opset, 20)

    def test_get_opset_main(self):
        g = _make_builder(18)
        self.assertEqual(g.get_opset(""), 18)

    def test_get_opset_secondary(self):
        from yobx.builder.spox import SpoxGraphBuilder

        g = SpoxGraphBuilder({"": 18, "ai.onnx.ml": 3})
        self.assertEqual(g.get_opset("ai.onnx.ml"), 3)

    def test_get_opset_missing_raises(self):
        g = _make_builder()
        with self.assertRaises(AssertionError):
            g.get_opset("unknown.domain", exc=True)

    def test_get_opset_missing_no_exc(self):
        g = _make_builder()
        self.assertIsNone(g.get_opset("unknown.domain", exc=False))

    def test_set_opset_new_domain(self):
        g = _make_builder()
        g.set_opset("ai.onnx.ml", 3)
        self.assertEqual(g.get_opset("ai.onnx.ml"), 3)

    def test_set_opset_same_version_noop(self):
        g = _make_builder(18)
        g.set_opset("", 18)  # should not raise
        self.assertEqual(g.main_opset, 18)

    def test_set_opset_version_mismatch_raises(self):
        g = _make_builder(18)
        with self.assertRaises(AssertionError):
            g.set_opset("", 20)

    def test_has_opset_present(self):
        g = _make_builder(18)
        self.assertEqual(g.has_opset(""), 18)

    def test_has_opset_missing(self):
        g = _make_builder()
        self.assertEqual(g.has_opset("nonexistent"), 0)


# ---------------------------------------------------------------------------
# Core builder API
# ---------------------------------------------------------------------------


@requires_spox()
class TestSpoxGraphBuilderCoreApi(ExtTestCase):
    """make_tensor_input, make_initializer, make_node, make_tensor_output, to_onnx."""

    def test_make_tensor_input_registers_name(self):
        g = _make_builder(18)
        g.make_tensor_input("X", TensorProto.FLOAT, (None, 4))
        self.assertIn("X", g.input_names)
        self.assertTrue(g.has_name("X"))

    def test_make_initializer_numpy(self):
        g = _make_builder(18)
        name = g.make_initializer("W", np.eye(4, dtype=np.float32))
        self.assertEqual(name, "W")
        self.assertTrue(g.has_name("W"))

    def test_make_initializer_empty_name(self):
        g = _make_builder(18)
        name = g.make_initializer("", np.ones(3, dtype=np.float32))
        self.assertIsInstance(name, str)
        self.assertTrue(len(name) > 0)
        self.assertTrue(g.has_name(name))

    def test_make_node_matmul_returns_name(self):
        g = _make_builder(18)
        g.make_tensor_input("X", TensorProto.FLOAT, (None, 4))
        w = g.make_initializer("W", np.eye(4, dtype=np.float32))
        y = g.make_node("MatMul", ["X", w], 1)
        self.assertIsInstance(y, str)

    def test_make_tensor_output_registers(self):
        g = _make_builder(18)
        g.make_tensor_input("X", TensorProto.FLOAT, (None, 4))
        w = g.make_initializer("W", np.eye(4, dtype=np.float32))
        y = g.make_node("MatMul", ["X", w], 1)
        g.make_tensor_output(y)
        self.assertIn(y, g.output_names)

    def test_to_onnx_returns_model_proto(self):
        import onnx

        g = _make_builder(18)
        g.make_tensor_input("X", TensorProto.FLOAT, (None, 4))
        w = g.make_initializer("W", np.eye(4, dtype=np.float32))
        y = g.make_node("MatMul", ["X", w], 1)
        g.make_tensor_output(y)
        proto = g.to_onnx()
        self.assertIsInstance(proto, onnx.ModelProto)
        op_types = [n.op_type for n in proto.graph.node]
        self.assertIn("MatMul", op_types)


# ---------------------------------------------------------------------------
# g.op.* dispatch
# ---------------------------------------------------------------------------


@requires_spox()
class TestSpoxGraphBuilderOpDispatch(ExtTestCase):
    """Validates the g.op.* shorthand API."""

    def test_op_relu(self):
        g = _make_builder(18)
        g.make_tensor_input("X", TensorProto.FLOAT, (None, 4))
        y = g.op.Relu("X")
        self.assertIsInstance(y, str)

    def test_op_gemm_with_attrs(self):
        g = _make_builder(18)
        g.make_tensor_input("X", TensorProto.FLOAT, (None, 4))
        w = g.make_initializer("W", np.ones((4, 3), dtype=np.float32))
        b = g.make_initializer("B", np.zeros(3, dtype=np.float32))
        y = g.op.Gemm("X", w, b, transB=1)
        self.assertIsInstance(y, str)

    def test_op_concat_variadic(self):
        """Concat takes Sequence[Var] — variadic inputs via *args must work."""
        g = _make_builder(18)
        g.make_tensor_input("X", TensorProto.FLOAT, (None, 4))
        g.make_tensor_input("Y", TensorProto.FLOAT, (None, 4))
        z = g.op.Concat("X", "Y", axis=1)
        self.assertIsInstance(z, str)

    def test_op_cast_with_onnx_dtype_int(self):
        """Cast.to is DTypeLike — passing an int ONNX elem-type must work."""
        g = _make_builder(18)
        g.make_tensor_input("X", TensorProto.FLOAT, (None, 4))
        y = g.op.ArgMax("X", axis=1, keepdims=0)
        z = g.op.Cast(y, to=TensorProto.INT64)
        self.assertIsInstance(z, str)

    def test_op_numpy_inline_input(self):
        """Passing a numpy array directly as an input creates a Constant."""
        g = _make_builder(18)
        g.make_tensor_input("X", TensorProto.FLOAT, (None, 4))
        scalar = np.array([0.0], dtype=np.float32)
        y = g.op.Sub(scalar, "X")
        self.assertIsInstance(y, str)

    def test_op_outputs_rename(self):
        """The 'outputs' kwarg renames the result tensors."""
        g = _make_builder(18)
        g.make_tensor_input("X", TensorProto.FLOAT, (None, 4))
        y = g.op.Relu("X", outputs=["my_relu"])
        self.assertEqual(y, "my_relu")

    def test_op_identity(self):
        g = _make_builder(18)
        g.make_tensor_input("X", TensorProto.FLOAT, (None, 4))
        y = g.op.Identity("X")
        self.assertIsInstance(y, str)


# ---------------------------------------------------------------------------
# Type / shape side-channel
# ---------------------------------------------------------------------------


@requires_spox()
class TestSpoxGraphBuilderTypeShape(ExtTestCase):
    """has_type, get_type, set_type, has_shape, get_shape, set_shape."""

    def test_has_type_after_input(self):
        g = _make_builder(18)
        g.make_tensor_input("X", TensorProto.FLOAT, (None, 4))
        self.assertTrue(g.has_type("X"))

    def test_get_type_after_input(self):
        g = _make_builder(18)
        g.make_tensor_input("X", TensorProto.FLOAT, (None, 4))
        self.assertEqual(g.get_type("X"), TensorProto.FLOAT)

    def test_set_type_get_type(self):
        g = _make_builder(18)
        g.make_tensor_input("X", TensorProto.FLOAT, (None, 4))
        y = g.op.Relu("X")
        g.set_type(y, TensorProto.FLOAT)
        self.assertTrue(g.has_type(y))
        self.assertEqual(g.get_type(y), TensorProto.FLOAT)

    def test_has_shape_after_input(self):
        g = _make_builder(18)
        g.make_tensor_input("X", TensorProto.FLOAT, (None, 4))
        self.assertTrue(g.has_shape("X"))

    def test_get_shape_after_input(self):
        g = _make_builder(18)
        g.make_tensor_input("X", TensorProto.FLOAT, (None, 4))
        shape = g.get_shape("X")
        self.assertEqual(shape, (None, 4))

    def test_set_shape_get_shape(self):
        g = _make_builder(18)
        g.make_tensor_input("X", TensorProto.FLOAT, (None, 4))
        y = g.op.Relu("X")
        g.set_shape(y, (None, 4))
        self.assertEqual(g.get_shape(y), (None, 4))

    def test_set_type_shape_unary_op(self):
        g = _make_builder(18)
        g.make_tensor_input("X", TensorProto.FLOAT, (None, 4))
        y = g.op.Relu("X")
        result = g.set_type_shape_unary_op(y, "X")
        # Should propagate type from X to y
        self.assertTrue(g.has_type(y) or result is not None)


# ---------------------------------------------------------------------------
# unique_name
# ---------------------------------------------------------------------------


@requires_spox()
class TestSpoxGraphBuilderUniqueName(ExtTestCase):
    def test_unique_name_first_call(self):
        g = _make_builder()
        name = g.unique_name("foo")
        self.assertEqual(name, "foo")

    def test_unique_name_collision(self):
        g = _make_builder()
        n1 = g.unique_name("foo")
        n2 = g.unique_name("foo")
        self.assertNotEqual(n1, n2)
        self.assertTrue(n2.startswith("foo"))


# ---------------------------------------------------------------------------
# ai.onnx.ml domain (TreeEnsembleClassifier)
# ---------------------------------------------------------------------------


@requires_spox()
@requires_sklearn("1.4")
class TestSpoxGraphBuilderMlDomain(ExtTestCase):
    def test_logistic_regression(self):
        from sklearn.linear_model import LogisticRegression
        from yobx.sklearn import to_onnx
        from yobx.builder.spox import SpoxGraphBuilder

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        dt = LogisticRegression()
        dt.fit(X, y)

        onx = to_onnx(dt, (X,), builder_cls=SpoxGraphBuilder)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Gemm", op_types)

    def test_logistic_regression_correctness(self):
        from sklearn.linear_model import LogisticRegression
        from yobx.sklearn import to_onnx
        from yobx.builder.spox import SpoxGraphBuilder
        from yobx.reference import ExtendedReferenceEvaluator

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 0])
        dt = LogisticRegression()
        dt.fit(X, y)

        onx = to_onnx(dt, (X,), builder_cls=SpoxGraphBuilder)
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label = results[0]
        self.assertEqualArray(dt.predict(X), label)


# ---------------------------------------------------------------------------
# Sklearn Pipeline end-to-end
# ---------------------------------------------------------------------------


@requires_spox()
@requires_sklearn("1.4")
class TestSpoxSklearnPipeline(ExtTestCase):
    """End-to-end sklearn Pipeline conversion with SpoxGraphBuilder."""

    def test_standard_scaler(self):
        from sklearn.preprocessing import StandardScaler
        from yobx.sklearn import to_onnx
        from yobx.builder.spox import SpoxGraphBuilder
        from yobx.reference import ExtendedReferenceEvaluator

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        ss = StandardScaler()
        ss.fit(X)

        onx = to_onnx(ss, (X,), builder_cls=SpoxGraphBuilder)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("Div", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = ss.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_logistic_regression_binary(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from yobx.sklearn import to_onnx
        from yobx.builder.spox import SpoxGraphBuilder
        from yobx.reference import ExtendedReferenceEvaluator

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y_arr = np.array([0, 0, 1, 1])
        ss = StandardScaler()
        X_scaled = ss.fit_transform(X).astype(np.float32)
        lr = LogisticRegression()
        lr.fit(X_scaled, y_arr)

        onx = to_onnx(lr, (X,), builder_cls=SpoxGraphBuilder)
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X_scaled})
        label, proba = results[0], results[1]

        self.assertEqualArray(lr.predict(X_scaled), label)
        self.assertEqualArray(lr.predict_proba(X_scaled).astype(np.float32), proba, atol=1e-5)

    def test_logistic_regression_multiclass(self):
        from sklearn.linear_model import LogisticRegression
        from yobx.sklearn import to_onnx
        from yobx.builder.spox import SpoxGraphBuilder
        from yobx.reference import ExtendedReferenceEvaluator

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        y_arr = np.array([0, 0, 1, 1, 2, 2])
        lr = LogisticRegression(max_iter=200)
        lr.fit(X, y_arr)

        onx = to_onnx(lr, (X,), builder_cls=SpoxGraphBuilder)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Gemm", op_types)
        self.assertIn("Softmax", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(lr.predict(X), label)
        self.assertEqualArray(lr.predict_proba(X).astype(np.float32), proba, atol=1e-5)

    def test_pipeline_scaler_logistic_regression(self):
        """Core requirement: sklearn Pipeline via SpoxGraphBuilder."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx
        from yobx.builder.spox import SpoxGraphBuilder
        from yobx.reference import ExtendedReferenceEvaluator

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y_arr = np.array([0, 0, 1, 1])
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        pipe.fit(X, y_arr)

        onx = to_onnx(pipe, (X,), builder_cls=SpoxGraphBuilder)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("Div", op_types)
        self.assertIn("Gemm", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        self.assertEqualArray(pipe.predict(X), label)
        self.assertEqualArray(pipe.predict_proba(X).astype(np.float32), proba, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

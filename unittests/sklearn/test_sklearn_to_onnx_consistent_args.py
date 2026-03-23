"""
Unit tests verifying that :func:`yobx.sklearn.to_onnx` accepts the full
range of supported arg types: numpy arrays, numpy dtypes / scalar types,
and :class:`onnx.ValueInfoProto` descriptors.
"""

import unittest
import numpy as np
import onnx
from yobx.ext_test_case import ExtTestCase, requires_sklearn


@requires_sklearn("1.4")
class TestSklearnToOnnxConsistentArgs(ExtTestCase):
    """Verify that yobx.sklearn.to_onnx accepts all supported arg types."""

    def _make_regressor(self):
        from sklearn.linear_model import LinearRegression

        X = np.random.randn(20, 4).astype(np.float32)
        y = X @ np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        return LinearRegression().fit(X, y), X

    def test_numpy_array(self):
        """Baseline: standard numpy-array arg works."""
        from yobx.sklearn import to_onnx

        reg, X = self._make_regressor()
        artifact = to_onnx(reg, (X,))
        self.assertEqual(len(artifact.proto.graph.input), 1)

    def test_numpy_scalar_type_float32(self):
        """numpy scalar type (e.g. np.float32) is accepted as arg."""
        from yobx.sklearn import to_onnx

        reg, _ = self._make_regressor()
        artifact = to_onnx(reg, (np.float32,))
        self.assertEqual(len(artifact.proto.graph.input), 1)
        inp = artifact.proto.graph.input[0]
        self.assertEqual(inp.type.tensor_type.elem_type, onnx.TensorProto.FLOAT)

    def test_numpy_dtype_instance(self):
        """numpy.dtype instance is accepted as arg."""
        from yobx.sklearn import to_onnx

        reg, _ = self._make_regressor()
        artifact = to_onnx(reg, (np.dtype("float32"),))
        self.assertEqual(len(artifact.proto.graph.input), 1)
        inp = artifact.proto.graph.input[0]
        self.assertEqual(inp.type.tensor_type.elem_type, onnx.TensorProto.FLOAT)

    def test_value_info_proto(self):
        """onnx.ValueInfoProto is accepted as arg (existing behaviour preserved)."""
        from yobx.sklearn import to_onnx

        reg, _ = self._make_regressor()
        vip = onnx.helper.make_tensor_value_info("my_X", onnx.TensorProto.FLOAT, [None, 4])
        artifact = to_onnx(reg, (vip,))
        self.assertEqual(len(artifact.proto.graph.input), 1)
        self.assertEqual(artifact.proto.graph.input[0].name, "my_X")

    def test_multiple_arg_types_mixed(self):
        """Multiple inputs with different arg types all work."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression
        from yobx.sklearn import to_onnx

        X = np.random.randn(20, 4).astype(np.float32)
        y = X.sum(axis=1)
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())])
        pipe.fit(X, y)

        # Pass a numpy array — pipeline has a single input
        artifact = to_onnx(pipe, (X,))
        self.assertEqual(len(artifact.proto.graph.input), 1)

        # Pass a dtype instead
        artifact2 = to_onnx(pipe, (np.float32,))
        self.assertEqual(len(artifact2.proto.graph.input), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

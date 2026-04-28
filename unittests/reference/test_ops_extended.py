import unittest
import numpy as np
import onnx
import onnx.helper as oh
from yobx.ext_test_case import ExtTestCase
from yobx.reference import ExtendedReferenceEvaluator

DOMAIN = "yaourt.ortops.fused_kernel.cuda"
TFLOAT = onnx.TensorProto.FLOAT
TFLOAT16 = onnx.TensorProto.FLOAT16
TINT64 = onnx.TensorProto.INT64


def _make_model(nodes, inputs, outputs, domain=DOMAIN):
    return oh.make_model(
        oh.make_graph(nodes, "test", inputs, outputs),
        opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(domain, 1)],
        ir_version=9,
    )


class TestExtendedOpsAddAddMulMul(ExtTestCase):
    """Tests for ops in op__extended_add_add_mul_mul.py."""

    def _xyz(self):
        rng = np.random.default_rng(0)
        x = rng.random((4,)).astype(np.float32)
        y = rng.random((4,)).astype(np.float32)
        z = rng.random((4,)).astype(np.float32)
        return x, y, z

    def test_add_add(self):
        model = _make_model(
            [oh.make_node("AddAdd", ["X", "Y", "Z"], ["out"], domain=DOMAIN)],
            [
                oh.make_tensor_value_info("X", TFLOAT, None),
                oh.make_tensor_value_info("Y", TFLOAT, None),
                oh.make_tensor_value_info("Z", TFLOAT, None),
            ],
            [oh.make_tensor_value_info("out", TFLOAT, None)],
        )
        x, y, z = self._xyz()
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": x, "Y": y, "Z": z})
        self.assertEqualArray(x + y + z, got[0])

    def test_mul_mul(self):
        model = _make_model(
            [oh.make_node("MulMul", ["X", "Y", "Z"], ["out"], domain=DOMAIN)],
            [
                oh.make_tensor_value_info("X", TFLOAT, None),
                oh.make_tensor_value_info("Y", TFLOAT, None),
                oh.make_tensor_value_info("Z", TFLOAT, None),
            ],
            [oh.make_tensor_value_info("out", TFLOAT, None)],
        )
        x, y, z = self._xyz()
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": x, "Y": y, "Z": z})
        self.assertEqualArray(x * y * z, got[0])

    def test_add_mul(self):
        model = _make_model(
            [oh.make_node("AddMul", ["X", "Y", "Z"], ["out"], domain=DOMAIN)],
            [
                oh.make_tensor_value_info("X", TFLOAT, None),
                oh.make_tensor_value_info("Y", TFLOAT, None),
                oh.make_tensor_value_info("Z", TFLOAT, None),
            ],
            [oh.make_tensor_value_info("out", TFLOAT, None)],
        )
        x, y, z = self._xyz()
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": x, "Y": y, "Z": z})
        self.assertEqualArray((x + y) * z, got[0])

    def test_add_mul_transpose_middle(self):
        model = _make_model(
            [oh.make_node("AddMul", ["X", "Y", "Z"], ["out"], domain=DOMAIN, transposeMiddle=1)],
            [
                oh.make_tensor_value_info("X", TFLOAT, None),
                oh.make_tensor_value_info("Y", TFLOAT, None),
                oh.make_tensor_value_info("Z", TFLOAT, None),
            ],
            [oh.make_tensor_value_info("out", TFLOAT, None)],
        )
        rng = np.random.default_rng(1)
        x = rng.random((2, 3, 4, 5)).astype(np.float32)
        y = rng.random((2, 3, 4, 5)).astype(np.float32)
        z = rng.random((2, 3, 4, 5)).astype(np.float32)
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": x, "Y": y, "Z": z})
        expected = np.transpose((x + y) * z, axes=[0, 2, 1, 3])
        self.assertEqualArray(expected, got[0])

    def test_mul_add(self):
        model = _make_model(
            [oh.make_node("MulAdd", ["X", "Y", "Z"], ["out"], domain=DOMAIN)],
            [
                oh.make_tensor_value_info("X", TFLOAT, None),
                oh.make_tensor_value_info("Y", TFLOAT, None),
                oh.make_tensor_value_info("Z", TFLOAT, None),
            ],
            [oh.make_tensor_value_info("out", TFLOAT, None)],
        )
        x, y, z = self._xyz()
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": x, "Y": y, "Z": z})
        self.assertEqualArray((x * y) + z, got[0])

    def test_mul_add_transpose_middle(self):
        model = _make_model(
            [oh.make_node("MulAdd", ["X", "Y", "Z"], ["out"], domain=DOMAIN, transposeMiddle=1)],
            [
                oh.make_tensor_value_info("X", TFLOAT, None),
                oh.make_tensor_value_info("Y", TFLOAT, None),
                oh.make_tensor_value_info("Z", TFLOAT, None),
            ],
            [oh.make_tensor_value_info("out", TFLOAT, None)],
        )
        rng = np.random.default_rng(2)
        x = rng.random((2, 3, 4, 5)).astype(np.float32)
        y = rng.random((2, 3, 4, 5)).astype(np.float32)
        z = rng.random((2, 3, 4, 5)).astype(np.float32)
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": x, "Y": y, "Z": z})
        expected = np.transpose((x * y) + z, axes=[0, 2, 1, 3])
        self.assertEqualArray(expected, got[0])

    def test_sub_mul(self):
        model = _make_model(
            [oh.make_node("SubMul", ["X", "Y", "Z"], ["out"], domain=DOMAIN)],
            [
                oh.make_tensor_value_info("X", TFLOAT, None),
                oh.make_tensor_value_info("Y", TFLOAT, None),
                oh.make_tensor_value_info("Z", TFLOAT, None),
            ],
            [oh.make_tensor_value_info("out", TFLOAT, None)],
        )
        x, y, z = self._xyz()
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": x, "Y": y, "Z": z})
        self.assertEqualArray((x - y) * z, got[0])

    def test_sub_mul_negative(self):
        model = _make_model(
            [oh.make_node("SubMul", ["X", "Y", "Z"], ["out"], domain=DOMAIN, negative=1)],
            [
                oh.make_tensor_value_info("X", TFLOAT, None),
                oh.make_tensor_value_info("Y", TFLOAT, None),
                oh.make_tensor_value_info("Z", TFLOAT, None),
            ],
            [oh.make_tensor_value_info("out", TFLOAT, None)],
        )
        x, y, z = self._xyz()
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": x, "Y": y, "Z": z})
        self.assertEqualArray((y - x) * z, got[0])

    def test_mul_sub(self):
        model = _make_model(
            [oh.make_node("MulSub", ["X", "Y", "Z"], ["out"], domain=DOMAIN)],
            [
                oh.make_tensor_value_info("X", TFLOAT, None),
                oh.make_tensor_value_info("Y", TFLOAT, None),
                oh.make_tensor_value_info("Z", TFLOAT, None),
            ],
            [oh.make_tensor_value_info("out", TFLOAT, None)],
        )
        x, y, z = self._xyz()
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": x, "Y": y, "Z": z})
        self.assertEqualArray((x * y) - z, got[0])

    def test_mul_sub_negative(self):
        model = _make_model(
            [oh.make_node("MulSub", ["X", "Y", "Z"], ["out"], domain=DOMAIN, negative=1)],
            [
                oh.make_tensor_value_info("X", TFLOAT, None),
                oh.make_tensor_value_info("Y", TFLOAT, None),
                oh.make_tensor_value_info("Z", TFLOAT, None),
            ],
            [oh.make_tensor_value_info("out", TFLOAT, None)],
        )
        x, y, z = self._xyz()
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": x, "Y": y, "Z": z})
        self.assertEqualArray(z - (x * y), got[0])

    def test_add_shared_input(self):
        model = _make_model(
            [oh.make_node("AddSharedInput", ["X", "Y", "Z"], ["out1", "out2"], domain=DOMAIN)],
            [
                oh.make_tensor_value_info("X", TFLOAT, None),
                oh.make_tensor_value_info("Y", TFLOAT, None),
                oh.make_tensor_value_info("Z", TFLOAT, None),
            ],
            [
                oh.make_tensor_value_info("out1", TFLOAT, None),
                oh.make_tensor_value_info("out2", TFLOAT, None),
            ],
        )
        x, y, z = self._xyz()
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": x, "Y": y, "Z": z})
        self.assertEqualArray(x + y, got[0])
        self.assertEqualArray(x + z, got[1])

    def test_mul_shared_input(self):
        model = _make_model(
            [oh.make_node("MulSharedInput", ["X", "Y", "Z"], ["out1", "out2"], domain=DOMAIN)],
            [
                oh.make_tensor_value_info("X", TFLOAT, None),
                oh.make_tensor_value_info("Y", TFLOAT, None),
                oh.make_tensor_value_info("Z", TFLOAT, None),
            ],
            [
                oh.make_tensor_value_info("out1", TFLOAT, None),
                oh.make_tensor_value_info("out2", TFLOAT, None),
            ],
        )
        x, y, z = self._xyz()
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": x, "Y": y, "Z": z})
        self.assertEqualArray(x * y, got[0])
        self.assertEqualArray(x * z, got[1])


class TestExtendedOpsMulSigmoid(ExtTestCase):
    """Tests for ops in op__extended_mul_sigmoid.py."""

    def test_mul_sigmoid_1d(self):
        model = _make_model(
            [oh.make_node("MulSigmoid", ["X"], ["out"], domain=DOMAIN)],
            [oh.make_tensor_value_info("X", TFLOAT, None)],
            [oh.make_tensor_value_info("out", TFLOAT, None)],
        )
        x = np.array([0.5, -1.0, 2.0, -0.5], dtype=np.float32)
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": x})

        def sigmoid(v):
            return np.where(v > 0, 1 / (1 + np.exp(-v)), np.exp(v) / (1 + np.exp(v)))

        expected = (x * sigmoid(x)).astype(np.float32)
        self.assertEqualArray(expected, got[0], atol=1e-6)

    def test_mul_sigmoid_empty(self):
        model = _make_model(
            [oh.make_node("MulSigmoid", ["X"], ["out"], domain=DOMAIN)],
            [oh.make_tensor_value_info("X", TFLOAT, None)],
            [oh.make_tensor_value_info("out", TFLOAT, None)],
        )
        x = np.array([], dtype=np.float32)
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": x})
        self.assertEqualArray(x, got[0])


class TestExtendedOpsNegXplus1(ExtTestCase):
    """Tests for ops in op__extended_negxplus1.py."""

    def test_neg_x_plus_1(self):
        model = _make_model(
            [oh.make_node("NegXplus1", ["X"], ["out"], domain=DOMAIN)],
            [oh.make_tensor_value_info("X", TFLOAT, None)],
            [oh.make_tensor_value_info("out", TFLOAT, None)],
        )
        x = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": x})
        self.assertEqualArray((1 - x).astype(np.float32), got[0])

    def test_neg_x_plus_1_preserves_dtype(self):
        model = _make_model(
            [oh.make_node("NegXplus1", ["X"], ["out"], domain=DOMAIN)],
            [oh.make_tensor_value_info("X", TFLOAT, None)],
            [oh.make_tensor_value_info("out", TFLOAT, None)],
        )
        x = np.array([0.25, 0.75], dtype=np.float32)
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": x})
        self.assertEqual(got[0].dtype, np.float32)
        self.assertEqualArray(np.array([0.75, 0.25], dtype=np.float32), got[0])


class TestExtendedOpsReplaceZero(ExtTestCase):
    """Tests for ops in op__extended_replace_zero.py."""

    def test_replace_zero_equal(self):
        model = _make_model(
            [oh.make_node("ReplaceZero", ["X"], ["out"], domain=DOMAIN, by=float(-1), equal=1)],
            [oh.make_tensor_value_info("X", TFLOAT, None)],
            [oh.make_tensor_value_info("out", TFLOAT, None)],
        )
        x = np.array([0.0, 1.0, 0.0, 2.0], dtype=np.float32)
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": x})
        expected = np.array([-1.0, 1.0, -1.0, 2.0], dtype=np.float32)
        self.assertEqualArray(expected, got[0])

    def test_replace_zero_not_equal(self):
        model = _make_model(
            [oh.make_node("ReplaceZero", ["X"], ["out"], domain=DOMAIN, by=float(-1), equal=0)],
            [oh.make_tensor_value_info("X", TFLOAT, None)],
            [oh.make_tensor_value_info("out", TFLOAT, None)],
        )
        x = np.array([0.0, 1.0, 0.0, 2.0], dtype=np.float32)
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": x})
        expected = np.array([0.0, -1.0, 0.0, -1.0], dtype=np.float32)
        self.assertEqualArray(expected, got[0])


class TestExtendedOpsRotary(ExtTestCase):
    """Tests for ops in op__extended_rotary.py."""

    def test_rotary_left(self):
        model = _make_model(
            [oh.make_node("Rotary", ["X"], ["out"], domain=DOMAIN, side="left")],
            [oh.make_tensor_value_info("X", TFLOAT, None)],
            [oh.make_tensor_value_info("out", TFLOAT, None)],
        )
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": x})
        last_dim = x.shape[-1] // 2
        expected = x.copy()
        expected[..., :last_dim] = x[..., last_dim:]
        expected[..., last_dim:] = -x[..., :last_dim]
        self.assertEqualArray(expected, got[0])

    def test_rotary_right(self):
        model = _make_model(
            [oh.make_node("Rotary", ["X"], ["out"], domain=DOMAIN, side="right")],
            [oh.make_tensor_value_info("X", TFLOAT, None)],
            [oh.make_tensor_value_info("out", TFLOAT, None)],
        )
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": x})
        last_dim = x.shape[-1] // 2
        expected = x.copy()
        expected[..., :last_dim] = -x[..., last_dim:]
        expected[..., last_dim:] = x[..., :last_dim]
        self.assertEqualArray(expected, got[0])


class TestExtendedOpsScatterNDOfShape(ExtTestCase):
    """Tests for ops in op__extended_scatternd_of_shape.py."""

    def test_scatter_nd_of_shape(self):
        model = _make_model(
            [
                oh.make_node(
                    "ScatterNDOfShape",
                    ["shape", "indices", "updates"],
                    ["out"],
                    domain=DOMAIN,
                    reduction="add",
                )
            ],
            [
                oh.make_tensor_value_info("shape", TINT64, None),
                oh.make_tensor_value_info("indices", TINT64, None),
                oh.make_tensor_value_info("updates", TFLOAT, None),
            ],
            [oh.make_tensor_value_info("out", TFLOAT, None)],
        )
        shape = np.array([3, 4], dtype=np.int64)
        indices = np.array([[0], [1], [0]], dtype=np.int64)
        updates = np.arange(12, dtype=np.float32).reshape(3, 4)
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"shape": shape, "indices": indices, "updates": updates})
        expected = np.zeros((3, 4), dtype=np.float32)
        expected[0] += updates[0] + updates[2]
        expected[1] += updates[1]
        self.assertEqualArray(expected, got[0])

    def test_masked_scatter_nd_of_shape(self):
        model = _make_model(
            [
                oh.make_node(
                    "MaskedScatterNDOfShape",
                    ["shape", "indices", "updates"],
                    ["out"],
                    domain=DOMAIN,
                    reduction="add",
                    maskedValue=-1,
                )
            ],
            [
                oh.make_tensor_value_info("shape", TINT64, None),
                oh.make_tensor_value_info("indices", TINT64, None),
                oh.make_tensor_value_info("updates", TFLOAT, None),
            ],
            [oh.make_tensor_value_info("out", TFLOAT, None)],
        )
        shape = np.array([3, 4], dtype=np.int64)
        indices = np.array([[0], [-1], [1]], dtype=np.int64)
        updates = np.ones((3, 4), dtype=np.float32)
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"shape": shape, "indices": indices, "updates": updates})
        expected = np.zeros((3, 4), dtype=np.float32)
        expected[0] += updates[0]
        expected[1] += updates[2]
        self.assertEqualArray(expected, got[0])


class TestExtendedOpsTransposeCast(ExtTestCase):
    """Tests for ops in op__extended_transpose_cast.py."""

    def test_transpose2d_cast_fp16(self):
        model = _make_model(
            [oh.make_node("Transpose2DCastFP16", ["X"], ["out"], domain=DOMAIN)],
            [oh.make_tensor_value_info("X", TFLOAT, None)],
            [oh.make_tensor_value_info("out", TFLOAT16, None)],
        )
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": x})
        self.assertEqualArray(x.T.astype(np.float16), got[0])

    def test_transpose2d_cast_fp32(self):
        model = _make_model(
            [oh.make_node("Transpose2DCastFP32", ["X"], ["out"], domain=DOMAIN)],
            [oh.make_tensor_value_info("X", TFLOAT16, None)],
            [oh.make_tensor_value_info("out", TFLOAT, None)],
        )
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float16)
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": x})
        self.assertEqualArray(x.T.astype(np.float32), got[0])


class TestExtendedOpsTriMatrix(ExtTestCase):
    """Tests for ops in op__extended_tri_matrix.py."""

    def test_tri_matrix(self):
        model = _make_model(
            [oh.make_node("TriMatrix", ["shape", "csts"], ["out"], domain=DOMAIN)],
            [
                oh.make_tensor_value_info("shape", TINT64, None),
                oh.make_tensor_value_info("csts", TFLOAT, None),
            ],
            [oh.make_tensor_value_info("out", TFLOAT, None)],
        )
        shape = np.array([3, 3], dtype=np.int64)
        csts = np.array([0.1, 1.0, 0.2], dtype=np.float32)
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"shape": shape, "csts": csts})
        expected = np.array([[1.0, 0.2, 0.2], [0.1, 1.0, 0.2], [0.1, 0.1, 1.0]], dtype=np.float32)
        self.assertEqualArray(expected, got[0])

    def test_tri_matrix_non_square(self):
        model = _make_model(
            [oh.make_node("TriMatrix", ["shape", "csts"], ["out"], domain=DOMAIN)],
            [
                oh.make_tensor_value_info("shape", TINT64, None),
                oh.make_tensor_value_info("csts", TFLOAT, None),
            ],
            [oh.make_tensor_value_info("out", TFLOAT, None)],
        )
        shape = np.array([2, 4], dtype=np.int64)
        csts = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"shape": shape, "csts": csts})
        expected = np.array([[0.0, 1.0, 1.0, 1.0], [-1.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        self.assertEqualArray(expected, got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)

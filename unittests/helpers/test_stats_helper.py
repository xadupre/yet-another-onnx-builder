import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from yobx.ext_test_case import ExtTestCase
from yobx.helpers.stats_helper import model_statistics

TFLOAT = onnx.TensorProto.FLOAT


def _make_mlp_model() -> onnx.ModelProto:
    """Creates a small MLP model: Z = Relu(MatMul(X, W) + b)."""
    X = oh.make_tensor_value_info("X", TFLOAT, [4, 8])
    Z = oh.make_tensor_value_info("Z", TFLOAT, [4, 16])
    W = onh.from_array(np.ones((8, 16), dtype=np.float32), name="W")
    b = onh.from_array(np.zeros((16,), dtype=np.float32), name="b")
    nodes = [
        oh.make_node("MatMul", ["X", "W"], ["mm_out"]),
        oh.make_node("Add", ["mm_out", "b"], ["add_out"]),
        oh.make_node("Relu", ["add_out"], ["Z"]),
    ]
    graph = oh.make_graph(nodes, "mlp", [X], [Z], [W, b])
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])


def _make_dynamic_model() -> onnx.ModelProto:
    """Creates a model with a named dynamic (batch) dimension."""

    def _make_vi(name: str, shape):
        vi = oh.make_tensor_value_info(name, TFLOAT, None)
        vi.type.tensor_type.shape.CopyFrom(onnx.TensorShapeProto())
        for d in shape:
            dim = vi.type.tensor_type.shape.dim.add()
            if d is None:
                dim.dim_param = "batch"
            else:
                dim.dim_value = d
        return vi

    X = _make_vi("X", [None, 8])
    Z = _make_vi("Z", [None, 16])
    W = onh.from_array(np.ones((8, 16), dtype=np.float32), name="W")
    nodes = [oh.make_node("MatMul", ["X", "W"], ["Z"])]
    graph = oh.make_graph(nodes, "dyn", [X], [Z], [W])
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])


class TestStatsHelper(ExtTestCase):
    def test_node_count(self):
        model = _make_mlp_model()
        stats = model_statistics(model)
        self.assertEqual(stats["n_nodes"], 3)
        self.assertEqual(stats["node_count_per_op_type"]["MatMul"], 1)
        self.assertEqual(stats["node_count_per_op_type"]["Add"], 1)
        self.assertEqual(stats["node_count_per_op_type"]["Relu"], 1)

    def test_flops_matmul(self):
        # MatMul [4,8] @ [8,16] → 2*4*8*16 = 1024
        model = _make_mlp_model()
        stats = model_statistics(model)
        self.assertEqual(stats["flops_per_op_type"]["MatMul"], 2 * 4 * 8 * 16)

    def test_flops_add(self):
        # Add [4,16] → 64
        model = _make_mlp_model()
        stats = model_statistics(model)
        self.assertEqual(stats["flops_per_op_type"]["Add"], 4 * 16)

    def test_flops_relu(self):
        # Relu [4,16] → 64
        model = _make_mlp_model()
        stats = model_statistics(model)
        self.assertEqual(stats["flops_per_op_type"]["Relu"], 4 * 16)

    def test_total_flops(self):
        model = _make_mlp_model()
        stats = model_statistics(model)
        expected = 2 * 4 * 8 * 16 + 4 * 16 + 4 * 16  # 1024 + 64 + 64 = 1152
        self.assertEqual(stats["total_estimated_flops"], expected)

    def test_dynamic_shapes_gives_none(self):
        model = _make_dynamic_model()
        stats = model_statistics(model)
        # Dynamic batch dimension → FLOPs cannot be estimated
        self.assertIsNone(stats["total_estimated_flops"])
        self.assertIsNone(stats["flops_per_op_type"]["MatMul"])

    def test_node_stats_entries(self):
        model = _make_mlp_model()
        stats = model_statistics(model)
        self.assertEqual(len(stats["node_stats"]), 3)
        for ns in stats["node_stats"]:
            self.assertIn("op_type", ns)
            self.assertIn("estimated_flops", ns)
            self.assertIn("inputs", ns)
            self.assertIn("outputs", ns)

    def test_reshape_zero_flops(self):
        X = oh.make_tensor_value_info("X", TFLOAT, [4, 8])
        Z = oh.make_tensor_value_info("Z", TFLOAT, [32])
        shape = onh.from_array(np.array([32], dtype=np.int64), name="shape")
        nodes = [oh.make_node("Reshape", ["X", "shape"], ["Z"])]
        graph = oh.make_graph(nodes, "reshape_test", [X], [Z], [shape])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
        stats = model_statistics(model)
        self.assertEqual(stats["flops_per_op_type"]["Reshape"], 0)
        self.assertEqual(stats["total_estimated_flops"], 0)

    def test_gemm_flops(self):
        # Gemm [M,K] @ [K,N] + bias: 2*M*K*N + M*N
        M, K, N = 3, 5, 7
        X = oh.make_tensor_value_info("X", TFLOAT, [M, K])
        Z = oh.make_tensor_value_info("Z", TFLOAT, [M, N])
        W = onh.from_array(np.ones((K, N), dtype=np.float32), name="W")
        bias = onh.from_array(np.zeros((N,), dtype=np.float32), name="bias")
        nodes = [oh.make_node("Gemm", ["X", "W", "bias"], ["Z"])]
        graph = oh.make_graph(nodes, "gemm_test", [X], [Z], [W, bias])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
        stats = model_statistics(model)
        expected = 2 * M * K * N + M * N
        self.assertEqual(stats["flops_per_op_type"]["Gemm"], expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)

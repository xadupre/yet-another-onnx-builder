import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from yobx.ext_test_case import ExtTestCase
from yobx.helpers.stats_helper import (
    ModelStatistics,
    model_statistics,
    extract_attributes,
    stats_tree_ensemble,
    enumerate_stats_nodes,
    NodeStatistics,
    TreeStatistics,
    HistTreeStatistics,
)

TFLOAT = onnx.TensorProto.FLOAT
TINT64 = onnx.TensorProto.INT64


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

    def test_reshape_flops(self):
        X = oh.make_tensor_value_info("X", TFLOAT, [4, 8])
        Z = oh.make_tensor_value_info("Z", TFLOAT, [32])
        shape = onh.from_array(np.array([32], dtype=np.int64), name="shape")
        nodes = [oh.make_node("Reshape", ["X", "shape"], ["Z"])]
        graph = oh.make_graph(nodes, "reshape_test", [X], [Z], [shape])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
        stats = model_statistics(model)
        # Reshape cost = rank of output shape [32] → rank = 1
        self.assertEqual(stats["flops_per_op_type"]["Reshape"], 1)
        self.assertEqual(stats["total_estimated_flops"], 1)

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

    def test_literal_fn_used_for_matmul_output(self):
        """MatMul where the output shape comes from a Reshape whose shape is a literal."""
        # X [4,8] → Reshape(X, [1,32]) → Y [1,32] → MatMul(Y, W) → Z [1,16]
        X = oh.make_tensor_value_info("X", TFLOAT, [4, 8])
        Z = oh.make_tensor_value_info("Z", TFLOAT, [1, 16])
        # Shape literal for Reshape: [1, 32]
        reshape_shape = onh.from_array(np.array([1, 32], dtype=np.int64), name="rshape")
        W = onh.from_array(np.ones((32, 16), dtype=np.float32), name="W")
        nodes = [
            oh.make_node("Reshape", ["X", "rshape"], ["Y"]),
            oh.make_node("MatMul", ["Y", "W"], ["Z"]),
        ]
        graph = oh.make_graph(nodes, "literal_test", [X], [Z], [reshape_shape, W])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
        stats = model_statistics(model)
        # Reshape cost = rank of output [1,32] → rank 2;
        # MatMul [1,32] @ [32,16] → 2*1*32*16 = 1024
        self.assertEqual(stats["flops_per_op_type"]["Reshape"], 2)
        self.assertEqual(stats["flops_per_op_type"]["MatMul"], 2 * 1 * 32 * 16)

    def test_model_statistics_class_node_count(self):
        """ModelStatistics.compute() returns the same node counts as model_statistics()."""
        model = _make_mlp_model()
        ms = ModelStatistics(model)
        stats = ms.compute()
        self.assertEqual(stats["n_nodes"], 3)
        self.assertEqual(stats["node_count_per_op_type"]["MatMul"], 1)

    def test_model_statistics_class_flops(self):
        """ModelStatistics.compute() returns the same FLOPs as model_statistics()."""
        model = _make_mlp_model()
        stats = ModelStatistics(model).compute()
        expected = 2 * 4 * 8 * 16 + 4 * 16 + 4 * 16
        self.assertEqual(stats["total_estimated_flops"], expected)

    def test_model_statistics_class_shape_fn(self):
        """After compute(), shape_fn is accessible on the instance."""
        model = _make_mlp_model()
        ms = ModelStatistics(model)
        ms.compute()
        # W is a known initializer → shape should be (8, 16)
        self.assertEqual(ms.shape_fn("W"), (8, 16))

    def test_model_statistics_class_literal_fn(self):
        """After compute(), literal_fn resolves int-constant tensors."""
        X = oh.make_tensor_value_info("X", TFLOAT, [4, 8])
        Z = oh.make_tensor_value_info("Z", TFLOAT, [32])
        shape = onh.from_array(np.array([32], dtype=np.int64), name="shape")
        nodes = [oh.make_node("Reshape", ["X", "shape"], ["Z"])]
        graph = oh.make_graph(nodes, "lit_fn_test", [X], [Z], [shape])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
        ms = ModelStatistics(model)
        ms.compute()
        self.assertEqual(ms.literal_fn("shape"), (32,))

    def test_model_statistics_from_graph_builder(self):
        """ModelStatistics accepts a GraphBuilderExtendedProtocol directly."""
        try:
            from yobx.xbuilder import GraphBuilder
        except ImportError:
            self.skipTest("yobx.xbuilder not available")

        gb = GraphBuilder(17)
        gb.make_tensor_input("X", TFLOAT, (4, 8))
        gb.make_initializer("W", np.ones((8, 16), dtype=np.float32))
        out = gb.make_node("MatMul", ["X", "W"], name="mm0")
        gb.make_tensor_output(out, TFLOAT, (4, 16))

        ms = ModelStatistics(gb)
        stats = ms.compute()
        self.assertEqual(stats["n_nodes"], 1)
        self.assertEqual(stats["node_count_per_op_type"]["MatMul"], 1)
        # MatMul [4,8] @ [8,16] → 2*4*8*16 = 1024
        self.assertEqual(stats["flops_per_op_type"]["MatMul"], 2 * 4 * 8 * 16)
        self.assertEqual(stats["total_estimated_flops"], 2 * 4 * 8 * 16)

    def test_no_op_handler_returns_zero_except_identity(self):
        """
        Every registered op handler must return a non-zero cost for a fixed
        non-empty input, except ``Identity`` which is the only allowed zero-cost op.
        """
        import onnx.helper as _oh
        from yobx.xshape.cost_inference import _OP_HANDLERS

        # A fixed non-empty shape used for all inputs.
        _SHAPE = (4, 8)

        def _shape_fn(name: str):
            return _SHAPE

        def _literal_fn(name: str):
            return _SHAPE

        zero_cost_ops = []
        for op_type, handler in sorted(_OP_HANDLERS.items()):
            node = _oh.make_node(op_type, inputs=["X"], outputs=["Y"])
            result = handler(node, _shape_fn, _literal_fn)
            if result == 0:
                zero_cost_ops.append(op_type)

        self.assertEqual(
            zero_cost_ops,
            ["Identity"],
            f"Only Identity should have cost=0, but got: {zero_cost_ops}",
        )


def _make_tree_ensemble_node(n_trees: int = 2) -> onnx.NodeProto:
    """Builds a small TreeEnsembleClassifier node with *n_trees* trees (1 split each)."""
    nodeids, treeids, featureids, modes, values = [], [], [], [], []
    truenodes, falsenodes, hitrates, missing = [], [], [], []
    class_ids, class_nodeids, class_treeids, class_weights = [], [], [], []

    for tid in range(n_trees):
        # node 0: split on feature tid % 2, threshold 0.5
        # node 1: left leaf  -> class 0
        # node 2: right leaf -> class 1
        nodeids += [0, 1, 2]
        treeids += [tid, tid, tid]
        featureids += [tid % 2, 0, 0]
        modes += ["BRANCH_LEQ", "LEAF", "LEAF"]
        values += [0.5, 0.0, 0.0]
        truenodes += [1, 0, 0]
        falsenodes += [2, 0, 0]
        hitrates += [1.0, 1.0, 1.0]
        missing += [0, 0, 0]
        class_ids += [0, 1]
        class_nodeids += [1, 2]
        class_treeids += [tid, tid]
        class_weights += [1.0, 1.0]

    return oh.make_node(
        "TreeEnsembleClassifier",
        inputs=["X"],
        outputs=["label", "probabilities"],
        domain="ai.onnx.ml",
        nodes_nodeids=nodeids,
        nodes_treeids=treeids,
        nodes_featureids=featureids,
        nodes_modes=modes,
        nodes_values=values,
        nodes_truenodeids=truenodes,
        nodes_falsenodeids=falsenodes,
        nodes_hitrates=hitrates,
        nodes_missing_value_tracks_true=missing,
        class_ids=class_ids,
        class_nodeids=class_nodeids,
        class_treeids=class_treeids,
        class_weights=class_weights,
        classlabels_int64s=[0, 1],
        post_transform="NONE",
    )


def _make_tree_model(n_trees: int = 2) -> onnx.ModelProto:
    """Wraps ``_make_tree_ensemble_node`` in a minimal ONNX model."""
    node = _make_tree_ensemble_node(n_trees)
    X_vi = oh.make_tensor_value_info("X", TFLOAT, [None, 2])
    label_vi = oh.make_tensor_value_info("label", TINT64, [None])
    proba_vi = oh.make_tensor_value_info("probabilities", TFLOAT, [None, 2])
    graph = oh.make_graph([node], "tree_test", [X_vi], [label_vi, proba_vi])
    return oh.make_model(
        graph,
        opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("ai.onnx.ml", 3)],
        ir_version=10,
    )


class TestTreeStatistics(ExtTestCase):
    def test_extract_attributes_ints(self):
        node = _make_tree_ensemble_node(1)
        atts = extract_attributes(node)
        self.assertIn("nodes_nodeids", atts)
        val = atts["nodes_nodeids"]
        np.testing.assert_array_equal(val, [0, 1, 2])

    def test_extract_attributes_floats(self):
        node = _make_tree_ensemble_node(1)
        atts = extract_attributes(node)
        val = atts["nodes_values"]
        self.assertIsInstance(val, np.ndarray)
        self.assertEqual(val.dtype, np.float32)

    def test_extract_attributes_strings(self):
        node = _make_tree_ensemble_node(1)
        atts = extract_attributes(node)
        val = atts["nodes_modes"]
        self.assertIn("BRANCH_LEQ", list(val))
        self.assertIn("LEAF", list(val))

    def test_stats_tree_ensemble_basic(self):
        node = _make_tree_ensemble_node(2)
        model = _make_tree_model(2)
        stats = stats_tree_ensemble(model.graph, node)
        self.assertIsInstance(stats, NodeStatistics)
        self.assertEqual(stats["kind"], "Classifier")
        self.assertEqual(stats["n_trees"], 2)

    def test_stats_tree_ensemble_n_outputs(self):
        node = _make_tree_ensemble_node(1)
        model = _make_tree_model(1)
        stats = stats_tree_ensemble(model.graph, node)
        # class_ids has [0, 1] -> 2 distinct output classes
        self.assertEqual(stats["n_outputs"], 2)

    def test_stats_tree_ensemble_features(self):
        node = _make_tree_ensemble_node(2)
        model = _make_tree_model(2)
        stats = stats_tree_ensemble(model.graph, node)
        # 2 trees: tree 0 uses feature 0, tree 1 uses feature 1
        self.assertEqual(stats["n_features"], 2)
        features = stats["features"]
        self.assertEqual(len(features), 2)
        for f in features:
            self.assertIsInstance(f, HistTreeStatistics)

    def test_stats_tree_ensemble_trees(self):
        node = _make_tree_ensemble_node(3)
        model = _make_tree_model(3)
        stats = stats_tree_ensemble(model.graph, node)
        trees = stats["trees"]
        self.assertEqual(len(trees), 3)
        for tr in trees:
            self.assertIsInstance(tr, TreeStatistics)
            self.assertEqual(tr["n_nodes"], 3)  # 1 split + 2 leaves
            self.assertEqual(tr["n_leaves"], 2)

    def test_stats_tree_ensemble_rules(self):
        node = _make_tree_ensemble_node(1)
        model = _make_tree_model(1)
        stats = stats_tree_ensemble(model.graph, node)
        self.assertEqual(stats["rules"], {"BRANCH_LEQ", "LEAF"})
        self.assertEqual(stats["n_rules"], 2)

    def test_stats_tree_ensemble_dict_values(self):
        node = _make_tree_ensemble_node(2)
        model = _make_tree_model(2)
        stats = stats_tree_ensemble(model.graph, node)
        dv = stats.dict_values
        self.assertIn("n_trees", dv)
        self.assertEqual(dv["n_trees"], 2)
        self.assertIn("kind", dv)

    def test_enumerate_stats_nodes(self):
        model = _make_tree_model(2)
        results = list(enumerate_stats_nodes(model))
        self.assertEqual(len(results), 1)
        _path, _parent, node_stats = results[0]
        self.assertIsInstance(node_stats, NodeStatistics)
        self.assertEqual(node_stats["n_trees"], 2)

    def test_enumerate_stats_nodes_custom_fcts(self):
        model = _make_tree_model(1)
        # Pass an empty dict → no nodes matched
        results = list(enumerate_stats_nodes(model, stats_fcts={}))
        self.assertEqual(results, [])

    def test_hist_tree_statistics(self):
        node = _make_tree_ensemble_node(1)
        values = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
        hist_stat = HistTreeStatistics(node, featureid=0, values=values)
        self.assertLess(abs(hist_stat["min"] - 0.1), 1e-5)
        self.assertLess(abs(hist_stat["max"] - 0.9), 1e-5)
        self.assertEqual(hist_stat["size"], 5)
        self.assertEqual(hist_stat["n_distinct"], 5)
        dv = hist_stat.dict_values
        self.assertIn("min", dv)
        self.assertIn("max", dv)

    def test_tree_statistics_dict_values(self):
        node = _make_tree_ensemble_node(1)
        tr = TreeStatistics(node, tree_id=0)
        tr.add("n_nodes", 3)
        tr.add("n_leaves", 2)
        tr.add("max_featureid", 1)
        tr.add("n_features", 1)
        tr.add("n_rules", 2)
        tr.add("rules", {"BRANCH_LEQ", "LEAF"})
        from collections import Counter

        tr.add("hist_rules", Counter({"LEAF": 2, "BRANCH_LEQ": 1}))
        dv = tr.dict_values
        self.assertIn("n_nodes", dv)
        self.assertEqual(dv["n_nodes"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

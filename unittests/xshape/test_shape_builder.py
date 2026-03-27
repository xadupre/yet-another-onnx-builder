import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from yobx.ext_test_case import ExtTestCase
from yobx.reference import ExtendedReferenceEvaluator
from yobx.xshape import ShapeBuilder, BasicShapeBuilder

TFLOAT = onnx.TensorProto.FLOAT
TFLOAT16 = onnx.TensorProto.FLOAT16
TINT64 = onnx.TensorProto.INT64
_mkv_ = oh.make_tensor_value_info


class TestShapeBuilder(ExtTestCase):
    def test_shape_builder(self):
        # Concrete methods in ShapeBuilder that should NOT raise NotImplementedError.
        _concrete_get = {"get_registered_constraints", "get_shape_renamed"}
        builder = ShapeBuilder()
        for me in dir(builder):
            if me.startswith("get_") and not me.startswith("get_att"):
                if me in _concrete_get:
                    # These are now concrete – they must not raise NotImplementedError.
                    continue
                self.assertRaise(lambda me=me: getattr(builder, me)(""), NotImplementedError)
            if me.startswith("set_"):
                self.assertRaise(
                    lambda me=me: getattr(builder, me)("", None), NotImplementedError
                )

    def test_basic_shape_builder(self):
        b = BasicShapeBuilder()
        msg = b.get_debug_msg()
        self.assertIn("--SHAPE--", msg)

    def test_check_shape(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [
                    _mkv_("X", TFLOAT, ["D32", "D128"]),
                    _mkv_("Y", TFLOAT, ["batch", "channel", "D128", "D64"]),
                ],
                [_mkv_("Z", TFLOAT, ["batch", "channel", "D32", "64"])],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 32, 128], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 128, 64], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"),
                ],
            )
        )
        onnx.shape_inference.infer_shapes(model)
        builder = BasicShapeBuilder()
        builder.run_model(model)
        self.assertEqual(builder._input_names, ["X", "Y"])
        self.assertEqual(
            builder._known_ranks,
            {
                "zero": 1,
                "un": 1,
                "shape1": 1,
                "shape2": 1,
                "shape3": 1,
                "X": 2,
                "Y": 4,
                "xu1": 3,
                "xu2": 4,
                "xm1": 3,
                "xm2c": 3,
                "xm2": 3,
                "xm": 3,
                "Z": 4,
            },
        )
        self.assertEqual(
            builder._known_shapes,
            {
                "zero": (1,),
                "un": (1,),
                "shape1": (3,),
                "shape2": (3,),
                "shape3": (4,),
                "X": ("D32", "D128"),
                "Y": ("batch", "channel", "D128", "D64"),
                "xu1": (1, "D32", "D128"),
                "xu2": (1, 1, "D32", "D128"),
                "xm1": (1, 32, 128),
                "xm2c": (15, 128, 64),
                "xm2": (15, 128, 64),
                "xm": (15, 32, 64),
                "Z": (3, 5, 32, 64),
            },
        )
        self.assertEqual(
            builder._known_types,
            {
                "zero": 7,
                "un": 7,
                "shape1": 7,
                "shape2": 7,
                "shape3": 7,
                "X": 1,
                "Y": 1,
                "xu1": 1,
                "xu2": 1,
                "xm1": 1,
                "xm2c": 1,
                "xm2": 1,
                "xm": 1,
                "Z": 1,
            },
        )
        self.assertEqualAny(
            builder.constants_computed_,
            {
                "shape1": np.array([1, 32, 128], dtype=np.int64),
                "shape2": np.array([15, 128, 64], dtype=np.int64),
                "shape3": np.array([3, 5, 32, 64], dtype=np.int64),
                "un": np.array([1], dtype=np.int64),
                "zero": np.array([0], dtype=np.int64),
            },
        )
        self.assertEqual(builder.constraints_, {})
        self.assertEqual(
            builder.dynamic_dimensions_,
            {
                "D128": {"D128"},
                "D32": {"D32"},
                "D64": {"D64"},
                "batch": {"batch"},
                "channel": {"channel"},
            },
        )
        self.assertEqual(builder._known_value_shape, {"zero": (0,), "un": (1,)})
        self.assertEqual(builder._output_names, ["Z"])

    def test_reshape_reshape(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "shape1"], ["xr"]),
                    oh.make_node("Reshape", ["xr", "shape2"], ["xrr"]),
                    oh.make_node("Add", ["xrr", "one"], ["Y"]),
                ],
                "dummy",
                [_mkv_("X", TFLOAT, ["a", "b", "c"])],
                [_mkv_("Y", TFLOAT, ["a", "b", "c"])],
                [
                    onh.from_array(np.array([0, 0, 2, -1], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([0, 0, -1], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([1], dtype=np.float32), name="one"),
                ],
            )
        )
        onnx.shape_inference.infer_shapes(model)
        builder = BasicShapeBuilder()
        builder.run_model(model)
        self.assertEqual(builder._input_names, ["X"])
        self.assertEqual(
            builder._known_ranks,
            {"X": 3, "Y": 3, "one": 1, "shape1": 1, "shape2": 1, "xr": 4, "xrr": 3},
        )
        self.assertEqual(
            builder._known_shapes,
            {
                "shape1": (4,),
                "shape2": (3,),
                "one": (1,),
                "X": ("a", "b", "c"),
                "xr": ("a", "b", 2, "c//2"),
                "xrr": ("a", "b", "c"),
                "Y": ("a", "b", "c"),
            },
        )
        self.assertEqual(
            builder._known_types,
            {"shape1": 7, "shape2": 7, "one": 1, "X": 1, "xr": 1, "xrr": 1, "Y": 1},
        )
        self.assertEqualAny(
            builder.constants_computed_,
            {"shape1": np.array([0, 0, 2, -1]), "shape2": np.array([0, 0, -1])},
        )
        self.assertEqual(builder.constraints_, {})
        self.assertEqual(builder.dynamic_dimensions_, {"a": {"a"}, "b": {"b"}, "c": {"c"}})
        self.assertEqual(builder._known_value_shape, {})
        self.assertEqual(builder._output_names, ["Y"])

    def test_value_as_shape(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["ids_weight"], ["shape"], start=0, end=2),
                    oh.make_node("Concat", ["shape", "init328"], ["new_shape"], axis=0),
                    oh.make_node("MatMul", ["ids_weight", "A"], ["A1"]),
                    oh.make_node("MatMul", ["ids_weight", "B"], ["B1"]),
                    oh.make_node("MatMul", ["ids_weight", "C"], ["C1"]),
                    oh.make_node("Reshape", ["A1", "new_shape"], ["Areshaped"]),
                    oh.make_node("Reshape", ["B1", "new_shape"], ["Breshaped"]),
                    oh.make_node("Reshape", ["C1", "new_shape"], ["Creshaped"]),
                    oh.make_node("Transpose", ["Areshaped"], ["At"], perm=[0, 2, 1, 3]),
                    oh.make_node("Transpose", ["Breshaped"], ["Bt"], perm=[0, 2, 1, 3]),
                    oh.make_node("Transpose", ["Creshaped"], ["Ct"], perm=[0, 2, 1, 3]),
                ],
                "dummy",
                [_mkv_("ids_weight", TFLOAT, ["batch", "seq", 256])],
                [
                    _mkv_("At", TFLOAT, ["batch", 32, "seq", 8]),
                    _mkv_("Bt", TFLOAT, ["batch", 32, "seq", 8]),
                    _mkv_("Ct", TFLOAT, ["batch", 32, "seq", 8]),
                ],
                [
                    onh.from_array(np.array([32, 8], dtype=np.int64), name="init328"),
                    onh.from_array(np.random.randn(256, 256).astype(np.float32), name="A"),
                    onh.from_array(np.random.randn(256, 256).astype(np.float32), name="B"),
                    onh.from_array(np.random.randn(256, 256).astype(np.float32), name="C"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        onnx.shape_inference.infer_shapes(model)
        builder = BasicShapeBuilder()
        builder.run_model(model)
        self.assertEqual(
            builder._known_value_shape,
            {"init328": (32, 8), "new_shape": ("batch", "seq", 32, 8), "shape": ("batch", "seq")},
        )
        self.assertEqual(builder._input_names, ["ids_weight"])
        self.assertEqual(
            builder._known_ranks,
            {
                "A": 2,
                "A1": 3,
                "Areshaped": 4,
                "At": 4,
                "B": 2,
                "B1": 3,
                "Breshaped": 4,
                "Bt": 4,
                "C": 2,
                "C1": 3,
                "Creshaped": 4,
                "Ct": 4,
                "ids_weight": 3,
                "init328": 1,
                "new_shape": 1,
                "shape": 1,
            },
        )
        self.assertEqual(
            builder._known_types,
            {
                "init328": 7,
                "A": 1,
                "B": 1,
                "C": 1,
                "ids_weight": 1,
                "A1": 1,
                "B1": 1,
                "C1": 1,
                "Areshaped": 1,
                "Breshaped": 1,
                "Creshaped": 1,
                "At": 1,
                "Bt": 1,
                "Ct": 1,
                "new_shape": 7,
                "shape": 7,
            },
        )
        self.assertEqual(
            builder._known_shapes,
            {
                "init328": (2,),
                "A": (256, 256),
                "B": (256, 256),
                "C": (256, 256),
                "ids_weight": ("batch", "seq", 256),
                "shape": (2,),
                "new_shape": (4,),
                "A1": ("batch", "seq", 256),
                "B1": ("batch", "seq", 256),
                "C1": ("batch", "seq", 256),
                "Areshaped": ("batch", "seq", 32, 8),
                "Breshaped": ("batch", "seq", 32, 8),
                "Creshaped": ("batch", "seq", 32, 8),
                "At": ("batch", 32, "seq", 8),
                "Bt": ("batch", 32, "seq", 8),
                "Ct": ("batch", 32, "seq", 8),
            },
        )
        self.assertEqualAny(
            builder.constants_computed_, {"init328": np.array([32, 8], dtype=np.int64)}
        )
        self.assertEqual(builder.constraints_, {})
        self.assertEqual(builder.dynamic_dimensions_, {"batch": {"batch"}, "seq": {"seq"}})
        self.assertEqual(builder._output_names, ["At", "Bt", "Ct"])

    def test_evaluate_shape(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Concat", ["X", "Y"], ["Z"], axis=1)],
                "dummy",
                [_mkv_("Y", TFLOAT, ["batch", "seq1"]), _mkv_("X", TFLOAT, ["batch", "seq2"])],
                [_mkv_("Z", TFLOAT, [None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        builder = BasicShapeBuilder()
        builder.run_model(model)
        self.assertEqual(
            builder._known_shapes,
            {"Y": ("batch", "seq1"), "X": ("batch", "seq2"), "Z": ("batch", "seq1+seq2")},
        )
        feeds = dict(
            X=np.random.rand(3, 5).astype(np.float32), Y=np.random.rand(3, 6).astype(np.float32)
        )
        got = ExtendedReferenceEvaluator(model).run(None, feeds)
        res = builder.compare_with_true_inputs(feeds, got)
        self.assertEqual(res, {"Z": (("batch", 3, 3), ("seq1+seq2", 11, 11))})

    def test_concat_split(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Concat", ["X", "Y"], ["xy"], axis=1),
                    oh.make_node("Split", ["xy"], ["S1", "S2"], axis=1, num_outputs=2),
                    oh.make_node("Concat", ["S2", "S1"], ["zs"], axis=1),
                    oh.make_node("Relu", ["zs"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, ["a", "b"]),
                    oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, ["a", "c"]),
                ],
                [oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, ["a", "e"])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        builder = BasicShapeBuilder()
        builder.run_model(model)
        builder.update_shapes(model)
        res = []
        for info in model.graph.value_info:
            t = info.type.tensor_type
            shape = tuple(d.dim_param or d.dim_value for d in t.shape.dim)
            res.append((t.elem_type, shape))
        expected = [
            (1, ("a", "b+c")),
            (1, ("a", "CeilToInt(b+c,2)")),
            (1, ("a", "b+c-CeilToInt(b+c,2)")),
            (1, ("a", "b+c")),
        ]
        self.assertEqual(expected, res)
        values = {
            name: builder.evaluate_shape(name, dict(a=3, b=4, c=6))
            for name in ["xy", "S1", "S2", "zs"]
        }
        self.assertEqual(values, {"S1": (3, 5), "S2": (3, 5), "xy": (3, 10), "zs": (3, 10)})

    def test_add_concat_reshape_computed_shapes(self):
        """Symbolic shapes through Add + Concat + Reshape; compare with onnx inference."""
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "Y"], ["added"]),
                    oh.make_node("Concat", ["added", "X"], ["concat_out"], axis=2),
                    oh.make_node("Reshape", ["concat_out", "reshape_shape"], ["Z"]),
                ],
                "add_concat_reshape",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq", "d_model"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["batch", "seq", "d_model"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [None, None, None])],
                [onh.from_array(np.array([0, 0, -1], dtype=np.int64), name="reshape_shape")],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )

        # onnx shape inference loses the symbolic link for Concat and Reshape outputs
        inferred = onnx.shape_inference.infer_shapes(model)
        onnx_shapes = {}
        for vi in list(inferred.graph.value_info) + list(inferred.graph.output):
            t = vi.type.tensor_type
            if t.HasField("shape"):
                onnx_shapes[vi.name] = tuple(
                    d.dim_param if d.dim_param else (d.dim_value if d.dim_value else None)
                    for d in t.shape.dim
                )
        # concat_out and Z get new unknown symbols instead of "2*d_model"
        self.assertTrue(
            onnx_shapes.get("concat_out", (None,))[-1] != "2*d_model",
            "onnx infer_shapes should not produce '2*d_model' for concat_out",
        )

        # BasicShapeBuilder tracks symbolic expressions
        builder = BasicShapeBuilder()
        builder.run_model(model)
        self.assertEqual(builder._known_shapes["added"], ("batch", "seq", "d_model"))
        self.assertEqual(builder._known_shapes["concat_out"], ("batch", "seq", "2*d_model"))
        self.assertEqual(builder._known_shapes["Z"], ("batch", "seq", "2*d_model"))

        # Evaluate symbolic shapes with concrete values
        context = dict(batch=2, seq=5, d_model=8)
        self.assertEqual(builder.evaluate_shape("concat_out", context), (2, 5, 16))
        self.assertEqual(builder.evaluate_shape("Z", context), (2, 5, 16))

    def _make_node_with_attrs(self, **attrs):
        node = oh.make_node("SomeOp", ["X"], ["Y"])
        for name, value in attrs.items():
            node.attribute.append(oh.make_attribute(name, value))
        return node

    def test_get_attribute_with_default_int(self):
        b = BasicShapeBuilder()
        node = self._make_node_with_attrs(axis=2)
        self.assertEqual(b.get_attribute_with_default(node, "axis", 0), 2)

    def test_get_attribute_with_default_ints(self):
        b = BasicShapeBuilder()
        node = self._make_node_with_attrs(perm=[0, 2, 1])
        self.assertEqual(b.get_attribute_with_default(node, "perm", []), [0, 2, 1])

    def test_get_attribute_with_default_float(self):
        b = BasicShapeBuilder()
        node = self._make_node_with_attrs(alpha=0.5)
        self.assertAlmostEqual(b.get_attribute_with_default(node, "alpha", 1.0), 0.5)

    def test_get_attribute_with_default_floats(self):
        b = BasicShapeBuilder()
        node = self._make_node_with_attrs(scales=[1.0, 2.0, 3.0])
        self.assertEqual(b.get_attribute_with_default(node, "scales", []), [1.0, 2.0, 3.0])

    def test_get_attribute_with_default_string(self):
        b = BasicShapeBuilder()
        node = self._make_node_with_attrs(mode=b"constant")
        self.assertEqual(b.get_attribute_with_default(node, "mode", b""), b"constant")

    def test_get_attribute_with_default_strings(self):
        b = BasicShapeBuilder()
        node = self._make_node_with_attrs(keys=[b"hello", b"world"])
        self.assertEqual(b.get_attribute_with_default(node, "keys", []), [b"hello", b"world"])

    def test_get_attribute_with_default_missing(self):
        b = BasicShapeBuilder()
        node = self._make_node_with_attrs(axis=1)
        self.assertEqual(b.get_attribute_with_default(node, "missing", 42), 42)

    def test_get_attribute_with_default_unsupported_type(self):
        b = BasicShapeBuilder()
        node = oh.make_node("SomeOp", ["X"], ["Y"])
        import onnx.numpy_helper as onh
        import numpy as np

        tensor = onh.from_array(np.array([1.0], dtype=np.float32))
        att = onnx.AttributeProto()
        att.name = "value"
        att.type = onnx.AttributeProto.TENSOR
        att.t.CopyFrom(tensor)
        node.attribute.append(att)
        self.assertRaise(lambda: b.get_attribute_with_default(node, "value", None), TypeError)

    def test_get_attributes_with_default_int(self):
        b = BasicShapeBuilder()
        node = self._make_node_with_attrs(axis=3)
        self.assertEqual(b.get_attributes_with_default(node, axis=0), {"axis": 3})

    def test_get_attributes_with_default_ints(self):
        b = BasicShapeBuilder()
        node = self._make_node_with_attrs(perm=[1, 0, 2])
        self.assertEqual(b.get_attributes_with_default(node, perm=[]), {"perm": [1, 0, 2]})

    def test_get_attributes_with_default_float(self):
        b = BasicShapeBuilder()
        node = self._make_node_with_attrs(alpha=0.25)
        result = b.get_attributes_with_default(node, alpha=1.0)
        self.assertAlmostEqual(result["alpha"], 0.25)

    def test_get_attributes_with_default_floats(self):
        b = BasicShapeBuilder()
        node = self._make_node_with_attrs(scales=[1.5, 2.5])
        self.assertEqual(b.get_attributes_with_default(node, scales=[]), {"scales": [1.5, 2.5]})

    def test_get_attributes_with_default_string(self):
        b = BasicShapeBuilder()
        node = self._make_node_with_attrs(mode=b"nearest")
        self.assertEqual(b.get_attributes_with_default(node, mode=b""), {"mode": b"nearest"})

    def test_get_attributes_with_default_strings(self):
        b = BasicShapeBuilder()
        node = self._make_node_with_attrs(keys=[b"a", b"b", b"c"])
        self.assertEqual(
            b.get_attributes_with_default(node, keys=[]), {"keys": [b"a", b"b", b"c"]}
        )

    def test_get_attributes_with_default_uses_default(self):
        b = BasicShapeBuilder()
        node = oh.make_node("SomeOp", ["X"], ["Y"])
        self.assertEqual(b.get_attributes_with_default(node, axis=5), {"axis": 5})

    def test_get_attributes_with_default_none_default_excluded(self):
        b = BasicShapeBuilder()
        node = oh.make_node("SomeOp", ["X"], ["Y"])
        self.assertEqual(b.get_attributes_with_default(node, axis=None), {})

    def test_get_attributes_with_default_unsupported_type(self):
        b = BasicShapeBuilder()
        node = oh.make_node("SomeOp", ["X"], ["Y"])
        import onnx.numpy_helper as onh
        import numpy as np

        tensor = onh.from_array(np.array([1.0], dtype=np.float32))
        att = onnx.AttributeProto()
        att.name = "value"
        att.type = onnx.AttributeProto.TENSOR
        att.t.CopyFrom(tensor)
        node.attribute.append(att)
        self.assertRaise(lambda: b.get_attributes_with_default(node, value=None), TypeError)

    def test_pretty_node_none(self):
        b = BasicShapeBuilder()
        self.assertEqual(b.pretty_node(None), "None")

    def test_pretty_node_simple(self):
        b = BasicShapeBuilder()
        node = oh.make_node("Add", ["X", "Y"], ["Z"])
        result = b.pretty_node(node)
        self.assertEqual(result, "Add: X, Y -> Z")

    def test_pretty_node_with_domain(self):
        b = BasicShapeBuilder()
        node = oh.make_node("CustomOp", ["X"], ["Y"], domain="custom.domain")
        result = b.pretty_node(node)
        self.assertEqual(result, "CustomOp[custom.domain]: X -> Y")

    def test_pretty_node_with_shape(self):
        b = BasicShapeBuilder()
        node = oh.make_node("Add", ["X", "Y"], ["Z"])
        b.set_type("X", TFLOAT)
        b.set_shape("X", (2, 3))
        b.set_type("Y", TFLOAT)
        b.set_shape("Y", (2, 3))
        b.set_type("Z", TFLOAT)
        b.set_shape("Z", (2, 3))
        result = b.pretty_node(node, shape=True)
        self.assertIn("X:1|2x3", result)
        self.assertIn("Y:1|2x3", result)
        self.assertIn("Z:1|2x3", result)
        self.assertIn("->", result)

    def test_pretty_node_with_shape_missing_info(self):
        b = BasicShapeBuilder()
        node = oh.make_node("Relu", ["X"], ["Y"])
        result = b.pretty_node(node, shape=True)
        self.assertIn("X:-|?", result)
        self.assertIn("Y:-|?", result)

    def test_pretty_node_short_false(self):
        b = BasicShapeBuilder()
        node = oh.make_node("Relu", ["X"], ["Y"])
        b.set_type("Y", TFLOAT)
        b.set_shape("Y", (4, 5))
        result = b.pretty_node(node, short=False)
        self.assertIn("Relu: X -> Y", result)
        self.assertIn("T1", result)
        self.assertIn("4 x 5", result)

    def test_pretty_node_short_false_with_name(self):
        b = BasicShapeBuilder()
        node = oh.make_node("Relu", ["X"], ["Y"], name="relu_node")
        result = b.pretty_node(node, short=False)
        self.assertIn("relu_node", result)

    def test_pretty_node_shape_op(self):
        b = BasicShapeBuilder()
        node = oh.make_node("Shape", ["X"], ["shape_out"])
        result = b.pretty_node(node)
        self.assertIn("Shape", result)
        self.assertIn("X", result)
        self.assertIn("shape_out", result)

    def test_pretty_node_shape_op_with_attributes(self):
        b = BasicShapeBuilder()
        node = oh.make_node("Shape", ["X"], ["shape_out"], start=1, end=3)
        result = b.pretty_node(node)
        self.assertIn("Shape", result)
        self.assertIn("X", result)
        self.assertIn("shape_out", result)

    def test_pretty_node_reshape_op(self):
        b = BasicShapeBuilder()
        node = oh.make_node("Reshape", ["data", "shape"], ["reshaped"])
        result = b.pretty_node(node)
        self.assertIn("Reshape", result)
        self.assertIn("data", result)
        self.assertIn("reshaped", result)

    def test_pretty_node_unsqueeze_op(self):
        b = BasicShapeBuilder()
        node = oh.make_node("Unsqueeze", ["X", "axes"], ["Y"])
        result = b.pretty_node(node)
        self.assertIn("Unsqueeze", result)
        self.assertIn("X", result)
        self.assertIn("Y", result)

    def test_pretty_node_squeeze_op(self):
        b = BasicShapeBuilder()
        node = oh.make_node("Squeeze", ["X"], ["Y"])
        result = b.pretty_node(node)
        self.assertIn("Squeeze", result)
        self.assertIn("X", result)
        self.assertIn("Y", result)

    def test_pretty_node_cast_op(self):
        b = BasicShapeBuilder()
        node = oh.make_node("Cast", ["X"], ["Y"], to=TFLOAT)
        result = b.pretty_node(node)
        self.assertIn("Cast", result)
        self.assertIn("X", result)
        self.assertIn("Y", result)

    def test_pretty_node_transpose_op(self):
        b = BasicShapeBuilder()
        node = oh.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 1])
        result = b.pretty_node(node)
        self.assertIn("Transpose", result)
        self.assertIn("X", result)
        self.assertIn("Y", result)

    def test_pretty_node_shape_op_with_shape_info(self):
        b = BasicShapeBuilder()
        node = oh.make_node("Shape", ["X"], ["shape_out"])
        b.set_type("X", TFLOAT)
        b.set_shape("X", (3, 4))
        b.set_type("shape_out", TINT64)
        b.set_shape("shape_out", (2,))
        result = b.pretty_node(node, shape=True)
        self.assertIn("X:1|3x4", result)
        self.assertIn("shape_out:7|2", result)
        self.assertIn("->", result)

    def test_pretty_node_cast_op_with_shape_info(self):
        b = BasicShapeBuilder()
        node = oh.make_node("Cast", ["X"], ["Y"], to=TFLOAT16)
        b.set_type("X", TFLOAT)
        b.set_shape("X", (2, 3))
        b.set_type("Y", TFLOAT16)
        b.set_shape("Y", (2, 3))
        result = b.pretty_node(node, shape=True)
        self.assertIn("X:1|2x3", result)
        self.assertIn("->", result)

    def test_pretty_node_transpose_op_with_shape_info(self):
        b = BasicShapeBuilder()
        node = oh.make_node("Transpose", ["X"], ["Y"], perm=[1, 0])
        b.set_type("X", TFLOAT)
        b.set_shape("X", (2, 3))
        b.set_type("Y", TFLOAT)
        b.set_shape("Y", (3, 2))
        result = b.pretty_node(node, shape=True)
        self.assertIn("X:1|2x3", result)
        self.assertIn("Y:1|3x2", result)
        self.assertIn("->", result)

    def test_pretty_node_reshape_op_short_false(self):
        b = BasicShapeBuilder()
        node = oh.make_node("Reshape", ["data", "shape"], ["reshaped"])
        b.set_type("reshaped", TFLOAT)
        b.set_shape("reshaped", (6,))
        result = b.pretty_node(node, short=False)
        self.assertIn("Reshape", result)
        self.assertIn("T1", result)
        self.assertIn("6", result)

    def test_pretty_node_unsqueeze_op_short_false(self):
        b = BasicShapeBuilder()
        node = oh.make_node("Unsqueeze", ["X", "axes"], ["Y"])
        b.set_type("Y", TFLOAT)
        b.set_shape("Y", (1, 4))
        result = b.pretty_node(node, short=False)
        self.assertIn("Unsqueeze", result)
        self.assertIn("T1", result)
        self.assertIn("1 x 4", result)

    def test_pretty_node_squeeze_op_short_false(self):
        b = BasicShapeBuilder()
        node = oh.make_node("Squeeze", ["X"], ["Y"])
        b.set_type("Y", TFLOAT)
        b.set_shape("Y", (4,))
        result = b.pretty_node(node, short=False)
        self.assertIn("Squeeze", result)
        self.assertIn("T1", result)
        self.assertIn("4", result)

    def test_get_registered_constraints(self):
        b = BasicShapeBuilder()
        self.assertEqual(b.get_registered_constraints(), {})
        b.register_constraint_dimension("batch", "s0")
        b.register_constraint_dimension("s0", "batch")
        constraints = b.get_registered_constraints()
        self.assertIn("batch", constraints)
        self.assertIn("s0", constraints)
        self.assertIn("s0", constraints["batch"])
        self.assertIn("batch", constraints["s0"])

    def test_get_shape_renamed_without_renaming(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", ("s0", "s1"))
        # Before any renaming, get_shape_renamed falls back to get_shape.
        self.assertEqual(b.get_shape_renamed("X"), ("s0", "s1"))

    def test_improves_dynamic_dimension_naming_basic(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", ("s0", "s1"))
        b.set_type("Y", TFLOAT)
        b.set_shape("Y", ("s0", "s1"))
        # Register constraints linking internal names to user-visible names.
        b.register_constraint_dimension("batch", "s0")
        b.register_constraint_dimension("s0", "batch")
        b.register_constraint_dimension("seq_length", "s1")
        b.register_constraint_dimension("s1", "seq_length")
        replacements = b._improves_dynamic_dimension_naming({"batch", "seq_length"})
        self.assertIn("s0", replacements)
        self.assertEqual(replacements["s0"], "batch")
        self.assertIn("s1", replacements)
        self.assertEqual(replacements["s1"], "seq_length")
        self.assertEqual(b.get_shape_renamed("X"), ("batch", "seq_length"))
        self.assertEqual(b.get_shape_renamed("Y"), ("batch", "seq_length"))

    def test_improves_dynamic_dimension_naming_no_constraints(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", ("s0", "s1"))
        # With no constraints linking s0/s1 to the original names, no renaming happens
        # for s0/s1 even though identity mappings are returned for the original names.
        replacements = b._improves_dynamic_dimension_naming({"batch", "seq_length"})
        # s0 and s1 are not constrained to batch/seq_length, so they stay unchanged.
        self.assertNotIn("s0", replacements)
        self.assertNotIn("s1", replacements)
        # Falls back to the original shape.
        self.assertEqual(b.get_shape_renamed("X"), ("s0", "s1"))

    def test_improves_dynamic_dimension_naming_from_model(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Add", ["X", "Y"], ["Z"])],
                "g",
                [_mkv_("X", TFLOAT, ["batch", "seq"]), _mkv_("Y", TFLOAT, ["batch", "seq"])],
                [_mkv_("Z", TFLOAT, ["batch", "seq"])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
        )
        b = BasicShapeBuilder()
        b.run_model(model)
        # The model uses "batch" and "seq" as symbolic names from the start.
        # No additional constraints are needed.
        self.assertEqual(b.get_shape("X"), ("batch", "seq"))
        # Since these names are already the "originals", renaming is identity.
        b._improves_dynamic_dimension_naming({"batch", "seq"})
        self.assertEqual(b.get_shape_renamed("Z"), ("batch", "seq"))

    def test_improves_dynamic_dimension_naming_partial(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", ("s0", 128))
        # Only the first dimension has a preferred name.
        b.register_constraint_dimension("batch", "s0")
        b.register_constraint_dimension("s0", "batch")
        replacements = b._improves_dynamic_dimension_naming({"batch"})
        self.assertIn("s0", replacements)
        self.assertEqual(replacements["s0"], "batch")
        # Static dimension 128 is unchanged.
        self.assertEqual(b.get_shape_renamed("X"), ("batch", 128))

    def test_improves_dynamic_dimension_naming_idempotent(self):
        b = BasicShapeBuilder()
        b.set_type("X", TFLOAT)
        b.set_shape("X", ("s0",))
        b.register_constraint_dimension("batch", "s0")
        b.register_constraint_dimension("s0", "batch")
        b._improves_dynamic_dimension_naming({"batch"})
        # Calling again should not raise and should return consistent results.
        b._improves_dynamic_dimension_naming({"batch"})
        self.assertEqual(b.get_shape_renamed("X"), ("batch",))

    def test_run_value_info_nonzero_registers_constraint_with_named_output(self):
        # NonZero introduces an internal dimension name (NEWDIM_nonzero_0).
        # When the graph output is declared with a user-visible symbolic name,
        # run_value_info should register a constraint linking the two names and
        # rename the internal placeholder throughout.
        model_named = oh.make_model(
            oh.make_graph(
                [oh.make_node("NonZero", ["X"], ["nz"])],
                "nonzero_named",
                [_mkv_("X", TFLOAT, ["batch", "seq"])],
                [_mkv_("nz", TINT64, ["rank", "nnz"])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        builder = BasicShapeBuilder()
        builder.run_model(model_named)

        # The internal placeholder should have been renamed to the user name.
        self.assertEqual(builder.get_shape("nz"), (2, "nnz"))
        # A constraint linking the internal name to the user name must be registered.
        constraints = builder.get_registered_constraints()
        self.assertIn("NEWDIM_nonzero_0", constraints)
        self.assertIn("nnz", constraints["NEWDIM_nonzero_0"])

    def test_run_value_info_nonzero_anonymous_output_no_constraint(self):
        # Without named output dimensions, no constraint should be registered
        # and the internal placeholder is kept as-is.
        model_anon = oh.make_model(
            oh.make_graph(
                [oh.make_node("NonZero", ["X"], ["nz"])],
                "nonzero_anon",
                [_mkv_("X", TFLOAT, ["batch", "seq"])],
                [_mkv_("nz", TINT64, [None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        builder = BasicShapeBuilder()
        builder.run_model(model_anon)

        shape = builder.get_shape("nz")
        self.assertEqual(shape[0], 2)
        self.assertIsInstance(shape[1], str)
        self.assertIn("NEWDIM_nonzero", shape[1])
        self.assertEqual(builder.get_registered_constraints(), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)

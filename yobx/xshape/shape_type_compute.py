from typing import List, Optional, Sequence, Tuple
import numpy as np
from onnx import NodeProto, TensorProto
from ._shape_helper import (
    DYNAMIC_SHAPE,
    is_static_shape,
    all_int,
    all_int_or_str,
)
from .simplify_expressions import simplify_expression
from .shape_builder import ShapeBuilder


def broadcast_shape(
    sh1: DYNAMIC_SHAPE,
    sh2: DYNAMIC_SHAPE,
    graph_builder: Optional[ShapeBuilder] = None,
) -> DYNAMIC_SHAPE:
    """
    Computes the shape for many broadcasting operators.
    This function should be used while converting the graph into ONNX
    because it assumes the broadcast is possible and adds the necessary constraints
    on the dynamic in the GraphBuilder shapes to make it work.

    :param sh1: first shape
    :param sh2: second shape
    :param graph_builder: if not None, the function register
        any constraint which might appear while applying the broadcast
    :return: resulting shape
    """
    if sh1 == sh2:
        return sh1
    if len(sh1) == 0:
        return sh2
    if len(sh2) == 0:
        return sh1
    if sh1 == (1,) and len(sh2) >= 1:
        return sh2
    if sh2 == (1,) and len(sh1) >= 1:
        return sh1
    if len(sh1) < len(sh2):
        sh1 = (1,) * (len(sh2) - len(sh1)) + sh1
    elif len(sh1) > len(sh2):
        sh2 = (1,) * (len(sh1) - len(sh2)) + sh2
    new_shape = []
    for a, b in zip(sh1, sh2):
        if isinstance(a, int):
            if a == 0:
                d = 0
            elif isinstance(b, int):
                d = max(a, b) if b > 0 else 0
            elif a == 1:
                d = b  # type: ignore[assignment]
            else:
                # We have two indications, let's take the most strict one.
                d = a
                if graph_builder:
                    # pyrefly: ignore [bad-argument-type]
                    graph_builder.register_constraint_dimension(b, a)  # type: ignore[arg-type]
        elif isinstance(b, int):
            # a is str
            if b == 0:
                d = 0
            elif b == 1:
                d = a  # type: ignore[assignment]
            elif b != 1:
                # a is not int, it is str
                d = b
                if graph_builder:
                    # pyrefly: ignore [bad-argument-type]
                    graph_builder.register_constraint_dimension(a, b)  # type: ignore[arg-type]
        else:
            # both str
            if a == b:
                d = a  # type: ignore[assignment]
            else:
                d = simplify_expression(f"({a})^({b})")  # type: ignore[assignment]
        # pyrefly: ignore [unbound-name]
        if d is None:
            raise RuntimeError(
                f"Not implemented for sh1={sh1}, sh2={sh2}, a={a}, b={b}, "
                f"type(a)={type(a)}, type(b)={type(b)}, a={a}, b={b}"
            )
        # pyrefly: ignore [unbound-name]
        new_shape.append(d)
    return tuple(new_shape)


def _set_shape_type_op_any_attention(g: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Attention."
    # pyrefly: ignore [missing-attribute]
    if g.has_type(node.input[0]):  # type: ignore[attr-defined]
        itype = g.get_type(node.input[0])
        for i, o in enumerate(node.output):
            if i != 2 and o:
                g.set_type(o, itype)
        # pyrefly: ignore [missing-attribute]
        if len(node.output) > 2 and node.output[2] and g.has_type(node.input[2]):  # type: ignore[attr-defined]
            g.set_type(node.output[2], g.get_type(node.input[2]))
    # pyrefly: ignore [missing-attribute]
    if g.has_shape(node.input[0]) and g.has_shape(node.input[1]) and g.has_shape(node.input[2]):  # type: ignore[attr-defined]
        rk = g.get_rank(node.input[0])
        if rk == 4:
            batch, q_head, seq, _size = g.get_shape(node.input[0])
            _batch, k_head, _seq, k_size = g.get_shape(node.input[1])
            _batch, v_head, _seq, v_size = g.get_shape(node.input[2])
            g.set_shape(node.output[0], (batch, q_head, seq, v_size))
        else:
            batch, seq, q_hidden_size = g.get_shape(node.input[0])
            _batch, _seq, k_hidden_size = g.get_shape(node.input[1])
            _batch, _seq, v_hidden_size = g.get_shape(node.input[2])
            q_head = g.get_attribute_with_default(node, "q_num_heads", 1)
            k_head = v_head = g.get_attribute_with_default(node, "kv_num_heads", 1)
            _q_size = q_hidden_size // q_head
            k_size = k_hidden_size // k_head
            v_size = v_hidden_size // v_head
            g.set_shape(node.output[0], (batch, seq, q_head * v_size))
        if len(node.output) > 1 and node.output[1]:
            batch, _v_head, past_seq, _size = (
                g.get_shape(node.input[4])
                if len(node.input) > 4 and node.input[4]
                else (batch, 0, 0, 0)
            )
            total_seq = (
                f"{seq}+{past_seq}"
                if isinstance(seq, str) or isinstance(past_seq, str)
                else (seq + past_seq)
            )
            g.set_shape(node.output[1], (batch, k_head, total_seq, k_size))
        if len(node.output) > 2 and node.output[2]:
            batch, _v_head, past_seq, _size = (
                g.get_shape(node.input[5])
                if len(node.input) > 5 and node.input[5]
                else (batch, 0, 0, 0)
            )
            total_seq = (
                f"{seq}+{past_seq}"
                if isinstance(seq, str) or isinstance(past_seq, str)
                else (seq + past_seq)
            )
            g.set_shape(node.output[2], (batch, v_head, total_seq, v_size))
        if len(node.output) > 3 and node.output[3]:
            g.set_shape(node.output[3], (batch, q_head, seq, total_seq))
    # pyrefly: ignore [missing-attribute]
    if g.has_rank(node.input[0]):  # type: ignore[attr-defined]
        rk = g.get_rank(node.input[0])
        g.set_rank(node.output[0], rk)
        for o in node.output[1:]:
            if o:
                g.set_rank(o, 4)


def set_type_shape_reshape(
    g: ShapeBuilder,
    name: str,
    input_name: str,
    new_shape: Sequence[int],
):
    "Sets the output shape for node type Reshape"
    g.set_type(name, g.get_type(input_name))
    # pyrefly: ignore [missing-attribute]
    if g.has_device(input_name):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        g.set_device(name, g.get_device(input_name))  # type: ignore[attr-defined]
    if isinstance(new_shape, str):
        # pyrefly: ignore [missing-attribute]
        if g.has_shape(new_shape):
            sh = g.get_shape(new_shape)
            assert len(sh) == 1, f"Unexpected value {sh} for shape={new_shape!r}"
            # pyrefly: ignore [bad-argument-type]
            g.set_rank(name, sh[0])
    # pyrefly: ignore [bad-argument-type]
    elif not is_static_shape(new_shape):  # type: ignore[arg-type]
        g.set_rank(name, len(new_shape))
    elif min(new_shape) == -1:
        # pyrefly: ignore [missing-attribute]
        if g.has_shape(input_name):  # type: ignore[attr-defined]
            shape = list(g.get_shape(input_name))
            arg_size = np.prod([a for a in new_shape if a >= 0])
            size = np.prod(shape)
            index = new_shape.index(-1)
            new_shape = list(new_shape)
            if arg_size == 0:
                assert size == 0, f"Unable to reshape {shape} into {new_shape}"
                new_shape[index] = 1
            else:
                new_shape[index] = int(size // arg_size)
            g.set_shape(name, tuple(new_shape))
        else:
            g.set_rank(name, len(new_shape))
    else:
        g.set_shape(name, tuple(new_shape))


def set_type_shape_unary_op(
    g: ShapeBuilder,
    name: str,
    input_name: str,
    itype: Optional[int] = None,
) -> bool:
    """Sets the shape and type for an unary operator (abs, exp, ...)."""
    # pyrefly: ignore [missing-attribute]
    if g.has_device(input_name):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        g.set_device(name, g.get_device(input_name))  # type: ignore[attr-defined]
    # pyrefly: ignore [missing-attribute]
    if not itype and not g.has_type(input_name):  # type: ignore[attr-defined]
        # pyrefly: ignore [bad-return]
        return  # type: ignore[return-value]
    g.set_type(name, itype or g.get_type(input_name))
    # pyrefly: ignore [missing-attribute]
    if g.has_shape(input_name):  # type: ignore[attr-defined]
        # pyrefly: ignore [unexpected-keyword]
        g.set_shape(name, g.get_shape(input_name), allow_zero=True)  # type: ignore[call-arg]
        # pyrefly: ignore [bad-return]
        return g.get_shape(input_name)  # type: ignore[return-value]
    # pyrefly: ignore [missing-attribute]
    if g.has_rank(input_name):  # type: ignore[attr-defined]
        g.set_rank(name, g.get_rank(input_name))
        return True
    # pyrefly: ignore [bad-return]
    return  # type: ignore[return-value]


def set_type_shape_unary_op_abs(
    g: ShapeBuilder,
    name: str,
    input_name: str,
    itype: Optional[int] = None,
) -> bool:
    """Sets the shape and type for an unary operator (abs, exp, ...)."""
    # pyrefly: ignore [missing-attribute]
    if g.has_device(input_name):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        g.set_device(name, g.get_device(input_name))  # type: ignore[attr-defined]
    # pyrefly: ignore [missing-attribute]
    if not itype and not g.has_type(input_name):  # type: ignore[attr-defined]
        # pyrefly: ignore [bad-return]
        return  # type: ignore[return-value]
    if not itype:
        itype = g.get_type(input_name)
    if itype in (TensorProto.COMPLEX64, TensorProto.COMPLEX128):
        if itype == TensorProto.COMPLEX64:
            rtype = TensorProto.FLOAT
        elif itype == TensorProto.COMPLEX128:
            rtype = TensorProto.DOUBLE
        else:
            # pyrefly: ignore [missing-attribute]
            raise AssertionError(f"Unexpected type {itype} for {input_name!r}{g.get_debug_msg()}")  # type: ignore[attr-defined]

        g.set_type(name, rtype)
        # pyrefly: ignore [missing-attribute]
        if g.has_shape(input_name):  # type: ignore[attr-defined]
            shape = g.get_shape(input_name)
            g.set_shape(name, shape)
            # pyrefly: ignore [bad-return]
            return shape  # type: ignore[return-value]
        # pyrefly: ignore [missing-attribute]
        if g.has_rank(input_name):  # type: ignore[attr-defined]
            g.set_rank(name, g.get_rank(input_name))
            return True
        # pyrefly: ignore [bad-return]
        return  # type: ignore[return-value]

    g.set_type(name, itype)
    # pyrefly: ignore [missing-attribute]
    if g.has_shape(input_name):  # type: ignore[attr-defined]
        sh = g.get_shape(input_name)
        # pyrefly: ignore [unexpected-keyword]
        g.set_shape(name, sh, allow_zero=0 in sh)  # type: ignore[call-arg]
        # pyrefly: ignore [bad-return]
        return g.get_shape(input_name)  # type: ignore[return-value]
    # pyrefly: ignore [missing-attribute]
    if g.has_rank(input_name):  # type: ignore[attr-defined]
        g.set_rank(name, g.get_rank(input_name))
        return True
    # pyrefly: ignore [bad-return]
    return  # type: ignore[return-value]


def set_type_shape_binary_op(
    g: ShapeBuilder,
    name: str,
    *input_names: List[str],
    begin: int = 0,
    cmp_op: bool = False,
    itype: Optional[int] = None,
) -> bool:
    """Sets the shape and type for a binary operator (add, mul, ...)."""
    # device
    # pyrefly: ignore [missing-attribute]
    if all(g.has_device(i) for i in input_names):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        devs = {g.get_device(i) for i in input_names}  # type: ignore[attr-defined]
        if len(devs) == 1:
            # pyrefly: ignore [missing-attribute]
            g.set_device(name, devs.pop())  # type: ignore[attr-defined]
    elif len(input_names) == 2:
        # pyrefly: ignore [missing-attribute]
        if g.has_device(input_names[0]) and not g.has_device(input_names[1]):  # type: ignore[attr-defined]
            # pyrefly: ignore [missing-attribute]
            g.set_device(name, g.get_device(input_names[0]))  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        elif not g.has_device(input_names[1]) and g.has_device(input_names[0]):  # type: ignore[attr-defined]
            # pyrefly: ignore [missing-attribute]
            g.set_device(name, g.get_device(input_names[1]))  # type: ignore[attr-defined]
    # type
    dtype = None
    if itype:
        g.set_type(name, itype)
    elif cmp_op:
        # operator comparing values
        g.set_type(name, TensorProto.BOOL)
    else:
        for input_name in input_names[begin:]:
            # pyrefly: ignore [missing-attribute]
            if g.has_type(input_name):  # type: ignore[attr-defined]
                # pyrefly: ignore [bad-argument-type]
                dtype = g.get_type(input_name)  # type: ignore[arg-type]
                break
        # pyrefly: ignore [missing-attribute]
        if not dtype and g.as_function:  # type: ignore[attr-defined]
            # pyrefly: ignore [bad-return]
            return  # type: ignore[return-value]
        # pyrefly: ignore [missing-attribute]
        assert dtype, f"Unable to guess type for {name!r} from {input_names}{g.get_debug_msg()}"  # type: ignore[attr-defined]
        g.set_type(name, dtype)

    # shape
    shape = None
    # pyrefly: ignore [bad-assignment]
    for input_name in input_names:
        if g.has_shape(input_name):  # type: ignore[attr-defined]
            input_shape = g.get_shape(input_name)  # type: ignore[arg-type]
            if None in input_shape:
                shape = None
                break
            shape = (
                input_shape
                if shape is None
                else broadcast_shape(shape, input_shape, graph_builder=g)
            )
        else:
            # one shape is missing
            shape = None
            break

    if shape is not None:
        g.set_shape(name, shape, allow_zero=True)  # type: ignore[call-arg]
        # pyrefly: ignore [bad-return]
        return shape  # type: ignore[return-value]

    # rank otherwise
    rank = None
    for input_name in input_names:
        if g.has_rank(input_name):  # type: ignore[attr-defined]
            rank = g.get_rank(input_name) if rank is None else max(rank, g.get_rank(input_name))  # type: ignore[arg-type]
            continue
        if rank is not None:
            rank = None
        # one shape is missing
        break

    if rank is not None:
        g.set_rank(name, rank)
        return True
    # pyrefly: ignore [bad-return]
    return  # type: ignore[return-value]


def set_type_shape_matmul(g: ShapeBuilder, name: str, x: str, y: str) -> bool:
    "Sets the output shape for node type MatMul."
    # device
    # pyrefly: ignore [missing-attribute]
    if g.has_device(x) and g.has_device(y) and g.get_device(x) == g.get_device(y):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        g.set_device(name, g.get_device(x))  # type: ignore[attr-defined]
    # pyrefly: ignore [missing-attribute]
    if not g.has_type(x):  # type: ignore[attr-defined]
        # pyrefly: ignore [bad-return]
        return  # type: ignore[return-value]
    g.set_type(name, g.get_type(x))
    # pyrefly: ignore [missing-attribute]
    if g.has_shape(x) and g.has_shape(y):  # type: ignore[attr-defined]
        sh1 = g.get_shape(x)
        sh2 = g.get_shape(y)
        if len(sh1) == len(sh2) == 1:
            g.set_shape(name, tuple())
            # pyrefly: ignore [bad-return]
            return  # type: ignore[return-value]
        if len(sh1) >= 2 and len(sh2) >= 2 and len(sh1) != len(sh2):
            if len(sh1) < len(sh2):
                sh1 = (1,) * (len(sh2) - len(sh1)) + sh1
            else:
                sh2 = (1,) * (len(sh1) - len(sh2)) + sh2
        elif len(sh1) == 1:
            sh1 = (1,) * (len(sh2) - len(sh1)) + sh1
        elif len(sh2) == 1:
            sh2 = (1,) * (len(sh1) - len(sh2) - 1) + sh2 + (1,)

        assert len(sh1) == len(sh2), (
            f"not implemented when shapes are {sh1} ({x!r}) and {sh2} ({y!r})"
            # pyrefly: ignore [missing-attribute]
            f"{g.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        new_shape = []
        for a, b in zip(sh1[:-2], sh2[:-2]):
            if all_int((a, b)) or a == b:
                new_shape.append(max(a, b))
                continue
            if a == 1:
                new_shape.append(b)
                continue
            if b == 1:
                new_shape.append(a)
                continue
            # We create a new dimension.
            # pyrefly: ignore [missing-attribute]
            new_shape.append(g.make_dimension_name_if_necessary(a, b, "^"))  # type: ignore[attr-defined]

        new_shape.append(sh1[-2])
        new_shape.append(sh2[-1])
        new_shape = tuple(new_shape)  # type: ignore[assignment]
        g.set_shape(name, new_shape)  # type: ignore[arg-type]
        # pyrefly: ignore [bad-return]
        return new_shape  # type: ignore[return-value]
    # pyrefly: ignore [missing-attribute]
    if g.has_rank(x) and g.has_rank(y):  # type: ignore[attr-defined]
        if g.get_rank(x) == g.get_rank(y) == 1:
            return g.set_shape(name, tuple())
        g.set_rank(name, max(g.get_rank(x), g.get_rank(y)))
        return True
    # pyrefly: ignore [bad-return]
    return  # type: ignore[return-value]


def set_type_shape_gemm(
    g: ShapeBuilder,
    name: str,
    x: str,
    y: str,
    transA: int,
    transB: int,
):
    "Sets the output shape for node type Gemm."
    if transA == 0 and transB == 0:
        return set_type_shape_matmul(g, name, x, y)
    # pyrefly: ignore [missing-attribute]
    if g.has_device(x) and g.has_device(y) and g.get_device(x) == g.get_device(y):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        g.set_device(name, g.get_device(x))  # type: ignore[attr-defined]
    g.set_type(name, g.get_type(x))
    # pyrefly: ignore [missing-attribute]
    if g.has_shape(x) and g.has_shape(y):  # type: ignore[attr-defined]
        sh1 = g.get_shape(x)
        sh2 = g.get_shape(y)
        assert len(sh1) == len(
            sh2
        # pyrefly: ignore [missing-attribute]
        ), f"not implemented when shapes are {sh1} and {sh2}{g.get_debug_msg()}"  # type: ignore[attr-defined]
        new_shape = (sh1[-1] if transA else sh1[-2], sh2[-2] if transB else sh2[-1])
        g.set_shape(name, new_shape)
        return new_shape
    # pyrefly: ignore [missing-attribute]
    elif g.has_rank(x) and g.has_rank(y):  # type: ignore[attr-defined]
        g.set_rank(name, max(g.get_rank(x), g.get_rank(y)))


def set_type_shape_reduce_op(
    g: ShapeBuilder,
    name: str,
    x: str,
    keepdim: int,
    axes: Optional[Tuple[int]] = None,
):
    "Sets the output shape for any Reduce type."
    assert keepdim in {None, 0, 1}, f"keepdim={keepdim!r} must be in {{0, 1}}"
    # pyrefly: ignore [missing-attribute]
    if g.has_device(x):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        g.set_device(name, g.get_device(x))  # type: ignore[attr-defined]
    if keepdim is None:
        keepdim = 1
    # pyrefly: ignore [missing-attribute]
    if g.has_type(x):  # type: ignore[attr-defined]
        g.set_type(name, g.get_type(x))
    if axes is None:
        new_shape = ((1,) * g.get_rank(x)) if keepdim else tuple()
        g.set_shape(name, new_shape)
        return new_shape
    # pyrefly: ignore [missing-attribute]
    elif not g.has_shape(x):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        if g.has_rank(x):  # type: ignore[attr-defined]
            g.set_rank(name, g.get_rank(x) - (1 - int(keepdim)) * len(axes))
            return True
    else:
        shape = list(g.get_shape(x))
        for d in axes:
            assert d < len(shape), (
                f"shape mismatch for a reduce op shape={shape}, "
                # pyrefly: ignore [missing-attribute]
                f"axes={axes}{g.get_debug_msg()}"  # type: ignore[attr-defined]
            )
            shape[d] = 1 if keepdim else None
        shape = tuple(_ for _ in shape if _ is not None)  # type: ignore[assignment]
        g.set_shape(name, shape)  # type: ignore[arg-type]
        return shape


########################################
# Implementation for the main algorithm.
########################################


def _set_shape_type_op_any_batch_normalization(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type BatchNormalization."
    res = []
    res.append(set_type_shape_unary_op(self, node.output[0], node.input[0]))
    if len(node.output) > 1:
        res.append(set_type_shape_unary_op(self, node.output[1], node.input[1]))
    if len(node.output) > 2:
        res.append(set_type_shape_unary_op(self, node.output[2], node.input[2]))
    return None if set(res) == {None} else res


def _set_shape_type_op_any_layer_normalization(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type LayerNormalization."
    res = []
    res.append(set_type_shape_unary_op(self, node.output[0], node.input[0]))
    if len(node.output) > 1:
        res.append(set_type_shape_unary_op(self, node.output[1], node.input[1]))
    if len(node.output) > 2:
        res.append(set_type_shape_unary_op(self, node.output[2], node.input[2]))
    return None if set(res) == {None} else res


def _set_shape_type_op_any_cast(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Cast."
    return set_type_shape_unary_op(
        self,
        node.output[0],
        node.input[0],
        # pyrefly: ignore [missing-attribute]
        itype=self.get_attribute(node, "to").i,  # type: ignore[union-attr]
    )


def _set_shape_type_op_any_rotary_embedding(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Cast."
    return set_type_shape_unary_op(self, node.output[0], node.input[0])


def _set_shape_type_op_any_castlike(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type CastLike."
    return set_type_shape_unary_op(
        self, node.output[0], node.input[0], itype=self.get_type(node.input[1])
    )


def _set_shape_type_op_any_compress(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Compress."
    return set_type_shape_binary_op(
        self, node.output[0], node.input[0], itype=self.get_type(node.input[0])
    )


def _set_shape_type_op_any_concat(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Concat."
    # pyrefly: ignore [missing-attribute]
    if all(self.has_device(i) for i in node.input):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        devs = {self.get_device(i) for i in node.input}  # type: ignore[attr-defined]
        if len(devs) == 1:
            # pyrefly: ignore [missing-attribute]
            self.set_device(node.output[0], devs.pop())  # type: ignore[attr-defined]
    # pyrefly: ignore [missing-attribute]
    if self.has_type(node.input[0]):  # type: ignore[attr-defined]
        self.set_type(node.output[0], self.get_type(node.input[0]))
    # pyrefly: ignore [missing-attribute]
    if all(self.has_shape(s) for s in node.input):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        axis = self.get_attribute(node, "axis").i  # type: ignore[union-attr]
        shapes = [self.get_shape(i) for i in node.input]
        new_shape = list(shapes[0])
        assert shapes and axis < min(len(sh) for sh in shapes), (
            f"axis={axis}, higher than a shape in {shapes}, "
            # pyrefly: ignore [missing-attribute]
            f"node={self.pretty_node(node)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        assert all(axis < len(sh) for sh in shapes), f"Unexpected shape in {shapes}, axis={axis}"
        dims = [sh[axis] for sh in shapes]
        if all_int(dims):
            new_shape[axis] = sum(dims)
        else:
            new_shape[axis] = "+".join(map(str, dims))
        new_shape = tuple(new_shape)  # type: ignore[assignment]
        self.set_shape(node.output[0], new_shape)  # type: ignore[arg-type]
        return new_shape
    # pyrefly: ignore [missing-attribute]
    elif all(map(self.has_rank, node.input)):  # type: ignore[attr-defined]
        ranks = [self.get_rank(i) for i in node.input]
        assert (
            len(set(ranks)) == 1
        # pyrefly: ignore [missing-attribute]
        ), f"Unexpected ranks={ranks} for node {node.op_type!r}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        self.set_rank(node.output[0], ranks[0])
        return True
    else:
        # pyrefly: ignore [missing-attribute]
        assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
            f"Unable to compute shape for node: "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )


def _set_shape_type_op_any_conv_max_pool(self: ShapeBuilder, node: NodeProto):
    """
    Sets the output shape for node types Conv, MaxPool.

    This function defines the following functions::

        conv_f1(d,s,stride) = s - (stride if d % stride == 0 else d % stride) // 2
        conv_f2(d,s,stride) = (
            s - (stride if d % stride == 0 else d % stride)) // 2 + stride % 2
        )
        conv_f3(d,s,stride,ceil_mode,p) = ... (see the code)
    """
    # pyrefly: ignore [missing-attribute]
    if self.has_device(node.input[0]):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        self.set_device(node.output[0], self.get_device(node.input[0]))  # type: ignore[attr-defined]
    # pyrefly: ignore [missing-attribute]
    if not self.has_type(node.input[0]):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
            f"Unable to compute shape for node: "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        return
    self.set_type(node.output[0], self.get_type(node.input[0]))
    if len(node.output) > 1:
        self.set_type(node.output[1], TensorProto.INT64)

    # pyrefly: ignore [missing-attribute]
    if not self.has_shape(node.input[0]) or (  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        len(node.input) > 1 and not self.has_shape(node.input[1])  # type: ignore[attr-defined]
    ):
        # pyrefly: ignore [missing-attribute]
        assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
            f"Unable to compute shape for node: "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        # pyrefly: ignore [missing-attribute]
        if self.has_rank(node.input[0]):  # type: ignore[attr-defined]
            self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return

    input_shape = self.get_shape(node.input[0])
    assert len(input_shape) >= 2, (
        f"Input tensor {node.input[0]!r} must have at least 2 dimensions for node "
        # pyrefly: ignore [missing-attribute]
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
    )

    n_input_dims = len(input_shape) - 2

    dilations = self.get_attribute_with_default(node, "dilations", [1] * n_input_dims)
    assert len(dilations) == n_input_dims, (
        f"Mismatch with dilations={dilations}, "
        # pyrefly: ignore [missing-attribute]
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
    )

    strides = self.get_attribute_with_default(node, "strides", [1] * n_input_dims)
    assert len(strides) == n_input_dims, (
        f"Mismatch with strides={strides}, "
        # pyrefly: ignore [missing-attribute]
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
    )

    # Gestion de kernel_shape
    kernel_shape = self.get_attribute_with_default(node, "kernel_shape", None)
    if kernel_shape:
        assert len(kernel_shape) == n_input_dims, (
            f"Mismatch with strides={kernel_shape}, "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
    if not kernel_shape:
        shape_w = self.get_shape(node.input[1])
        kernel_shape = shape_w[2:]
        assert all_int(kernel_shape), (
            f"kernel_shape is not provided and its shape is unknown "
            f"for sure kernel_shape={kernel_shape}, "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )

    effective_kernel_shape = [(k - 1) * d + 1 for k, d in zip(kernel_shape, dilations)]

    pads = self.get_attribute_with_default(node, "pads", [0] * (n_input_dims * 2))
    assert len(pads) == n_input_dims * 2, (
        f"Mismatch with pads={pads}, "
        # pyrefly: ignore [missing-attribute]
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
    )
    _ = lambda v: v if isinstance(v, int) or "," not in v else f"({v})"  # noqa: E731

    auto_pad_attr = self.get_attribute_with_default(node, "auto_pad", "NOTSET")
    if auto_pad_attr and auto_pad_attr != "VALID":
        for i in range(n_input_dims):
            stride = strides[i]
            if stride > 1:
                input_dim = input_shape[2 + i]
                if isinstance(input_dim, str):
                    if stride == 1:
                        residual = 0
                    else:
                        residual = None
                else:
                    residual = input_dim % stride

                if residual is not None:
                    total_pad = (
                        (effective_kernel_shape[i] - stride)
                        if residual == 0
                        else (effective_kernel_shape[i] - residual)
                    )
                    total_pad = max(total_pad, 0)
                    half_pad_small = total_pad // 2
                    half_pad_big = total_pad - half_pad_small
                    if auto_pad_attr == "SAME_UPPER":
                        pads[i] = half_pad_small
                        pads[i + n_input_dims] = half_pad_big
                    elif auto_pad_attr == "SAME_LOWER":
                        pads[i] = half_pad_big
                        pads[i + n_input_dims] = half_pad_small
                else:
                    # conv_f1=(d,s,stride) = (
                    #   s - (stride if d % stride == 0 else d % stride)) // 2
                    # )
                    pads[i] = f"conv_f1({_(input_dim)},{effective_kernel_shape[i]},{_(stride)})"
                    # conv_f2=(d,s,stride) = (
                    #   s - (stride if d % stride == 0 else d % stride)) // 2 + stride % 2
                    # )
                    pads[i + n_input_dims] = (
                        f"conv_f2({_(input_dim)},{effective_kernel_shape[i]},{_(stride)})"
                    )

    require_kernel_shape = node.op_type in {"MaxPool"}
    output_shape = []
    output_shape.append(input_shape[0])
    if require_kernel_shape:
        output_shape.append(input_shape[1])
    else:
        w_shape = self.get_shape(node.input[1])
        output_shape.append(w_shape[0])

    for i in range(len(kernel_shape)):
        ceil_mode = self.get_attribute_with_default(node, "ceil_mode", 0)
        input_size = input_shape[2 + i]
        if (
            isinstance(pads[i], int)
            and isinstance(pads[i + len(kernel_shape)], int)
            and isinstance(input_size, int)
        ):
            effective_input_size = input_size + pads[i] + pads[i + len(kernel_shape)]
            if ceil_mode:
                output_size = (
                    (effective_input_size - effective_kernel_shape[i] + (strides[i] - 1))
                    // strides[i]
                ) + 1
                if (output_size - 1) * strides[i] >= input_size + pads[i]:
                    # ~ pads[i + len(kernel_shape)] - effective_kernel_shape[i] +
                    #   (strides[i] - 1) >= 0
                    output_size -= 1
            else:
                output_size = (
                    (effective_input_size - effective_kernel_shape[i]) // strides[i]
                ) + 1
            output_shape.append(output_size)
            continue

        if (
            isinstance(pads[i], int)
            and isinstance(pads[i + len(kernel_shape)], int)
            and effective_kernel_shape[i] == 1
            and strides[i] == 1
        ):
            if ceil_mode:
                output_size = ((input_size + pads[i] + pads[i + len(kernel_shape)] - 1)) + 1
                if pads[i + len(kernel_shape)] >= 1:
                    # ~ pads[i + len(kernel_shape)] - effective_kernel_shape[i] +
                    #   (strides[i] - 1) >= 0
                    output_size = f"{input_size}+{pads[i] + pads[i + len(kernel_shape)]-1}"
                else:
                    output_size = f"{input_size}+{pads[i] + pads[i + len(kernel_shape)]-2}"
            else:
                output_size = f"{input_size}+{pads[i] + pads[i + len(kernel_shape)]}"
            output_shape.append(output_size)
            continue

        # conv_f3(d,s,stride,ceil_mode,p) = (
        #       d + (stride if d % stride == 0 else d % stride) +
        #       (stride - 1) * (ceil_mode == 1)
        #   ) // stride + 1 + ...

        output_size = (
            f"conv_f3_{ceil_mode}({_(input_size)},{effective_kernel_shape[i]},{_(strides[i])})"
        )
        output_shape.append(output_size)

    self.set_shape(node.output[0], tuple(output_shape))
    res = [tuple(output_shape)]

    # Gestion de la deuxième sortie pour MaxPool
    if node.op_type == "MaxPool" and len(node.output) > 1:
        second_output_shape = []
        second_output_shape.extend(output_shape)
        self.set_shape(node.output[1], tuple(second_output_shape))
        res.append(tuple(second_output_shape))
    return res


def _set_shape_type_op_any_gather(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Gather."
    # pyrefly: ignore [missing-attribute]
    if self.has_device(node.input[0]):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        self.set_device(node.output[0], self.get_device(node.input[0]))  # type: ignore[attr-defined]
    # pyrefly: ignore [missing-attribute]
    if self.has_type(node.input[0]):  # type: ignore[attr-defined]
        self.set_type(node.output[0], self.get_type(node.input[0]))
    # pyrefly: ignore [missing-attribute]
    if self.has_shape(node.input[0]) and self.has_shape(node.input[1]):  # type: ignore[attr-defined]
        sh1 = self.get_shape(node.input[0])
        sh2 = self.get_shape(node.input[1])
        att = self.get_attribute(node, "axis", exc=False)
        axis = 0 if att is None else att.i
        if len(sh2) == 0:
            new_shape = tuple(s for i, s in enumerate(sh1) if i != axis)
            # pyrefly: ignore [unexpected-keyword]
            self.set_shape(node.output[0], new_shape, allow_zero=True)  # type: ignore[call-arg]
            return new_shape
        if len(sh1) == len(sh2) == 2 and axis == 0:
            new_shape = (*sh2, sh1[-1])
            self.set_shape(node.output[0], new_shape)
            return new_shape
        if len(sh1) == len(sh2) == 1:
            self.set_shape(node.output[0], sh2)
            return sh2
        self.set_rank(node.output[0], len(sh1) + len(sh2) - 1)
        return True
    # pyrefly: ignore [missing-attribute]
    if self.has_rank(node.input[0]) and self.has_rank(node.input[1]):  # type: ignore[attr-defined]
        rk1 = self.get_rank(node.input[0])
        rk2 = self.get_rank(node.input[1])
        self.set_rank(node.output[0], rk1 + rk2 - 1)
        return True
    # pyrefly: ignore [missing-attribute]
    assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
        f"Unable to compute shape for node: "
        # pyrefly: ignore [missing-attribute]
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
    )


def _set_shape_type_op_any_gather_elements(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type GatherElements."
    # pyrefly: ignore [missing-attribute]
    if self.has_device(node.input[0]):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        self.set_device(node.output[0], self.get_device(node.input[0]))  # type: ignore[attr-defined]
    # pyrefly: ignore [missing-attribute]
    if self.has_type(node.input[0]):  # type: ignore[attr-defined]
        self.set_type(node.output[0], self.get_type(node.input[0]))
    # pyrefly: ignore [missing-attribute]
    if self.has_shape(node.input[0]) and self.has_shape(node.input[1]):  # type: ignore[attr-defined]
        shape = self.get_shape(node.input[0])
        att_axis = self.get_attribute(node, "axis", exc=False)
        axis = 0 if att_axis is None else att_axis.i
        i_shape = self.get_shape(node.input[1])
        new_shape = list(shape)
        new_shape[axis] = i_shape[axis]
        self.set_shape(node.output[0], tuple(new_shape))
        return tuple(new_shape)
    # pyrefly: ignore [missing-attribute]
    if self.has_rank(node.input[0]):  # type: ignore[attr-defined]
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return True
    # pyrefly: ignore [missing-attribute]
    assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
        f"Unable to compute shape for node: "
        # pyrefly: ignore [missing-attribute]
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
    )


def _set_shape_type_op_any_gemm(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Gemm."
    transA = self.get_attribute(node, "transA", exc=False)
    transB = self.get_attribute(node, "transB", exc=False)
    assert (
        len(node.input) >= 2
    ), f"Unexpected number of input {node.input} for node {node.op_type} name {node.name!r}"
    return set_type_shape_gemm(  # type: ignore[misc]
        self,
        node.output[0],
        *node.input[:2],
        # pyrefly: ignore [bad-keyword-argument]
        transA=0 if transA is None else transA.i,
        # pyrefly: ignore [bad-keyword-argument]
        transB=0 if transB is None else transB.i,
    )


def _set_shape_type_op_any_matmul(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type MatMul."
    r = set_type_shape_matmul(self, node.output[0], *node.input)
    # pyrefly: ignore [missing-attribute]
    assert r is not None or not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        # pyrefly: ignore [missing-attribute]
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
    )
    return r


def _set_shape_type_op_any_non_zero(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type NonZro."
    self.set_type(node.output[0], TensorProto.INT64)
    # pyrefly: ignore [missing-attribute]
    if self.has_rank(node.input[0]):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        new_shape = (self.get_rank(node.input[0]), self.unique_dimension_name("NEWDIM_nonzero"))  # type: ignore[attr-defined]
        self.set_shape(node.output[0], new_shape)
        return new_shape
    # pyrefly: ignore [missing-attribute]
    assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
        f"Unable to compute shape for node: "
        # pyrefly: ignore [missing-attribute]
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
    )


def _set_shape_type_op_any_pad(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Pad."
    # pyrefly: ignore [missing-attribute]
    if self.has_device(node.input[0]):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        self.set_device(node.output[0], self.get_device(node.input[0]))  # type: ignore[attr-defined]
    # pyrefly: ignore [missing-attribute]
    if self.has_type(node.input[0]):  # type: ignore[attr-defined]
        self.set_type(node.output[0], self.get_type(node.input[0]))
    else:
        # pyrefly: ignore [missing-attribute]
        assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
            f"Unable to compute shape for node: "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )

    # pyrefly: ignore [missing-attribute]
    if self.has_shape(node.input[0]) and self.is_constant(node.input[1]):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        pads = self.compute_constant(node.input[1])[0]  # type: ignore[attr-defined]
        assert pads is not None or not self._debug_shape_missing, (  # type: ignore[attr-defined]
            f"Unable to evaluate pad={node.input[1]!r}: "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        if pads is None:
            return
        pads = pads.tolist()
        if len(node.input) > 3 and node.input[3]:
            # pyrefly: ignore [missing-attribute]
            axes = self.compute_constant(node.input[1])[0]  # type: ignore[attr-defined]
            assert axes is not None or not self._debug_shape_missing, (  # type: ignore[attr-defined]
                f"Unable to evaluate axes={node.input[1]!r}: "
                # pyrefly: ignore [missing-attribute]
                f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
            )
            axes = axes.tolist()
        else:
            axes = list(range(len(pads) // 2))

        shape = self.get_shape(node.input[0])
        new_shape = list(shape)
        for i in range(len(axes)):
            a = axes[i]
            d = shape[a]
            p1, p2 = pads[i], pads[i + len(axes)]
            new_shape[a] = (d + p1 + p2) if isinstance(d, int) else f"{d}+{p1+p2}"
        self.set_shape(node.output[0], tuple(new_shape))
        return tuple(new_shape)
    # pyrefly: ignore [missing-attribute]
    if self.has_rank(node.input[0]):  # type: ignore[attr-defined]
        self.set_rank(node.input[0], self.get_rank(node.input[0]))
        return True
    assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
        f"Unable to compute shape for node: "
        # pyrefly: ignore [missing-attribute]
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
    )


def _set_shape_type_op_any_range(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for for node type Range."
    # pyrefly: ignore [missing-attribute]
    types = [self.get_type(i) for i in node.input if self.has_type(i)]  # type: ignore[attr-defined]
    assert types and len(set(types)) == 1, (
        f"Mixed type for node {self.pretty_node(node)}, types={types}, "
        f"unable to set shape and types."
    )
    self.set_type(node.output[0], types[0])
    self.set_rank(node.output[0], 1)
    # pyrefly: ignore [missing-attribute]
    if self.is_constant(node.input[2]) and self.get_constant(node.input[2]) == 1:  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        v1 = self.value_as_shape(node.input[0])  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        v2 = self.value_as_shape(node.input[1])  # type: ignore[attr-defined]
        assert not isinstance(
            v1, tuple
        # pyrefly: ignore [missing-attribute]
        ), f"Unexpected tuple for {node.input[0]!r} (v1={v1}){self.get_debug_msg()}"  # type: ignore[attr-defined]
        assert not isinstance(
            v2, tuple
        # pyrefly: ignore [missing-attribute]
        ), f"Unexpected tuple for {node.input[1]!r} (v2={v2}){self.get_debug_msg()}"  # type: ignore[attr-defined]
        if v1 is not None and v2 is not None:
            if isinstance(v1, int):
                if isinstance(v2, int):
                    dim = v2 - v1
                elif v1 == 0:
                    dim = v2
                else:
                    dim = simplify_expression(f"{v2}-{v1}")  # type: ignore[assignment]
            elif v2 == 0:
                dim = simplify_expression(f"-({v1})")  # type: ignore[assignment]
            else:
                dim = simplify_expression(f"{v2}-({v1})")  # type: ignore[assignment]
            self.set_shape(node.output[0], (dim,))
            return
    # pyrefly: ignore [missing-attribute]
    new_shape = (self.unique_dimension_name("NEWDIM_range"),)  # type: ignore[attr-defined]
    self.set_shape(node.output[0], new_shape)
    return new_shape


def _set_shape_type_op_any_reduce(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for Reduce node type."
    keepdim = self.get_attribute(node, "keepdims", exc=False)
    axes = self.get_attribute(node, "axes", exc=False)
    keepdim = None if keepdim is None else keepdim.i
    if axes is None:
        if len(node.input) == 2:
            # pyrefly: ignore [missing-attribute]
            if self.is_constant(node.input[1]):  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                cst = self.get_constant(node.input[1])  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                if isinstance(cst, NodeProto) and self.is_constant(cst.output[0]):  # type: ignore[attr-defined]
                    # pyrefly: ignore [missing-attribute]
                    cst = self.get_constant(node.input[1], computed_value=True)  # type: ignore[attr-defined]
                if isinstance(cst, np.ndarray):
                    iaxes = (int(cst),) if len(cst.shape) == 0 else tuple(int(i) for i in cst)
                else:
                    import torch

                    assert isinstance(cst, torch.Tensor), (
                        f"Unexpected type {type(cst)} for {node.input[1]!r}, "
                        f"unable to set type and shape for node {node.op_type} "
                        # pyrefly: ignore [missing-attribute]
                        f"with name={node.name!r}{self.get_debug_msg()}"  # type: ignore[attr-defined]
                    )
                    # pyrefly: ignore [missing-attribute]
                    with self.maybe_disable_fake_tensor_mode():  # type: ignore[attr-defined]
                        cst = cst.cpu()
                    iaxes = (int(cst),) if len(cst.shape) == 0 else tuple(int(i) for i in cst)
            elif keepdim is not None:
                self.set_rank(node.output[0], self.get_rank(node.input[0]))
                return True
            # pyrefly: ignore [missing-attribute]
            elif self.has_shape(node.input[1]) and self.has_rank(node.input[0]):  # type: ignore[attr-defined]
                shape = self.get_shape(node.input[1])
                assert (
                    len(shape) == 1
                # pyrefly: ignore [missing-attribute]
                ), f"Wrong shape={shape!r} for axes={node.input[1]!r}{self.get_debug_msg()}"  # type: ignore[attr-defined]
                if isinstance(shape[0], int):
                    self.set_rank(node.output[0], self.get_rank(node.input[0]) - shape[0])
                    return True
            else:
                assert (
                    # pyrefly: ignore [missing-attribute]
                    self._debug_shape_missing  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                ), f"Unable to determine shape for node {node}\n---\n{self.get_debug_msg()}"  # type: ignore[attr-defined]
                return
        else:
            iaxes = None
    else:
        iaxes = tuple(axes.ints)

    return set_type_shape_reduce_op(
        self,
        node.output[0],
        node.input[0],
        # pyrefly: ignore [bad-argument-type]
        keepdim=keepdim,  # type: ignore[arg-type]
        # pyrefly: ignore [bad-argument-type, unbound-name]
        axes=iaxes,  # type: ignore[arg-type]
    )


def _set_shape_type_op_any_reshape(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Reshape."
    # pyrefly: ignore [missing-attribute]
    if self.has_device(node.input[0]):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        self.set_device(node.output[0], self.get_device(node.input[0]))  # type: ignore[attr-defined]
    k = node.output[0]
    # pyrefly: ignore [missing-attribute]
    if self.has_type(node.input[0]):  # type: ignore[attr-defined]
        self.set_type(k, self.get_type(node.input[0]))
    value = None
    # pyrefly: ignore [missing-attribute]
    if self.is_constant(node.input[1]):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        value = self.get_constant(node.input[1], computed_value=True, as_shape=True, exc=False)  # type: ignore[attr-defined]
    if value is None:
        # pyrefly: ignore [missing-attribute]
        value = self.value_as_shape(node.input[1])  # type: ignore[attr-defined]
    if value is not None:
        cst = tuple(value)
        if all_int_or_str(cst):
            if -1 not in cst and 0 not in cst:
                self.set_shape(k, cst)
                return cst
            # pyrefly: ignore [missing-attribute]
            if all_int(cst) and self.has_shape(node.input[0]):  # type: ignore[attr-defined]
                sh = self.get_shape(node.input[0])
                # pyrefly: ignore [missing-attribute]
                new_shape = self._apply_reshape_to_shape(sh, cst)  # type: ignore[attr-defined]
                if new_shape is not None:
                    # pyrefly: ignore [unexpected-keyword]
                    self.set_shape(k, new_shape, allow_zero=0 in sh)  # type: ignore[call-arg]
                    return new_shape

    # pyrefly: ignore [missing-attribute]
    if self.has_shape(node.input[1]):  # type: ignore[attr-defined]
        rk = self.get_shape(node.input[1])
        # pyrefly: ignore [bad-argument-type]
        self.set_rank(k, rk[0])  # type: ignore[arg-type]
        return True
    # pyrefly: ignore [missing-attribute]
    assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
        f"Unable to compute shape for node: "
        # pyrefly: ignore [missing-attribute]
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
    )


def _set_shape_type_op_any_expand(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Reshape."
    # pyrefly: ignore [missing-attribute]
    if self.has_device(node.input[0]):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        self.set_device(node.output[0], self.get_device(node.input[0]))  # type: ignore[attr-defined]
    k = node.output[0]
    # pyrefly: ignore [missing-attribute]
    if self.has_type(node.input[0]):  # type: ignore[attr-defined]
        self.set_type(k, self.get_type(node.input[0]))
    value = None
    # pyrefly: ignore [missing-attribute]
    if self.is_constant(node.input[1]):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        value = self.get_constant(node.input[1], computed_value=True, as_shape=True, exc=False)  # type: ignore[attr-defined]
    if value is None:
        # pyrefly: ignore [missing-attribute]
        value = self.value_as_shape(node.input[1])  # type: ignore[attr-defined]
    if value is not None:
        cst = tuple(value)
        if all_int_or_str(cst):
            if -1 not in cst and 1 not in cst and 0 not in cst:
                self.set_shape(k, cst)
                return cst
            # pyrefly: ignore [missing-attribute]
            if self.has_shape(node.input[0]):  # type: ignore[attr-defined]
                sh = self.get_shape(node.input[0])
                # pyrefly: ignore [missing-attribute]
                new_shape = self._apply_expand_to_shape(sh, cst)  # type: ignore[attr-defined]
                if new_shape is not None:
                    # pyrefly: ignore [unexpected-keyword]
                    self.set_shape(k, new_shape, allow_zero=0 in sh or 0 in new_shape)  # type: ignore[call-arg]
                    return new_shape

    # pyrefly: ignore [missing-attribute]
    if self.has_shape(node.input[1]):  # type: ignore[attr-defined]
        rk = self.get_shape(node.input[1])
        if isinstance(rk[0], int):
            self.set_rank(k, rk[0])
            return True
    # pyrefly: ignore [missing-attribute]
    assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
        f"Unable to compute shape for node: "
        # pyrefly: ignore [missing-attribute]
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
    )


def _set_shape_type_op_any_sign(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Sign."
    return set_type_shape_unary_op(self, node.output[0], node.input[0])


def _set_shape_type_op_any_slice(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Slice."
    # pyrefly: ignore [missing-attribute]
    if self.has_device(node.input[0]):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        self.set_device(node.output[0], self.get_device(node.input[0]))  # type: ignore[attr-defined]
    # pyrefly: ignore [missing-attribute]
    if self.has_type(node.input[0]):  # type: ignore[attr-defined]
        self.set_type(node.output[0], self.get_type(node.input[0]))
    # pyrefly: ignore [missing-attribute]
    if self.has_rank(node.input[0]):  # type: ignore[attr-defined]
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return True
    # pyrefly: ignore [missing-attribute]
    assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
        f"Unable to compute shape for node: "
        # pyrefly: ignore [missing-attribute]
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
    )


def _set_shape_type_op_any_split(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Split."
    # pyrefly: ignore [missing-attribute]
    if self.has_device(node.input[0]):  # type: ignore[attr-defined]
        for o in node.output:
            # pyrefly: ignore [missing-attribute]
            self.set_device(o, self.get_device(node.input[0]))  # type: ignore[attr-defined]
    num_outputs = self.get_attribute(node, "num_outputs", exc=False)
    assert num_outputs is None or num_outputs.i == len(
        node.output
    ), f"Unexpected number of outputs (should be {num_outputs.i}) for node {node}"
    # pyrefly: ignore [missing-attribute]
    if not self.has_type(node.input[0]):  # type: ignore[attr-defined]
        # the main type is missing, cannot continue
        # pyrefly: ignore [missing-attribute]
        assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
            f"Unable to compute shape for node: "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        return
    dtype = self.get_type(node.input[0])
    # pyrefly: ignore [missing-attribute]
    device = self.get_device(node.input[0]) if self.has_device(node.input[0]) else None  # type: ignore[attr-defined]
    for o in node.output:
        self.set_type(o, dtype)
        if device is not None:
            # pyrefly: ignore [missing-attribute]
            self.set_device(o, device)  # type: ignore[attr-defined]
    att = self.get_attribute(node, "axis", exc=False)
    axis = 0 if att is None else att.i
    # pyrefly: ignore [missing-attribute]
    if self.has_shape(node.input[0]) and len(node.input) > 1 and self.is_constant(node.input[1]):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        splits = list(self.get_constant(node.input[1]))  # type: ignore[attr-defined]
        assert len(splits) == len(
            node.output
        ), f"Unexpected number of outputs, output={node.output} splits={splits}"
        sh = list(self.get_shape(node.input[0]))
        for i, o in enumerate(node.output):
            sh[axis] = int(splits[i])
            # pyrefly: ignore [unexpected-keyword]
            self.set_shape(o, tuple(sh), allow_zero=True)  # type: ignore[call-arg]
        return [self.get_shape(o) for o in node.output]
    num_outputs = self.get_attribute(node, "num_outputs", exc=False)
    if num_outputs is not None:
        no = num_outputs.i
        # pyrefly: ignore [missing-attribute]
        if self.has_shape(node.input[0]):  # type: ignore[attr-defined]
            dim = self.get_shape(node.input[0])[axis]
            if isinstance(dim, int):
                if dim % no == 0:
                    dims = [dim // no for i in range(no)]
                else:
                    d = dim // no + 1
                    dims = [d for i in range(no - 1)]
                    dims.append(dim - d * (no - 1))
            else:
                dims = [f"CeilToInt({dim},{no})" for i in range(no)]
                dims[-1] = (
                    f"{dim}-{no-1}*CeilToInt({dim},{no})"
                    if no > 2
                    else f"{dim}-CeilToInt({dim},{no})"
                )
            li = list(self.get_shape(node.input[0]))
            for d, o in zip(dims, node.output):
                li[axis] = d
                self.set_shape(o, tuple(li))
            return [self.get_shape(o) for o in node.output]
    # pyrefly: ignore [missing-attribute]
    if self.has_rank(node.input[0]):  # type: ignore[attr-defined]
        rank = self.get_rank(node.input[0])
        for o in node.output:
            self.set_rank(o, rank)
        return True
    # pyrefly: ignore [missing-attribute]
    assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
        f"Unable to compute shape for node: "
        # pyrefly: ignore [missing-attribute]
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
    )


def _set_shape_type_op_any_scatternd(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type ScatterND."
    # pyrefly: ignore [missing-attribute]
    if self.has_device(node.input[0]):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        self.set_device(node.output[0], self.get_device(node.input[0]))  # type: ignore[attr-defined]
    # pyrefly: ignore [missing-attribute]
    if not self.has_type(node.input[0]):  # type: ignore[attr-defined]
        # the main type is missing, cannot continue
        # pyrefly: ignore [missing-attribute]
        assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
            f"Unable to compute shape for node: "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        return
    dtype = self.get_type(node.input[0])
    self.set_type(node.output[0], dtype)
    # pyrefly: ignore [missing-attribute]
    if self.has_shape(node.input[0]):  # type: ignore[attr-defined]
        sh = self.get_shape(node.input[0])
        # pyrefly: ignore [unexpected-keyword]
        self.set_shape(node.output[0], sh, allow_zero=0 in sh)  # type: ignore[call-arg]
        return self.get_shape(node.input[0])
    # pyrefly: ignore [missing-attribute]
    if self.has_rank(node.input[0]):  # type: ignore[attr-defined]
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return True
    # pyrefly: ignore [missing-attribute]
    assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
        f"Unable to compute shape for node: "
        # pyrefly: ignore [missing-attribute]
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
    )


def _set_shape_type_op_any_sequence_empty(self: ShapeBuilder, node: NodeProto):
    for att in node.attribute:
        if att.name == "dtype":
            # pyrefly: ignore [missing-attribute]
            self.set_sequence(node.output[0], dtype=att.i)  # type: ignore[attr-defined]
            return True
    # pyrefly: ignore [missing-attribute]
    raise AssertionError(f"Attribute 'dtype' is missong from node {node}{self.get_debug_msg()}")  # type: ignore[attr-defined]


def _set_shape_type_op_any_transpose(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Transpose."
    # pyrefly: ignore [missing-attribute]
    if self.has_device(node.input[0]):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        self.set_device(node.output[0], self.get_device(node.input[0]))  # type: ignore[attr-defined]
    # pyrefly: ignore [missing-attribute]
    if not self.has_type(node.input[0]):  # type: ignore[attr-defined]
        # the main type is missing, cannot continue
        # pyrefly: ignore [missing-attribute]
        assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
            f"Unable to compute shape for node: "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        return
    dtype = self.get_type(node.input[0])
    self.set_type(node.output[0], dtype)
    # pyrefly: ignore [missing-attribute]
    if self.has_shape(node.input[0]):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        perm = list(self.get_attribute(node, "perm").ints)  # type: ignore[union-attr]
        shape = self.get_shape(node.input[0])
        assert len(perm) == len(shape), (
            f"Mismatch between perm={perm} and shape={shape}, "
            f"for op {node.op_type!r} and name={node.name!r}"
            # pyrefly: ignore [missing-attribute]
            f"{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        new_shape = list(range(len(perm)))
        for i, p in enumerate(perm):
            new_shape[i] = shape[p]
        # pyrefly: ignore [unexpected-keyword]
        self.set_shape(node.output[0], tuple(new_shape), allow_zero=0 in shape)  # type: ignore[call-arg]
        return tuple(new_shape)
    # pyrefly: ignore [missing-attribute]
    if self.has_rank(node.input[0]):  # type: ignore[attr-defined]
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return True
    # pyrefly: ignore [missing-attribute]
    assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
        f"Unable to compute shape for node: "
        # pyrefly: ignore [missing-attribute]
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
    )


def _set_shape_type_op_any_tile(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Tile."
    # pyrefly: ignore [missing-attribute]
    if self.has_device(node.input[0]):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        self.set_device(node.output[0], self.get_device(node.input[0]))  # type: ignore[attr-defined]
    self.set_type(node.output[0], self.get_type(node.input[0]))
    # pyrefly: ignore [missing-attribute]
    if self.has_rank(node.input[0]):  # type: ignore[attr-defined]
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return True
    # pyrefly: ignore [missing-attribute]
    assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
        f"Unable to compute shape for node: "
        # pyrefly: ignore [missing-attribute]
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
    )


def _set_shape_type_op_any_topk(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type TopK."
    # pyrefly: ignore [missing-attribute]
    if self.has_device(node.input[0]):  # type: ignore[attr-defined]
        for o in node.output:
            # pyrefly: ignore [missing-attribute]
            self.set_device(o, self.get_device(node.input[0]))  # type: ignore[attr-defined]
    # pyrefly: ignore [missing-attribute]
    is_scalar = self.is_constant(node.input[1])  # type: ignore[attr-defined]
    # pyrefly: ignore [missing-attribute]
    if is_scalar and self.has_shape(node.input[0]):  # type: ignore[attr-defined]
        att = self.get_attribute(node, "axis", exc=False)
        axis = att.i if att is not None else -1
        shape = list(self.get_shape(node.input[0]))
        # pyrefly: ignore [missing-attribute]
        k = self.get_constant(node.input[1], computed_value=True)  # type: ignore[attr-defined]
        ki = int(k) if k.shape == tuple() else int(k[0])
        shape[axis] = ki
        shape = tuple(shape)  # type: ignore[assignment]
    else:
        shape = None

    ret_shapes = []
    if node.output[0]:
        self.set_type(node.output[0], self.get_type(node.input[0]))
        if shape is not None:
            self.set_shape(node.output[0], shape)  # type: ignore[arg-type]
            ret_shapes.append(shape)
        # pyrefly: ignore [missing-attribute]
        elif self.has_rank(node.input[0]):  # type: ignore[attr-defined]
            self.set_rank(node.output[0], self.get_rank(node.input[0]))
        else:
            # pyrefly: ignore [missing-attribute]
            assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
                f"Unable to compute shape for node: "
                # pyrefly: ignore [missing-attribute]
                f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
            )
    if node.output[1]:
        self.set_type(node.output[1], TensorProto.INT64)
        if shape is not None:
            self.set_shape(node.output[1], shape)  # type: ignore[arg-type]
            ret_shapes.append(shape)
        # pyrefly: ignore [missing-attribute]
        elif self.has_rank(node.input[0]):  # type: ignore[attr-defined]
            self.set_rank(node.output[1], self.get_rank(node.input[0]))
            return True
        else:
            # pyrefly: ignore [missing-attribute]
            assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
                f"Unable to compute shape for node: "
                # pyrefly: ignore [missing-attribute]
                f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
            )
    if ret_shapes:
        return ret_shapes


def _set_shape_type_op_any_unsqueeze(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Unsqueeze."
    # pyrefly: ignore [missing-attribute]
    if self.has_device(node.input[0]):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        self.set_device(node.output[0], self.get_device(node.input[0]))  # type: ignore[attr-defined]
    # pyrefly: ignore [missing-attribute]
    if not self.has_type(node.input[0]):  # type: ignore[attr-defined]
        # the main type is missing, cannot continue
        # pyrefly: ignore [missing-attribute]
        assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
            f"Unable to compute shape for node: "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        return
    dtype = self.get_type(node.input[0])
    self.set_type(node.output[0], dtype)
    # pyrefly: ignore [missing-attribute]
    if self.has_shape(node.input[0]):  # type: ignore[attr-defined]
        if len(node.input) == 1:
            c = self.get_attribute(node, "axes")
            # pyrefly: ignore [missing-attribute]
            cst = np.array(c.ints, dtype=np.int64)  # type: ignore[union-attr]
        else:
            # pyrefly: ignore [missing-attribute]
            assert self.is_constant(node.input[1]), (  # type: ignore[attr-defined]
                f"axes {node.input[1]!r} from node {node.op_type}, "
                f"name={node.name!r} is not a constant, "
                # pyrefly: ignore [missing-attribute]
                f"the new shape cannot be inferred{self.get_debug_msg()}"  # type: ignore[attr-defined]
            )
            # pyrefly: ignore [missing-attribute]
            cst = self.get_constant(node.input[1])  # type: ignore[attr-defined]
            if isinstance(cst, NodeProto) and cst.op_type in (
                "Constant",
                "Identity",
                "ConstantOfShape",
            ):
                # pyrefly: ignore [missing-attribute]
                cst = self.get_constant(node.input[1], computed_value=True)  # type: ignore[attr-defined]

        if isinstance(cst, np.ndarray):
            iaxes = (int(cst),) if len(cst.shape) == 0 else tuple(int(i) for i in cst)
            shape0 = self.get_shape(node.input[0])
            shape = list(shape0)
            for i in iaxes:
                shape.insert((i + len(shape) + 1) if i < 0 else i, 1)
            # pyrefly: ignore [unexpected-keyword]
            self.set_shape(node.output[0], tuple(shape), allow_zero=0 in shape0)  # type: ignore[call-arg]
            return tuple(shape)
        # pyrefly: ignore [missing-attribute]
        elif isinstance(cst, self.torch.Tensor):  # type: ignore[attr-defined]
            # pyrefly: ignore [missing-attribute]
            with self.maybe_disable_fake_tensor_mode():  # type: ignore[attr-defined]
                iaxes = (int(cst),) if len(cst.shape) == 0 else tuple(int(i) for i in cst)
                shape = list(self.get_shape(node.input[0]))
                for i in iaxes:
                    shape.insert((i + len(shape) + 1) if i < 0 else i, 1)
                self.set_shape(node.output[0], tuple(shape))
                return tuple(shape)
        else:
            raise AssertionError(
                f"Unexpected type {type(cst)} for {node.input[1]!r}, "
                f"unable to set type and shape for node {node.op_type} "
                # pyrefly: ignore [missing-attribute]
                f"with name={node.name!r}{self.get_debug_msg()}"  # type: ignore[attr-defined]
            )
    # pyrefly: ignore [missing-attribute]
    elif self.has_rank(node.input[0]) and self.is_constant(node.input[1]):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        cst = self.get_constant(node.input[1], computed_value=True)  # type: ignore[attr-defined]
        assert cst is not None, (
            f"unable to extract constant {node.input[1]!r} in node "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        self.set_rank(node.output[0], self.get_rank(node.input[0]) + cst.size)
        return True
    else:
        # pyrefly: ignore [missing-attribute]
        assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
            f"Unable to compute shape for node: "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )


def _set_shape_type_op_any_squeeze(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Squeeze."
    # pyrefly: ignore [missing-attribute]
    if self.has_device(node.input[0]):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        self.set_device(node.output[0], self.get_device(node.input[0]))  # type: ignore[attr-defined]
    # pyrefly: ignore [missing-attribute]
    if not self.has_type(node.input[0]):  # type: ignore[attr-defined]
        # the main type is missing, cannot continue
        # pyrefly: ignore [missing-attribute]
        assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
            f"Unable to compute shape for node: "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        return
    dtype = self.get_type(node.input[0])
    self.set_type(node.output[0], dtype)
    if len(node.input) == 1 and not node.attribute:
        # No axes specified.
        # pyrefly: ignore [missing-attribute]
        if self.has_shape(node.input[0]):  # type: ignore[attr-defined]
            shape_x = self.get_shape(node.input[0])
            if all_int(shape_x):
                new_shape = tuple(s for s in shape_x if s != 1)
                self.set_shape(node.output[0], new_shape)
                return new_shape
        # In other cases, we cannot really determine the new shape for sure.
    # pyrefly: ignore [missing-attribute]
    elif self.has_shape(node.input[0]):  # type: ignore[attr-defined]
        if len(node.input) == 1:
            c = self.get_attribute(node, "axes")
            # pyrefly: ignore [missing-attribute]
            cst = np.array(c.ints, dtype=np.int64)  # type: ignore[union-attr]
        else:
            # pyrefly: ignore [missing-attribute]
            assert self.is_constant(node.input[1]), (  # type: ignore[attr-defined]
                f"axes from node {node.op_type}, "
                f"name={node.name!r} is not a constant, "
                # pyrefly: ignore [missing-attribute]
                f"the new shape cannot be inferred{self.get_debug_msg()}"  # type: ignore[attr-defined]
            )
            # pyrefly: ignore [missing-attribute]
            cst = self.get_constant(node.input[1])  # type: ignore[attr-defined]
            if isinstance(cst, NodeProto) and cst.op_type in (
                "Constant",
                "Identity",
                "ConstantOfShape",
            ):
                # pyrefly: ignore [missing-attribute]
                cst = self.get_constant(node.input[1], computed_value=True)  # type: ignore[attr-defined]
        if isinstance(cst, np.ndarray):
            iaxes = set((int(cst),) if len(cst.shape) == 0 else tuple(int(i) for i in cst))
            shape = list(self.get_shape(node.input[0]))
            iaxes = set((i + len(shape)) % len(shape) for i in iaxes)  # for negative value
            new_shape = tuple(s for i, s in enumerate(shape) if i not in iaxes)
            self.set_shape(node.output[0], new_shape)
            return new_shape
        # pyrefly: ignore [missing-attribute]
        elif isinstance(cst, self.torch.Tensor):  # type: ignore[attr-defined]
            # pyrefly: ignore [missing-attribute]
            with self.maybe_disable_fake_tensor_mode():  # type: ignore[attr-defined]
                iaxes = set((int(cst),) if len(cst.shape) == 0 else tuple(int(i) for i in cst))
                shape = list(self.get_shape(node.input[0]))
                iaxes = set((i + len(shape)) % len(shape) for i in iaxes)  # for negative value
                new_shape = tuple(s for i, s in enumerate(shape) if i not in iaxes)
                self.set_shape(node.output[0], new_shape)
                return new_shape
        else:
            raise AssertionError(
                f"Unexpected type {type(cst)} for {node.input[1]!r}, "
                f"unable to set type and shape for node {node.op_type} "
                # pyrefly: ignore [missing-attribute]
                f"with name={node.name!r}{self.get_debug_msg()}"  # type: ignore[attr-defined]
            )
    # pyrefly: ignore [missing-attribute]
    elif self.has_rank(node.input[0]) and self.is_constant(node.input[1]):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        cst = self.get_constant(node.input[1], computed_value=True)  # type: ignore[attr-defined]
        self.set_rank(
            node.output[0],
            self.get_rank(node.input[0])
            - int(cst.numel() if hasattr(cst, "numel") else cst.size),
        )
        return True
    else:
        # pyrefly: ignore [missing-attribute]
        assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
            f"Unable to compute shape for node: "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )


def _set_shape_type_op_any_where(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Where."
    # pyrefly: ignore [missing-attribute]
    if self.has_device(node.input[0]):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        self.set_device(node.output[0], self.get_device(node.input[0]))  # type: ignore[attr-defined]
    # pyrefly: ignore [missing-attribute]
    if not self.has_type(node.input[2]):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
            f"Unable to compute shape for node: "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        return
    self.set_type(node.output[0], self.get_type(node.input[2]))
    if (
        # pyrefly: ignore [missing-attribute]
        self.has_shape(node.input[0])  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        and self.has_shape(node.input[1])  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        and self.has_shape(node.input[2])  # type: ignore[attr-defined]
    ):
        sh1 = broadcast_shape(
            self.get_shape(node.input[0]),
            self.get_shape(node.input[1]),
            graph_builder=self,
        )
        sh = broadcast_shape(sh1, self.get_shape(node.input[2]), graph_builder=self)
        self.set_shape(node.output[0], sh)
        return sh
    # pyrefly: ignore [missing-attribute]
    elif all(self.has_rank(i) for i in node.input):  # type: ignore[attr-defined]
        self.set_rank(node.output[0], max(self.get_rank(i) for i in node.input))
        return True
    else:
        # pyrefly: ignore [missing-attribute]
        assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
            f"Unable to compute shape for node: "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )


def _set_shape_type_op_any_unary(
    self: ShapeBuilder,
    node: NodeProto,
    itype: Optional[int] = None,
):
    "Sets the output shape for any unary type."
    return set_type_shape_unary_op(self, node.output[0], node.input[0], itype=itype)


def _set_shape_type_op_any_arg_max_min(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for ArgMax and ArgMin."
    self.set_type(node.output[0], TensorProto.INT64)
    axis_att = self.get_attribute(node, "axis", exc=False)
    axis = 0 if axis_att is None else axis_att.i
    keepdim_att = self.get_attribute(node, "keepdims", exc=False)
    keepdim = 1 if keepdim_att is None else keepdim_att.i
    # pyrefly: ignore [missing-attribute]
    if self.has_shape(node.input[0]):  # type: ignore[attr-defined]
        shape = list(self.get_shape(node.input[0]))
        if keepdim:
            shape[axis] = 1
        else:
            del shape[axis]
        new_shape = tuple(shape)
        self.set_shape(node.output[0], new_shape)
        return new_shape
    # pyrefly: ignore [missing-attribute]
    if self.has_rank(node.input[0]):  # type: ignore[attr-defined]
        rk = self.get_rank(node.input[0])
        self.set_rank(node.output[0], rk if keepdim else rk - 1)
        return True
    # pyrefly: ignore [missing-attribute]
    assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
        f"Unable to compute shape for node: "
        # pyrefly: ignore [missing-attribute]
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
    )


def _set_shape_type_op_any_global_pool(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for GlobalAveragePool and GlobalMaxPool."
    # pyrefly: ignore [missing-attribute]
    if self.has_type(node.input[0]):  # type: ignore[attr-defined]
        self.set_type(node.output[0], self.get_type(node.input[0]))
    # pyrefly: ignore [missing-attribute]
    if self.has_shape(node.input[0]):  # type: ignore[attr-defined]
        shape = self.get_shape(node.input[0])
        new_shape = shape[:2] + (1,) * (len(shape) - 2)
        self.set_shape(node.output[0], new_shape)
        return new_shape
    # pyrefly: ignore [missing-attribute]
    if self.has_rank(node.input[0]):  # type: ignore[attr-defined]
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return True
    # pyrefly: ignore [missing-attribute]
    assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
        f"Unable to compute shape for node: "
        # pyrefly: ignore [missing-attribute]
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
    )


def _set_shape_type_op_any_flatten(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for Flatten."
    # pyrefly: ignore [missing-attribute]
    if self.has_type(node.input[0]):  # type: ignore[attr-defined]
        self.set_type(node.output[0], self.get_type(node.input[0]))
    axis_att = self.get_attribute(node, "axis", exc=False)
    axis = 1 if axis_att is None else axis_att.i
    # pyrefly: ignore [missing-attribute]
    if self.has_shape(node.input[0]):  # type: ignore[attr-defined]
        shape = self.get_shape(node.input[0])
        if axis < 0:
            axis = len(shape) + axis
        dims_before = shape[:axis]
        dims_after = shape[axis:]
        if all_int(dims_before):
            d1 = 1
            # pyrefly: ignore [bad-assignment]
            for d in dims_before:
                d1 *= d  # type: ignore[assignment]
        else:
            d1 = "*".join(map(str, dims_before)) if dims_before else "1"  # type: ignore[assignment]
        if all_int(dims_after):
            d2 = 1
            # pyrefly: ignore [bad-assignment]
            for d in dims_after:
                d2 *= d  # type: ignore[assignment]
        else:
            d2 = "*".join(map(str, dims_after)) if dims_after else "1"  # type: ignore[assignment]
        new_shape = (d1, d2)
        self.set_shape(node.output[0], new_shape)
        return new_shape
    self.set_rank(node.output[0], 2)
    return True


def _set_shape_type_op_any_eyelike(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for EyeLike."
    dtype_att = self.get_attribute(node, "dtype", exc=False)
    if dtype_att is not None:
        self.set_type(node.output[0], dtype_att.i)
    # pyrefly: ignore [missing-attribute]
    elif self.has_type(node.input[0]):  # type: ignore[attr-defined]
        self.set_type(node.output[0], self.get_type(node.input[0]))
    # pyrefly: ignore [missing-attribute]
    if self.has_shape(node.input[0]):  # type: ignore[attr-defined]
        self.set_shape(node.output[0], self.get_shape(node.input[0]))
        return self.get_shape(node.input[0])
    # pyrefly: ignore [missing-attribute]
    if self.has_rank(node.input[0]):  # type: ignore[attr-defined]
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return True
    # pyrefly: ignore [missing-attribute]
    assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
        f"Unable to compute shape for node: "
        # pyrefly: ignore [missing-attribute]
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
    )


def _set_shape_type_op_any_depth_to_space(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for DepthToSpace."
    # pyrefly: ignore [missing-attribute]
    blocksize = self.get_attribute(node, "blocksize").i  # type: ignore[union-attr]
    # pyrefly: ignore [missing-attribute]
    if self.has_type(node.input[0]):  # type: ignore[attr-defined]
        self.set_type(node.output[0], self.get_type(node.input[0]))
    # pyrefly: ignore [missing-attribute]
    if self.has_shape(node.input[0]):  # type: ignore[attr-defined]
        shape = self.get_shape(node.input[0])
        n, c = shape[0], shape[1]
        spatial = shape[2:]
        b_pow = blocksize ** len(spatial)
        new_c = (c // b_pow) if isinstance(c, int) else f"{c}//{b_pow}"
        new_spatial = tuple(
            (s * blocksize if isinstance(s, int) else f"{s}*{blocksize}") for s in spatial
        )
        new_shape = (n, new_c, *new_spatial)
        self.set_shape(node.output[0], new_shape)
        return new_shape
    # pyrefly: ignore [missing-attribute]
    if self.has_rank(node.input[0]):  # type: ignore[attr-defined]
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return True
    # pyrefly: ignore [missing-attribute]
    assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
        f"Unable to compute shape for node: "
        # pyrefly: ignore [missing-attribute]
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
    )


def _set_shape_type_op_any_space_to_depth(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for SpaceToDepth."
    # pyrefly: ignore [missing-attribute]
    blocksize = self.get_attribute(node, "blocksize").i  # type: ignore[union-attr]
    # pyrefly: ignore [missing-attribute]
    if self.has_type(node.input[0]):  # type: ignore[attr-defined]
        self.set_type(node.output[0], self.get_type(node.input[0]))
    # pyrefly: ignore [missing-attribute]
    if self.has_shape(node.input[0]):  # type: ignore[attr-defined]
        shape = self.get_shape(node.input[0])
        n, c = shape[0], shape[1]
        spatial = shape[2:]
        b_pow = blocksize ** len(spatial)
        new_c = (c * b_pow) if isinstance(c, int) else f"{c}*{b_pow}"
        new_spatial = tuple(
            (s // blocksize if isinstance(s, int) else f"{s}//{blocksize}") for s in spatial
        )
        new_shape = (n, new_c, *new_spatial)
        self.set_shape(node.output[0], new_shape)
        return new_shape
    # pyrefly: ignore [missing-attribute]
    if self.has_rank(node.input[0]):  # type: ignore[attr-defined]
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return True
    # pyrefly: ignore [missing-attribute]
    assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
        f"Unable to compute shape for node: "
        # pyrefly: ignore [missing-attribute]
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
    )


_set_shape_type_op_any_known = {
    "ArgMax": _set_shape_type_op_any_arg_max_min,
    "ArgMin": _set_shape_type_op_any_arg_max_min,
    "Attention": _set_shape_type_op_any_attention,
    "BatchNormalization": _set_shape_type_op_any_batch_normalization,
    "Cast": _set_shape_type_op_any_cast,
    "Compress": _set_shape_type_op_any_compress,
    "Concat": _set_shape_type_op_any_concat,
    "Conv": _set_shape_type_op_any_conv_max_pool,
    "DepthToSpace": _set_shape_type_op_any_depth_to_space,
    "EyeLike": _set_shape_type_op_any_eyelike,
    "Expand": _set_shape_type_op_any_expand,
    "Flatten": _set_shape_type_op_any_flatten,
    "Gather": _set_shape_type_op_any_gather,
    "GatherElements": _set_shape_type_op_any_gather_elements,
    "Gelu": _set_shape_type_op_any_unary,
    "Gemm": _set_shape_type_op_any_gemm,
    "GlobalAveragePool": _set_shape_type_op_any_global_pool,
    "GlobalMaxPool": _set_shape_type_op_any_global_pool,
    # pyrefly: ignore [bad-keyword-argument]
    "IsInf": lambda *args: _set_shape_type_op_any_unary(*args, itype=TensorProto.BOOL),  # type: ignore[misc]
    # pyrefly: ignore [bad-keyword-argument]
    "IsNaN": lambda *args: _set_shape_type_op_any_unary(*args, itype=TensorProto.BOOL),  # type: ignore[misc]
    "LayerNormalization": _set_shape_type_op_any_layer_normalization,
    "Log": _set_shape_type_op_any_unary,
    "MatMul": _set_shape_type_op_any_matmul,
    "MaxPool": _set_shape_type_op_any_conv_max_pool,
    "NonZero": _set_shape_type_op_any_non_zero,
    "Pad": _set_shape_type_op_any_pad,
    "Range": _set_shape_type_op_any_range,
    "Reshape": _set_shape_type_op_any_reshape,
    "RotaryEmbedding": _set_shape_type_op_any_rotary_embedding,
    "ScatterND": _set_shape_type_op_any_scatternd,
    "SequenceEmpty": _set_shape_type_op_any_sequence_empty,
    "Sign": _set_shape_type_op_any_sign,
    "Slice": _set_shape_type_op_any_slice,
    "SpaceToDepth": _set_shape_type_op_any_space_to_depth,
    "Split": _set_shape_type_op_any_split,
    "Squeeze": _set_shape_type_op_any_squeeze,
    "Tile": _set_shape_type_op_any_tile,
    "TopK": _set_shape_type_op_any_topk,
    "Transpose": _set_shape_type_op_any_transpose,
    "Unsqueeze": _set_shape_type_op_any_unsqueeze,
    "Where": _set_shape_type_op_any_where,
}


def set_shape_type_op_any(self: ShapeBuilder, node: NodeProto, exc: bool = False):
    """Sets the shape and type if it can."""
    if node.op_type.startswith("Reduce"):
        return _set_shape_type_op_any_reduce(self, node)
    if node.op_type in _set_shape_type_op_any_known:
        return _set_shape_type_op_any_known[node.op_type](self, node)  # type: ignore[operator]
    if node.op_type in self._op_type_element_wise_cmp_types:
        r = set_type_shape_binary_op(self, node.output[0], *node.input, cmp_op=True)
        # pyrefly: ignore [missing-attribute]
        assert r is not None or not self._debug_shape_missing, (
            f"No function to compute shape for node: "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        return r
    if node.op_type in self._op_type_element_wise_types:
        r = set_type_shape_binary_op(self, node.output[0], *node.input)
        # pyrefly: ignore [missing-attribute]
        assert r is not None or not self._debug_shape_missing, (
            f"No function to compute shape for node: "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        return r
    if node.op_type in {"DequantizeLinear", "DynamicQuantizeLinear"}:
        raise AssertionError(
            f"set_shape_type_op_any not implemented for "
            # pyrefly: ignore [missing-attribute]
            f"{node.op_type!r}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
    if node.op_type in {"CastLike"}:
        r = set_type_shape_binary_op(
            self, node.output[0], node.input[0], itype=self.get_type(node.input[1])
        )
        # pyrefly: ignore [missing-attribute]
        assert r is not None or not self._debug_shape_missing, (
            f"No function to compute shape for node: "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        return r
    if node.op_type in {"Pow"}:
        r = set_type_shape_binary_op(self, node.output[0], *node.input)
        # pyrefly: ignore [missing-attribute]
        assert r is not None or not self._debug_shape_missing, (
            f"No function to compute shape for node: "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        return r
    if node.op_type in self._op_type_unary_like:
        if node.op_type == "Abs":
            r = set_type_shape_unary_op_abs(self, node.output[0], node.input[0])
        else:
            r = set_type_shape_unary_op(self, node.output[0], node.input[0])
        # pyrefly: ignore [missing-attribute]
        assert r is not None or not self._debug_shape_missing, (
            f"No function to compute shape for node: "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        return r
    if node.op_type in {"ScatterElements", "ScatterND"}:
        r = set_type_shape_unary_op(self, node.output[0], node.input[0])
        # pyrefly: ignore [missing-attribute]
        assert r is not None or not self._debug_shape_missing, (
            f"No function to compute shape for node: "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        return r
    if node.op_type not in {"Constant", "ConstantOfShape", "Identity", "Reshape", "Shape"}:
        # Some nodes are handled when the node is created such as Identity.
        # pyrefly: ignore [missing-attribute]
        assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
            f"No function to compute shape for node: "
            # pyrefly: ignore [missing-attribute]
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
    if exc:
        raise NotImplementedError(f"No shape function for node type {node.op_type!r}")


def set_type_shape_fused_matmul(self: ShapeBuilder, node: NodeProto):
    name = node.output[0]
    x, y = node.input[:2]
    transA = self.get_attribute(node, "transA", exc=False)
    transA = transA.i if transA else 0
    transB = self.get_attribute(node, "transB", exc=False)
    transB = transB.i if transB else 0
    if transA == 0 and transB == 0:
        return set_type_shape_matmul(self, name, x, y)
    # pyrefly: ignore [missing-attribute]
    if self.has_type(x):  # type: ignore[attr-defined]
        self.set_type(name, self.get_type(x))
    # pyrefly: ignore [missing-attribute]
    elif self.has_type(y):  # type: ignore[attr-defined]
        self.set_type(name, self.get_type(y))
    # pyrefly: ignore [missing-attribute]
    if self.has_shape(x) and self.has_shape(y):  # type: ignore[attr-defined]
        sh1 = self.get_shape(x)
        sh2 = self.get_shape(y)
        if len(sh1) != len(sh2):
            if len(sh1) < len(sh2):
                sh1 = ((1,) * (len(sh2) - len(sh1))) + sh1
            else:
                sh2 = ((1,) * (len(sh1) - len(sh2))) + sh2
        prefix = (
            broadcast_shape(sh1[:-2], sh2[:-2], graph_builder=self) if len(sh1) > 2 else tuple()
        )
        new_shape = (sh1[-1] if transA else sh1[-2], sh2[-2] if transB else sh2[-1])
        new_shape = prefix + new_shape  # type: ignore[assignment]
        self.set_shape(name, new_shape)
        return new_shape
    # pyrefly: ignore [missing-attribute]
    elif self.has_rank(x) and self.has_rank(y):  # type: ignore[attr-defined]
        self.set_rank(name, max(self.get_rank(x), self.get_rank(y)))


def set_type_shape_tree_ensemble(self: ShapeBuilder, node: NodeProto):
    self.set_type(node.output[0], self.get_type(node.input[0]))
    n_targets = self.get_attribute(node, "n_targets", exc=False)
    assert n_targets is not None, (
        f"Unable to extract the dimension of the output for node type "
        f"{node.op_type!r} and name={node.name!r}"
    )
    # pyrefly: ignore [missing-attribute]
    if self.has_shape(node.input[0]):  # type: ignore[attr-defined]
        shape = self.get_shape(node.input[0])
        new_shape = (shape[0], n_targets.i)
        self.set_shape(node.output[0], new_shape)
        return new_shape
    else:
        self.set_rank(node.output[0], 2)


def set_type_shape_to_complex(self: ShapeBuilder, node: NodeProto):
    # pyrefly: ignore [missing-attribute]
    if self.has_type(node.input[0]):  # type: ignore[attr-defined]
        dtype = self.get_type(node.input[0])
        mapping = {
            TensorProto.FLOAT: TensorProto.COMPLEX64,
            TensorProto.DOUBLE: TensorProto.COMPLEX128,
        }
        assert (
            dtype in mapping
        # pyrefly: ignore [missing-attribute]
        ), f"Unexpected type {dtype} for node {node.op_type}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        self.set_type(node.output[0], mapping[dtype])
    # pyrefly: ignore [missing-attribute]
    if self.has_shape(node.input[0]):  # type: ignore[attr-defined]
        new_shape = self.get_shape(node.input[0])[:-1]
        self.set_shape(node.output[0], new_shape)
        return new_shape
    # pyrefly: ignore [missing-attribute]
    if self.has_rank(node.input[0]):  # type: ignore[attr-defined]
        self.set_rank(node.output[0], self.get_rank(node.input[0]) - 1)
    return True


def set_type_shape_complex_module(self: ShapeBuilder, node: NodeProto):
    # pyrefly: ignore [missing-attribute]
    if self.has_type(node.input[0]):  # type: ignore[attr-defined]
        dtype = self.get_type(node.input[0])
        mapping = {
            TensorProto.COMPLEX64: TensorProto.FLOAT,
            TensorProto.COMPLEX128: TensorProto.DOUBLE,
        }
        assert (
            dtype in mapping
        # pyrefly: ignore [missing-attribute]
        ), f"Unexpected type {dtype} for node {node.op_type}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        self.set_type(node.output[0], mapping[dtype])
    # pyrefly: ignore [missing-attribute]
    if self.has_shape(node.input[0]):  # type: ignore[attr-defined]
        self.set_shape(node.output[0], self.get_shape(node.input[0]))
    # pyrefly: ignore [missing-attribute]
    elif self.has_rank(node.input[0]):  # type: ignore[attr-defined]
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
    return True


def set_type_shape_shared_input(self: ShapeBuilder, node: NodeProto):
    r1 = set_type_shape_binary_op(self, node.output[0], *node.input[:2])
    r2 = set_type_shape_binary_op(self, node.output[1], *node.input[::2])
    if r1 or r2:
        return [r1, r2]


def set_type_shape_scatter_nd_of_shape(self: ShapeBuilder, node: NodeProto):
    # pyrefly: ignore [missing-attribute]
    if self.has_type(node.input[2]):  # type: ignore[attr-defined]
        self.set_type(node.output[0], self.get_type(node.input[2]))
    # pyrefly: ignore [missing-attribute]
    value = self.value_as_shape(node.input[0])  # type: ignore[attr-defined]
    if value is not None:
        self.set_shape(node.output[0], tuple(value))
        return tuple(value)


def set_type_shape_tri_matrix(self: ShapeBuilder, node: NodeProto):
    # pyrefly: ignore [missing-attribute]
    if self.has_type(node.input[1]):  # type: ignore[attr-defined]
        self.set_type(node.output[0], self.get_type(node.input[1]))
    # pyrefly: ignore [missing-attribute]
    value = self.value_as_shape(node.input[0])  # type: ignore[attr-defined]
    if value is not None:
        tvalue = tuple(value)
        self.set_shape(node.output[0], tvalue)
        return tvalue


def set_type_shape_transpose_2d_cast_fp16(self: ShapeBuilder, node: NodeProto):
    self.set_type(node.output[0], TensorProto.FLOAT16)
    # pyrefly: ignore [missing-attribute]
    if self.has_shape(node.input[0]):  # type: ignore[attr-defined]
        shape = self.get_shape(node.input[0])
        self.set_shape(node.output[0], shape[::-1])
        return shape[::-1]


def set_type_shape_transpose_2d_cast_fp32(self: ShapeBuilder, node: NodeProto):
    self.set_type(node.output[0], TensorProto.FLOAT)
    # pyrefly: ignore [missing-attribute]
    if self.has_shape(node.input[0]):  # type: ignore[attr-defined]
        shape = self.get_shape(node.input[0])
        self.set_shape(node.output[0], shape[::-1])
        return shape[::-1]


def set_type_shape_multi_head_attention(self: ShapeBuilder, node: NodeProto):
    itype = self.get_type(node.input[0])
    for o in node.output:
        self.set_type(o, itype)
    if (
        # pyrefly: ignore [missing-attribute]
        self.has_shape(node.input[0])  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        and self.has_shape(node.input[1])  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        and self.has_shape(node.input[2])  # type: ignore[attr-defined]
    ):
        assert (
            self.get_rank(node.input[0]) == 3
        # pyrefly: ignore [missing-attribute]
        ), f"rank(query)={self.get_rank(node.input[0])} != 3{self.get_debug_msg()}"  # type: ignore[attr-defined]
        q_shape, _k_shape, _v_shape = [self.get_shape(i) for i in node.input[:3]]
        pk_shape = (
            self.get_shape(node.input[6])
            # pyrefly: ignore [missing-attribute]
            if len(node.input) > 6 and node.input[6] and self.has_shape(node.input[6])  # type: ignore[attr-defined]
            else None
        )
        up = []
        self.set_shape(node.output[0], q_shape)
        up.append(q_shape)
        # pyrefly: ignore [unsupported-operation]
        d1, d2 = q_shape[1], pk_shape[2]  # type: ignore[index]
        if isinstance(d1, int) and isinstance(d2, int):
            d = d1 + d2
        else:
            d = simplify_expression(f"({d1})+({d2})")  # type: ignore[assignment]
        # pyrefly: ignore [unsupported-operation]
        shape = (*pk_shape[:2], d, pk_shape[-1])  # type: ignore[index]
        for o in node.output[1:]:
            if o:
                self.set_shape(o, shape)
                up.append(shape)
        return up
    else:
        self.set_rank(node.output[0], 3)
        for o in node.output[1:]:
            if o:
                self.set_rank(o, 4)


_set_shape_type_op_any_custom = {
    "AddAdd": lambda g, node: set_type_shape_binary_op(g, node.output[0], *node.input),
    "AddMul": lambda g, node: set_type_shape_binary_op(g, node.output[0], *node.input),
    "AddSharedInput": set_type_shape_shared_input,
    "BiasGelu": lambda g, node: set_type_shape_unary_op(g, node.output[0], node.input[0]),
    "BiasSoftmax": lambda g, node: set_type_shape_unary_op(g, node.output[0], node.input[0]),
    "ComplexModule": set_type_shape_complex_module,
    "FastGelu": lambda g, node: set_type_shape_unary_op(g, node.output[0], node.input[0]),
    "FusedMatMul": set_type_shape_fused_matmul,
    "FusedConv": _set_shape_type_op_any_conv_max_pool,
    "Gelu": lambda g, node: set_type_shape_unary_op(g, node.output[0], node.input[0]),
    "MaskedScatterNDOfShape": set_type_shape_scatter_nd_of_shape,
    "MulAdd": lambda g, node: set_type_shape_binary_op(g, node.output[0], *node.input),
    "MulMul": lambda g, node: set_type_shape_binary_op(g, node.output[0], *node.input),
    "MulSharedInput": set_type_shape_shared_input,
    "MulSigmoid": lambda g, node: set_type_shape_binary_op(g, node.output[0], *node.input),
    "MulMulSigmoid": lambda g, node: set_type_shape_binary_op(g, node.output[0], *node.input),
    "MultiHeadAttention": set_type_shape_multi_head_attention,
    "MulSub": lambda g, node: set_type_shape_binary_op(g, node.output[0], *node.input),
    "QuickGelu": lambda g, node: set_type_shape_unary_op(g, node.output[0], node.input[0]),
    "Rotary": lambda g, node: set_type_shape_unary_op(g, node.output[0], node.input[0]),
    "RotaryEmbedding": lambda g, node: set_type_shape_unary_op(g, node.output[0], node.input[0]),
    "ScatterNDOfShape": set_type_shape_scatter_nd_of_shape,
    "SkipLayerNormalization": lambda g, node: set_type_shape_unary_op(
        g, node.output[0], node.input[0]
    ),
    "SkipSimplifiedLayerNormalization": lambda g, node: set_type_shape_unary_op(
        g, node.output[0], node.input[0]
    ),
    "SubMul": lambda g, node: set_type_shape_binary_op(g, node.output[0], *node.input),
    "ToComplex": set_type_shape_to_complex,
    "Transpose2DCastFP16": set_type_shape_transpose_2d_cast_fp16,
    "Transpose2DCastFP32": set_type_shape_transpose_2d_cast_fp32,
    "TriMatrix": set_type_shape_tri_matrix,
}


def set_shape_type_custom(self: ShapeBuilder, node: NodeProto, exc: bool = False):
    """Sets the shape and type if it can."""
    if node.domain == "ai.onnx.ml":
        if node.op_type in ("TreeEnsembleRegressor", "TreeEnsemble"):
            return set_type_shape_tree_ensemble(self, node)
        return None
    if node.op_type in {"ReplaceZero", "NegXplus1"}:
        return set_type_shape_unary_op(self, node.output[0], node.input[0])
    if node.op_type in _set_shape_type_op_any_custom:
        return _set_shape_type_op_any_custom[node.op_type](self, node)
    # pyrefly: ignore [missing-attribute]
    if self.has_local_function(node.op_type, domain=node.domain, builder=True):  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        local_function_builder = self.get_local_function(  # type: ignore[attr-defined]
            node.op_type, domain=node.domain, builder=True
        )
        assert local_function_builder is not None, (
            f"Missing local function for node {(node.domain, node.op_type)}"
            # pyrefly: ignore [missing-attribute]
            f"{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        assert isinstance(local_function_builder, self.__class__), (
            f"Unexpected type {type(local_function_builder)} "
            f"for node {(node.domain, node.op_type)} "
            # pyrefly: ignore [missing-attribute]
            f"and the local_function it refers to{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        # pyrefly: ignore [missing-attribute]
        shapes = [self.get_shape(i) if self.has_shape(i) else None for i in node.input]  # type: ignore[attr-defined]
        if None in shapes:
            # Nothing we can do.
            return
        # pyrefly: ignore [missing-attribute]
        proto_local_function = self.get_local_function(node.op_type, domain=node.domain)  # type: ignore[attr-defined]
        local_shapes = [
            # pyrefly: ignore [missing-attribute]
            local_function_builder.get_shape(i) if local_function_builder.has_shape(i) else None  # type: ignore[attr-defined]
            for i in proto_local_function.input
        ]
        # The builder creating the local function may have less inputs because
        # when exported to FunctionProto, constants were promoted as inputs.
        assert len(shapes) == len(local_shapes), (
            f"Mismatch between the number of inputs, node '{node.domain}.{node.op_type}' "
            # pyrefly: ignore [missing-attribute]
            f"has {node.input}, function has {proto_local_function.input}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        if local_shapes != shapes:
            # pyrefly: ignore [missing-attribute]
            local_function_builder.reset_types_and_shapes()  # type: ignore[attr-defined]
            for ni, i, sh in zip(node.input, proto_local_function.input, shapes):
                # pyrefly: ignore [missing-attribute]
                if self.has_type(ni):  # type: ignore[attr-defined]
                    local_function_builder.set_type(i, self.get_type(ni))
                # pyrefly: ignore [missing-attribute]
                if self.has_device(ni):  # type: ignore[attr-defined]
                    # pyrefly: ignore [missing-attribute]
                    local_function_builder.set_device(i, self.get_device(ni))  # type: ignore[attr-defined]
                # pyrefly: ignore [bad-argument-type]
                local_function_builder.set_shape(i, sh)  # type: ignore[arg-type]
            # pyrefly: ignore [missing-attribute]
            local_function_builder.infer_shapes()  # type: ignore[attr-defined]
        assert len(local_function_builder.output_names) == len(node.output), (
            f"Mismatch between the number of outputs, node has {node.output}, "
            # pyrefly: ignore [missing-attribute]
            f"function has {local_function_builder.output_names}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        for o, lo in zip(node.output, local_function_builder.output_names):
            # pyrefly: ignore [missing-attribute]
            if local_function_builder.has_type(lo):  # type: ignore[attr-defined]
                self.set_type(o, local_function_builder.get_type(lo))
            # pyrefly: ignore [missing-attribute]
            if local_function_builder.has_shape(lo):  # type: ignore[attr-defined]
                self.set_shape(o, local_function_builder.get_shape(lo))
            # pyrefly: ignore [missing-attribute]
            elif local_function_builder.has_rank(lo):  # type: ignore[attr-defined]
                self.set_rank(o, local_function_builder.get_rank(lo))
        return None

    # to be improved later
    if node.op_type in {"PackedMultiHeadAttention"} and node.domain == "com.microsoft":
        # pyrefly: ignore [missing-attribute]
        if self.has_type(node.input[0]):  # type: ignore[attr-defined]
            self.set_type(node.output[0], self.get_type(node.input[0]))
        # pyrefly: ignore [missing-attribute]
        if self.has_rank(node.input[0]):  # type: ignore[attr-defined]
            self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return None

    # to be improved later
    if node.op_type in {"Attention"} and node.domain == "com.microsoft":
        # pyrefly: ignore [missing-attribute]
        if self.has_type(node.input[0]):  # type: ignore[attr-defined]
            self.set_type(node.output[0], self.get_type(node.input[0]))
        # pyrefly: ignore [missing-attribute]
        if self.has_shape(node.input[0]):  # type: ignore[attr-defined]
            self.set_shape(node.output[0], self.get_shape(node.input[0]))
        # pyrefly: ignore [missing-attribute]
        elif self.has_rank(node.input[0]):  # type: ignore[attr-defined]
            self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return None

    # to be improved later
    if node.op_type in {"GroupQueryAttention"} and node.domain == "com.microsoft":
        # pyrefly: ignore [missing-attribute]
        if self.has_type(node.input[0]):  # type: ignore[attr-defined]
            self.set_type(node.output[0], self.get_type(node.input[0]))
        # pyrefly: ignore [missing-attribute]
        if self.has_rank(node.input[0]):  # type: ignore[attr-defined]
            self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return None

    assert node.op_type in {"GatherGrad", "SoftmaxGrad", "ConcatTraining"} or node.domain not in {
        "ai.onnx.ml",
        "intermediate",
        "ai.onnx.complex",
        "com.microsoft",
        "local_domain",
        "SimplifyingFunction",
        "onnx_extended.ortops.optim.cuda",
    }, (
        f"Unable to find a function computing the output shape of node "
        f"{(node.domain, node.op_type)}, list of functions is "
        # pyrefly: ignore [missing-attribute]
        f"{sorted(self.functions)}, list of functions with graph is "  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        f"{sorted(self.functions_builder)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
    )
    # pyrefly: ignore [missing-attribute]
    assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
        f"No function to compute shape for node: "
        # pyrefly: ignore [missing-attribute]
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
    )
    if exc:
        raise NotImplementedError(
            f"No shape function for node type {node.op_type!r} from domain {node.domain!r}"
        )

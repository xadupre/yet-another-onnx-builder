from collections import Counter
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple
import numpy as np
import onnx
from onnx import NodeProto, TensorProto
from ..helpers.onnx_helper import pretty_onnx
from ..xexpressions import simplify_expression
from ._shape_helper import DYNAMIC_SHAPE, is_static_shape, all_int, all_int_or_str
from .shape_builder import ShapeBuilder
from .type_inference import _infer_types_body


def broadcast_shape(
    sh1: DYNAMIC_SHAPE, sh2: DYNAMIC_SHAPE, graph_builder: Optional[ShapeBuilder] = None
) -> DYNAMIC_SHAPE:
    """
    Computes the output shape for broadcasting operators (e.g. ``Add``, ``Mul``, ``Where``).

    The function follows NumPy/ONNX broadcasting rules.  Shapes are right-aligned
    and each pair of dimensions ``(a, b)`` is resolved according to the table below:

    +-------------------+-------------------+--------+----------------------------------------+
    | ``a``             | ``b``             | Result | Side effect                            |
    +===================+===================+========+========================================+
    | int (any)         | int (any)         | max    | none                                   |
    +-------------------+-------------------+--------+----------------------------------------+
    | int ``n ≠ 0, 1``  | str (symbolic)    | ``n``  | :meth:`register_constraint_dimension   |
    |                   |                   |        | <yobx.xshape.ShapeBuilder.             |
    |                   |                   |        | register_constraint_dimension>`        |
    |                   |                   |        | ``(b, n)`` if *graph_builder*          |
    +-------------------+-------------------+--------+----------------------------------------+
    | ``1``             | str (symbolic)    | ``b``  | none                                   |
    +-------------------+-------------------+--------+----------------------------------------+
    | str (symbolic)    | int ``n ≠ 0, 1``  | ``n``  | :meth:`register_constraint_dimension   |
    |                   |                   |        | <yobx.xshape.ShapeBuilder.             |
    |                   |                   |        | register_constraint_dimension>`        |
    |                   |                   |        | ``(a, n)`` if *graph_builder*          |
    +-------------------+-------------------+--------+----------------------------------------+
    | str (symbolic)    | ``1``             | ``a``  | none                                   |
    +-------------------+-------------------+--------+----------------------------------------+
    | str ``a == b``    | str ``a == b``    | ``a``  | none                                   |
    +-------------------+-------------------+--------+----------------------------------------+
    | str ``a != b``    | str ``a != b``    | ``a^b``| none (``^`` means ``max``)             |
    +-------------------+-------------------+--------+----------------------------------------+

    When a symbolic dimension is paired with a concrete integer ``n ≠ 1``, the concrete
    value is chosen as the output dimension *and* the equality is stored as a constraint
    via :meth:`register_constraint_dimension
    <yobx.xshape.ShapeBuilder.register_constraint_dimension>`.  This avoids the need to
    backtrack through earlier nodes when the concrete value is discovered later:
    downstream operations immediately see a precise integer shape.

    :param sh1: first shape (tuple of ints and/or symbolic strings)
    :param sh2: second shape (tuple of ints and/or symbolic strings)
    :param graph_builder: if not None, constraints are registered on this builder
        whenever a symbolic dimension is equated to a concrete integer
    :return: resulting broadcast shape
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
                d = b
            else:
                # We have two indications, let's take the most strict one.
                d = a
                if graph_builder:
                    graph_builder.register_constraint_dimension(b, a)
        elif isinstance(b, int):
            # a is str
            if b == 0:
                d = 0
            elif b == 1:
                d = a
            elif b != 1:
                # a is not int, it is str
                d = b
                if graph_builder:
                    graph_builder.register_constraint_dimension(a, b)
        else:
            # both str
            if a == b:
                d = a
            else:
                d = simplify_expression(f"({a})^({b})")
        if d is None:
            raise RuntimeError(
                f"Not implemented for sh1={sh1}, sh2={sh2}, a={a}, b={b}, "
                f"type(a)={type(a)}, type(b)={type(b)}, a={a}, b={b}"
            )
        new_shape.append(d)
    return tuple(new_shape)


def _set_shape_type_op_any_attention(g: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Attention."
    if g.has_device(node.input[0]):
        for o in node.output:
            if o:
                g.set_device(o, g.get_device(node.input[0]))
    if g.has_type(node.input[0]):
        itype = g.get_type(node.input[0])
        for i, o in enumerate(node.output):
            if i != 2 and o:
                g.set_type(o, itype)
        if len(node.output) > 2 and node.output[2] and g.has_type(node.input[2]):
            g.set_type(node.output[2], g.get_type(node.input[2]))
    if g.has_shape(node.input[0]) and g.has_shape(node.input[1]) and g.has_shape(node.input[2]):
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
    if g.has_rank(node.input[0]):
        rk = g.get_rank(node.input[0])
        g.set_rank(node.output[0], rk)
        for o in node.output[1:]:
            if o:
                g.set_rank(o, 4)


def set_type_shape_reshape(g: ShapeBuilder, name: str, input_name: str, new_shape: Sequence[int]):
    "Sets the output shape for node type Reshape"
    g.set_type(name, g.get_type(input_name))
    if g.has_device(input_name):
        g.set_device(name, g.get_device(input_name))
    if isinstance(new_shape, str):
        if g.has_shape(new_shape):
            sh = g.get_shape(new_shape)
            assert len(sh) == 1, f"Unexpected value {sh} for shape={new_shape!r}"
            g.set_rank(name, sh[0])
    elif not is_static_shape(new_shape):
        g.set_rank(name, len(new_shape))
    elif min(new_shape) == -1:
        if g.has_shape(input_name):
            shape = list(g.get_shape(input_name))
            if not all_int(shape):
                # Input shape contains dynamic dimensions (strings); we cannot compute
                # the product statically.  Try to propagate a single dynamic dim when
                # arg_size == 1 and the input is effectively 1-D (one dynamic dim,
                # all other dims == 1).  Fall back to rank-only otherwise.
                arg_size = np.prod([a for a in new_shape if a >= 0])
                non_one = [d for d in shape if not isinstance(d, int) or d != 1]
                if arg_size == 1 and len(non_one) == 1 and not isinstance(non_one[0], int):
                    index = new_shape.index(-1)
                    new_shape = list(new_shape)
                    new_shape[index] = non_one[0]
                    g.set_shape(name, tuple(new_shape))
                else:
                    g.set_rank(name, len(new_shape))
            else:
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
    g: ShapeBuilder, name: str, input_name: str, itype: Optional[int] = None
) -> bool:
    """Sets the shape and type for an unary operator (abs, exp, ...)."""
    if g.has_device(input_name):
        g.set_device(name, g.get_device(input_name))
    if not itype and not g.has_type(input_name):
        return
    g.set_type(name, itype or g.get_type(input_name))
    if g.has_shape(input_name):
        g.set_shape(name, g.get_shape(input_name), allow_zero=True)
        return g.get_shape(input_name)
    if g.has_rank(input_name):
        g.set_rank(name, g.get_rank(input_name))
        return True
    return


def set_type_shape_unary_op_abs(
    g: ShapeBuilder, name: str, input_name: str, itype: Optional[int] = None
) -> bool:
    """Sets the shape and type for an unary operator (abs, exp, ...)."""
    if g.has_device(input_name):
        g.set_device(name, g.get_device(input_name))
    if not itype and not g.has_type(input_name):
        return
    if not itype:
        itype = g.get_type(input_name)
    if itype in (TensorProto.COMPLEX64, TensorProto.COMPLEX128):
        if itype == TensorProto.COMPLEX64:
            rtype = TensorProto.FLOAT
        elif itype == TensorProto.COMPLEX128:
            rtype = TensorProto.DOUBLE
        else:
            raise AssertionError(f"Unexpected type {itype} for {input_name!r}{g.get_debug_msg()}")

        g.set_type(name, rtype)
        if g.has_shape(input_name):
            shape = g.get_shape(input_name)
            g.set_shape(name, shape)
            return shape
        if g.has_rank(input_name):
            g.set_rank(name, g.get_rank(input_name))
            return True
        return

    g.set_type(name, itype)
    if g.has_shape(input_name):
        sh = g.get_shape(input_name)
        g.set_shape(name, sh, allow_zero=0 in sh)
        return g.get_shape(input_name)
    if g.has_rank(input_name):
        g.set_rank(name, g.get_rank(input_name))
        return True
    return


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
    if all(g.has_device(i) for i in input_names):
        devs = {g.get_device(i) for i in input_names}
        if len(devs) == 1:
            g.set_device(name, devs.pop())
    elif len(input_names) == 2:
        if g.has_device(input_names[0]) and not g.has_device(input_names[1]):
            g.set_device(name, g.get_device(input_names[0]))
        elif not g.has_device(input_names[1]) and g.has_device(input_names[0]):
            g.set_device(name, g.get_device(input_names[1]))
    # type
    dtype = None
    if itype:
        g.set_type(name, itype)
    elif cmp_op:
        # operator comparing values
        g.set_type(name, TensorProto.BOOL)
    else:
        for input_name in input_names[begin:]:
            if g.has_type(input_name):
                dtype = g.get_type(input_name)
                break
        if not dtype and g.as_function:
            return
        assert dtype, f"Unable to guess type for {name!r} from {input_names}{g.get_debug_msg()}"
        g.set_type(name, dtype)

    # shape
    shape = None
    for input_name in input_names:
        if g.has_shape(input_name):
            input_shape = g.get_shape(input_name)
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
        g.set_shape(name, shape, allow_zero=True)
        return shape

    # rank otherwise
    rank = None
    for input_name in input_names:
        if g.has_rank(input_name):
            rank = g.get_rank(input_name) if rank is None else max(rank, g.get_rank(input_name))
            continue
        if rank is not None:
            rank = None
        # one shape is missing
        break

    if rank is not None:
        g.set_rank(name, rank)
        return True
    return


def set_type_shape_matmul(g: ShapeBuilder, name: str, x: str, y: str) -> bool:
    "Sets the output shape for node type MatMul."
    # device
    if g.has_device(x) and g.has_device(y) and g.get_device(x) == g.get_device(y):
        g.set_device(name, g.get_device(x))
    if not g.has_type(x):
        return
    g.set_type(name, g.get_type(x))
    if g.has_shape(x) and g.has_shape(y):
        sh1 = g.get_shape(x)
        sh2 = g.get_shape(y)
        if len(sh1) == len(sh2) == 1:
            g.set_shape(name, tuple())
            return
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
            f"{g.get_debug_msg()}"
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
            new_shape.append(g.make_dimension_name_if_necessary(a, b, "^"))

        new_shape.append(sh1[-2])
        new_shape.append(sh2[-1])
        new_shape = tuple(new_shape)
        g.set_shape(name, new_shape)
        return new_shape
    if g.has_rank(x) and g.has_rank(y):
        if g.get_rank(x) == g.get_rank(y) == 1:
            return g.set_shape(name, tuple())
        g.set_rank(name, max(g.get_rank(x), g.get_rank(y)))
        return True
    return


def set_type_shape_gemm(g: ShapeBuilder, name: str, x: str, y: str, transA: int, transB: int):
    "Sets the output shape for node type Gemm."
    if transA == 0 and transB == 0:
        return set_type_shape_matmul(g, name, x, y)
    if g.has_device(x) and g.has_device(y) and g.get_device(x) == g.get_device(y):
        g.set_device(name, g.get_device(x))
    g.set_type(name, g.get_type(x))
    if g.has_shape(x) and g.has_shape(y):
        sh1 = g.get_shape(x)
        sh2 = g.get_shape(y)
        assert len(sh1) == len(
            sh2
        ), f"not implemented when shapes are {sh1} and {sh2}{g.get_debug_msg()}"
        new_shape = (sh1[-1] if transA else sh1[-2], sh2[-2] if transB else sh2[-1])
        g.set_shape(name, new_shape)
        return new_shape
    elif g.has_rank(x) and g.has_rank(y):
        g.set_rank(name, max(g.get_rank(x), g.get_rank(y)))


def set_type_shape_reduce_op(
    g: ShapeBuilder, name: str, x: str, keepdim: int, axes: Optional[Tuple[int]] = None
):
    "Sets the output shape for any Reduce type."
    assert keepdim in {None, 0, 1}, f"keepdim={keepdim!r} must be in {{0, 1}}"
    if g.has_device(x):
        g.set_device(name, g.get_device(x))
    if keepdim is None:
        keepdim = 1
    if g.has_type(x):
        g.set_type(name, g.get_type(x))
    if axes is None:
        new_shape = ((1,) * g.get_rank(x)) if keepdim else tuple()
        g.set_shape(name, new_shape)
        return new_shape
    elif not g.has_shape(x):
        if g.has_rank(x):
            g.set_rank(name, g.get_rank(x) - (1 - int(keepdim)) * len(axes))
            return True
    else:
        shape = list(g.get_shape(x))
        for d in axes:
            assert d < len(shape), (
                f"shape mismatch for a reduce op shape={shape}, "
                f"axes={axes}{g.get_debug_msg()}"
            )
            shape[d] = 1 if keepdim else None
        shape = tuple(_ for _ in shape if _ is not None)
        g.set_shape(name, shape)
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


def _set_shape_type_op_any_instance_normalization(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type InstanceNormalization."
    return set_type_shape_unary_op(self, node.output[0], node.input[0])


def _set_shape_type_op_any_lp_normalization(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type LpNormalization."
    return set_type_shape_unary_op(self, node.output[0], node.input[0])


def _set_shape_type_op_any_cast(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Cast."
    r = set_type_shape_unary_op(
        self, node.output[0], node.input[0], itype=self.get_attribute(node, "to").i
    )
    assert not self.has_type(node.input[0]) or self.has_type(
        node.output[0]
    ), f"Missing output for node {pretty_onnx(node)}."
    return r


def _set_shape_type_op_any_dropout(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Dropout."
    res = set_type_shape_unary_op(self, node.output[0], node.input[0])
    if len(node.output) > 1 and node.output[1]:
        set_type_shape_unary_op(self, node.output[1], node.input[0], itype=TensorProto.BOOL)
    return res


def _set_shape_type_op_any_rotary_embedding(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type RotaryEmbedding."
    return set_type_shape_unary_op(self, node.output[0], node.input[0])


def _set_shape_type_op_any_gemma_rotary_embedding(self: ShapeBuilder, node: NodeProto):
    """Sets the output shapes for node type GemmaRotaryEmbedding.

    Inputs: ``emb`` (batch, seq, dim), ``q`` (batch, heads_q, seq, dim),
    ``q_rot``, ``k`` (batch, heads_k, seq, dim), ``k_rot``.
    Outputs: ``q_embed`` matching ``q``, ``k_embed`` matching ``k``.
    """
    if len(node.output) >= 1 and node.output[0]:
        set_type_shape_unary_op(self, node.output[0], node.input[1])
    if len(node.output) >= 2 and node.output[1]:
        set_type_shape_unary_op(self, node.output[1], node.input[3])


def _set_shape_type_op_any_castlike(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type CastLike."
    return set_type_shape_unary_op(
        self, node.output[0], node.input[0], itype=self.get_type(node.input[1])
    )


def _set_shape_type_op_any_compress(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Compress."
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))
    att = self.get_attribute(node, "axis", exc=False)
    if att is not None:
        axis = att.i
        if self.has_shape(node.input[0]):
            shape = list(self.get_shape(node.input[0]))
            if axis < 0:
                axis = len(shape) + axis
            shape[axis] = self.unique_dimension_name("NEWDIM_compress")
            new_shape = tuple(shape)
            self.set_shape(node.output[0], new_shape)
            return new_shape
        if self.has_rank(node.input[0]):
            self.set_rank(node.output[0], self.get_rank(node.input[0]))
            return True
    else:
        # No axis: input is flattened first, output is 1-D with unknown size
        new_shape = (self.unique_dimension_name("NEWDIM_compress"),)
        self.set_shape(node.output[0], new_shape)
        return new_shape


def _set_shape_type_op_any_concat(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Concat."
    if all(self.has_device(i) for i in node.input):
        devs = {self.get_device(i) for i in node.input}
        if len(devs) == 1:
            self.set_device(node.output[0], devs.pop())
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))
    if all(self.value_as_shape(s) is not None for s in node.input):
        axis = self.get_attribute(node, "axis").i
        assert (
            axis == 0
        ), f"Unexpected value for axis={axis} in {self.pretty_node(node)}{self.get_debug_msg()}"
        shape = []
        for s in node.input:
            shape.extend(self.value_as_shape(s))
        self.set_value_shape(node.output[0], tuple(shape))
    if all(self.has_shape(s) for s in node.input):
        axis = self.get_attribute(node, "axis").i
        shapes = [self.get_shape(i) for i in node.input]
        new_shape = list(shapes[0])
        assert shapes and axis < min(len(sh) for sh in shapes), (
            f"axis={axis}, higher than a shape in {shapes}, "
            f"node={self.pretty_node(node)}{self.get_debug_msg()}"
        )
        assert all(axis < len(sh) for sh in shapes), f"Unexpected shape in {shapes}, axis={axis}"
        dims = [sh[axis] for sh in shapes]
        if all_int(dims):
            new_shape[axis] = sum(dims)
        else:
            new_shape[axis] = "+".join(map(str, dims))
        new_shape = tuple(new_shape)
        self.set_shape(node.output[0], new_shape)
        return new_shape

    if all(map(self.has_rank, node.input)):
        ranks = [self.get_rank(i) for i in node.input]
        assert (
            len(set(ranks)) == 1
        ), f"Unexpected ranks={ranks} for node {node.op_type!r}{self.get_debug_msg()}"
        self.set_rank(node.output[0], ranks[0])
        return True
    assert not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
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
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if not self.has_type(node.input[0]):
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
        return
    self.set_type(node.output[0], self.get_type(node.input[0]))
    if len(node.output) > 1:
        self.set_type(node.output[1], TensorProto.INT64)

    if not self.has_shape(node.input[0]) or (
        len(node.input) > 1 and not self.has_shape(node.input[1])
    ):
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
        if self.has_rank(node.input[0]):
            self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return

    input_shape = self.get_shape(node.input[0])
    assert len(input_shape) >= 2, (
        f"Input tensor {node.input[0]!r} must have at least 2 dimensions for node "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )

    n_input_dims = len(input_shape) - 2

    dilations = self.get_attribute_with_default(node, "dilations", [1] * n_input_dims)
    assert len(dilations) == n_input_dims, (
        f"Mismatch with dilations={dilations}, "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )

    strides = self.get_attribute_with_default(node, "strides", [1] * n_input_dims)
    assert len(strides) == n_input_dims, (
        f"Mismatch with strides={strides}, "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )

    # Gestion de kernel_shape
    kernel_shape = self.get_attribute_with_default(node, "kernel_shape", None)
    if kernel_shape:
        assert len(kernel_shape) == n_input_dims, (
            f"Mismatch with strides={kernel_shape}, "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
    if not kernel_shape:
        shape_w = self.get_shape(node.input[1])
        kernel_shape = shape_w[2:]
        assert all_int(kernel_shape), (
            f"kernel_shape is not provided and its shape is unknown "
            f"for sure kernel_shape={kernel_shape}, "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )

    effective_kernel_shape = [(k - 1) * d + 1 for k, d in zip(kernel_shape, dilations)]

    pads = self.get_attribute_with_default(node, "pads", [0] * (n_input_dims * 2))
    assert len(pads) == n_input_dims * 2, (
        f"Mismatch with pads={pads}, "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )
    _ = lambda v: v if isinstance(v, int) or "," not in v else f"({v})"  # noqa: E731

    auto_pad_attr = self.get_attribute_with_default(node, "auto_pad", "NOTSET")
    if isinstance(auto_pad_attr, bytes):
        auto_pad_attr = auto_pad_attr.decode("utf-8")
    if auto_pad_attr and auto_pad_attr != "VALID" and auto_pad_attr != "NOTSET":
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
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))
    if self.has_shape(node.input[0]) and self.has_shape(node.input[1]):
        sh1 = self.get_shape(node.input[0])
        sh2 = self.get_shape(node.input[1])
        att = self.get_attribute(node, "axis", exc=False)
        axis = 0 if att is None else att.i
        new_shape = sh1[:axis] + sh2 + sh1[axis + 1 :]
        self.set_shape(node.output[0], new_shape, allow_zero=len(sh2) == 0)
        return new_shape
    if self.has_rank(node.input[0]) and self.has_rank(node.input[1]):
        rk1 = self.get_rank(node.input[0])
        rk2 = self.get_rank(node.input[1])
        self.set_rank(node.output[0], rk1 + rk2 - 1)
        return True
    assert not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )


def _set_shape_type_op_any_gather_elements(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type GatherElements."
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))
    if self.has_shape(node.input[0]) and self.has_shape(node.input[1]):
        shape = self.get_shape(node.input[0])
        att_axis = self.get_attribute(node, "axis", exc=False)
        axis = 0 if att_axis is None else att_axis.i
        i_shape = self.get_shape(node.input[1])
        new_shape = list(shape)
        new_shape[axis] = i_shape[axis]
        self.set_shape(node.output[0], tuple(new_shape))
        return tuple(new_shape)
    if self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return True
    assert not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )


def _set_shape_type_op_any_gemm(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Gemm."
    transA = self.get_attribute(node, "transA", exc=False)
    transB = self.get_attribute(node, "transB", exc=False)
    assert (
        len(node.input) >= 2
    ), f"Unexpected number of input {node.input} for node {node.op_type} name {node.name!r}"
    return set_type_shape_gemm(
        self,
        node.output[0],
        *node.input[:2],
        transA=0 if transA is None else transA.i,
        transB=0 if transB is None else transB.i,
    )


def _set_shape_type_op_any_matmul(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type MatMul."
    r = set_type_shape_matmul(self, node.output[0], *node.input)
    assert r is not None or not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )
    return r


def _set_shape_type_op_any_einsum(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Einsum."
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))

    equation_attr = self.get_attribute(node, "equation", exc=False)
    if equation_attr is None:
        return
    equation = equation_attr.s.decode("utf-8").strip().replace(" ", "")

    # Parse equation into input subscripts and output subscript.
    if "->" in equation:
        lhs, output_subscript = equation.split("->")
    else:
        # Implicit output: sorted unique labels that appear exactly once.
        # If ellipsis is present in any input, it is prepended to the output.
        lhs = equation
        counts = Counter(ch for ch in lhs if ch.isalpha())
        unique_labels = "".join(sorted(k for k, v in counts.items() if v == 1))
        output_subscript = ("..." if "..." in lhs else "") + unique_labels

    input_subscripts = lhs.split(",")

    # Build a mapping from label character to its dimension size.
    dim_map = {}
    ellipsis_shape = None
    for inp_name, subscript in zip(node.input, input_subscripts):
        if not self.has_shape(inp_name):
            continue
        shape = self.get_shape(inp_name)
        if "..." in subscript:
            prefix, suffix = subscript.split("...")
            prefix_len = len(prefix)
            suffix_len = len(suffix)
            ell_len = len(shape) - prefix_len - suffix_len
            if ell_len < 0:
                continue
            for i, ch in enumerate(prefix):
                dim_map[ch] = shape[i]
            if ellipsis_shape is None:
                ellipsis_shape = shape[prefix_len : prefix_len + ell_len]
            for i, ch in enumerate(suffix):
                dim_map[ch] = shape[prefix_len + ell_len + i]
        else:
            if len(subscript) == len(shape):
                for i, ch in enumerate(subscript):
                    dim_map[ch] = shape[i]

    # Compute the output shape.
    if "..." in output_subscript:
        if ellipsis_shape is not None:
            prefix, suffix = output_subscript.split("...")
            output_shape = []
            ok = True
            for ch in prefix:
                if ch not in dim_map:
                    ok = False
                    break
                output_shape.append(dim_map[ch])
            if ok:
                output_shape.extend(ellipsis_shape)
                for ch in suffix:
                    if ch not in dim_map:
                        ok = False
                        break
                    output_shape.append(dim_map[ch])
            if ok:
                self.set_shape(node.output[0], tuple(output_shape))
                return tuple(output_shape)
        # Cannot determine shape or rank when ellipsis length is unknown.
    else:
        if all(ch in dim_map for ch in output_subscript):
            output_shape = tuple(dim_map[ch] for ch in output_subscript)
            self.set_shape(node.output[0], output_shape)
            return output_shape
        # Rank is known from equation even when dimension sizes are not.
        self.set_rank(node.output[0], len(output_subscript))
        return True

    assert not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )


def _set_shape_type_op_any_non_zero(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type NonZero."
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    self.set_type(node.output[0], TensorProto.INT64)
    if self.has_rank(node.input[0]):
        new_shape = (self.get_rank(node.input[0]), self.unique_dimension_name("NEWDIM_nonzero"))
        self.set_shape(node.output[0], new_shape)
        return new_shape
    # Output is always 2D: (rank_of_input, num_nonzero_elements)
    self.set_rank(node.output[0], 2)
    assert not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )


def _set_shape_type_op_any_pad(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Pad."
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))
    else:
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )

    if self.has_shape(node.input[0]) and self.is_constant(node.input[1]):
        pads = self.compute_constant(node.input[1])[0]
        assert pads is not None or not self._debug_shape_missing, (
            f"Unable to evaluate pad={node.input[1]!r}: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
        if pads is None:
            return
        if isinstance(pads, onnx.TensorProto):
            pads = onnx.numpy_helper.to_array(pads)
        pads = pads.tolist()
        if len(node.input) > 3 and node.input[3]:
            axes = self.compute_constant(node.input[3])[0]
            assert axes is not None or not self._debug_shape_missing, (
                f"Unable to evaluate axes={node.input[3]!r}: "
                f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
            )
            if isinstance(axes, onnx.TensorProto):
                axes = onnx.numpy_helper.to_array(axes)
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
    if self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return True
    assert not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )


def _set_shape_type_op_any_range(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for for node type Range."
    # device for Range is not necessary known. It makes to host the inputs
    # on CPU but produce the output on CUDA.
    types = [self.get_type(i) for i in node.input if self.has_type(i)]
    assert types and len(set(types)) == 1, (
        f"Mixed type for node {self.pretty_node(node)}, types={types}, "
        f"unable to set shape and types."
    )
    self.set_type(node.output[0], types[0])
    self.set_rank(node.output[0], 1)
    if self.is_constant(node.input[2]) and self.get_constant(node.input[2]) == 1:
        v1 = self.value_as_shape(node.input[0])
        v2 = self.value_as_shape(node.input[1])
        assert not isinstance(
            v1, tuple
        ), f"Unexpected tuple for {node.input[0]!r} (v1={v1}){self.get_debug_msg()}"
        assert not isinstance(
            v2, tuple
        ), f"Unexpected tuple for {node.input[1]!r} (v2={v2}){self.get_debug_msg()}"
        if v1 is not None and v2 is not None:
            if isinstance(v1, int):
                if isinstance(v2, int):
                    dim = v2 - v1
                elif v1 == 0:
                    dim = v2
                else:
                    dim = simplify_expression(f"{v2}-{v1}")
            elif v2 == 0:
                dim = simplify_expression(f"-({v1})")
            else:
                dim = simplify_expression(f"{v2}-({v1})")
            self.set_shape(node.output[0], (dim,))
            return
    new_shape = (self.unique_dimension_name("NEWDIM_range"),)
    self.set_shape(node.output[0], new_shape)
    return new_shape


def _set_shape_type_op_any_reduce(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for Reduce node type."
    keepdim = self.get_attribute(node, "keepdims", exc=False)
    axes = self.get_attribute(node, "axes", exc=False)
    keepdim = None if keepdim is None else keepdim.i
    iaxes = None
    if axes is None:
        if len(node.input) == 2:
            if self.is_constant(node.input[1]):
                cst = self.get_constant(node.input[1])
                if isinstance(cst, NodeProto) and self.is_constant(cst.output[0]):
                    cst = self.get_constant(node.input[1], computed_value=True)
                if isinstance(cst, np.ndarray):
                    iaxes = (int(cst),) if len(cst.shape) == 0 else tuple(int(i) for i in cst)
                else:
                    import torch

                    assert isinstance(cst, torch.Tensor), (
                        f"Unexpected type {type(cst)} for {node.input[1]!r}, "
                        f"unable to set type and shape for node {node.op_type} "
                        f"with name={node.name!r}{self.get_debug_msg()}"
                    )
                    with self.maybe_disable_fake_tensor_mode():
                        cst = cst.cpu()
                    iaxes = (int(cst),) if len(cst.shape) == 0 else tuple(int(i) for i in cst)
            elif keepdim is not None:
                if self.has_device(node.input[0]):
                    self.set_device(node.output[0], self.get_device(node.input[0]))
                self.set_rank(node.output[0], self.get_rank(node.input[0]))
                return True
            elif self.has_shape(node.input[1]) and self.has_rank(node.input[0]):
                shape = self.get_shape(node.input[1])
                assert (
                    len(shape) == 1
                ), f"Wrong shape={shape!r} for axes={node.input[1]!r}{self.get_debug_msg()}"
                if isinstance(shape[0], int):
                    if self.has_device(node.input[0]):
                        self.set_device(node.output[0], self.get_device(node.input[0]))
                    self.set_rank(node.output[0], self.get_rank(node.input[0]) - shape[0])
                    return True
            else:
                assert (
                    self._debug_shape_missing
                ), f"Unable to determine shape for node {node}\n---\n{self.get_debug_msg()}"
                return
    else:
        iaxes = tuple(axes.ints)

    if iaxes is None:
        if len(node.input) > 1:
            if not self.is_constant(node.input[1]):
                # No constant axis.
                if self.has_device(node.input[0]):
                    self.set_device(node.output[0], self.get_device(node.input[0]))
                if self.has_shape(node.input[1]):
                    self.set_rank(node.output[0], len(self.get_shape(node.input[1])))
                if self.has_type(node.input[0]):
                    self.set_type(node.output[0], self.get_type(node.input[0]))
                return
            assert iaxes is not None, (
                f"iaxes=None when {axes=} (constant), "
                f"node.input={node.input}{self.get_debug_msg()}"
            )
        # Full reduction
        if self.has_device(node.input[0]):
            self.set_device(node.output[0], self.get_device(node.input[0]))
        self.set_shape(node.output[0], tuple())
        if self.has_type(node.input[0]):
            self.set_type(node.output[0], self.get_type(node.input[0]))
        return

    assert (
        iaxes is not None
    ), f"iaxes=None when {axes=}, node.input={node.input}{self.get_debug_msg()}"
    return set_type_shape_reduce_op(
        self, node.output[0], node.input[0], keepdim=keepdim, axes=iaxes
    )


def _compute_reshape_shape(shape1: DYNAMIC_SHAPE, shape2: DYNAMIC_SHAPE):
    if 0 in shape2:
        # 0 means keeping the dimension coming from shape1
        new_shape2 = []
        for s1, s2 in zip(shape1, shape2):
            new_shape2.append(s1 if s2 == 0 else s2)
        new_shape2.extend(shape2[len(new_shape2) :])
        shape2 = tuple(new_shape2)

    if -1 not in shape2:
        return shape2

    total_int1 = int(np.prod([i for i in shape1 if isinstance(i, int)]))
    if total_int1 == 0:
        return tuple(s if s != -1 else 0 for s in shape2)

    total_int2 = int(np.prod([i for i in shape2 if isinstance(i, int) and i != -1]))
    if total_int1 >= total_int2 and total_int1 % total_int2 == 0:
        intpart = total_int1 // total_int2
    else:
        intpart = simplify_expression(f"({total_int1})//({total_int2})")

    exist1 = {s for s in shape1 if isinstance(s, str)}
    exist2 = {s for s in shape2 if isinstance(s, str)}
    common = exist1 & exist2
    left1 = exist1 - common
    left2 = exist2 - common
    if left1 and left2:
        resp = "*".join(f"({s})" for s in sorted(left1))
        resm = "*".join(f"({s})" for s in sorted(left2))
        ok = simplify_expression(f"({resp})//({resm})")
    elif left1:
        ok = "*".join(f"({s})" for s in sorted(left1))
    elif left2:
        resm = "*".join(f"({s})" for s in sorted(left2))
        ok = simplify_expression(f"1//({resm})")
    else:
        ok = ""
    if not ok and isinstance(intpart, int):
        return tuple(s if s != -1 else intpart for s in shape2)
    if intpart != 1:
        ok = simplify_expression(f"({ok})*({intpart})") if ok else intpart
    assert ok, f"Unable to compute a shape with {shape1=} and {shape2=}."
    return tuple(s if s != -1 else ok for s in shape2)


def _set_shape_type_op_any_reshape(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Reshape."
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    k = node.output[0]
    if self.has_type(node.input[0]):
        self.set_type(k, self.get_type(node.input[0]))
    value = None
    if self.is_constant(node.input[1]):
        value = self.get_constant(node.input[1], computed_value=True, as_shape=True, exc=False)
    if value is None:
        value = self.value_as_shape(node.input[1])
    if value is not None:
        cst = tuple(value)
        if all_int_or_str(cst):
            if -1 not in cst and 0 not in cst:
                self.set_shape(k, cst)
                return cst
            if all_int(cst) and self.has_shape(node.input[0]):
                sh = self.get_shape(node.input[0])
                new_shape = self._apply_reshape_to_shape(sh, cst)
                if new_shape is not None:
                    self.set_shape(k, new_shape, allow_zero=0 in sh)
                    return new_shape
        if self.has_shape(node.input[0]):
            combined_shape = _compute_reshape_shape(self.get_shape(node.input[0]), value)
            if combined_shape is not None:
                self.set_shape(node.output[0], combined_shape, allow_zero=0 in combined_shape)
                return combined_shape

    if self.has_shape(node.input[1]):
        rk = self.get_shape(node.input[1])
        assert len(rk) == 1, f"A shape must have a rank==1 but it is {rk}{self.get_debug_msg()}"
        if isinstance(rk[0], int):
            self.set_rank(k, rk[0])
            return True
        return False
    assert not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )


def _set_shape_type_op_any_expand(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Reshape."
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    k = node.output[0]
    if self.has_type(node.input[0]):
        self.set_type(k, self.get_type(node.input[0]))
    value = None
    if self.is_constant(node.input[1]):
        value = self.get_constant(node.input[1], computed_value=True, as_shape=True, exc=False)
    if value is None:
        value = self.value_as_shape(node.input[1])
    if value is not None:
        cst = tuple(value)
        if all_int_or_str(cst):
            if -1 not in cst and 1 not in cst and 0 not in cst:
                self.set_shape(k, cst)
                return cst
            if self.has_shape(node.input[0]):
                sh = self.get_shape(node.input[0])
                new_shape = self._apply_expand_to_shape(sh, cst)
                if new_shape is not None:
                    self.set_shape(k, new_shape, allow_zero=0 in sh or 0 in new_shape)
                    return new_shape

    if self.has_shape(node.input[1]):
        rk = self.get_shape(node.input[1])
        if isinstance(rk[0], int):
            self.set_rank(k, rk[0])
            return True
    assert not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )


def _set_shape_type_op_any_sign(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Sign."
    return set_type_shape_unary_op(self, node.output[0], node.input[0])


def _set_shape_type_op_any_size(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Sign."
    self.set_type(node.output[0], TensorProto.INT64)
    self.set_shape(node.output[0], tuple())
    if self.has_shape(node.input[0]):
        shape = self.get_shape(node.input[0])
        if all(isinstance(s, int) for s in shape):
            self.set_value_shape(node.output[0], int(np.prod(shape)))
        else:
            int_part = [i for i in shape if isinstance(i, int)]
            s_part = [i for i in shape if isinstance(i, str)]
            t = "*".join(f"({s})" for s in s_part)
            self.set_value_shape(
                node.output[0],
                simplify_expression(f"{int(np.prod(int_part))}*{t}" if int_part else t),
            )
    return True


def _resolve_int_tuple_or_shape(self: ShapeBuilder, name: str):
    if self.is_constant(name):
        cst = self.get_constant(name, exc=False, computed_value=True)
        if cst is not None:
            return tuple(int(v) for v in cst)
    sv = self.value_as_shape(name)
    if isinstance(sv, tuple) and all(isinstance(x, (str, int)) for x in sv):
        return sv
    return None


def _set_shape_type_op_any_slice(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Slice."
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))

    if self.has_shape(node.input[0]):
        input_shape = self.get_shape(node.input[0])
        rank = len(input_shape)

        starts = _resolve_int_tuple_or_shape(self, node.input[1])
        ends = _resolve_int_tuple_or_shape(self, node.input[2])
        axes = _resolve_int_tuple_or_shape(self, node.input[3]) if len(node.input) > 3 else None

        if starts is None or ends is None:
            if axes is None:
                self.set_rank(node.output[0], rank)
                return True
            output_shape = list(input_shape)
            for a in axes:
                output_shape[a] = self.unique_dimension_name("NEWDIM_slice")
            new_shape = tuple(output_shape)
            self.set_shape(node.output[0], new_shape, allow_zero=0 in new_shape)
            return new_shape

        axes = (
            _resolve_int_tuple_or_shape(self, node.input[3])
            if len(node.input) > 3
            else np.arange(len(starts))
        )
        steps = _resolve_int_tuple_or_shape(self, node.input[4]) if len(node.input) > 4 else None

        output_shape = list(input_shape)
        for idx, (s, e, a) in enumerate(zip(starts, ends, axes)):
            st = steps[idx] if steps is not None and idx < len(steps) else 1
            if isinstance(e, int) and e >= 922337203685477580:
                e = input_shape[a]
            # Sentinel for "go to the beginning" with negative step (e.g. from flip).
            if isinstance(e, int) and e < -922337203685477580:
                if isinstance(st, int) and st < 0 and isinstance(s, int):
                    d = input_shape[a] if isinstance(input_shape[a], int) else None
                    s_n = s + d if s < 0 and d is not None else s
                    if isinstance(s_n, int):
                        # Number of elements: ceil((s_n + 1) / (-st))
                        output_shape[a] = (s_n + 1 + (-st) - 1) // (-st)
                        continue
                # Fallback: unable to compute size for this axis.
                output_shape[a] = self.unique_dimension_name("NEWDIM_slice")
                continue
            if isinstance(s, int) and isinstance(e, int):
                if e >= s >= 0:
                    if isinstance(st, int) and st > 0:
                        output_shape[a] = (e - s + st - 1) // st
                    elif isinstance(st, int) and st < 0:
                        # Negative step going from s down to e (exclusive).
                        output_shape[a] = max(0, (s - e - 1) // (-st) + 1)
                    else:
                        output_shape[a] = simplify_expression(f"({e-s})//({st})")
                    continue
                if isinstance(input_shape[a], int):
                    d = input_shape[a]
                    if s < 0:
                        s += d
                    if e < 0:
                        e += d
                    if isinstance(st, int) and st > 0:
                        output_shape[a] = max(0, (e - s + st - 1) // st)
                    elif isinstance(st, int) and st < 0:
                        # Negative step going from s down to e (exclusive).
                        output_shape[a] = max(0, (s - e - 1) // (-st) + 1)
                    else:
                        output_shape[a] = simplify_expression(f"({e-s})//({st})")
                    continue
            d = input_shape[a]
            if isinstance(s, int) and s < 0:
                s = f"(({d})+({e}))"
            if isinstance(e, int) and e < 0:
                e = f"(({d})+({e}))"
            output_shape[a] = simplify_expression(f"{e}-{s}" if st == 1 else f"({e}-{s})//({st})")
            continue

        new_shape = tuple(output_shape)
        self.set_shape(node.output[0], new_shape, allow_zero=0 in new_shape)
        return new_shape

    if self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return True
    assert not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )


def _set_shape_type_op_any_split(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Split."
    if self.has_device(node.input[0]):
        for o in node.output:
            self.set_device(o, self.get_device(node.input[0]))
    num_outputs = self.get_attribute(node, "num_outputs", exc=False)
    assert num_outputs is None or num_outputs.i == len(
        node.output
    ), f"Unexpected number of outputs (should be {num_outputs.i}) for node {node}"
    if not self.has_type(node.input[0]):
        # the main type is missing, cannot continue
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
        return
    dtype = self.get_type(node.input[0])
    device = self.get_device(node.input[0]) if self.has_device(node.input[0]) else None
    for o in node.output:
        self.set_type(o, dtype)
        if device is not None:
            self.set_device(o, device)
    att = self.get_attribute(node, "axis", exc=False)
    axis = 0 if att is None else att.i
    if self.has_shape(node.input[0]) and len(node.input) > 1:
        _splits_cst = None
        if self.is_constant(node.input[1]):
            spl_cst = self.get_constant(node.input[1])
            if isinstance(spl_cst, NodeProto):
                # constant is not yet evaluated; try value_as_shape instead
                sv = self.value_as_shape(node.input[1])
                if isinstance(sv, tuple) and all(isinstance(x, int) for x in sv):
                    _splits_cst = list(sv)
            elif len(spl_cst.shape) == 1:
                _splits_cst = list(self.get_constant(node.input[1]))
            else:
                split_size = int(spl_cst)
                if not self.has_shape(node.input[0]):
                    _splits_cst = None
                else:
                    shape = self.get_shape(node.input[0])
                    if isinstance(shape[axis], int):
                        div = shape[axis] // split_size
                        _splits_cst = [split_size for _ in range(div)]
                        if shape[axis] % split_size > 0:
                            _splits_cst.append(shape[axis] - div * split_size)
                    else:
                        _splits_cst = None
        else:
            sv = self.value_as_shape(node.input[1])
            if isinstance(sv, tuple) and all(isinstance(x, int) for x in sv):
                _splits_cst = list(sv)
        if _splits_cst is not None:
            splits = _splits_cst
            assert len(splits) == len(
                node.output
            ), f"Unexpected number of outputs, output={node.output} splits={splits}"
            sh = list(self.get_shape(node.input[0]))
            for i, o in enumerate(node.output):
                sh[axis] = int(splits[i])
                self.set_shape(o, tuple(sh), allow_zero=True)
            return [self.get_shape(o) for o in node.output]
    num_outputs = self.get_attribute(node, "num_outputs", exc=False)
    if num_outputs is not None:
        no = num_outputs.i
        if self.has_shape(node.input[0]):
            dim = self.get_shape(node.input[0])[axis]
            if isinstance(dim, int):
                if dim % no == 0:
                    dims = [dim // no for i in range(no)]
                else:
                    d = dim // no + 1
                    dims = [d for i in range(no - 1)]
                    dims.append(dim - d * (no - 1))
            else:
                dims = [f"({dim}+{no-1})//{no}" for i in range(no)]
                dims[-1] = (
                    f"{dim}-{no-1}*(({dim}+{no-1})//{no})"
                    if no > 2
                    else f"{dim}-({dim}+{no-1})//{no}"
                )
            li = list(self.get_shape(node.input[0]))
            for d, o in zip(dims, node.output):
                li[axis] = d
                self.set_shape(o, tuple(li))
            return [self.get_shape(o) for o in node.output]
    if self.has_rank(node.input[0]):
        rank = self.get_rank(node.input[0])
        for o in node.output:
            self.set_rank(o, rank)
        return True
    assert not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )


def _set_shape_type_op_any_scatternd(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type ScatterND."
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if not self.has_type(node.input[0]):
        # the main type is missing, cannot continue
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
        return
    dtype = self.get_type(node.input[0])
    self.set_type(node.output[0], dtype)
    if self.has_shape(node.input[0]):
        sh = self.get_shape(node.input[0])
        self.set_shape(node.output[0], sh, allow_zero=0 in sh)
        return self.get_shape(node.input[0])
    if self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return True
    assert not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )


def _set_shape_type_op_any_sequence_empty(self: ShapeBuilder, node: NodeProto):
    for att in node.attribute:
        if att.name == "dtype":
            self.set_sequence(node.output[0], dtype=att.i)
            return True
    raise AssertionError(f"Attribute 'dtype' is missong from node {node}{self.get_debug_msg()}")


def _set_shape_type_op_any_transpose(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Transpose."
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if not self.has_type(node.input[0]):
        # the main type is missing, cannot continue
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
        return
    dtype = self.get_type(node.input[0])
    self.set_type(node.output[0], dtype)
    if self.has_shape(node.input[0]):
        att = self.get_attribute(node, "perm", exc=False)
        if att is None:
            assert self.has_rank(
                node.input[0]
            ), f"Missing rank for {node.input[0]!r}{self.get_debug_msg()}"
            perm = list(range(self.get_rank(node.input[0]))[::-1])
        else:
            perm = list(att.ints)
        shape = self.get_shape(node.input[0])
        assert len(perm) == len(shape), (
            f"Mismatch between perm={perm} and shape={shape}, "
            f"for op {node.op_type!r} and name={node.name!r}"
            f"{self.get_debug_msg()}"
        )
        new_shape = list(range(len(perm)))
        for i, p in enumerate(perm):
            new_shape[i] = shape[p]
        self.set_shape(node.output[0], tuple(new_shape), allow_zero=0 in shape)
        return tuple(new_shape)
    if self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return True
    assert not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )


def _set_shape_type_op_any_tile(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Tile."
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    self.set_type(node.output[0], self.get_type(node.input[0]))
    if self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return True
    assert not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )


def _set_shape_type_op_any_topk(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type TopK."
    if self.has_device(node.input[0]):
        for o in node.output:
            self.set_device(o, self.get_device(node.input[0]))
    is_scalar = self.is_constant(node.input[1])
    if is_scalar and self.has_shape(node.input[0]):
        att = self.get_attribute(node, "axis", exc=False)
        axis = att.i if att is not None else -1
        shape = list(self.get_shape(node.input[0]))
        k = self.get_constant(node.input[1], computed_value=True)
        ki = int(k) if k.shape == tuple() else int(k[0])
        shape[axis] = ki
        shape = tuple(shape)
    else:
        shape = None

    ret_shapes = []
    if node.output[0]:
        if self.has_type(node.input[0]):
            self.set_type(node.output[0], self.get_type(node.input[0]))
        if shape is not None:
            self.set_shape(node.output[0], shape)
            ret_shapes.append(shape)
        elif self.has_rank(node.input[0]):
            self.set_rank(node.output[0], self.get_rank(node.input[0]))
        else:
            assert not self._debug_shape_missing, (
                f"Unable to compute shape for node: "
                f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
            )
    if node.output[1]:
        self.set_type(node.output[1], TensorProto.INT64)
        if shape is not None:
            self.set_shape(node.output[1], shape)
            ret_shapes.append(shape)
        elif self.has_rank(node.input[0]):
            self.set_rank(node.output[1], self.get_rank(node.input[0]))
            return True
        else:
            assert not self._debug_shape_missing, (
                f"Unable to compute shape for node: "
                f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
            )
    if ret_shapes:
        return ret_shapes


def _set_shape_type_op_any_unsqueeze(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Unsqueeze."
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if not self.has_type(node.input[0]):
        # the main type is missing, cannot continue
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
        return
    dtype = self.get_type(node.input[0])
    self.set_type(node.output[0], dtype)
    if self.has_shape(node.input[0]):
        if len(node.input) == 1:
            c = self.get_attribute(node, "axes")
            cst = np.array(c.ints, dtype=np.int64)
        elif self.is_constant(node.input[1]):
            cst = self.get_constant(node.input[1])
            if isinstance(cst, NodeProto) and cst.op_type in (
                "Constant",
                "Identity",
                "ConstantOfShape",
            ):
                cst = self.get_constant(node.input[1], computed_value=True)
        else:
            # axes is not a constant but its shape may be known
            if self.has_shape(node.input[1]):
                n_axes = self.get_shape(node.input[1])
                if n_axes and isinstance(n_axes[0], int):
                    self.set_rank(node.output[0], self.get_rank(node.input[0]) + n_axes[0])
                    return True
            assert not self._debug_shape_missing, (
                f"Unable to compute shape for node: "
                f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
            )
            return

        if isinstance(cst, np.ndarray):
            iaxes = (int(cst),) if len(cst.shape) == 0 else tuple(int(i) for i in cst)
            shape0 = self.get_shape(node.input[0])
            shape = list(shape0)
            for i in iaxes:
                shape.insert((i + len(shape) + 1) if i < 0 else i, 1)
            self.set_shape(node.output[0], tuple(shape), allow_zero=0 in shape0)
            return tuple(shape)
        elif isinstance(cst, self.torch.Tensor):
            with self.maybe_disable_fake_tensor_mode():
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
                f"with name={node.name!r}{self.get_debug_msg()}"
            )
    elif self.has_rank(node.input[0]):
        if len(node.input) == 1:
            c = self.get_attribute(node, "axes")
            n_axes = len(c.ints)
        elif self.is_constant(node.input[1]):
            cst = self.get_constant(node.input[1], computed_value=True)
            assert cst is not None, (
                f"unable to extract constant {node.input[1]!r} in node "
                f"{self.pretty_node(node)}{self.get_debug_msg()}"
            )
            n_axes = cst.size
        else:
            assert not self._debug_shape_missing, (
                f"Unable to compute shape for node: "
                f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
            )
            return
        self.set_rank(node.output[0], self.get_rank(node.input[0]) + n_axes)
        return True
    elif self.has_rank(node.input[0]) and len(node.input) > 1 and self.has_shape(node.input[1]):
        n_axes = self.get_shape(node.input[1])
        if n_axes and isinstance(n_axes[0], int):
            self.set_rank(node.output[0], self.get_rank(node.input[0]) + n_axes[0])
            return True
    else:
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )


def _set_shape_type_op_any_squeeze(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Squeeze."
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if not self.has_type(node.input[0]):
        # the main type is missing, cannot continue
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
        return
    dtype = self.get_type(node.input[0])
    self.set_type(node.output[0], dtype)
    if len(node.input) == 1 and not node.attribute:
        # No axes specified.
        if self.has_shape(node.input[0]):
            shape_x = self.get_shape(node.input[0])
            if all_int(shape_x):
                new_shape = tuple(s for s in shape_x if s != 1)
                self.set_shape(node.output[0], new_shape)
                return new_shape
        # In other cases, we cannot really determine the new shape for sure.
    elif self.has_shape(node.input[0]):
        if len(node.input) == 1:
            c = self.get_attribute(node, "axes")
            cst = np.array(c.ints, dtype=np.int64)
        elif self.is_constant(node.input[1]):
            cst = self.get_constant(node.input[1])
            if isinstance(cst, NodeProto) and cst.op_type in (
                "Constant",
                "Identity",
                "ConstantOfShape",
            ):
                cst = self.get_constant(node.input[1], computed_value=True)
        else:
            # axes is not a constant but its shape may be known
            if self.has_shape(node.input[1]):
                n_axes = self.get_shape(node.input[1])
                if n_axes and isinstance(n_axes[0], int):
                    self.set_rank(node.output[0], self.get_rank(node.input[0]) - n_axes[0])
                    return True
            assert not self._debug_shape_missing, (
                f"Unable to compute shape for node: "
                f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
            )
            return
        if isinstance(cst, np.ndarray):
            iaxes = set((int(cst),) if len(cst.shape) == 0 else tuple(int(i) for i in cst))
            shape = list(self.get_shape(node.input[0]))
            iaxes = set((i + len(shape)) % len(shape) for i in iaxes)  # for negative value
            new_shape = tuple(s for i, s in enumerate(shape) if i not in iaxes)
            self.set_shape(node.output[0], new_shape)
            return new_shape
        elif isinstance(cst, self.torch.Tensor):
            with self.maybe_disable_fake_tensor_mode():
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
                f"with name={node.name!r}{self.get_debug_msg()}"
            )
    elif self.has_rank(node.input[0]):
        if len(node.input) == 1 and node.attribute:
            c = self.get_attribute(node, "axes")
            n_axes = len(c.ints)
        elif len(node.input) > 1 and self.is_constant(node.input[1]):
            cst = self.get_constant(node.input[1], computed_value=True)
            n_axes = int(cst.numel() if hasattr(cst, "numel") else cst.size)
        else:
            assert not self._debug_shape_missing, (
                f"Unable to compute shape for node: "
                f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
            )
            return
        self.set_rank(node.output[0], self.get_rank(node.input[0]) - n_axes)
        return True
    elif self.has_rank(node.input[0]) and len(node.input) > 1 and self.has_shape(node.input[1]):
        n_axes = self.get_shape(node.input[1])
        if n_axes and isinstance(n_axes[0], int):
            self.set_rank(node.output[0], self.get_rank(node.input[0]) - n_axes[0])
            return True
    else:
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )


def _set_shape_type_op_any_where(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Where."
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if not self.has_type(node.input[2]):
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
        return
    self.set_type(node.output[0], self.get_type(node.input[2]))
    if (
        self.has_shape(node.input[0])
        and self.has_shape(node.input[1])
        and self.has_shape(node.input[2])
    ):
        sh1 = broadcast_shape(
            self.get_shape(node.input[0]), self.get_shape(node.input[1]), graph_builder=self
        )
        sh = broadcast_shape(sh1, self.get_shape(node.input[2]), graph_builder=self)
        self.set_shape(node.output[0], sh)
        return sh
    elif all(self.has_rank(i) for i in node.input):
        self.set_rank(node.output[0], max(self.get_rank(i) for i in node.input))
        return True
    else:
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )


def _set_shape_type_op_any_unary(
    self: ShapeBuilder, node: NodeProto, itype: Optional[int] = None
):
    "Sets the output shape for any unary type."
    return set_type_shape_unary_op(self, node.output[0], node.input[0], itype=itype)


def _set_shape_type_op_any_arg_max_min(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for ArgMax and ArgMin."
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    self.set_type(node.output[0], TensorProto.INT64)
    axis_att = self.get_attribute(node, "axis", exc=False)
    axis = 0 if axis_att is None else axis_att.i
    keepdim_att = self.get_attribute(node, "keepdims", exc=False)
    keepdim = 1 if keepdim_att is None else keepdim_att.i
    if self.has_shape(node.input[0]):
        shape = list(self.get_shape(node.input[0]))
        if keepdim:
            shape[axis] = 1
        else:
            del shape[axis]
        new_shape = tuple(shape)
        self.set_shape(node.output[0], new_shape)
        return new_shape
    if self.has_rank(node.input[0]):
        rk = self.get_rank(node.input[0])
        self.set_rank(node.output[0], rk if keepdim else rk - 1)
        return True
    assert not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )


def _set_shape_type_op_any_global_pool(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for GlobalAveragePool and GlobalMaxPool."
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))
    if self.has_shape(node.input[0]):
        shape = self.get_shape(node.input[0])
        new_shape = shape[:2] + (1,) * (len(shape) - 2)
        self.set_shape(node.output[0], new_shape)
        return new_shape
    if self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return True
    assert not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )


def _set_shape_type_op_any_flatten(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for Flatten."
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))
    axis_att = self.get_attribute(node, "axis", exc=False)
    axis = 1 if axis_att is None else axis_att.i
    if self.has_shape(node.input[0]):
        shape = self.get_shape(node.input[0])
        if axis < 0:
            axis = len(shape) + axis
        dims_before = shape[:axis]
        dims_after = shape[axis:]
        if all_int(dims_before):
            d1 = 1
            for d in dims_before:
                d1 *= d
        else:
            d1 = "*".join(map(str, dims_before)) if dims_before else "1"
        if all_int(dims_after):
            d2 = 1
            for d in dims_after:
                d2 *= d
        else:
            d2 = "*".join(map(str, dims_after)) if dims_after else "1"
        new_shape = (d1, d2)
        self.set_shape(node.output[0], new_shape)
        return new_shape
    self.set_rank(node.output[0], 2)
    return True


def _set_shape_type_op_any_eyelike(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for EyeLike."
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    dtype_att = self.get_attribute(node, "dtype", exc=False)
    if dtype_att is not None:
        self.set_type(node.output[0], dtype_att.i)
    elif self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))
    if self.has_shape(node.input[0]):
        self.set_shape(node.output[0], self.get_shape(node.input[0]))
        return self.get_shape(node.input[0])
    if self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return True
    assert not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )


def _set_shape_type_op_any_depth_to_space(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for DepthToSpace."
    blocksize = self.get_attribute(node, "blocksize").i
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))
    if self.has_shape(node.input[0]):
        shape = self.get_shape(node.input[0])
        n, c = shape[0], shape[1]
        spatial = shape[2:]
        b_pow = blocksize ** len(spatial)
        new_c = (c // b_pow) if isinstance(c, int) else simplify_expression(f"({c})//({b_pow})")
        new_spatial = tuple(
            (s * blocksize if isinstance(s, int) else simplify_expression(f"({s})*({blocksize})"))
            for s in spatial
        )
        new_shape = (n, new_c, *new_spatial)
        self.set_shape(node.output[0], new_shape)
        return new_shape
    if self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return True
    assert not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )


def _set_shape_type_op_any_gridsample(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for GridSample."
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))
    if self.has_shape(node.input[0]) and self.has_shape(node.input[1]):
        x_shape = self.get_shape(node.input[0])
        grid_shape = self.get_shape(node.input[1])
        # Output: (N, C, *grid_spatial) where grid_spatial = grid[1:-1]
        new_shape = (x_shape[0], x_shape[1], *grid_shape[1:-1])
        self.set_shape(node.output[0], new_shape)
        return new_shape
    if self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return True
    assert not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )


def _set_shape_type_op_any_space_to_depth(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for SpaceToDepth."
    blocksize = self.get_attribute(node, "blocksize").i
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))
    if self.has_shape(node.input[0]):
        shape = self.get_shape(node.input[0])
        n, c = shape[0], shape[1]
        spatial = shape[2:]
        b_pow = blocksize ** len(spatial)
        new_c = (c * b_pow) if isinstance(c, int) else simplify_expression(f"({c})*({b_pow})")
        new_spatial = tuple(
            (
                s // blocksize
                if isinstance(s, int)
                else simplify_expression(f"({s})//({blocksize})")
            )
            for s in spatial
        )
        new_shape = (n, new_c, *new_spatial)
        self.set_shape(node.output[0], new_shape)
        return new_shape
    if self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return True
    assert not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )


def _set_shape_type_op_any_window(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for BlackmanWindow, HannWindow, HammingWindow."
    dtype = self.get_attribute_with_default(node, "output_datatype", TensorProto.FLOAT)
    self.set_type(node.output[0], dtype)
    if self.is_constant(node.input[0]):
        cst = self.get_constant(node.input[0], exc=False, computed_value=True)
        if cst is not None:
            size = int(cst.flat[0])
            self.set_shape(node.output[0], (size,))
            return
    self.set_rank(node.output[0], 1)


def _set_shape_type_op_any_resize(self: ShapeBuilder, node: NodeProto):
    "Sets the output shape for node type Resize."
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))
    else:
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )

    # input[3] = sizes (takes priority over scales when non-empty)
    if len(node.input) > 3 and node.input[3] and self.is_constant(node.input[3]):
        sizes = self.get_constant(node.input[3], computed_value=True)
        if sizes is not None and sizes.size > 0:
            new_shape = tuple(int(d) for d in sizes.tolist())
            self.set_shape(node.output[0], new_shape)
            return new_shape

    # input[2] = scales
    if (
        len(node.input) > 2
        and node.input[2]
        and self.is_constant(node.input[2])
        and self.has_shape(node.input[0])
    ):
        scales = self.get_constant(node.input[2], computed_value=True)
        if scales is not None and scales.size > 0:
            shape = self.get_shape(node.input[0])
            new_shape = tuple(
                (
                    int(np.floor(d * s))
                    if isinstance(d, int)
                    else f"int(floor({simplify_expression(f'({d})*({s})')}))"
                )
                for d, s in zip(shape, scales.tolist())
            )
            self.set_shape(node.output[0], new_shape)
            return new_shape

    if self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return True
    assert not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )


def _set_shape_type_op_any_identity(self: ShapeBuilder, node: NodeProto):
    "Sets the output type and shape for Identity (passthrough)."
    return set_type_shape_unary_op(self, node.output[0], node.input[0])


def _set_shape_type_op_any_shape(self: ShapeBuilder, node: NodeProto):
    "Sets the output type and rank for Shape op."
    if not self.has_device(node.output[0]):
        self.set_device(node.output[0], -1)
    if not self.has_type(node.output[0]):
        self.set_type(node.output[0], TensorProto.INT64)
    if self.has_rank(node.input[0]):
        rk = self.get_rank(node.input[0])
        self.set_shape(node.output[0], (rk,))
        return (rk,)
    self.set_rank(node.output[0], 1)
    return True


def _set_shape_type_op_any_constantofshape(self: ShapeBuilder, node: NodeProto):
    "Sets the output type for ConstantOfShape."
    if node.attribute and node.attribute[0].name == "value":
        itype = node.attribute[0].t.data_type
    else:
        itype = TensorProto.FLOAT
    if not self.has_type(node.output[0]):
        self.set_type(node.output[0], itype)
    return True


def _set_shape_type_op_any_gathernd(self: ShapeBuilder, node: NodeProto):
    "Sets the output type for GatherND."
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))
    return True


def _set_shape_type_op_any_onehot(self: ShapeBuilder, node: NodeProto):
    "Sets the output type for OneHot (type from values input)."
    if len(node.input) >= 3 and self.has_type(node.input[2]):
        self.set_type(node.output[0], self.get_type(node.input[2]))
    else:
        self.set_type(node.output[0], TensorProto.FLOAT)
    return True


def _set_shape_type_op_any_qlinear_matmul(self: ShapeBuilder, node: NodeProto):
    "Sets the output type for QLinearMatMul."
    if len(node.input) >= 6 and self.has_type(node.input[5]):
        self.set_type(node.output[0], self.get_type(node.input[5]))
    elif len(node.input) >= 8 and self.has_type(node.input[7]):
        self.set_type(node.output[0], self.get_type(node.input[7]))
    else:
        self.set_type(node.output[0], TensorProto.UINT8)
    return True


def _set_shape_type_op_any_nms(self: ShapeBuilder, node: NodeProto):
    "Sets the output type for NonMaxSuppression (output is INT64)."
    self.set_type(node.output[0], TensorProto.INT64)
    return True


def _set_shape_type_op_any_rnn(self: ShapeBuilder, node: NodeProto):
    "Sets the output types for RNN/GRU/LSTM."
    itype = self.get_type(node.input[0]) if self.has_type(node.input[0]) else TensorProto.FLOAT
    for out in node.output:
        if out:
            self.set_type(out, itype)
    return True


def _set_shape_type_op_any_mel_weight_matrix(self: ShapeBuilder, node: NodeProto):
    "Sets the output type for MelWeightMatrix."
    for att in node.attribute:
        if att.name == "output_datatype":
            self.set_type(node.output[0], att.i)
            return True
    self.set_type(node.output[0], TensorProto.FLOAT)
    return True


def _set_shape_type_op_any_loop(self: ShapeBuilder, node: NodeProto):
    """Sets output types and shapes for the Loop operator from its body subgraph.

    The Loop body subgraph has the following output order:
    ``[cond_out, v_out_0, ..., v_out_K-1, scan_0, ..., scan_S-1]``

    The Loop node outputs are:
    ``[v_final_0, ..., v_final_K-1, scan_0_stacked, ..., scan_S-1_stacked]``

    Loop-carried output ``v_final_i`` has the same type and shape as
    ``body.output[i+1]``.  Each scan output has the same element type as the
    corresponding body scan output but with an extra leading dimension (the
    number of iterations, which is usually unknown at compile time).
    """
    body_graph = None
    for att in node.attribute:
        if att.type == onnx.AttributeProto.GRAPH:
            body_graph = att.g
            break
    if body_graph is None:
        return None

    # Number of loop-carried dependencies (excluding M and cond from inputs)
    n_loop_carried = max(0, len(node.input) - 2)

    # body.output[0] is cond_out; body.output[1..] maps to loop node outputs
    body_outputs = list(body_graph.output)
    # skip cond_out at index 0
    mapped_body_outputs = body_outputs[1:]

    # Pre-infer body output types when any are missing (elem_type == 0).
    # Body inputs: iter→INT64, cond_in→BOOL, loop-carried→from node inputs[2+]
    body_out_elem_types = [
        bo.type.tensor_type.elem_type if bo.type.HasField("tensor_type") else 0
        for bo in body_graph.output
    ]
    # body_out_elem_types[0] is cond_out (always BOOL); check the remaining
    # output types (index 1+) which map directly to Loop node outputs.
    if any(t == 0 for t in body_out_elem_types[1:]):
        body_input_types = [TensorProto.INT64, TensorProto.BOOL] + [  # iter  # cond_in
            # inp may be "" for optional omitted inputs; treat those as unknown (0)
            self.get_type(inp) if inp else 0
            for inp in list(node.input)[2:]
        ]
        inferred = _infer_types_body(body_graph, body_input_types)
        body_out_elem_types = [
            inferred.get(
                bo.name, bo.type.tensor_type.elem_type if bo.type.HasField("tensor_type") else 0
            )
            for bo in body_graph.output
        ]
    # mapped_body_elem_types[i] → elem type for Loop output[i]
    mapped_body_elem_types = body_out_elem_types[1:]

    for i, out_name in enumerate(node.output):
        if not out_name:
            continue
        if i >= len(mapped_body_outputs):
            break
        body_out = mapped_body_outputs[i]
        elem_type = mapped_body_elem_types[i] if i < len(mapped_body_elem_types) else 0
        if elem_type:
            self.set_type(out_name, elem_type)

        if body_out.type.HasField("tensor_type") and body_out.type.tensor_type.HasField("shape"):
            body_shape = tuple(
                d.dim_param if d.dim_param else d.dim_value
                for d in body_out.type.tensor_type.shape.dim
            )
            if i < n_loop_carried:
                # v_final: same shape as the body output
                self.set_shape(out_name, body_shape)
            else:
                # scan output: prepend an unknown leading dimension
                scan_dim = self.unique_dimension_name(f"loop_{out_name}_iters")
                self.set_shape(out_name, (scan_dim, *body_shape))

    return True


def _set_shape_type_op_any_softmax_cross_entropy_loss(
    self: "ShapeBuilder", node: NodeProto
) -> bool:
    """Sets shape/type for SoftmaxCrossEntropyLoss.

    Output 0 shape depends on the ``reduction`` attribute:

    * ``"none"``: batch shape = input[0] shape without the class dimension (dim 1),
      i.e. ``(N,)`` for 2-D input or ``(N, d1, ..., dk)`` for higher-rank input.
    * ``"mean"`` / ``"sum"``: scalar ``()``.

    Output 1 (log_prob, optional): same shape as input[0].
    """
    if not node.input or not self.has_type(node.input[0]):
        return False
    itype = self.get_type(node.input[0])
    reduction_attr = self.get_attribute(node, "reduction", exc=False)
    reduction = reduction_attr.s.decode() if reduction_attr is not None else "mean"

    # Output 0
    out0 = node.output[0]
    self.set_type(out0, itype)
    if reduction == "none":
        if self.has_shape(node.input[0]):
            scores_shape = self.get_shape(node.input[0])
            # Remove the class dimension (index 1); output shape is (N, d1,...,dk).
            if len(scores_shape) > 2:
                out_shape = (scores_shape[0], *scores_shape[2:])
            else:
                out_shape = (scores_shape[0],)
            self.set_shape(out0, out_shape)
        elif self.has_rank(node.input[0]):
            self.set_rank(out0, self.get_rank(node.input[0]) - 1)
    else:
        self.set_shape(out0, ())

    # Output 1 (log_prob) — optional
    if len(node.output) > 1 and node.output[1]:
        out1 = node.output[1]
        self.set_type(out1, itype)
        if self.has_shape(node.input[0]):
            self.set_shape(out1, self.get_shape(node.input[0]))
        elif self.has_rank(node.input[0]):
            self.set_rank(out1, self.get_rank(node.input[0]))
    return True


_set_shape_type_op_any_known = {
    "ArgMax": _set_shape_type_op_any_arg_max_min,
    "ArgMin": _set_shape_type_op_any_arg_max_min,
    "Attention": _set_shape_type_op_any_attention,
    "BatchNormalization": _set_shape_type_op_any_batch_normalization,
    "BlackmanWindow": _set_shape_type_op_any_window,
    "Cast": _set_shape_type_op_any_cast,
    "BitCast": _set_shape_type_op_any_cast,
    "Clip": _set_shape_type_op_any_unary,
    "Compress": _set_shape_type_op_any_compress,
    "Concat": _set_shape_type_op_any_concat,
    "ConstantOfShape": _set_shape_type_op_any_constantofshape,
    "Conv": _set_shape_type_op_any_conv_max_pool,
    "CumProd": _set_shape_type_op_any_unary,
    "CumSum": _set_shape_type_op_any_unary,
    "DepthToSpace": _set_shape_type_op_any_depth_to_space,
    "Dropout": _set_shape_type_op_any_dropout,
    "Einsum": _set_shape_type_op_any_einsum,
    "EyeLike": _set_shape_type_op_any_eyelike,
    "Expand": _set_shape_type_op_any_expand,
    "Flatten": _set_shape_type_op_any_flatten,
    "Gather": _set_shape_type_op_any_gather,
    "GatherElements": _set_shape_type_op_any_gather_elements,
    "GatherND": _set_shape_type_op_any_gathernd,
    "Gelu": _set_shape_type_op_any_unary,
    "HammingWindow": _set_shape_type_op_any_window,
    "HannWindow": _set_shape_type_op_any_window,
    "Gemm": _set_shape_type_op_any_gemm,
    "GlobalAveragePool": _set_shape_type_op_any_global_pool,
    "GlobalMaxPool": _set_shape_type_op_any_global_pool,
    "GridSample": _set_shape_type_op_any_gridsample,
    "Identity": _set_shape_type_op_any_identity,
    "InstanceNormalization": _set_shape_type_op_any_instance_normalization,
    "IsInf": lambda *args: _set_shape_type_op_any_unary(*args, itype=TensorProto.BOOL),
    "IsNaN": lambda *args: _set_shape_type_op_any_unary(*args, itype=TensorProto.BOOL),
    "LayerNormalization": _set_shape_type_op_any_layer_normalization,
    "LpNormalization": _set_shape_type_op_any_lp_normalization,
    "Log": _set_shape_type_op_any_unary,
    "LogSoftmax": _set_shape_type_op_any_unary,
    "MatMul": _set_shape_type_op_any_matmul,
    "MaxPool": _set_shape_type_op_any_conv_max_pool,
    "MelWeightMatrix": _set_shape_type_op_any_mel_weight_matrix,
    "Loop": _set_shape_type_op_any_loop,
    "NonMaxSuppression": _set_shape_type_op_any_nms,
    "NonZero": _set_shape_type_op_any_non_zero,
    "OneHot": _set_shape_type_op_any_onehot,
    "Pad": _set_shape_type_op_any_pad,
    "QLinearMatMul": _set_shape_type_op_any_qlinear_matmul,
    "Range": _set_shape_type_op_any_range,
    "Reshape": _set_shape_type_op_any_reshape,
    "Resize": _set_shape_type_op_any_resize,
    "RNN": _set_shape_type_op_any_rnn,
    "RotaryEmbedding": _set_shape_type_op_any_rotary_embedding,
    "RMSNormalization": _set_shape_type_op_any_layer_normalization,
    "ScatterND": _set_shape_type_op_any_scatternd,
    "SequenceEmpty": _set_shape_type_op_any_sequence_empty,
    "Shape": _set_shape_type_op_any_shape,
    "Sign": _set_shape_type_op_any_sign,
    "SimplifiedLayerNormalization": _set_shape_type_op_any_layer_normalization,
    "Size": _set_shape_type_op_any_size,
    "Slice": _set_shape_type_op_any_slice,
    "Softmax": _set_shape_type_op_any_unary,
    "SoftmaxCrossEntropyLoss": _set_shape_type_op_any_softmax_cross_entropy_loss,
    "Swish": _set_shape_type_op_any_unary,
    "SpaceToDepth": _set_shape_type_op_any_space_to_depth,
    "Split": _set_shape_type_op_any_split,
    "Squeeze": _set_shape_type_op_any_squeeze,
    "Tile": _set_shape_type_op_any_tile,
    "TopK": _set_shape_type_op_any_topk,
    "Transpose": _set_shape_type_op_any_transpose,
    "Unsqueeze": _set_shape_type_op_any_unsqueeze,
    "Upsample": _set_shape_type_op_any_resize,
    "Where": _set_shape_type_op_any_where,
}


def set_shape_type_op_any(self: ShapeBuilder, node: NodeProto, exc: bool = False):
    """Sets the shape and type if it can."""
    if node.op_type.startswith("Reduce"):
        return _set_shape_type_op_any_reduce(self, node)
    if node.op_type in _set_shape_type_op_any_known:
        fct = _set_shape_type_op_any_known[node.op_type]
        return fct(self, node)
    if node.op_type in self._op_type_element_wise_cmp_types:
        r = set_type_shape_binary_op(self, node.output[0], *node.input, cmp_op=True)
        assert r is not None or not self._debug_shape_missing, (
            f"No function to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
        return r
    if node.op_type in self._op_type_element_wise_types:
        r = set_type_shape_binary_op(self, node.output[0], *node.input)
        assert r is not None or not self._debug_shape_missing, (
            f"No function to compute shape for node {node.op_type!r}"
            f"\ninput shapes are "
            f"{[(self.get_shape(i) if self.has_shape(i) else '?') for i in node.input if i]}"
            f"\nvalue shapes are "
            f"{[self.value_as_shape(i) for i in node.input if i]}"
            f"\nnode is {self.pretty_node(node)}{self.get_debug_msg()}"
        )
        return r
    if node.op_type in {"DequantizeLinear", "DynamicQuantizeLinear"}:
        raise AssertionError(
            f"set_shape_type_op_any not implemented for "
            f"{node.op_type!r}{self.get_debug_msg()}"
        )
    if node.op_type in {"CastLike"}:
        r = set_type_shape_binary_op(
            self, node.output[0], node.input[0], itype=self.get_type(node.input[1])
        )
        assert r is not None or not self._debug_shape_missing, (
            f"No function to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
        return r
    if node.op_type in {"Pow"}:
        r = set_type_shape_binary_op(self, node.output[0], *node.input)
        assert r is not None or not self._debug_shape_missing, (
            f"No function to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
        return r
    if node.op_type in self._op_type_unary_like:
        if node.op_type == "Abs":
            r = set_type_shape_unary_op_abs(self, node.output[0], node.input[0])
        else:
            r = set_type_shape_unary_op(self, node.output[0], node.input[0])
        assert r is not None or not self._debug_shape_missing, (
            f"No function to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
        return r
    if node.op_type in {"ScatterElements", "ScatterND"}:
        r = set_type_shape_unary_op(self, node.output[0], node.input[0])
        assert r is not None or not self._debug_shape_missing, (
            f"No function to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
        return r
    if node.op_type not in {"Constant", "ConstantOfShape", "Identity", "Reshape", "Shape"}:
        # Some nodes are handled when the node is created such as Identity.
        assert not self._debug_shape_missing, (
            f"No function to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
    if exc:
        raise NotImplementedError(f"No shape function for node type {node.op_type!r}")


def set_type_shape_fused_matmul(self: ShapeBuilder, node: NodeProto):
    """Sets the output shape for node type FusedMatMul."""
    x, y = node.input[:2]
    transA = self.get_attribute(node, "transA", exc=False)
    transA = transA.i if transA else 0
    transB = self.get_attribute(node, "transB", exc=False)
    transB = transB.i if transB else 0
    name = node.output[0]
    if transA == 0 and transB == 0:
        return set_type_shape_matmul(self, name, x, y)
    if self.has_device(x) and self.has_device(y) and self.get_device(x) == self.get_device(y):
        self.set_device(name, self.get_device(x))
    elif self.has_device(x):
        self.set_device(name, self.get_device(x))
    if self.has_type(x):
        self.set_type(name, self.get_type(x))
    elif self.has_type(y):
        self.set_type(name, self.get_type(y))
    if self.has_shape(x) and self.has_shape(y):
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
        new_shape = prefix + new_shape
        self.set_shape(name, new_shape)
        return new_shape
    elif self.has_rank(x) and self.has_rank(y):
        self.set_rank(name, max(self.get_rank(x), self.get_rank(y)))


def set_type_shape_tree_ensemble(self: ShapeBuilder, node: NodeProto):
    """Sets the output shape for node types TreeEnsemble and TreeEnsembleRegressor."""
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
        if len(node.output) > 1:
            self.set_device(node.output[1], self.get_device(node.input[0]))
    assert self.has_opset("ai.onnx.ml"), f"Opset ai.onnx.ml is missing{self.get_debug_msg()}"
    assert self.get_opset("ai.onnx.ml") >= 1, (
        f"Only opset >= 1 is supported but opset for ai.onnx.ml is "
        f"{self.get_opset('ai.onnx.ml')}{self.get_debug_msg()}"
    )
    if node.op_type == "TreeEnsembleClassifier":
        self.set_type(node.output[0], TensorProto.INT64)
        self.set_type(node.output[1], TensorProto.FLOAT)
        att = self.get_attribute(node, "classlabels_int64s", exc=False)
        if not att:
            att = self.get_attribute(node, "classlabels_strings", exc=False)
            assert att is not None, (
                f"No attribute classlabels_int64s or classlabels_strings "
                f"is set for node {self.pretty_node(node)}{self.get_debug_msg()}"
            )
            n_targets = len(att.strings)
        else:
            n_targets = len(att.ints)
    elif node.op_type == "TreeEnsembleRegressor":
        self.set_type(node.output[0], TensorProto.FLOAT)
        n_targets = self.get_attribute(node, "n_targets").i
    elif node.op_type == "TreeEnsemble":
        self.set_type(node.output[0], self.get_type(node.input[0]))
        n_targets = self.get_attribute(node, "n_targets").i
    else:
        raise AssertionError(f"Unexpected type {node.op_type!r}{self.get_debug_msg()}")

    assert n_targets is not None, (
        f"Unable to extract the dimension of the output for node type "
        f"{node.op_type!r} and name={node.name!r}"
    )
    if self.has_shape(node.input[0]):
        shape = self.get_shape(node.input[0])
        new_shape = (shape[0], n_targets)
        self.set_shape(node.output[0], new_shape)
        return new_shape
    else:
        self.set_rank(node.output[0], 2)


def set_type_shape_to_complex(self: ShapeBuilder, node: NodeProto):
    """Sets the output shape for node type ToComplex (converts float to complex)."""
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if self.has_type(node.input[0]):
        dtype = self.get_type(node.input[0])
        mapping = {
            TensorProto.FLOAT: TensorProto.COMPLEX64,
            TensorProto.DOUBLE: TensorProto.COMPLEX128,
        }
        assert (
            dtype in mapping
        ), f"Unexpected type {dtype} for node {node.op_type}{self.get_debug_msg()}"
        self.set_type(node.output[0], mapping[dtype])
    if self.has_shape(node.input[0]):
        new_shape = self.get_shape(node.input[0])[:-1]
        self.set_shape(node.output[0], new_shape)
        return new_shape
    if self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]) - 1)
    return True


def set_type_shape_complex_module(self: ShapeBuilder, node: NodeProto):
    """Sets the output shape for node type ComplexModule (extracts real/imaginary part)."""
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if self.has_type(node.input[0]):
        dtype = self.get_type(node.input[0])
        mapping = {
            TensorProto.COMPLEX64: TensorProto.FLOAT,
            TensorProto.COMPLEX128: TensorProto.DOUBLE,
        }
        assert (
            dtype in mapping
        ), f"Unexpected type {dtype} for node {node.op_type}{self.get_debug_msg()}"
        self.set_type(node.output[0], mapping[dtype])
    if self.has_shape(node.input[0]):
        self.set_shape(node.output[0], self.get_shape(node.input[0]))
    elif self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
    return True


def set_type_shape_shared_input(self: ShapeBuilder, node: NodeProto):
    """Sets the output shapes for nodes with two outputs sharing the same inputs."""
    r1 = set_type_shape_binary_op(self, node.output[0], *node.input[:2])
    r2 = set_type_shape_binary_op(self, node.output[1], *node.input[::2])
    if r1 or r2:
        return [r1, r2]


def set_type_shape_scatter_nd_of_shape(self: ShapeBuilder, node: NodeProto):
    """Sets the output shape for node types ScatterNDOfShape and MaskedScatterNDOfShape."""
    if self.has_device(node.input[2]):
        self.set_device(node.output[0], self.get_device(node.input[2]))
    if self.has_type(node.input[2]):
        self.set_type(node.output[0], self.get_type(node.input[2]))
    value = self.value_as_shape(node.input[0])
    if value is not None:
        self.set_shape(node.output[0], tuple(value))
        return tuple(value)


def set_type_shape_tri_matrix(self: ShapeBuilder, node: NodeProto):
    """Sets the output shape for node type TriMatrix."""
    if self.has_device(node.input[1]):
        self.set_device(node.output[0], self.get_device(node.input[1]))
    if self.has_type(node.input[1]):
        self.set_type(node.output[0], self.get_type(node.input[1]))
    value = self.value_as_shape(node.input[0])
    if value is not None:
        tvalue = tuple(value)
        self.set_shape(node.output[0], tvalue)
        return tvalue


def set_type_shape_transpose_2d_cast_fp16(self: ShapeBuilder, node: NodeProto):
    """
    Sets the output shape for node type Transpose2DCastFP16
    (transposes and casts to float16).
    """
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    self.set_type(node.output[0], TensorProto.FLOAT16)
    if self.has_shape(node.input[0]):
        shape = self.get_shape(node.input[0])
        self.set_shape(node.output[0], shape[::-1])
        return shape[::-1]


def set_type_shape_transpose_2d_cast_fp32(self: ShapeBuilder, node: NodeProto):
    """
    Sets the output shape for node type Transpose2DCastFP32
    (transposes and casts to float32).
    """
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    self.set_type(node.output[0], TensorProto.FLOAT)
    if self.has_shape(node.input[0]):
        shape = self.get_shape(node.input[0])
        self.set_shape(node.output[0], shape[::-1])
        return shape[::-1]


def set_type_shape_multi_head_attention(self: ShapeBuilder, node: NodeProto):
    """Sets the output shape for node type MultiHeadAttention."""
    if self.has_device(node.input[0]):
        for o in node.output:
            if o:
                self.set_device(o, self.get_device(node.input[0]))
    itype = self.get_type(node.input[0])
    for o in node.output:
        self.set_type(o, itype)
    if (
        self.has_shape(node.input[0])
        and self.has_shape(node.input[1])
        and self.has_shape(node.input[2])
    ):
        assert (
            self.get_rank(node.input[0]) == 3
        ), f"rank(query)={self.get_rank(node.input[0])} != 3{self.get_debug_msg()}"
        q_shape, _k_shape, _v_shape = [self.get_shape(i) for i in node.input[:3]]
        pk_shape = (
            self.get_shape(node.input[6])
            if len(node.input) > 6 and node.input[6] and self.has_shape(node.input[6])
            else None
        )
        if pk_shape is not None:
            up = []
            self.set_shape(node.output[0], q_shape)
            up.append(q_shape)
            d1, d2 = q_shape[1], pk_shape[2]
            if isinstance(d1, int) and isinstance(d2, int):
                d = d1 + d2
            else:
                d = simplify_expression(f"({d1})+({d2})")
            shape = (*pk_shape[:2], d, pk_shape[-1])
            for o in node.output[1:]:
                if o:
                    self.set_shape(o, shape)
                    up.append(shape)
            return up
    self.set_rank(node.output[0], 3)
    for o in node.output[1:]:
        if o:
            self.set_rank(o, 4)


def set_type_shape_bias_split_gelu(g: "ShapeBuilder", node: "NodeProto"):  # noqa: F821
    """Sets the shape and type for ``com.microsoft.BiasSplitGelu``.

    The operator computes ``Y = left * Gelu(right)`` after adding a bias and
    splitting the last dimension into two equal halves, so the output shape
    equals the input shape with the last dimension halved.
    """
    out = node.output[0]
    inp = node.input[0]
    if g.has_device(inp):
        g.set_device(out, g.get_device(inp))
    if g.has_type(inp):
        g.set_type(out, g.get_type(inp))
    if g.has_shape(inp):
        shape = g.get_shape(inp)
        last = shape[-1]
        if isinstance(last, int):
            new_last = last // 2
        else:
            new_last = f"({last})//2"
        g.set_shape(out, (*shape[:-1], new_last))
    elif g.has_rank(inp):
        g.set_rank(out, g.get_rank(inp))


_set_shape_type_op_any_custom = {
    "AddAdd": lambda g, node: set_type_shape_binary_op(g, node.output[0], *node.input),
    "AddMul": lambda g, node: set_type_shape_binary_op(g, node.output[0], *node.input),
    "AddSharedInput": set_type_shape_shared_input,
    "BiasGelu": lambda g, node: set_type_shape_unary_op(g, node.output[0], node.input[0]),
    "BiasSoftmax": lambda g, node: set_type_shape_unary_op(g, node.output[0], node.input[0]),
    "BiasSplitGelu": lambda g, node: set_type_shape_bias_split_gelu(g, node),
    "ComplexModule": set_type_shape_complex_module,
    "ComplexMul": lambda g, node: set_type_shape_binary_op(g, node.output[0], *node.input),
    "ComplexMulConj": lambda g, node: set_type_shape_binary_op(g, node.output[0], *node.input),
    "FastGelu": lambda g, node: set_type_shape_unary_op(g, node.output[0], node.input[0]),
    "FusedMatMul": set_type_shape_fused_matmul,
    "FusedMatMulActivation": set_type_shape_fused_matmul,
    "FusedConv": _set_shape_type_op_any_conv_max_pool,
    "Gelu": lambda g, node: set_type_shape_unary_op(g, node.output[0], node.input[0]),
    "GemmFastGelu": lambda g, node: set_type_shape_matmul(g, node.output[0], *node.input[:2]),
    "GemmaRotaryEmbedding": _set_shape_type_op_any_gemma_rotary_embedding,
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


def _set_shape_type_packed_multi_head_attention(self: ShapeBuilder, node: NodeProto):
    """Sets the output shape for ``com.microsoft.PackedMultiHeadAttention``."""
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))
    if self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]))


def _set_shape_type_attention_microsoft(self: ShapeBuilder, node: NodeProto):
    """Sets the output shape for ``com.microsoft.Attention``."""
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))
    if self.has_shape(node.input[0]):
        self.set_shape(node.output[0], self.get_shape(node.input[0]))
    elif self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]))


def _set_shape_type_group_query_attention(self: ShapeBuilder, node: NodeProto):
    """Sets the output shape for ``com.microsoft.GroupQueryAttention``."""
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))
    if self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]))


def _set_shape_type_murmur_hash3(self: ShapeBuilder, node: NodeProto):
    """Sets the output shape for ``com.microsoft.MurmurHash3``."""
    self.set_type(node.output[0], TensorProto.INT32)
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if self.has_shape(node.input[0]):
        self.set_shape(node.output[0], self.get_shape(node.input[0]))
    elif self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]))


def _set_shape_type_cdist(self: ShapeBuilder, node: NodeProto):
    """Sets the output shape for ``com.microsoft.CDist``.

    Input A has shape ``(N, D)`` and input B has shape ``(M, D)``, so the
    output has shape ``(N, M)``.
    """
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if self.has_shape(node.input[0]) and self.has_shape(node.input[1]):
        shape_a = self.get_shape(node.input[0])
        shape_b = self.get_shape(node.input[1])
        self.set_shape(node.output[0], (shape_a[0], shape_b[0]))
    elif self.has_rank(node.input[0]):
        self.set_rank(node.output[0], 2)


def _set_shape_type_relative_position_bias(self: ShapeBuilder, node: NodeProto):
    """Sets the output shape for ``com.microsoft.RelativePositionBias``.

    Inputs: ``bias_table (num_heads, num_buckets)``, ``query_length ()``,
    ``key_length ()``.  Output: ``(1, num_heads, query_length, key_length)``.
    """
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if self.has_shape(node.input[0]):
        bias_shape = self.get_shape(node.input[0])
        num_heads = bias_shape[0]
        q_val = self.value_as_shape(node.input[1])
        k_val = self.value_as_shape(node.input[2])
        q_dim = q_val[0] if q_val is not None and len(q_val) == 1 else node.input[1]
        k_dim = k_val[0] if k_val is not None and len(k_val) == 1 else node.input[2]
        self.set_shape(node.output[0], (1, num_heads, q_dim, k_dim))
    else:
        self.set_rank(node.output[0], 4)


def _set_shape_type_embed_layer_normalization(self: ShapeBuilder, node: NodeProto):
    """Sets the output shape for ``com.microsoft.EmbedLayerNormalization``.

    Inputs: ``input_ids [B, S]``, ``segment_ids [B, S]`` (optional),
    ``word_embedding [V, D]``, ``position_embedding [P, D]``,
    ``segment_embedding [NS, D]`` (optional), ``gamma [D]``, ``beta [D]``,
    ``mask [B, S]`` (optional), ``position_ids [B, S]`` (optional).
    Outputs: ``output [B, S, D]``, ``mask_index [B]``.
    """
    word_emb_input = node.input[2] if len(node.input) > 2 else ""
    dtype = (
        self.get_type(word_emb_input)
        if word_emb_input and self.has_type(word_emb_input)
        else None
    )
    if dtype is not None and node.output[0]:
        self.set_type(node.output[0], dtype)
    if len(node.output) > 1 and node.output[1]:
        self.set_type(node.output[1], TensorProto.INT32)
    if self.has_device(node.input[0]) and node.output[0]:
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if self.has_shape(node.input[0]) and word_emb_input and self.has_shape(word_emb_input):
        ids_shape = self.get_shape(node.input[0])
        emb_shape = self.get_shape(word_emb_input)
        if len(ids_shape) >= 2 and len(emb_shape) >= 2 and node.output[0]:
            hidden = emb_shape[1]
            output_shape = (*ids_shape, hidden)
            self.set_shape(node.output[0], output_shape)
        if len(ids_shape) >= 1 and len(node.output) > 1 and node.output[1]:
            self.set_shape(node.output[1], (ids_shape[0],))
    elif self.has_rank(node.input[0]) and node.output[0]:
        rank = self.get_rank(node.input[0])
        self.set_rank(node.output[0], rank + 1)
        if len(node.output) > 1 and node.output[1]:
            self.set_rank(node.output[1], 1)


def _set_shape_type_gated_relative_position_bias(self: ShapeBuilder, node: NodeProto):
    """Sets the output shape for ``com.microsoft.GatedRelativePositionBias``.

    Inputs: ``query_layer (batch, seq_len, num_heads*head_size)``,
    ``query_bias``, ``rel_pos (1, num_heads, seq_len, seq_len)``, ``weight``,
    ``bias``, ``eco_a (1, num_heads, 1, 1)``, ``[token_offset]``.
    Output: ``(batch_size, num_heads, seq_len, seq_len)``.
    """
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    rel_pos_idx = 2
    if self.has_shape(node.input[0]) and self.has_shape(node.input[rel_pos_idx]):
        query_shape = self.get_shape(node.input[0])
        rel_pos_shape = self.get_shape(node.input[rel_pos_idx])
        batch_dim = query_shape[0]
        num_heads_dim = rel_pos_shape[1]
        seq_q_dim = rel_pos_shape[2]
        seq_k_dim = rel_pos_shape[3]
        self.set_shape(node.output[0], (batch_dim, num_heads_dim, seq_q_dim, seq_k_dim))
    elif self.has_rank(node.input[0]):
        self.set_rank(node.output[0], 4)


def _set_shape_type_causal_conv_with_state(self: ShapeBuilder, node: NodeProto):
    """Sets the output shape for ``com.microsoft.CausalConvWithState``.

    Inputs: ``input (N, C, L)``, ``weight (C, 1, K)``, ``bias (C)``
    (optional), ``past_state (N, C, K-1)`` (optional).
    Outputs: ``output`` (same shape as input),
    ``present_state (N, C, K-1)`` (optional).
    """
    dtype = self.get_type(node.input[0]) if self.has_type(node.input[0]) else None
    if dtype is not None:
        self.set_type(node.output[0], dtype)
        if len(node.output) > 1 and node.output[1]:
            self.set_type(node.output[1], dtype)
    if self.has_device(node.input[0]):
        dev = self.get_device(node.input[0])
        self.set_device(node.output[0], dev)
        if len(node.output) > 1 and node.output[1]:
            self.set_device(node.output[1], dev)
    if self.has_shape(node.input[0]):
        input_shape = self.get_shape(node.input[0])
        self.set_shape(node.output[0], input_shape)
        if len(node.output) > 1 and node.output[1]:
            if len(node.input) > 1 and self.has_shape(node.input[1]):
                weight_shape = self.get_shape(node.input[1])
                kernel_size = weight_shape[2]
                state_len = kernel_size - 1 if isinstance(kernel_size, int) else kernel_size
                self.set_shape(node.output[1], (input_shape[0], input_shape[1], state_len))
            else:
                self.set_rank(node.output[1], len(input_shape))
    elif self.has_rank(node.input[0]):
        rk = self.get_rank(node.input[0])
        self.set_rank(node.output[0], rk)
        if len(node.output) > 1 and node.output[1]:
            self.set_rank(node.output[1], rk)


def _set_shape_type_greedy_search(self: ShapeBuilder, node: NodeProto):
    """Sets the output shape for ``com.microsoft.GreedySearch``.

    Input ``input_ids`` has shape ``(batch_size, sequence_length)`` and type
    INT32.  Input ``max_length`` is a scalar INT32.  Output ``sequences`` has
    shape ``(batch_size, max_length_value)`` and type INT32.
    """
    self.set_type(node.output[0], TensorProto.INT32)
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if self.has_shape(node.input[0]):
        input_shape = self.get_shape(node.input[0])
        batch_size = input_shape[0]
        has_max_length = len(node.input) > 1 and node.input[1]
        max_len_val = self.value_as_shape(node.input[1]) if has_max_length else None
        if max_len_val is not None and len(max_len_val) == 1:
            max_len = max_len_val[0]
        else:
            max_len = node.input[1] if has_max_length else "max_len"
        self.set_shape(node.output[0], (batch_size, max_len))
    else:
        self.set_rank(node.output[0], 2)


def _set_shape_type_moe(self: ShapeBuilder, node: NodeProto):
    """Sets the output shape for ``com.microsoft.MoE`` (Mixture of Experts).

    Input ``input`` has shape ``(num_tokens, hidden_size)`` or
    ``(batch_size, seq_len, hidden_size)``.  Output ``output`` has the same
    shape and dtype as the input.
    """
    dtype = self.get_type(node.input[0]) if self.has_type(node.input[0]) else None
    if dtype is not None:
        self.set_type(node.output[0], dtype)
    if self.has_device(node.input[0]):
        self.set_device(node.output[0], self.get_device(node.input[0]))
    if self.has_shape(node.input[0]):
        self.set_shape(node.output[0], self.get_shape(node.input[0]))
    elif self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]))


_set_shape_type_com_microsoft_ops: Dict[str, Any] = {
    "Attention": _set_shape_type_attention_microsoft,
    "CausalConvWithState": _set_shape_type_causal_conv_with_state,
    "CDist": _set_shape_type_cdist,
    "EmbedLayerNormalization": _set_shape_type_embed_layer_normalization,
    "GatedRelativePositionBias": _set_shape_type_gated_relative_position_bias,
    "GreedySearch": _set_shape_type_greedy_search,
    "GroupQueryAttention": _set_shape_type_group_query_attention,
    "MoE": _set_shape_type_moe,
    "MurmurHash3": _set_shape_type_murmur_hash3,
    "PackedMultiHeadAttention": _set_shape_type_packed_multi_head_attention,
    "RelativePositionBias": _set_shape_type_relative_position_bias,
}

_SUPPORTED_AI_ONNX_ML_OPS: FrozenSet[str] = frozenset(
    {"TreeEnsemble", "TreeEnsembleClassifier", "TreeEnsembleRegressor"}
)

_SUPPORTED_UNARY_CUSTOM_OPS: FrozenSet[str] = frozenset({"NegXplus1", "ReplaceZero"})


def supported_ops_in_set_shape_type_custom() -> Dict[str, FrozenSet[str]]:
    """Returns the ops supported by :func:`set_shape_type_custom` grouped by domain.

    Returns a dictionary mapping each ONNX domain name to a :class:`frozenset`
    of op type names for which :func:`set_shape_type_custom` provides shape and
    type inference.

    The special key ``""`` (empty string) groups ops that are handled
    regardless of their domain (i.e. no domain check is performed for them).
    Local functions registered at runtime are not included because they are
    determined dynamically.

    Returns:
        Dictionary mapping domain name to a frozenset of supported op types.
    """
    return {
        "ai.onnx.ml": _SUPPORTED_AI_ONNX_ML_OPS,
        "": _SUPPORTED_UNARY_CUSTOM_OPS | frozenset(_set_shape_type_op_any_custom),
        "com.microsoft": frozenset(_set_shape_type_com_microsoft_ops),
    }


def set_shape_type_custom(self: ShapeBuilder, node: NodeProto, exc: bool = False):
    """Sets the shape and type if it can."""
    if node.domain == "ai.onnx.ml":
        if node.op_type in _SUPPORTED_AI_ONNX_ML_OPS:
            return set_type_shape_tree_ensemble(self, node)
        return None
    if node.op_type in _SUPPORTED_UNARY_CUSTOM_OPS:
        return set_type_shape_unary_op(self, node.output[0], node.input[0])
    if node.op_type in _set_shape_type_op_any_custom:
        return _set_shape_type_op_any_custom[node.op_type](self, node)
    if self.has_local_function(node.op_type, domain=node.domain, builder=True):
        local_function_builder = self.get_local_function(
            node.op_type, domain=node.domain, builder=True
        )
        assert local_function_builder is not None, (
            f"Missing local function for node {(node.domain, node.op_type)}"
            f"{self.get_debug_msg()}"
        )
        assert isinstance(local_function_builder, self.__class__), (
            f"Unexpected type {type(local_function_builder)} "
            f"for node {(node.domain, node.op_type)} "
            f"and the local_function it refers to{self.get_debug_msg()}"
        )
        shapes = [self.get_shape(i) if self.has_shape(i) else None for i in node.input]
        if None in shapes:
            # Nothing we can do.
            return
        proto_local_function = self.get_local_function(node.op_type, domain=node.domain)
        function_inputs = list(proto_local_function.input)
        # The builder creating the local function can have a different number of
        # inputs because constants can be promoted as FunctionProto inputs at export.
        if (
            len(function_inputs) != len(node.input)
            and hasattr(local_function_builder, "input_names")
            and len(local_function_builder.input_names) == len(node.input)
        ):
            function_inputs = list(local_function_builder.input_names)

        local_shapes = [
            local_function_builder.get_shape(i) if local_function_builder.has_shape(i) else None
            for i in function_inputs
        ]
        assert len(shapes) == len(local_shapes), (
            f"Mismatch in the number of provided input shapes for "
            f"node '{node.domain}.{node.op_type}': node has {node.input} ({len(node.input)}), "
            f"function has {proto_local_function.input} ({len(proto_local_function.input)}), "
            f"matched-function-inputs={function_inputs} ({len(function_inputs)})"
            f"{self.get_debug_msg()}"
        )
        if local_shapes != shapes:
            local_function_builder.reset_types_and_shapes()
            for ni, i, sh in zip(node.input, function_inputs, shapes):
                if self.has_type(ni):
                    local_function_builder.set_type(i, self.get_type(ni))
                if self.has_device(ni):
                    local_function_builder.set_device(i, self.get_device(ni))
                if self.is_constant(ni):
                    cst = self.get_constant(ni, exc=exc, computed_value=True)
                    if cst is not None:
                        local_function_builder.constants_[i] = self.constants_[ni]
                        local_function_builder.constants_computed_[i] = cst
                local_function_builder.set_shape(i, sh)
            if hasattr(local_function_builder, "infer_shapes"):
                # A GraphBuilder
                local_function_builder.infer_shapes()
            else:
                # A ShapeBuilder
                for n in proto_local_function.node:
                    local_function_builder.run_node(n, exc=exc)
                for o in proto_local_function.output:
                    local_function_builder._output_names.append(o)

        assert len(local_function_builder.output_names) == len(node.output), (
            f"Mismatch between the number of outputs, node has {node.output}, "
            f"function has {local_function_builder.output_names}{self.get_debug_msg()}"
        )
        for o, lo in zip(node.output, local_function_builder.output_names):
            if local_function_builder.has_type(lo):
                self.set_type(o, local_function_builder.get_type(lo))
            if local_function_builder.has_shape(lo):
                self.set_shape(o, local_function_builder.get_shape(lo))
            elif local_function_builder.has_rank(lo):
                self.set_rank(o, local_function_builder.get_rank(lo))
        return None

    if node.op_type in _set_shape_type_com_microsoft_ops and node.domain == "com.microsoft":
        return _set_shape_type_com_microsoft_ops[node.op_type](self, node)

    assert node.domain not in {
        "ai.onnx.ml",
        "intermediate",
        "ai.onnx.complex",
        "com.microsoft",
        "local_domain",
        "SimplifyingFunction",
        "yaourt.ortops.fused_kernel.cuda",
    }, (
        f"Unable to find a function computing the output shape of node "
        f"{(node.domain, node.op_type)}, list of functions is "
        f"{sorted(self.functions)}, list of functions with graph is "
        f"{sorted(self.functions_builder)}{self.get_debug_msg()}"
    )
    assert not self._debug_shape_missing, (
        f"No function to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )
    if exc:
        raise NotImplementedError(
            f"No shape function for node type {node.op_type!r} from domain {node.domain!r}"
        )

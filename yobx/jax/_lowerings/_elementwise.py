"""Lowers elementwise JAX primitives."""

from functools import partial

from ._registry import jax_dtype, register_lowering

_ELEMENTWISE = {
    "add": "Add",
    "sub": "Sub",
    "mul": "Mul",
    "div": "Div",
    "max": "Max",
    "min": "Min",
    "pow": "Pow",
    "neg": "Neg",
    "abs": "Abs",
    "exp": "Exp",
    "log": "Log",
    "sqrt": "Sqrt",
    "tanh": "Tanh",
    "sin": "Sin",
    "cos": "Cos",
    "sign": "Sign",
    "ceil": "Ceil",
    "floor": "Floor",
    "logistic": "Sigmoid",
    "gt": "Greater",
    "lt": "Less",
    "ge": "GreaterOrEqual",
    "le": "LessOrEqual",
    "eq": "Equal",
}


def _lower_elementwise(op_type, state, eqn, args):
    """Lowers an elementwise primitive to its ONNX counterpart."""
    return getattr(state.builder.op, op_type)(*args)


def _lower_not_equal(state, eqn, args):
    """Lowers inequality through Equal and Not."""
    return state.builder.op.Not(state.builder.op.Equal(*args))


def _lower_integer_pow(state, eqn, args):
    """Lowers a power whose exponent is a static integer."""
    exponent = state.constant(eqn.params["y"])
    return state.builder.op.Pow(args[0], exponent)


def _lower_square(state, eqn, args):
    """Lowers elementwise squaring."""
    return state.builder.op.Mul(args[0], args[0])


def _lower_select_n(state, eqn, args):
    """Lowers a boolean choice between two values."""
    if len(args) != 3:
        raise NotImplementedError("select_n currently supports exactly two choices.")
    return state.builder.op.Where(args[0], args[2], args[1])


def _lower_identity_or_cast(state, eqn, args):
    """Lowers identity-like and element-type conversion primitives."""
    if eqn.primitive.name == "stop_gradient":
        return state.builder.op.Identity(args[0])
    return state.builder.op.Cast(args[0], to=jax_dtype(eqn.outvars[0]))


for primitive_name, onnx_op_type in _ELEMENTWISE.items():
    register_lowering(primitive_name, partial(_lower_elementwise, onnx_op_type))
register_lowering("ne", _lower_not_equal)
register_lowering("integer_pow", _lower_integer_pow)
register_lowering("square", _lower_square)
register_lowering("select_n", _lower_select_n)
register_lowering("stop_gradient", _lower_identity_or_cast)
register_lowering("convert_element_type", _lower_identity_or_cast)

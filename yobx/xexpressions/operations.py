from typing import Union
import numpy as np
from .simplify_expressions import simplify_expression

DIM_TYPE = Union[int, str]


def dim_mul(a: DIM_TYPE, b: DIM_TYPE) -> DIM_TYPE:
    """Multiplies dimensions."""
    if isinstance(a, int) and isinstance(b, int):
        return a * b
    return simplify_expression(f"({a})*({b})")


def dim_multi_mul(*args: DIM_TYPE) -> DIM_TYPE:
    """Multiplies dimensions."""
    if all(isinstance(a, int) for a in args):
        return int(np.prod(args))  # type: ignore
    return simplify_expression("*".join(f"({a})" for a in args))


def dim_add(a: DIM_TYPE, b: DIM_TYPE) -> DIM_TYPE:
    """Adds dimensions."""
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    return simplify_expression(f"({a})+({b})")


def dim_sub(a: DIM_TYPE, b: DIM_TYPE) -> DIM_TYPE:
    """Subtracts dimensions."""
    if isinstance(a, int) and isinstance(b, int):
        return a - b
    return simplify_expression(f"({a})-({b})")


def dim_div(a: DIM_TYPE, b: DIM_TYPE) -> DIM_TYPE:
    """Divides dimensions. This assumes both values are positive."""
    if isinstance(a, int) and isinstance(b, int):
        return a // b
    return simplify_expression(f"({a})//({b})")


def dim_mod(a: DIM_TYPE, b: DIM_TYPE) -> DIM_TYPE:
    """Modulo."""
    if isinstance(a, int) and isinstance(b, int):
        return a % b
    return simplify_expression(f"({a})%({b})")


def dim_max(a: DIM_TYPE, b: DIM_TYPE) -> DIM_TYPE:
    """Maximum of too dimensions."""
    if isinstance(a, int) and isinstance(b, int):
        return max(a, b)
    return simplify_expression(f"({a})^({b})")


def dim_min(a: DIM_TYPE, b: DIM_TYPE) -> DIM_TYPE:
    """Maximum of too dimensions."""
    if isinstance(a, int) and isinstance(b, int):
        return min(a, b)
    return simplify_expression(f"({a})&({b})")

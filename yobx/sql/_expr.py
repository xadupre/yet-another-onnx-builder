"""
ONNX expression emitter for SQL AST nodes.

:class:`_ExprEmitter` translates parsed SQL expression trees (column
references, literals, binary operators, aggregate functions) into ONNX nodes
inside a :class:`~yobx.xbuilder.GraphBuilder`.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from ..xbuilder import GraphBuilder
from ..xtracing.tracing import trace_numpy_function
from .parse import AggExpr, BinaryExpr, ColumnRef, Condition, FuncCallExpr, Literal


class _ExprEmitter:
    """Emits ONNX nodes for SQL expression AST nodes."""

    def __init__(
        self,
        g: GraphBuilder,
        col_map: Dict[str, str],
        custom_functions: Optional[Dict[str, Callable]] = None,
    ):
        self._g = g
        # col_map: column_name → current ONNX tensor name for that column
        self._col_map = col_map
        # custom_functions: function_name → Python callable (traced via xtracing)
        self._custom_functions: Dict[str, Callable] = custom_functions or {}

    def emit(self, node: object, name: str = "sql") -> str:
        """Emit *node* into the graph and return the output tensor name."""
        if isinstance(node, ColumnRef):
            col = node.column
            if col not in self._col_map:
                raise KeyError(
                    f"Column {col!r} not found in inputs. Available: {list(self._col_map)}"
                )
            return self._col_map[col]

        if isinstance(node, Literal):
            v = node.value
            if isinstance(v, bool):
                arr = np.array([v], dtype=np.bool_)
            elif isinstance(v, int):
                arr = np.array([v], dtype=np.int64)
            elif isinstance(v, float):
                arr = np.array([v], dtype=np.float32)
            else:
                raise TypeError(f"Unsupported literal type {type(v)}")
            cst = self._g.make_initializer(f"{name}_lit", arr)
            return cst

        if isinstance(node, BinaryExpr):
            left = self.emit(node.left, name=f"{name}_l")
            right = self.emit(node.right, name=f"{name}_r")
            op_map = {
                "+": "Add",
                "-": "Sub",
                "*": "Mul",
                "/": "Div",
                "=": "Equal",
                "<": "Less",
                ">": "Greater",
                "<=": "LessOrEqual",
                ">=": "GreaterOrEqual",
                "<>": "Not",  # handled below
            }
            # Cast the right operand to match the left operand type when one
            # side is a column (has a known type) and the other is a literal.
            if isinstance(node.right, Literal) and self._g.has_type(left):
                right = self._g.op.CastLike(right, left, name=f"{name}_cast")
            elif isinstance(node.left, Literal) and self._g.has_type(right):
                left = self._g.op.CastLike(left, right, name=f"{name}_cast")
            if node.op == "<>":
                eq = self._g.op.Equal(left, right, name=f"{name}_eq")
                return self._g.op.Not(eq, name=name)  # type: ignore
            onnx_op = op_map.get(node.op)
            if onnx_op is None:
                raise ValueError(f"Unsupported binary operator: {node.op!r}")
            return getattr(self._g.op, onnx_op)(left, right, name=name)

        if isinstance(node, Condition):
            op = node.op
            if op == "and":
                left = self.emit(node.left, name=f"{name}_l")
                right = self.emit(node.right, name=f"{name}_r")
                return self._g.op.And(left, right, name=name)  # type: ignore
            if op == "or":
                left = self.emit(node.left, name=f"{name}_l")
                right = self.emit(node.right, name=f"{name}_r")
                return self._g.op.Or(left, right, name=name)  # type: ignore
            # leaf comparison — treat it as a BinaryExpr
            return self.emit(BinaryExpr(left=node.left, op=node.op, right=node.right), name=name)

        if isinstance(node, AggExpr):
            func = node.func
            if func == "count":
                # COUNT(*) → total row count of the first known column
                first_col = next(iter(self._col_map.values()))
                size = self._g.op.Shape(first_col, name=f"{name}_shape")
                return self._g.op.Gather(  # type: ignore
                    size, np.array(0, dtype=np.int64), axis=0, name=f"{name}_count"
                )
            arg_tensor = self.emit(node.arg, name=f"{name}_arg")
            reduce_axes = self._g.make_initializer(f"{name}_axes", np.array([0], dtype=np.int64))
            agg_map = {
                "sum": "ReduceSum",
                "avg": "ReduceMean",
                "min": "ReduceMin",
                "max": "ReduceMax",
            }
            onnx_op = agg_map.get(func)
            if onnx_op is None:
                raise ValueError(f"Unsupported aggregation function: {func!r}")
            return getattr(self._g.op, onnx_op)(arg_tensor, reduce_axes, keepdims=0, name=name)

        if isinstance(node, FuncCallExpr):
            func_name = node.func
            if func_name not in self._custom_functions:
                raise KeyError(
                    f"Unknown function {func_name!r} in SQL expression. "
                    f"Register it via the custom_functions parameter. "
                    f"Available: {list(self._custom_functions)}"
                )
            func = self._custom_functions[func_name]
            # Emit the argument expressions first
            arg_tensors = [self.emit(arg, name=f"{name}_arg{i}") for i, arg in enumerate(node.args)]
            # Allocate a unique output tensor name
            out_name = self._g.unique_name(f"{name}_{func_name}")
            trace_numpy_function(self._g, {}, [out_name], func, arg_tensors, name=out_name)
            return out_name

        raise TypeError(f"Cannot emit expression of type {type(node)}")

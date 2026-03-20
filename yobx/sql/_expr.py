"""
ONNX expression emitter for SQL AST nodes.

:class:`_ExprEmitter` translates parsed SQL expression trees (column
references, literals, binary operators, aggregate functions) into ONNX nodes
inside a :class:`~yobx.typing.GraphBuilderExtendedProtocol`.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import onnx.numpy_helper as onh

from ..typing import GraphBuilderExtendedProtocol
from ..xtracing.tracing import trace_numpy_function
from .parse import AggExpr, BinaryExpr, ColumnRef, Condition, FuncCallExpr, Literal


class _ExprEmitter:
    """Emits ONNX nodes for SQL expression AST nodes."""

    def __init__(
        self,
        g: GraphBuilderExtendedProtocol,
        col_map: Dict[str, str],
        custom_functions: Optional[Dict[str, Callable]] = None,
        group_by_inverse_indices: Optional[str] = None,
        group_by_first_indices: Optional[str] = None,
        group_by_n_groups: Optional[str] = None,
    ):
        self._g = g
        # col_map: column_name → current ONNX tensor name for that column
        self._col_map = col_map
        # custom_functions: function_name → Python callable (traced via xtracing)
        self._custom_functions: Dict[str, Callable] = custom_functions or {}
        # GROUP BY support: when set, aggregate functions produce one value per group
        # instead of a global scalar.
        # inverse_indices: shape (n_rows,), int64 — group index for each input row
        self._group_by_inverse_indices = group_by_inverse_indices
        # first_indices: shape (n_groups,), int64 — first occurrence row index per group
        self._group_by_first_indices = group_by_first_indices
        # n_groups: scalar int64 — number of unique groups
        self._group_by_n_groups = group_by_n_groups

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
            cst = self._g.make_initializer(f"{name}_lit", arr, give_unique_name=True)
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
            if self._group_by_inverse_indices is not None:
                return self._emit_grouped_agg(node, name)
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
            arg_tensors = [
                self.emit(arg, name=f"{name}_arg{i}") for i, arg in enumerate(node.args)
            ]
            # Allocate a unique output tensor name
            out_name = self._g.unique_name(f"{name}_{func_name}")
            trace_numpy_function(self._g, {}, [out_name], func, arg_tensors, name=out_name)
            return out_name

        raise TypeError(f"Cannot emit expression of type {type(node)}")

    def _emit_grouped_agg(self, node: AggExpr, name: str) -> str:
        """Emit a per-group aggregation using ``ScatterElements``.

        For each unique group (determined by the GROUP BY key), computes the
        aggregate function over the rows belonging to that group and returns a
        1-D tensor of shape ``(n_groups,)`` — one value per group, in the
        order produced by ``Unique`` (sorted by default).

        :param node: the :class:`~yobx.sql.parse.AggExpr` to evaluate.
        :param name: base name used when generating ONNX node names.
        :return: tensor name of the per-group result, shape ``(n_groups,)``.
        """
        g = self._g
        inv_idx = self._group_by_inverse_indices  # shape: (n_rows,), int64
        first_idx = self._group_by_first_indices  # shape: (n_groups,), int64
        n_groups = self._group_by_n_groups  # scalar int64
        func = node.func

        # n_groups wrapped as a 1-D tensor for use with ConstantOfShape
        n_groups_1d = g.op.Unsqueeze(
            n_groups, np.array([0], dtype=np.int64), name=f"{name}_n_groups_1d"
        )

        if func == "count":
            # COUNT(*): count the number of rows mapped to each group
            n_rows = g.op.Gather(
                g.op.Shape(inv_idx, name=f"{name}_inv_shape"),
                np.array(0, dtype=np.int64),
                axis=0,
                name=f"{name}_n_rows",
            )
            n_rows_1d = g.op.Unsqueeze(
                n_rows, np.array([0], dtype=np.int64), name=f"{name}_n_rows_1d"
            )
            ones = g.op.ConstantOfShape(
                n_rows_1d,
                value=onh.from_array(np.array([1], dtype=np.int64)),
                name=f"{name}_ones",
            )
            zeros = g.op.ConstantOfShape(
                n_groups_1d,
                value=onh.from_array(np.array([0], dtype=np.int64)),
                name=f"{name}_zeros",
            )
            return g.op.ScatterElements(  # type: ignore[return-value]
                zeros, inv_idx, ones, reduction="add", axis=0, name=name
            )

        arg_tensor = self.emit(node.arg, name=f"{name}_arg")

        # Create a zero-filled buffer of shape (n_groups,) with arg_tensor's dtype
        zeros_f32 = g.op.ConstantOfShape(n_groups_1d, name=f"{name}_zeros_f32")
        zeros = g.op.CastLike(zeros_f32, arg_tensor, name=f"{name}_zeros")

        if func == "sum":
            return g.op.ScatterElements(  # type: ignore[return-value]
                zeros, inv_idx, arg_tensor, reduction="add", axis=0, name=name
            )

        if func == "avg":
            sum_vals = g.op.ScatterElements(
                zeros,
                inv_idx,
                arg_tensor,
                reduction="add",
                axis=0,
                name=f"{name}_sum",
            )
            # Count per group using float32 ones so that the division stays in float
            n_rows = g.op.Gather(
                g.op.Shape(arg_tensor, name=f"{name}_arg_shape"),
                np.array(0, dtype=np.int64),
                axis=0,
                name=f"{name}_n_rows",
            )
            n_rows_1d = g.op.Unsqueeze(
                n_rows, np.array([0], dtype=np.int64), name=f"{name}_n_rows_1d"
            )
            ones_f32 = g.op.ConstantOfShape(
                n_rows_1d,
                value=onh.from_array(np.array([1.0], dtype=np.float32)),
                name=f"{name}_ones_f32",
            )
            zeros_cnt = g.op.ConstantOfShape(n_groups_1d, name=f"{name}_zeros_cnt")
            cnt_vals = g.op.ScatterElements(
                zeros_cnt,
                inv_idx,
                ones_f32,
                reduction="add",
                axis=0,
                name=f"{name}_cnt",
            )
            sum_f32 = g.op.CastLike(sum_vals, cnt_vals, name=f"{name}_sum_f32")
            return g.op.Div(sum_f32, cnt_vals, name=name)  # type: ignore[return-value]

        # For min/max: initialise each group's buffer cell with the first-occurrence
        # value for that group.  This is type-agnostic and always valid: because
        # ScatterElements scatters *all* rows (including each first-occurrence row),
        # the result is min/max over the full set of group values regardless of
        # the initial seed.
        init_vals = g.op.Gather(
            arg_tensor, first_idx, axis=0, name=f"{name}_init"
        )

        if func == "min":
            return g.op.ScatterElements(  # type: ignore[return-value]
                init_vals, inv_idx, arg_tensor, reduction="min", axis=0, name=name
            )

        if func == "max":
            return g.op.ScatterElements(  # type: ignore[return-value]
                init_vals, inv_idx, arg_tensor, reduction="max", axis=0, name=name
            )

        raise ValueError(f"Unsupported aggregation function: {func!r}")

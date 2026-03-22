"""
Functions to compute statistics on an ONNX model such as number of nodes
per op_type and estimation of computational cost.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
from ..xexpressions.operations import DIM_TYPE
from ..xshape.cost_inference import (
    _estimate_node_flops,
    _ShapeFn,
    _LiteralFn,
)
from ..typing import GraphBuilderExtendedProtocol


class ModelStatistics:
    """
    Computes statistics on an ONNX model, including node counts per op_type
    and estimated FLOPs.

    *model* may be either an :class:`onnx.ModelProto` or a
    :class:`~yobx.typing.GraphBuilderExtendedProtocol` instance.  When a graph
    builder is provided its already-computed shape information is used directly
    (no second shape-inference pass is run) and the ONNX model is obtained via
    :meth:`~yobx.typing.GraphBuilderProtocol.to_onnx`.

    :param model: ONNX model or graph builder
    :param verbose: verbosity level passed to :class:`BasicShapeBuilder`
        (ignored when *model* is a graph builder)

    Usage::

        stats = ModelStatistics(model).compute()
    """

    def __init__(
        self, model: Union[onnx.ModelProto, GraphBuilderExtendedProtocol], verbose: int = 0
    ) -> None:
        if isinstance(model, GraphBuilderExtendedProtocol):
            self._builder: Any = model
            onnx_model = model.to_onnx()
            # to_onnx() may return an ExportArtifact wrapping the ModelProto
            if not isinstance(onnx_model, onnx.ModelProto):
                proto = getattr(onnx_model, "proto", None)
                if isinstance(proto, onnx.ModelProto):
                    onnx_model = proto
                else:
                    raise TypeError(
                        f"GraphBuilderExtendedProtocol.to_onnx() returned "
                        f"{type(onnx_model).__name__!r}, expected ModelProto"
                    )
            self.model: onnx.ModelProto = onnx_model
        else:
            self._builder = None
            self.model = model
        self.verbose = verbose
        self._bs: Any = None
        self._shape_map: Dict[str, Optional[Tuple]] = {}
        self._node_count: Dict[str, int] = {}
        self._flops_by_op: Dict[str, List[Optional[DIM_TYPE]]] = {}
        self._node_stats: List[Dict[str, Any]] = []

    def _collect_names(self, graph: onnx.GraphProto, all_names: set) -> None:
        """Recursively gathers every tensor name referenced in *graph*."""
        for vi in list(graph.input) + list(graph.output):
            all_names.add(vi.name)
        for init in graph.initializer:
            all_names.add(init.name)
        for node in graph.node:
            for n in list(node.input) + list(node.output):
                if n:
                    all_names.add(n)
            for att in node.attribute:
                if att.type == onnx.AttributeProto.GRAPH:
                    self._collect_names(att.g, all_names)
                elif att.type == onnx.AttributeProto.GRAPHS:
                    for g in att.graphs:
                        self._collect_names(g, all_names)

    def _build_shape_map(self) -> None:
        """Populates :attr:`_shape_map` from the builder (if provided) or by
        running :class:`BasicShapeBuilder` on the stored model."""
        all_names: set = set()
        self._collect_names(self.model.graph, all_names)

        if self._builder is not None:
            # Use shape information already computed by the graph builder.
            for name in all_names:
                if not self._builder.has_shape(name):
                    continue
                sh = self._builder.get_shape(name)
                if all(isinstance(d, int) for d in sh):
                    self._shape_map[name] = tuple(int(d) for d in sh)
                else:
                    self._shape_map[name] = sh
        else:
            from ..xshape import BasicShapeBuilder

            self._bs = BasicShapeBuilder(verbose=self.verbose)
            self._bs.run_model(self.model)

            for name in all_names:
                if not self._bs.has_shape(name):
                    continue
                sh = self._bs.get_shape(name)
                if all(isinstance(d, int) for d in sh):
                    self._shape_map[name] = tuple(int(d) for d in sh)
                else:
                    self._shape_map[name] = sh

    def shape_fn(self, name: str) -> Optional[Tuple]:
        """Returns the inferred shape of *name*, or ``None`` if unknown."""
        return self._shape_map.get(name)

    def literal_fn(self, name: str) -> Optional[Tuple[int, ...]]:
        """
        Returns the integer values stored in a 1-D integer constant tensor
        (e.g. the ``shape`` input of a Reshape node), or ``None`` when *name*
        is not a known constant.
        """
        if not name:
            return None
        if self._builder is not None:
            # Use the builder's constant API via duck typing.
            is_const_fn = getattr(self._builder, "is_constant", None)
            if is_const_fn is not None and not is_const_fn(name):
                return None
            get_fn = getattr(self._builder, "get_computed_constant", None)
            if get_fn is not None:
                try:
                    val = get_fn(name)
                    if val is not None:
                        return tuple(int(v) for v in np.asarray(val).flat)
                except Exception:
                    pass
            return None
        if not self._bs.is_constant(name):
            return None
        val = self._bs.get_constant(name, exc=False, computed_value=True, as_shape=True)
        if val is None:
            return None
        return tuple(int(v) for v in val)  # type: ignore

    def _collect_nodes(self, graph: onnx.GraphProto) -> None:
        """Recursively visits all nodes in *graph* and accumulates statistics."""
        for node in graph.node:
            op = node.op_type
            self._node_count[op] = self._node_count.get(op, 0) + 1

            f = _estimate_node_flops(node, self.shape_fn, self.literal_fn)
            if op not in self._flops_by_op:
                self._flops_by_op[op] = []
            self._flops_by_op[op].append(f)

            input_shapes = [(n, self._shape_map.get(n)) for n in node.input if n]
            output_shapes = [(n, self._shape_map.get(n)) for n in node.output if n]
            self._node_stats.append(
                {
                    "op_type": op,
                    "name": node.name,
                    "inputs": input_shapes,
                    "outputs": output_shapes,
                    "estimated_flops": f,
                }
            )
            # Recurse into subgraphs (e.g. inside If / Loop / Scan)
            for att in node.attribute:
                if att.type == onnx.AttributeProto.GRAPH:
                    self._collect_nodes(att.g)
                elif att.type == onnx.AttributeProto.GRAPHS:
                    for g in att.graphs:
                        self._collect_nodes(g)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self) -> Dict[str, Any]:
        """
        Runs the full analysis and returns a statistics dictionary.

        :return: dictionary with the following keys:

            - ``"n_nodes"`` – total number of nodes
            - ``"node_count_per_op_type"`` – dict mapping op_type → count
            - ``"total_estimated_flops"`` – total estimated FLOPs (int or None)
            - ``"flops_per_op_type"`` – dict mapping op_type → estimated FLOPs
              (None means the cost could not be estimated for some nodes)
            - ``"node_stats"`` – list of per-node dicts with keys
              ``op_type``, ``name``, ``inputs``, ``outputs``, ``estimated_flops``
        """
        self._build_shape_map()
        self._collect_nodes(self.model.graph)

        # Aggregate FLOPs per op_type
        flops_per_op: Dict[str, Optional[int]] = {}
        total: Optional[int] = 0
        for op, values in self._flops_by_op.items():
            op_total: Optional[int] = 0
            for v in values:
                if v is None or not isinstance(v, int):
                    op_total = None
                    break
                op_total += v  # type: ignore
            flops_per_op[op] = op_total
            if op_total is None:
                total = None
            elif total is not None:
                total += op_total

        return {
            "n_nodes": sum(self._node_count.values()),
            "node_count_per_op_type": self._node_count,
            "total_estimated_flops": total,
            "flops_per_op_type": flops_per_op,
            "node_stats": self._node_stats,
        }


def model_statistics(
    model: Union[onnx.ModelProto, GraphBuilderExtendedProtocol], verbose: int = 0
) -> Dict[str, Any]:
    """
    Computes statistics on an ONNX model.

    This is a convenience wrapper around :class:`ModelStatistics`.

    :param model: ONNX model or graph builder
    :param verbose: verbosity level
    :return: statistics dictionary — see :meth:`ModelStatistics.compute` for details
    """
    return ModelStatistics(model, verbose=verbose).compute()

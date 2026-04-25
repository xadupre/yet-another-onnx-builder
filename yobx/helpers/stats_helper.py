"""
Functions to compute statistics on an ONNX model such as number of nodes
per op_type and estimation of computational cost.  Also provides classes
and helpers for computing per-tree statistics on ``TreeEnsemble*`` operators
(adapted from :mod:`onnx_extended.tools.stats_nodes`).
"""

import pprint
from collections import Counter
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import numpy as np
import onnx
import onnx.numpy_helper as onh
from onnx import (
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    SparseTensorProto,
    TensorProto,
)
from ..xexpressions.operations import DIM_TYPE
from ..xshape.cost_inference import estimate_node_flops
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

            f = estimate_node_flops(node, self.shape_fn, self.literal_fn)
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


# ---------------------------------------------------------------------------
# Tree-ensemble statistics
# Adapted from onnx_extended.tools.stats_nodes
# (https://github.com/sdpython/onnx-extended/blob/main/onnx_extended/tools/stats_nodes.py)
# ---------------------------------------------------------------------------


def extract_attributes(node: NodeProto) -> Dict[str, Tuple[AttributeProto, Any]]:
    """
    Extracts all attributes of a node into a plain Python dictionary.

    :param node: node to inspect
    :return: dictionary mapping attribute name to ``(AttributeProto, value)``
        where *value* is a Python/NumPy scalar or array, or ``None`` for graph
        and ref-attribute entries.
    """
    atts: Dict[str, Tuple[AttributeProto, Any]] = {}
    for att in node.attribute:
        if hasattr(att, "ref_attr_name") and att.ref_attr_name:
            atts[att.name] = (att, None)
            continue
        if att.type == AttributeProto.INT:
            atts[att.name] = (att, att.i)
            continue
        if att.type == AttributeProto.FLOAT:
            atts[att.name] = (att, att.f)
            continue
        if att.type == AttributeProto.INTS:
            atts[att.name] = (att, np.array(att.ints))
            continue
        if att.type == AttributeProto.FLOATS:
            atts[att.name] = (att, np.array(att.floats, dtype=np.float32))
            continue
        if att.type == AttributeProto.GRAPH and hasattr(att, "g") and att.g is not None:
            atts[att.name] = (att, None)
            continue
        if att.type == AttributeProto.TENSOR:
            atts[att.name] = (att, onh.to_array(att.t))
            continue
        if att.type == AttributeProto.TENSORS:
            atts[att.name] = (att, [onh.to_array(t) for t in att.tensors])
            continue
        if att.type == AttributeProto.STRING:
            atts[att.name] = (att, att.s.decode("utf-8"))
            continue
        if att.type == AttributeProto.STRINGS:
            atts[att.name] = (att, np.array([s.decode("utf-8") for s in att.strings]))
            continue
    return atts


class _Statistics:
    """
    Common base class for all statistics containers used by the tree-statistics API.
    """

    def __init__(self) -> None:
        self._statistics: Dict[str, Any] = {}

    def __len__(self) -> int:
        """Returns the number of statistics stored."""
        return len(self._statistics)

    def add(self, name: str, value: Any) -> None:
        """Adds a single named statistic.

        :param name: statistic name (must be unique within this instance)
        :param value: value to store
        """
        if name in self._statistics:
            raise ValueError(f"Statistics {name!r} was already added.")
        self._statistics[name] = value

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """Iterates over ``(name, value)`` pairs."""
        yield from self._statistics.items()

    def __getitem__(self, name: str) -> Any:
        """Returns the statistic identified by *name*."""
        return self._statistics[name]

    def get(self, name: str, default_value: Optional[Any] = None) -> Any:
        """Returns the statistic identified by *name*, or *default_value* if absent."""
        return self._statistics.get(name, default_value)

    def __str__(self) -> str:
        """Returns a human-readable representation."""
        return f"{self.__class__.__name__}(\n{pprint.pformat(self._statistics)})"

    @property
    def dict_values(self) -> Dict[str, Any]:
        """
        Converts the stored statistics into a flat dictionary suitable for building
        a pandas DataFrame row.

        Raises :class:`NotImplementedError` for the base class — subclasses must
        override this property.
        """
        raise NotImplementedError(
            f"Property 'dict_values' not implemented for class {type(self)}."
        )


class NodeStatistics(_Statistics):
    """
    Stores per-node statistics for a :class:`onnx.NodeProto`.

    :param parent: the :class:`~onnx.GraphProto` or
        :class:`~onnx.FunctionProto` that contains *node*
    :param node: the ONNX node being described
    """

    def __init__(self, parent: Union[GraphProto, FunctionProto], node: NodeProto) -> None:
        _Statistics.__init__(self)
        self.parent = parent
        self.node = node

    def __str__(self) -> str:
        """Returns a human-readable representation."""
        return (
            f"{self.__class__.__name__}(<{self.parent.name}>, <{self.node.op_type}>,\n"
            f"{pprint.pformat(self._statistics)})"
        )

    @property
    def dict_values(self) -> Dict[str, Any]:
        """Returns the statistics as a flat dictionary for DataFrame construction."""
        obs: Dict[str, Any] = {}
        for k, v in self._statistics.items():
            if isinstance(v, (int, float, str, np.int64, np.int32, np.float32, np.float64)):
                obs[k] = v
            elif isinstance(v, set):
                obs[k] = ",".join(map(str, sorted(v)))
            elif isinstance(v, Counter):
                for kk, vv in v.items():
                    obs[f"{k}__{kk}"] = vv
            elif isinstance(v, list):
                if len(v) == 0:
                    continue
                if isinstance(v[0], (HistTreeStatistics, TreeStatistics)):
                    # Per-tree statistics are intentionally skipped here.
                    continue
                raise TypeError(
                    f"Unexpected type {type(v)} for statistics {k!r} "
                    f"with element {type(v[0])}."
                )
            elif isinstance(v, _Statistics):
                dv = v.dict_values
                for kk, vv in dv.items():
                    if isinstance(vv, (int, float, str)):
                        obs[f"{k}__{kk}"] = vv
            else:
                raise TypeError(f"Unexpected type {type(v)} for statistics {k!r}: {v}.")
        return obs


class TreeStatistics(_Statistics):
    """
    Stores per-tree statistics extracted from ``TreeEnsemble*`` operators.

    :param node: the ``TreeEnsembleClassifier`` or ``TreeEnsembleRegressor`` node
    :param tree_id: zero-based index of this tree within the ensemble
    """

    def __init__(self, node: NodeProto, tree_id: int) -> None:
        _Statistics.__init__(self)
        self.node = node
        self.tree_id = tree_id

    def __str__(self) -> str:
        """Returns a human-readable representation."""
        return (
            f"{self.__class__.__name__}(<{self.node.op_type}>, {self.tree_id},\n"
            f"{pprint.pformat(self._statistics)})"
        )

    @property
    def dict_values(self) -> Dict[str, Any]:
        """Returns the statistics as a flat dictionary for DataFrame construction."""
        obs: Dict[str, Any] = {}
        for k, v in self._statistics.items():
            if isinstance(v, (int, float, str, np.int64, np.int32, np.float32, np.float64)):
                obs[k] = v
            elif isinstance(v, set):
                obs[k] = ",".join(map(str, sorted(v)))
            elif isinstance(v, Counter):
                for kk, vv in v.items():
                    obs[f"{k}__{kk}"] = vv
        return obs


class HistTreeStatistics(_Statistics):
    """
    Stores threshold-distribution statistics for a single feature across all
    trees in a ``TreeEnsemble*`` node.

    :param node: the ``TreeEnsembleClassifier`` or ``TreeEnsembleRegressor`` node
    :param featureid: zero-based feature index
    :param values: array of threshold values for *featureid*
    :param bins: number of histogram bins (default ``20``)
    """

    def __init__(
        self, node: NodeProto, featureid: int, values: np.ndarray, bins: int = 20
    ) -> None:
        _Statistics.__init__(self)
        self.node = node
        self.featureid = featureid
        self.add("min", float(values.min()))
        self.add("max", float(values.max()))
        self.add("mean", float(values.mean()))
        self.add("median", float(np.median(values)))
        self.add("size", len(values))
        n_distinct = len(set(values))
        self.add("n_distinct", n_distinct)
        self.add("hist", np.histogram(values, bins))
        if n_distinct <= 50:
            self.add("v_distinct", set(values))

    def __str__(self) -> str:
        """Returns a human-readable representation."""
        return (
            f"{self.__class__.__name__}(<{self.node.op_type}>, {self.featureid},\n"
            f"{pprint.pformat(self._statistics)})"
        )

    @property
    def dict_values(self) -> Dict[str, Any]:
        """Returns the statistics as a flat dictionary for DataFrame construction."""
        obs: Dict[str, Any] = {}
        for k in ("min", "max", "mean", "median", "size", "n_distinct"):
            obs[k] = self[k]
        return obs


def enumerate_nodes(
    onx: Union[FunctionProto, GraphProto, ModelProto], recursive: bool = True
) -> Iterable[
    Tuple[
        Tuple[str, ...],
        Union[GraphProto, FunctionProto],
        Union[NodeProto, TensorProto, SparseTensorProto],
    ]
]:
    """
    Enumerates all nodes in a model.

    :param onx: the model, graph, or function to traverse
    :param recursive: if ``True``, recurse into sub-graphs
        (e.g. inside ``If`` / ``Loop`` / ``Scan``)
    :return: yields tuples ``(path, parent, node)`` where *path* is a tuple of
        name strings identifying the location of *node* in the model, *parent*
        is the containing :class:`~onnx.GraphProto` or
        :class:`~onnx.FunctionProto`, and *node* is a
        :class:`~onnx.NodeProto`, :class:`~onnx.TensorProto`, or
        :class:`~onnx.SparseTensorProto`.
    """
    if isinstance(onx, ModelProto):
        for c, parent, node in enumerate_nodes(onx.graph, recursive=recursive):
            yield (onx.graph.name, *c), parent, node
        for f in onx.functions:
            for c, parent, node in enumerate_nodes(f, recursive=recursive):
                yield (f.name, *c), parent, node
    elif isinstance(onx, (GraphProto, FunctionProto)):
        if isinstance(onx, GraphProto):
            for init in onx.initializer:
                yield (init.name,), onx, init
            for initp in onx.sparse_initializer:
                yield (initp.indices.name or initp.values.name,), onx, initp
        for i, node in enumerate(onx.node):
            if node.op_type == "Constant":
                yield (node.output[0],), onx, node
            else:
                yield (node.name or f"#{i}",), onx, node
            if recursive:
                for att in node.attribute:
                    if att.g:
                        for c, parent, inner in enumerate_nodes(att.g, recursive=recursive):
                            if isinstance(inner, NodeProto):
                                n = inner.name or f"#{i}"
                            elif isinstance(inner, TensorProto):
                                n = inner.name
                            elif isinstance(inner, SparseTensorProto):
                                n = inner.indices.name or inner.values.name
                            else:
                                raise TypeError(f"Unexpected type {type(inner)}.")
                            yield (f"{n}/{att.name}", *c), parent, inner


def stats_tree_ensemble(
    parent: Union[GraphProto, FunctionProto], node: NodeProto
) -> NodeStatistics:
    """
    Computes statistics on every tree of a ``TreeEnsembleClassifier`` or
    ``TreeEnsembleRegressor`` node.

    The returned :class:`NodeStatistics` instance contains the following entries:

    - ``"kind"`` – ``"Classifier"`` or ``"Regressor"``
    - ``"n_trees"`` – total number of trees
    - ``"n_outputs"`` – number of outputs / classes
    - ``"max_featureid"`` – maximum feature index used across all nodes
    - ``"n_features"`` – number of distinct features used across all nodes
    - ``"n_rules"`` – number of distinct node modes (split types) used
    - ``"rules"`` – :class:`set` of node mode strings (e.g. ``{"BRANCH_LEQ", "LEAF"}``)
    - ``"hist_rules"`` – :class:`collections.Counter` of node mode frequencies
    - ``"features"`` – list of :class:`HistTreeStatistics`, one per feature
    - ``"trees"`` – list of :class:`TreeStatistics`, one per tree

    Each :class:`TreeStatistics` in ``"trees"`` contains:

    - ``"n_nodes"`` – total nodes in the tree
    - ``"n_leaves"`` – leaf nodes
    - ``"max_featureid"`` – maximum feature index
    - ``"n_features"`` – distinct feature count
    - ``"n_rules"`` – distinct split-mode count
    - ``"rules"`` – :class:`set` of mode strings
    - ``"hist_rules"`` – :class:`collections.Counter` of mode frequencies

    :param parent: the :class:`~onnx.GraphProto` or :class:`~onnx.FunctionProto`
        that contains *node*
    :param node: a ``TreeEnsembleClassifier`` or ``TreeEnsembleRegressor`` node
    :return: :class:`NodeStatistics` populated with the statistics listed above
    :raises KeyError: if required tree-structure attributes are missing from *node*
    """
    stats = NodeStatistics(parent, node)
    atts = {k: v[1] for k, v in extract_attributes(node).items()}
    unique = set(atts["nodes_treeids"])
    stats.add("kind", "Regressor" if "n_targets" in atts else "Classifier")
    stats.add("n_trees", len(unique))
    stats.add(
        "n_outputs", atts["n_targets"] if "n_targets" in atts else len(set(atts["class_ids"]))
    )
    stats.add("max_featureid", int(max(atts["nodes_featureids"])))
    stats.add("n_features", len(set(atts["nodes_featureids"])))
    stats.add("n_rules", len(set(atts["nodes_modes"])))
    stats.add("rules", set(atts["nodes_modes"]))
    stats.add("hist_rules", Counter(atts["nodes_modes"]))

    features = []
    for fid in sorted(set(atts["nodes_featureids"])):
        indices = atts["nodes_featureids"] == fid
        features.append(HistTreeStatistics(node, fid, atts["nodes_values"][indices]))
    stats.add("features", features)

    atts_nodes = {k: v for k, v in atts.items() if k.startswith("nodes")}
    tree_stats = []
    for treeid in sorted(unique):
        tr = TreeStatistics(node, treeid)
        indices = atts_nodes["nodes_treeids"] == treeid
        atts_tree = {k: v[indices] for k, v in atts_nodes.items()}
        tr.add("n_nodes", len(atts_tree["nodes_nodeids"]))
        tr.add("n_leaves", int(np.sum(atts_tree["nodes_modes"] == "LEAF")))
        tr.add("max_featureid", int(max(atts_tree["nodes_featureids"])))
        tr.add("n_features", len(set(atts_tree["nodes_featureids"])))
        tr.add("n_rules", len(set(atts_tree["nodes_modes"])))
        tr.add("rules", set(atts_tree["nodes_modes"]))
        tr.add("hist_rules", Counter(atts_tree["nodes_modes"]))
        tree_stats.append(tr)
    stats.add("trees", tree_stats)
    return stats


def enumerate_stats_nodes(
    onx: Union[FunctionProto, GraphProto, ModelProto],
    recursive: bool = True,
    stats_fcts: Optional[
        Dict[
            Tuple[str, str],
            Callable[
                [
                    Union[GraphProto, FunctionProto],
                    Union[NodeProto, TensorProto, SparseTensorProto],
                ],
                Union[NodeStatistics, "HistStatistics"],
            ],
        ]
    ] = None,
) -> Iterable[
    Tuple[
        Tuple[str, ...], Union[GraphProto, FunctionProto], Union[NodeStatistics, "HistStatistics"]
    ]
]:
    """
    Iterates over nodes in *onx*, yielding statistics for those that match
    entries in *stats_fcts*.

    By default the function handles both ``TreeEnsembleClassifier`` and
    ``TreeEnsembleRegressor`` nodes in the ``"ai.onnx.ml"`` domain via
    :func:`stats_tree_ensemble`.

    :param onx: the model, graph, or function to traverse
    :param recursive: if ``True``, recurse into sub-graphs
    :param stats_fcts: mapping of ``(domain, op_type)`` to a callable that
        accepts ``(parent, node)`` and returns a statistics object.  When
        ``None`` the default handlers for tree-ensemble operators are used.
    :return: yields tuples ``(path, parent, statistics)`` for every matched node
    """
    if stats_fcts is None:
        stats_fcts = {
            ("ai.onnx.ml", "TreeEnsembleRegressor"): stats_tree_ensemble,
            ("ai.onnx.ml", "TreeEnsembleClassifier"): stats_tree_ensemble,
        }
    for name, parent, node in enumerate_nodes(onx, recursive=recursive):
        if isinstance(node, NodeProto):
            if (node.domain, node.op_type) in stats_fcts:
                stat = stats_fcts[node.domain, node.op_type](parent, node)
                yield name, parent, stat


class HistStatistics(_Statistics):
    """
    Stores distribution statistics for a constant tensor (initializer or
    ``Constant`` node).

    :param parent: the :class:`~onnx.GraphProto` or :class:`~onnx.FunctionProto`
        that contains *node*
    :param node: a :class:`~onnx.NodeProto` (``Constant`` op),
        :class:`~onnx.TensorProto`, or :class:`~onnx.SparseTensorProto`
    :param bins: number of histogram bins (default ``20``)
    """

    def __init__(
        self,
        parent: Union[GraphProto, FunctionProto],
        node: Union[NodeProto, TensorProto, SparseTensorProto],
        bins: int = 20,
    ) -> None:
        _Statistics.__init__(self)
        self.parent = parent
        self.node = node
        values = self._values

        self.add("shape", values.shape)
        self.add("dtype", values.dtype)
        self.add("min", float(values.min()))
        self.add("max", float(values.max()))
        self.add("mean", float(values.mean()))
        self.add("median", float(np.median(values)))
        self.add("size", int(values.size))
        n_distinct = len(set(values.ravel()))
        self.add("n_distinct", n_distinct)
        if values.size > 1:
            self.add("hist", np.histogram(values, bins))
        else:
            self.add("hist", (values, np.array([1], dtype=np.int64)))
        if n_distinct <= 50:
            self.add("v_distinct", set(values.ravel()))

    @property
    def _values(self) -> np.ndarray:
        """Returns the tensor values as a NumPy array."""
        if isinstance(self.node, NodeProto):
            # Constant node: extract the value from its attribute.
            for att in self.node.attribute:
                if att.type == AttributeProto.TENSOR:
                    return onh.to_array(att.t)
            raise ValueError(f"Constant node {self.node.name!r} has no TENSOR attribute.")
        if isinstance(self.node, SparseTensorProto):
            return onh.to_array(self.node.values)
        return onh.to_array(self.node)

    @property
    def name(self) -> str:
        """Returns the tensor name."""
        if isinstance(self.node, SparseTensorProto):
            return self.node.indices.name or self.node.values.name
        if isinstance(self.node, NodeProto):
            return self.node.output[0]
        return self.node.name

    def __str__(self) -> str:
        """Returns a human-readable representation."""
        if isinstance(self.node, NodeProto):
            return (
                f"{self.__class__.__name__}(<{self.parent.name}>, "
                f"<{self.node.op_type}>,\n"
                f"{pprint.pformat(self._statistics)})"
            )
        return (
            f"{self.__class__.__name__}(<{self.parent.name}>, <{self.name}>,\n"
            f"{pprint.pformat(self._statistics)})"
        )

    @property
    def dict_values(self) -> Dict[str, Any]:
        """Returns the statistics as a flat dictionary for DataFrame construction."""
        obs: Dict[str, Any] = {}
        for k in ("size", "shape", "dtype", "min", "max", "mean", "median", "n_distinct"):
            obs[k] = self[k]
        hist = self["hist"]
        if hist[0].size > 0 and len(hist[0].shape) > 0:
            for i, v in enumerate(hist[0]):
                obs[f"hist_y_{i}"] = v
            for i, v in enumerate(hist[1]):
                obs[f"hist_x_{i}"] = v
        return obs

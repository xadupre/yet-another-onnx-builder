"""
:class:`ExportArtifact` and :class:`ExportReport` — standard output
of every :func:`to_onnx` conversion function.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import onnx
from ..typing import GraphBuilderExtendedProtocol, ExportArtifactProtocol
from .model_container import ExtendedModelContainer
from .build_stats import BuildStats


class ExportReport:
    """
    Holds statistics and metadata gathered during an ONNX export.

    The :attr:`_stats` attribute stores the per-pattern optimization statistics
    returned by :meth:`~yobx.xbuilder.GraphBuilder.to_onnx` when called
    with ``return_optimize_report=True``.  Each element of the list is a
    dict with at least the keys ``"pattern"``, ``"added"``, ``"removed"``,
    and ``"time_in"``.

    Additional arbitrary key-value pairs can be recorded via the
    :meth:`update` method and are stored in :attr:`extra`.

    Node-level statistics (op-type counts and estimated FLOPs) are stored in
    :attr:`node_stats` and can be populated by calling
    :meth:`compute_node_stats`.

    Per-input-set discrepancy records from
    :meth:`~yobx.torch.InputObserver.check_discrepancies` can be stored in
    :attr:`discrepancies`.  When present, :meth:`to_excel` writes a
    ``"discrepancies"`` sheet to the workbook.

    Example::

        report = ExportReport()
        report.update({"time_total": 0.42})
        print(report)
    """

    def __init__(
        self,
        stats: Optional[List[Dict[str, Any]]] = None,
        extra: Optional[Dict[str, Any]] = None,
        build_stats: Optional[BuildStats] = None,
        node_stats: Optional[List[Dict[str, Any]]] = None,
        symbolic_flops: Optional[List[Dict[str, Any]]] = None,
        discrepancies: Optional[List[Dict[str, Any]]] = None,
    ):
        self.stats: List[Dict[str, Any]] = stats if stats is not None else []
        self.extra: Dict[str, Any] = extra if extra is not None else {}
        self.build_stats = build_stats
        self.node_stats: List[Dict[str, Any]] = node_stats if node_stats is not None else []
        self.symbolic_flops: List[Dict[str, Any]] = (
            symbolic_flops if symbolic_flops is not None else []
        )
        self.discrepancies: List[Dict[str, Any]] = (
            discrepancies if discrepancies is not None else []
        )

    def update(self, data: Any):
        """
        Appends *data* to the report.

        :param data: anything
        :return: self
        """
        if isinstance(data, dict):
            self.extra.update(data)
        elif isinstance(data, list):
            self.stats.extend(data)
        elif isinstance(data, BuildStats):
            self.build_stats = data
        elif isinstance(data, ExportReport):
            self.update(data.stats)
            self.update(data.extra)
            if data.build_stats:
                self.update(data.build_stats)
            if data.node_stats:
                self.node_stats.extend(data.node_stats)
            if data.symbolic_flops:
                self.symbolic_flops.extend(data.symbolic_flops)
            if data.discrepancies:
                self.discrepancies.extend(data.discrepancies)
        else:
            raise TypeError(f"Unexpected type {type(data)} for data.")
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary representation of this report.

        :return: dictionary with keys ``"stats"`` and ``"extra"``.
        """
        return {
            "stats": self.stats,
            "extra": self.extra,
            "build_stats": self.build_stats,
            "node_stats": self.node_stats,
            "symbolic_flops": self.symbolic_flops,
            "discrepancies": self.discrepancies,
        }

    def to_string(self) -> str:
        """Return a human-readable text summary of this report.

        The output includes:

        * Any extra key/value pairs stored in :attr:`extra`.
        * A per-pattern optimization table built from :attr:`stats` (when
          non-empty), grouped and sorted by the number of nodes removed.
        * Timing entries from :attr:`build_stats` (when available).

        :return: multi-line string suitable for printing.

        .. runpython::
            :showcode:

            from yobx.container import ExportReport

            report = ExportReport(
                stats=[
                    {"pattern": "p1", "added": 1, "removed": 2, "time_in": 0.01},
                    {"pattern": "p1", "added": 0, "removed": 1, "time_in": 0.02},
                ],
                extra={"time_total": 0.42},
            )
            print(report.to_string())
        """
        lines: List[str] = []

        if self.extra:
            lines.append("-- extra --")
            for k, v in sorted(self.extra.items()):
                lines.append(f"  {k}: {v}")

        if self.stats:
            import pandas

            df = pandas.DataFrame(self.stats)
            for col in ["added", "removed"]:
                if col in df.columns:
                    df[col] = df[col].fillna(0).astype(int)
            num_cols = [c for c in ["added", "removed", "time_in"] if c in df.columns]
            if "pattern" in df.columns and num_cols:
                agg = df.groupby("pattern")[num_cols].sum()
                mask = pandas.Series(False, index=agg.index)
                for col in ["added", "removed"]:
                    if col in agg.columns:
                        mask |= agg[col] > 0  # type: ignore
                agg = agg[mask]
                if "removed" in agg.columns:
                    agg = agg.sort_values("removed", ascending=False)
                if not agg.empty:
                    lines.append("-- stats (aggregated by pattern) --")
                    lines.append(agg.to_string())
            else:
                lines.append("-- stats --")
                lines.append(df.to_string())

        if self.build_stats:
            d = self.build_stats.to_dict()
            if d:
                lines.append("-- build_stats --")
                for k, v in sorted(d.items()):
                    lines.append(f"  {k}: {v}")

        if self.node_stats:
            import pandas

            df_ns = pandas.DataFrame(self.node_stats)
            lines.append("-- node_stats --")
            lines.append(df_ns.to_string(index=False))

        if self.symbolic_flops:
            import pandas

            df_sf = pandas.DataFrame(self.symbolic_flops)
            lines.append("-- symbolic_flops --")
            lines.append(df_sf.to_string(index=False))

        if self.discrepancies:
            import pandas

            df_disc = pandas.DataFrame(self.discrepancies)
            lines.append("-- discrepancies --")
            lines.append(df_disc.to_string(index=False))

        return "\n".join(lines)

    def to_excel(self, path: str) -> None:
        """Write the report contents to an Excel workbook at *path*.

        The workbook contains up to seven sheets:

        ``stats``
            Every row from :attr:`stats` as a raw :class:`~pandas.DataFrame`.
            Only written when :attr:`stats` is non-empty.
        ``stats_agg``
            The same data aggregated (summed) by ``"pattern"``, sorted by the
            number of removed nodes in descending order.  Only written when
            :attr:`stats` is non-empty and contains a ``"pattern"`` column.
        ``extra``
            :attr:`extra` rendered as a two-column table (``key``, ``value``).
            Only written when :attr:`extra` is non-empty.
        ``build_stats``
            :attr:`build_stats` rendered as a two-column table (``key``,
            ``value``).  Only written when :attr:`build_stats` is not ``None``
            and contains at least one entry.
        ``node_stats``
            Per-op-type node counts and estimated FLOPs from
            :attr:`node_stats`.  Only written when :attr:`node_stats` is
            non-empty.  Populated by :meth:`compute_node_stats`.
        ``symbolic_flops``
            Per-node symbolic FLOPs expressions from :attr:`symbolic_flops`.
            Each row has ``op_type``, ``node_name``, and ``symbolic_flops``
            columns.  Only written when :attr:`symbolic_flops` is non-empty.
            Populated by :meth:`compute_symbolic_flops`.
        ``discrepancies``
            Per-input-set discrepancy records from
            :meth:`~yobx.torch.InputObserver.check_discrepancies`.  Each row
            corresponds to one forward call and contains at least the columns
            ``index``, ``SUCCESS``, ``abs``, and ``rel``.  Only written when
            :attr:`discrepancies` is non-empty.

        :param path: destination file path (e.g. ``"report.xlsx"``).

        .. runpython::
            :showcode:

            import tempfile, os
            from yobx.container import ExportReport

            report = ExportReport(
                stats=[
                    {"pattern": "p1", "added": 1, "removed": 2, "time_in": 0.01},
                ],
                extra={"time_total": 0.42},
            )
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "report.xlsx")
                report.to_excel(path)
                print(f"Saved to {os.path.basename(path)!r}")
        """
        import pandas

        with pandas.ExcelWriter(path, engine="openpyxl") as writer:
            if self.stats:
                df_stats = pandas.DataFrame(self.stats)
                for col in ["added", "removed"]:
                    if col in df_stats.columns:
                        df_stats[col] = df_stats[col].fillna(0).astype(int)
                df_stats.to_excel(writer, sheet_name="stats", index=False)

                num_cols = [c for c in ["added", "removed", "time_in"] if c in df_stats.columns]
                if "pattern" in df_stats.columns and num_cols:
                    agg = df_stats.groupby("pattern")[num_cols].sum()
                    if "removed" in agg.columns:
                        agg = agg.sort_values("removed", ascending=False)
                    agg.reset_index().to_excel(writer, sheet_name="stats_agg", index=False)

            if self.extra:
                df_extra = pandas.DataFrame(list(self.extra.items()), columns=["key", "value"])
                df_extra.to_excel(writer, sheet_name="extra", index=False)

            if self.build_stats:
                d = self.build_stats.to_dict()
                if d:
                    df_bs = pandas.DataFrame(list(d.items()), columns=["key", "value"])
                    df_bs.to_excel(writer, sheet_name="build_stats", index=False)

            if self.node_stats:
                df_ns = pandas.DataFrame(self.node_stats)
                df_ns.to_excel(writer, sheet_name="node_stats", index=False)

            if self.symbolic_flops:
                df_sf = pandas.DataFrame(self.symbolic_flops)
                df_sf.to_excel(writer, sheet_name="symbolic_flops", index=False)

            if self.discrepancies:
                df_disc = pandas.DataFrame(self.discrepancies)
                df_disc.to_excel(writer, sheet_name="discrepancies", index=False)

    def compute_node_stats(self, model: "onnx.ModelProto", verbose: int = 0) -> "ExportReport":
        """Compute per-op-type node counts and estimated FLOPs from *model*.

        Delegates to :func:`~yobx.helpers.stats_helper.model_statistics` for
        shape inference and FLOPs estimation.  Results are stored in
        :attr:`node_stats` as a list of dicts with the keys:

        ``op_type``
            ONNX operator type name (e.g. ``"MatMul"``).
        ``count``
            Number of nodes with this op_type.
        ``flops``
            Estimated total floating-point operations for all nodes of this
            type.  ``None`` when the estimate is unavailable (shapes not
            fully known or the op is not covered by the estimator).

        The rows are sorted by ``count`` descending.

        :param model: exported ONNX model proto.
        :param verbose: verbosity level passed to
            :class:`~yobx.helpers.stats_helper.ModelStatistics`.
        :return: ``self``, to allow method chaining.

        .. runpython::
            :showcode:

            import onnx.helper as oh
            import onnx
            from yobx.container import ExportReport

            X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [4, 8])
            Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [4, 8])
            node = oh.make_node("Relu", ["X"], ["Y"])
            graph = oh.make_graph([node], "g", [X], [Y])
            model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])

            report = ExportReport()
            report.compute_node_stats(model)
            print(report.node_stats)
        """
        from ..helpers.stats_helper import model_statistics

        result = model_statistics(model, verbose=verbose)
        counts = result["node_count_per_op_type"]
        flops = result["flops_per_op_type"]

        rows = [
            {"op_type": op_type, "count": counts[op_type], "flops": flops.get(op_type)}
            for op_type in counts
        ]
        rows.sort(key=lambda r: -r["count"])
        self.node_stats = rows
        return self

    def compute_symbolic_flops(
        self, model: "onnx.ModelProto", verbose: int = 0
    ) -> "ExportReport":
        """Compute per-node symbolic FLOPs from *model* using symbolic shape inference.

        Uses :class:`~yobx.xshape.BasicShapeBuilder` with
        ``inference=InferenceMode.COST`` to walk every node and record the
        FLOPs expression.  When the model's input shapes contain symbolic
        dimensions (strings such as ``"batch"`` or ``"seq"``), the returned
        expressions are symbolic arithmetic strings
        (e.g. ``"2*batch*d_head*seq*seq"``).  For models with fully static
        shapes the values are plain integers.  Operators whose cost cannot be
        estimated produce ``None``.

        Results are stored in :attr:`symbolic_flops` as a list of dicts with
        the keys:

        ``op_type``
            ONNX operator type name (e.g. ``"MatMul"``).
        ``node_name``
            The name of the ONNX node (may be an empty string).
        ``symbolic_flops``
            Symbolic FLOPs expression (str), concrete integer, or ``None``
            when the estimate is unavailable.

        :param model: exported ONNX model proto.
        :param verbose: verbosity level passed to
            :class:`~yobx.xshape.BasicShapeBuilder`.
        :return: ``self``, to allow method chaining.

        .. runpython::
            :showcode:

            import onnx.helper as oh
            import onnx
            from yobx.container import ExportReport

            X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, ["batch", "d"])
            Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, ["batch", "d"])
            node = oh.make_node("Relu", ["X"], ["Y"])
            graph = oh.make_graph([node], "g", [X], [Y])
            model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])

            report = ExportReport()
            report.compute_symbolic_flops(model)
            print(report.symbolic_flops)
        """
        from ..xshape import BasicShapeBuilder, InferenceMode

        bs = BasicShapeBuilder(verbose=verbose)
        cost = bs.run_model(model, inference=InferenceMode.COST)

        rows = []
        for node, (op_type, flops, _) in zip(model.graph.node, cost):
            rows.append({"op_type": op_type, "node_name": node.name, "symbolic_flops": flops})
        self.symbolic_flops = rows
        return self

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_stats={len(self.stats)}, "
            f"extra={sorted(self.extra)}, "
            f"has_build_stats={self.build_stats is not None}, "
            f"n_node_stats={len(self.node_stats)}, "
            f"n_symbolic_flops={len(self.symbolic_flops)}, "
            f"n_discrepancies={len(self.discrepancies)})"
        )


class FunctionPieces:
    """
    Holds the function-specific data produced by
    :meth:`~yobx.xbuilder.GraphBuilder._to_onnx_function`.

    An instance of this class is stored as :attr:`ExportArtifact.function`
    when :meth:`~yobx.xbuilder.GraphBuilder.to_onnx` is called with
    ``function_options.export_as_function=True``.  It is ``None`` for regular
    model exports.

    Args:
        initializers_name : list[str] | None
            Ordered list of extra input names that were promoted from
            initializers into function inputs when ``return_initializer=True``
            was set in :class:`~yobx.xbuilder.FunctionOptions`.  ``None``
            when no initializers were promoted.
        initializers_dict : dict[str, Any] | None
            Mapping from (possibly renamed) initializer name to tensor value
            for every name listed in :attr:`initializers_name`.  ``None``
            when no initializers were promoted.
        initializers_renaming : dict[str, str] | None
            Mapping from original initializer name to the (possibly renamed)
            name used in :attr:`initializers_name`.  ``None`` when no
            initializers were promoted.
        nested_functions : list[FunctionProto] | None
            Local :class:`~onnx.FunctionProto` objects defined inside the
            exported function that are needed to evaluate it.  ``None`` when
            there are no nested functions.
        nested_function_names: set of (domain, name) used by the GraphBuilder
            owning the functions
    """

    def __init__(
        self,
        initializers_name: Optional[List[str]] = None,
        initializers_dict: Optional[Dict[str, Any]] = None,
        initializers_renaming: Optional[Dict[str, str]] = None,
        nested_functions: Optional[List[onnx.FunctionProto]] = None,
        nested_function_names: Optional[Set[Tuple[str, str]]] = None,
    ):
        self.initializers_name = initializers_name
        self.initializers_dict = initializers_dict
        self.initializers_renaming = initializers_renaming
        self.nested_functions = nested_functions
        self.nested_function_names = nested_function_names

    def __repr__(self) -> str:
        n_nested = len(self.nested_functions or self.nested_function_names or [])
        return (
            f"{self.__class__.__name__}("
            f"n_initializers={len(self.initializers_name) if self.initializers_name else 0}, "
            f"n_nested_functions={n_nested})"
        )


class ExportArtifact(ExportArtifactProtocol):
    """
    Standard output of every :func:`to_onnx` conversion function.

    Every top-level ``to_onnx`` function (sklearn, tensorflow, litert,
    torch, sql …) returns an :class:`ExportArtifact` instead of a bare
    :class:`~onnx.ModelProto` or
    :class:`~yobx.container.ExtendedModelContainer`.
    The instance bundles the exported proto, the optional large-model
    container, an :class:`ExportReport` describing the export process,
    and an optional filename.

    Args:
        proto : ModelProto | FunctionProto | GraphProto | None
            The ONNX proto produced by the export.  When *large_model* was
            requested the proto contains *placeholders* for external data;
            use :meth:`get_proto` to obtain a fully self-contained proto.
        container : ExtendedModelContainer | None
            The :class:`~yobx.container.ExtendedModelContainer` produced when
            the conversion was called with ``large_model=True``.  ``None``
            otherwise.
        report : ExportReport | None
            Statistics and metadata about the export.
        filename : str | None
            Path where the model was last saved, or ``None`` if never saved.
        builder: GraphBuilderExtendedProtocol
            Keeps the builder building the onnx model.
        function : FunctionPieces | None
            When the export was performed with
            ``function_options.export_as_function=True``, this holds a
            :class:`FunctionPieces` instance with function-specific data
            (initializer names/values and nested local functions).  ``None``
            for regular model exports.

    Example::

        import numpy as np
        from sklearn.linear_model import LinearRegression
        from yobx.sklearn import to_onnx
        from yobx.container import ExportArtifact, ExportReport

        X = np.random.randn(20, 4).astype(np.float32)
        y = X @ np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        reg = LinearRegression().fit(X, y)

        artifact = to_onnx(reg, (X,))
        assert isinstance(artifact, ExportArtifact)
        assert isinstance(artifact.report, ExportReport)

        proto = artifact.get_proto()
        artifact.save("model.onnx")
    """

    def __init__(
        self,
        proto: Optional[Union[onnx.ModelProto, onnx.GraphProto, onnx.FunctionProto]] = None,
        container: Optional[ExtendedModelContainer] = None,
        report: Optional[ExportReport] = None,
        filename: Optional[str] = None,
        builder: Optional[GraphBuilderExtendedProtocol] = None,
        function: Optional["FunctionPieces"] = None,
    ):
        assert not proto or isinstance(
            proto, (onnx.ModelProto, onnx.GraphProto, onnx.FunctionProto)
        ), f"Unexpected type {proto} for proto"
        assert not container or isinstance(
            container, ExtendedModelContainer
        ), f"Unexpected type {proto} for container"
        self.proto = proto
        self.container = container
        self.report = report
        self.filename = filename
        self.builder = builder
        self.function = function
        if self.container and self.container._stats:
            if not self.report:
                self.report = ExportReport(build_stats=self.container._stats)
            else:
                self.report.build_stats = self.container._stats

    def update(self, data: Any):
        """Updates report."""
        if not self.report:
            self.report = ExportReport()
        self.report.update(data)

    def compute_node_stats(self, verbose: int = 0) -> "ExportArtifact":
        """Compute per-op-type node counts and estimated FLOPs and store them in the report.

        Delegates to :meth:`ExportReport.compute_node_stats` using this
        artifact's :attr:`proto`.  Does nothing when :attr:`proto` is not a
        :class:`~onnx.ModelProto`.

        :param verbose: verbosity level passed to
            :class:`~yobx.helpers.stats_helper.ModelStatistics`.
        :return: ``self``, to allow method chaining.
        """
        if not isinstance(self.proto, onnx.ModelProto):
            return self
        if self.report is None:
            self.report = ExportReport()
        self.report.compute_node_stats(self.proto, verbose=verbose)
        return self

    def compute_symbolic_flops(self, verbose: int = 0) -> "ExportArtifact":
        """Compute per-node symbolic FLOPs and store them in the report.

        Delegates to :meth:`ExportReport.compute_symbolic_flops` using this
        artifact's :attr:`proto`.  Does nothing when :attr:`proto` is not a
        :class:`~onnx.ModelProto`.

        :param verbose: verbosity level passed to
            :class:`~yobx.xshape.BasicShapeBuilder`.
        :return: ``self``, to allow method chaining.
        """
        if not isinstance(self.proto, onnx.ModelProto):
            return self
        if self.report is None:
            self.report = ExportReport()
        self.report.compute_symbolic_flops(self.proto, verbose=verbose)
        return self

    def save_report(self, onnx_path: str) -> None:
        """Save the report as an Excel file alongside *onnx_path* when available."""
        if self.report is None:
            return
        import os

        if not self.report.node_stats and isinstance(self.proto, onnx.ModelProto):
            self.compute_node_stats()

        if not self.report.symbolic_flops and isinstance(self.proto, onnx.ModelProto):
            self.compute_symbolic_flops()

        excel_path = os.path.splitext(onnx_path)[0] + ".xlsx"
        self.report.to_excel(excel_path)

    def save(self, file_path: str, all_tensors_to_one_file: bool = True) -> Any:
        """Save the exported model to *file_path*.

        When a :class:`~yobx.container.ExtendedModelContainer` is present
        (``large_model=True`` was used during export) the model and its
        external weight files are saved via
        :meth:`~yobx.container.ExtendedModelContainer.save`.  Otherwise
        the proto is saved with :func:`onnx.save_model`.

        :param file_path: destination file path (including ``.onnx``
            extension).
        :param all_tensors_to_one_file: when saving a large model, write
            all external tensors into a single companion data file.
        :return: the saved :class:`~onnx.ModelProto`.

        Example::

            artifact = to_onnx(estimator, (X,))
            artifact.save("model.onnx")
        """
        assert not self.function, f"save is not implemented when function is not empty {self!r}"
        if self.container is not None:
            result = self.container.save(
                file_path, all_tensors_to_one_file=all_tensors_to_one_file
            )
            self.filename = file_path
            self.save_report(file_path)
            return result
        import onnx

        if isinstance(self.proto, onnx.ModelProto):
            onnx.save_model(self.proto, file_path)
            self.filename = file_path
            self.save_report(file_path)
            return self.proto
        raise TypeError(
            f"Cannot save a proto of type {type(self.proto).__name__}. "
            "Only ModelProto is directly saveable; for FunctionProto or "
            "GraphProto embed it in a ModelProto first."
        )

    def get_proto(self, include_weights: bool = True) -> Any:
        """Return the ONNX proto, optionally with all weights inlined.

        When the export was performed with ``large_model=True`` (i.e.
        :attr:`container` is set), the raw :attr:`proto` has
        *external-data* placeholders instead of embedded weight tensors.
        Passing ``include_weights=True`` (the default) uses
        :meth:`~yobx.container.ExtendedModelContainer.to_ir` to build a
        fully self-contained :class:`~onnx.ModelProto`.

        :param include_weights: when ``True`` (default) embed the large
            initializers stored in :attr:`container` into the returned
            proto.  When ``False`` return the raw proto as-is.
        :return: :class:`~onnx.ModelProto`,
            :class:`~onnx.FunctionProto`, or
            :class:`~onnx.GraphProto`.

        Example::

            artifact = to_onnx(estimator, (X,), large_model=True)
            # Fully self-contained proto (weights embedded):
            proto_with_weights = artifact.get_proto(include_weights=True)
            # Proto with external-data placeholders:
            proto_no_weights = artifact.get_proto(include_weights=False)
        """
        if self.container is None:
            return self.proto

        if not include_weights:
            return self.container.model_proto
        return self.container.get_model_with_data()

    @classmethod
    def load(cls, file_path: str, load_large_initializers: bool = True) -> "ExportArtifact":
        """Load a saved model from *file_path*.

        If the file references external data (i.e. the model was saved
        with ``large_model=True``) an
        :class:`~yobx.container.ExtendedModelContainer` is created and
        returned in :attr:`container`.  Otherwise the proto is loaded
        directly with :func:`onnx.load` and :attr:`container` is ``None``.

        :param file_path: path to the ``.onnx`` file.
        :param load_large_initializers: when ``True`` (default) also load
            the large initializers stored alongside the model file.
        :return: :class:`ExportArtifact` with :attr:`filename` set to
            *file_path*.

        Example::

            artifact = ExportArtifact.load("model.onnx")
            proto = artifact.get_proto()
        """
        import onnx
        from onnx.external_data_helper import _get_all_tensors, uses_external_data

        from .model_container import ExtendedModelContainer

        # Load without external data first to inspect the proto.
        proto = onnx.load(file_path, load_external_data=False)
        has_external = any(uses_external_data(t) for t in _get_all_tensors(proto))

        if has_external:
            container = ExtendedModelContainer()
            container.load(file_path, load_large_initializers=load_large_initializers)
            return cls(proto=container.model_proto, container=container, filename=file_path)

        # Regular model — no external data, proto is already complete.
        return cls(proto=proto, filename=file_path)

    def __repr__(self) -> str:
        proto_type = type(self.proto).__name__ if self.proto is not None else "None"
        container_repr = (
            f"{type(self.container).__name__}()" if self.container is not None else "None"
        )
        return (
            f"{self.__class__.__name__}("
            f"proto={proto_type}, "
            f"container={container_repr}, "
            f"filename={self.filename!r}, "
            f"report={self.report!r})"
        )

    @property
    def graph(self) -> onnx.GraphProto:
        """Returns the GraphProto is the model is available. Fails otherwise."""
        if self.proto:
            if isinstance(self.proto, onnx.ModelProto):
                return self.proto.graph
            if isinstance(self.proto, onnx.GraphProto):
                return self.proto
            raise TypeError(f"Unable to return a GraphProto from type {self.proto}.")
        if self.container and self.container.model_proto_:
            return self.container.model_proto_.graph
        raise AttributeError(f"The artifact do not contain any model {self!r}.")

    def SerializeToString(self) -> bytes:
        """
        Serializes the model to bytes.
        It does not includes weights if the model is stored in a container.
        """
        if self.proto:
            return self.proto.SerializeToString()
        if self.container and self.container.model_proto_:
            return self.container.model_proto_.SerializeToString()
        raise ValueError(f"There is nothing to serialize in {self!r}.")

    @property
    def opset_import(self) -> Sequence[onnx.OperatorSetIdProto]:
        """Returns the opset import."""
        if self.proto:
            if isinstance(self.proto, (onnx.ModelProto, onnx.FunctionProto)):
                return self.proto.opset_import
            raise TypeError(f"Unable to return opsets from type {self.proto}.")
        if self.container and self.container.model_proto_:
            return self.container.model_proto_.opset_import
        raise AttributeError(f"The artifact do not contain any model or function {self!r}.")

    @property
    def functions(self) -> Sequence[onnx.FunctionProto]:
        """Returns the opset import."""
        if self.proto:
            if isinstance(self.proto, onnx.ModelProto):
                return self.proto.functions
            raise TypeError(f"Unable to return a GraphProto from type {self.proto}.")
        if self.container and self.container.model_proto_:
            return self.container.model_proto_.functions
        raise AttributeError(f"The artifact do not contain any model {self!r}.")

    @property
    def metadata_props(self) -> Sequence[onnx.StringStringEntryProto]:
        """Returns the opset import."""
        if self.proto:
            if hasattr(self.proto, "metadata_props"):
                return self.proto.metadata_props
            raise TypeError(f"Unable to return metadata_props from type {self.proto}.")
        if self.container and self.container.model_proto_:
            return self.container.model_proto_.metadata_props
        raise AttributeError(f"The artifact do not contain any model {self!r}.")

    @property
    def ir_version(self) -> int:
        """Returns the opset import."""
        if self.proto:
            if hasattr(self.proto, "ir_version"):
                return self.proto.ir_version
            raise TypeError(f"Unable to return ir_version from type {self.proto}.")
        if self.container and self.container.model_proto_:
            return self.container.model_proto_.ir_version
        raise AttributeError(f"The artifact do not contain any model {self!r}.")

    @property
    def input_names(self) -> List[str]:
        """Returns the input names."""
        return [i.name for i in self.graph.input]

    @property
    def output_names(self) -> List[str]:
        """Returns the output names."""
        return [i.name for i in self.graph.output]

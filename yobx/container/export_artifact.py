"""
:class:`ExportArtifact` and :class:`ExportReport` — standard output
of every :func:`to_onnx` conversion function.
"""

from typing import Any, Dict, List, Optional, Union


class ExportReport:
    """
    Holds statistics and metadata gathered during an ONNX export.

    The *stats* attribute stores the per-pattern optimization statistics
    returned by :meth:`~yobx.xbuilder.GraphBuilder.to_onnx` when called
    with ``return_optimize_report=True``.  Each element of the list is a
    dict with at least the keys ``"pattern"``, ``"added"``, ``"removed"``,
    and ``"time_in"``.

    Additional arbitrary key-value pairs can be recorded via the
    :meth:`update` method.

    Example::

        report = ExportReport()
        report.update({"time_total": 0.42})
        print(report)
    """

    def __init__(
        self,
        stats: Optional[List[Dict[str, Any]]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self._stats: List[Dict[str, Any]] = stats if stats is not None else []
        self._extra: Dict[str, Any] = extra if extra is not None else {}

    @property
    def stats(self) -> List[Dict[str, Any]]:
        """Per-pattern optimization statistics (list of dicts)."""
        return self._stats

    @property
    def extra(self) -> Dict[str, Any]:
        """Additional export metadata (arbitrary key-value pairs)."""
        return self._extra

    def update(self, data: Dict[str, Any]) -> None:
        """Merge *data* into :attr:`extra`.

        :param data: key-value pairs to add to :attr:`extra`.
        """
        self._extra.update(data)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary representation of this report.

        :return: dictionary with keys ``"stats"`` and ``"extra"``.
        """
        return {"stats": self._stats, "extra": self._extra}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_stats={len(self._stats)}, "
            f"extra={sorted(self._extra)})"
        )


class ExportArtifact:
    """
    Standard output of every :func:`to_onnx` conversion function.

    Every top-level ``to_onnx`` function (sklearn, tensorflow, litert,
    torch, sql …) returns an :class:`ExportArtifact` instead of a bare
    :class:`~onnx.ModelProto` or
    :class:`~yobx.container.ExtendedModelContainer`.
    The instance bundles the exported proto, the optional large-model
    container, and an :class:`ExportReport` describing the export process.

    Attributes
    ----------
    proto : ModelProto | FunctionProto | GraphProto | None
        The ONNX proto produced by the export.  When *large_model* was
        requested the proto contains *placeholders* for external data;
        use :meth:`get_proto` to obtain a fully self-contained proto.
    container : ExtendedModelContainer | None
        The :class:`~yobx.container.ExtendedModelContainer` produced when
        the conversion was called with ``large_model=True``.  ``None``
        otherwise.
    report : ExportReport
        Statistics and metadata about the export.

    Backward Compatibility
    ----------------------
    :class:`ExportArtifact` transparently proxies attribute access to its
    underlying *proto* so that code that previously received a raw
    :class:`~onnx.ModelProto` continues to work without modification::

        onx = to_onnx(estimator, (X,))

        # New API:
        print(onx.proto)
        onx.save("model.onnx")

        # Legacy API still works:
        print(onx.graph.node)  # delegated to proto.graph.node
        onx.SerializeToString()  # delegated to proto.SerializeToString()

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
        proto: Any = None,
        container: Optional[Any] = None,
        report: Optional[ExportReport] = None,
    ):
        self.proto = proto
        self.container = container
        self.report = report if report is not None else ExportReport()

    # ------------------------------------------------------------------
    # Backward-compatibility proxy
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attribute access to the underlying proto."""
        # Avoid infinite recursion: access stored attributes directly.
        proto = object.__getattribute__(self, "proto")
        if proto is not None:
            try:
                return getattr(proto, name)
            except AttributeError:
                pass
        container = object.__getattribute__(self, "container")
        if container is not None:
            try:
                return getattr(container, name)
            except AttributeError:
                pass
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute {name!r}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        file_path: str,
        all_tensors_to_one_file: bool = True,
    ) -> Any:
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
        if self.container is not None:
            return self.container.save(
                file_path, all_tensors_to_one_file=all_tensors_to_one_file
            )
        import onnx

        if isinstance(self.proto, onnx.ModelProto):
            onnx.save_model(self.proto, file_path)
            return self.proto
        raise TypeError(
            f"Cannot save a proto of type {type(self.proto).__name__}. "
            "Only ModelProto is directly saveable; for FunctionProto or "
            "GraphProto embed it in a ModelProto first."
        )

    def get_proto(
        self, include_weights: bool = True
    ) -> Any:
        """Return the ONNX proto, optionally with all weights inlined.

        When the export was performed with ``large_model=True`` (i.e.
        :attr:`container` is set), the raw :attr:`proto` has
        *external-data* placeholders instead of embedded weight tensors.
        Passing ``include_weights=True`` (the default) returns a deep copy
        of the :class:`~onnx.ModelProto` with all external tensors
        replaced by their in-memory values so the result is fully
        self-contained.

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

        proto = self.container.model_proto
        if not include_weights:
            return proto

        # Embed large initializers into a copy of the ModelProto.
        import copy

        import numpy as np
        import onnx
        from onnx.external_data_helper import _get_all_tensors, uses_external_data

        proto_copy = copy.deepcopy(proto)
        large_inits = self.container.large_initializers

        for tensor in _get_all_tensors(proto_copy):
            if not uses_external_data(tensor):
                continue
            location: Optional[str] = None
            for ext in tensor.external_data:
                if ext.key == "location":
                    location = ext.value
                    break
            if location is None or location not in large_inits:
                continue
            val = large_inits[location]

            tensor.data_location = onnx.TensorProto.DEFAULT
            del tensor.external_data[:]

            if isinstance(val, np.ndarray):
                tensor.raw_data = val.tobytes()
            elif isinstance(val, onnx.TensorProto):
                tensor.raw_data = val.raw_data
            else:
                # Assume a torch tensor.
                import torch

                if isinstance(val, (torch.nn.Parameter, torch.Tensor)):
                    arr = val.detach().cpu().numpy()
                    tensor.raw_data = arr.tobytes()
                else:
                    raise TypeError(
                        f"Unsupported large initializer type {type(val)!r}"
                    )

        return proto_copy

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        proto_type = type(self.proto).__name__ if self.proto is not None else "None"
        has_container = self.container is not None
        return (
            f"{self.__class__.__name__}("
            f"proto={proto_type}, "
            f"container={has_container}, "
            f"report={self.report!r})"
        )

"""
:class:`ExportArtifact` and :class:`ExportReport` — standard output
of every :func:`to_onnx` conversion function.
"""

from typing import Any, Dict, List, Optional, Sequence, Union
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
    ):
        self.stats: List[Dict[str, Any]] = stats if stats is not None else []
        self.extra: Dict[str, Any] = extra if extra is not None else {}
        self.build_stats = build_stats

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
        else:
            raise TypeError(f"Unexpected tppe {type(data)} for data.")
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary representation of this report.

        :return: dictionary with keys ``"stats"`` and ``"extra"``.
        """
        return {"stats": self.stats, "extra": self.extra, "build_stats": self.build_stats}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_stats={len(self.stats)}, "
            f"extra={sorted(self.extra)}, "
            f"has_build_stats={self.build_stats is not None})"
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
    filename : str | None
        Path where the model was last saved, or ``None`` if never saved.
    builder: GraphBuilderExtendedProtocol
        Keeps the builder building the onnx model

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
        if self.container is not None:
            result = self.container.save(
                file_path, all_tensors_to_one_file=all_tensors_to_one_file
            )
            self.filename = file_path
            return result
        import onnx

        if isinstance(self.proto, onnx.ModelProto):
            onnx.save_model(self.proto, file_path)
            self.filename = file_path
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

        import onnx_ir.serde as oirs

        return oirs.serialize_model(self.container.to_ir())

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

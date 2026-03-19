"""
:class:`ExportArtifact` and :class:`ExportReport` — standard output
of every :func:`to_onnx` conversion function.
"""

from typing import Any, Dict, List, Optional, Union
import onnx
from ..typing import GraphBuilderExtendedProtocol
from .model_container import ExtendedModelContainer


class ExportReport:
    """
    Holds statistics and metadata gathered during an ONNX export.

    The :attr:`_stats` attribute stores the per-pattern optimization statistics
    returned by :meth:`~yobx.xbuilder.GraphBuilder.to_onnx` when called
    with ``return_optimize_report=True``.  Each element of the list is a
    dict with at least the keys ``"pattern"``, ``"added"``, ``"removed"``,
    and ``"time_in"``.

    Additional arbitrary key-value pairs can be recorded via the
    :meth:`update` method and are stored in :attr:`_extra`.

    Example::

        report = ExportReport()
        report.update({"time_total": 0.42})
        print(report)
    """

    def __init__(
        self, stats: Optional[List[Dict[str, Any]]] = None, extra: Optional[Dict[str, Any]] = None
    ):
        self._stats: List[Dict[str, Any]] = stats if stats is not None else []
        self._extra: Dict[str, Any] = extra if extra is not None else {}

    def update(self, data: Dict[str, Any]) -> None:
        """Merge *data* into :attr:`_extra`.

        :param data: key-value pairs to add to :attr:`_extra`.
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
    <<<<<<< HEAD
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
    =======
    >>>>>>> 4e17a857e37cad067b15fbc6bc7d7b953888f7a2
    """

    def __init__(
        self,
        proto: Optional[Union[onnx.ModelProto, onnx.GraphProto, onnx.FunctionProto]] = None,
        container: Optional[ExtendedModelContainer] = None,
        report: Optional[ExportReport] = None,
        filename: Optional[str] = None,
        builder: Optional[GraphBuilderExtendedProtocol] = None,
    ):
        self.proto = proto
        self.container = container
        self.report = report if report is not None else ExportReport()
        self.filename = filename
        self.builder = builder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

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

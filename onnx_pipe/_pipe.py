"""Core OnnxPipe class implementing the pipe operator for ONNX models."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import onnx
import onnx.compose

if TYPE_CHECKING:
    pass

_counter = itertools.count(1)


class OnnxPipe:
    """Wraps an ONNX model and supports the ``|`` operator to chain models.

    When two :class:`OnnxPipe` instances are combined with ``|``,
    the outputs of the left-hand model are connected to the inputs of the
    right-hand model (matched in order), producing a new :class:`OnnxPipe`.

    Example::

        from onnx_pipe import OnnxPipe, op

        pipe = op("Abs") | op("Relu")
        model = pipe.to_onnx()
    """

    def __init__(self, model: onnx.ModelProto):
        self._model = model

    @property
    def model(self) -> onnx.ModelProto:
        """Returns the underlying :class:`onnx.ModelProto`."""
        return self._model

    @property
    def input_names(self) -> list[str]:
        """Names of the graph inputs."""
        return [inp.name for inp in self._model.graph.input]

    @property
    def output_names(self) -> list[str]:
        """Names of the graph outputs."""
        return [out.name for out in self._model.graph.output]

    def __or__(self, other: "OnnxPipe") -> "OnnxPipe":
        """Chain this pipe with *other* using ``|``.

        The outputs of *self* are connected to the inputs of *other* in
        order.  If *self* has fewer outputs than *other* has inputs, the
        remaining inputs of *other* remain as free inputs of the combined
        model.

        Args:
            other: The :class:`OnnxPipe` to append to this one.

        Returns:
            A new :class:`OnnxPipe` representing the combined model.
        """
        if not isinstance(other, OnnxPipe):
            raise TypeError(
                f"unsupported operand type(s) for |: 'OnnxPipe' and '{type(other).__name__}'"
            )

        left_outputs = self.output_names
        right_inputs = other.input_names

        n = min(len(left_outputs), len(right_inputs))
        io_map = list(zip(left_outputs[:n], right_inputs[:n]))

        prefix2 = f"_p{next(_counter)}_"

        merged = onnx.compose.merge_models(
            self._model,
            other._model,
            io_map=io_map,
            prefix2=prefix2,
        )
        return OnnxPipe(merged)

    def to_onnx(self) -> onnx.ModelProto:
        """Return the underlying :class:`onnx.ModelProto`."""
        return self._model

    def __repr__(self) -> str:
        inputs = ", ".join(self.input_names)
        outputs = ", ".join(self.output_names)
        return f"OnnxPipe(inputs=[{inputs}], outputs=[{outputs}])"

from __future__ import annotations
import numpy as np
from onnx.reference.op_run import OpRun


class Gather(OpRun):
    """Overrides the ONNX reference ``Gather`` op to ensure contiguous arrays.

    ONNX 1.20.x has a bug where ``indices.ascontiguousarray()`` is called instead
    of ``np.ascontiguousarray(indices)``.  Non-contiguous index arrays (e.g. produced
    by ``Split`` + ``Squeeze``) trigger an ``AttributeError`` in the upstream
    implementation.  This override guards against that by making both inputs
    contiguous before calling ``np.take``.
    """

    def _run(self, x, indices, axis=None):
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
        if not indices.flags["C_CONTIGUOUS"]:
            indices = np.ascontiguousarray(indices)
        if indices.size == 0:
            return (np.empty((0,), dtype=x.dtype),)
        try:
            return (np.take(x, indices, axis=axis),)
        except TypeError:
            # distribution x86 requires int32.
            return (np.take(x, indices.astype(int), axis=axis),)

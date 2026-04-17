"""
Tests exporting torch ops from :mod:`torch.testing._internal.common_methods_invocations`
to ONNX using :func:`yobx.torch.interpreter.to_onnx` with
``ExportOptions(tracing=TracingMode.NEW_TRACING)``.

One test method is generated automatically for every (op, dtype) pair collected
from ``op_db``, following the naming convention
``test_export_<op_name>_<dtype>`` where dots in the op name are replaced by
underscores and *dtype* is one of ``float32``, ``float16``, ``bfloat16``,
``int32``, or ``int64``.
"""

import unittest
import warnings
from typing import Any, Callable, Dict, FrozenSet, List, Sequence, Tuple

import torch
from torch.testing._internal import common_methods_invocations

from yobx.ext_test_case import ExtTestCase, has_onnxruntime, ignore_warnings, requires_torch
from yobx.helpers import max_diff
from yobx.reference import ExtendedReferenceEvaluator
from yobx.torch import ExportOptions, TracingMode
from yobx.torch.interpreter import to_onnx
from yobx.torch.coverage.op_coverage import (
    NO_CONVERTER_OPS,
    NON_DETERMINISTIC_OPS,
    XFAIL_OPS,
    XFAIL_OPS_BFLOAT16,
    XFAIL_OPS_FLOAT16,
    XFAIL_OPS_INT32,
    XFAIL_OPS_INT64,
    ATOL_OPS_FLOAT16,
    ATOL_OPS_FLOAT32,
    ATOL_OPS_BFLOAT16,
)
from yobx.torch.torch_helper import to_numpy

# Human-readable label for each tested dtype, used as the suffix in generated
# test method names (e.g. ``test_export_add_float32``).
_DTYPE_NAMES: Dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int32: "int32",
    torch.int64: "int64",
}

# Absolute tolerance used when comparing eager vs exported outputs.
# bfloat16 has only 7 mantissa bits (vs 10 for float16) so needs the widest
# tolerance; integer and float32 dtypes use the tightest tolerance.
_ATOL_DEFAULT: float = 1e-3
_ATOL_FLOAT16: float = 1e-2
_ATOL_BFLOAT16: float = 2e-2
_DTYPE_ATOL: Dict[torch.dtype, float] = {
    torch.float16: _ATOL_FLOAT16,
    torch.bfloat16: _ATOL_BFLOAT16,
}

# Per-dtype per-op atol overrides (op name → float).  Ops that accumulate
# more floating-point error than the global dtype tolerance are listed here.
_DTYPE_ATOL_OPS: Dict[torch.dtype, Dict[str, float]] = {
    torch.float16: ATOL_OPS_FLOAT16,
    torch.float32: ATOL_OPS_FLOAT32,
    torch.bfloat16: ATOL_OPS_BFLOAT16,
}


def _tensor_is_exportable(t: torch.Tensor) -> bool:
    """Checks whether a single tensor can be safely converted to NumPy.

    Tensors with the conjugate bit set require :meth:`torch.Tensor.resolve_conj`
    before calling ``.numpy()``.  Rather than patching up every comparison,
    ops that always produce such tensors are skipped here.

    Returns:
        ``True`` when *t* is a real, non-conjugate tensor.
    """
    return not t.is_conj() and not t.dtype.is_complex


def _result_is_exportable(result: Any) -> bool:
    """Checks whether *result* can be safely exported and compared.

    Accepts a single :class:`torch.Tensor` or a tuple/list of tensors.
    Returns ``True`` only when every tensor in *result* passes
    :func:`_tensor_is_exportable`.

    Returns:
        ``True`` when all tensors in *result* are real and non-conjugate.
    """
    if isinstance(result, torch.Tensor):
        return _tensor_is_exportable(result)
    if isinstance(result, (tuple, list)) and result:
        return all(isinstance(r, torch.Tensor) and _tensor_is_exportable(r) for r in result)
    return False


def _collect_ops(dtype: torch.dtype) -> List[Any]:
    """Collects ops from op_db that are suitable for ONNX export testing with *dtype*.

    Filters to ops that:

    - Support *dtype*
    - Have no variant test name
    - Are not non-deterministic (random-output) ops
    - Are not in :data:`NO_CONVERTER_OPS` (missing aten converter)
    - Are not in the dtype-specific xfail set (known failures for *dtype*)
    - Have at least one sample input whose ``.input`` is a tensor of *dtype*
    - Have all positional sample args that are also tensors (or none at all)
    - Have no non-trivial keyword arguments (to avoid unsupported ONNX kwargs)

    Args:
        dtype: The :class:`torch.dtype` to collect ops for.  Must be one of
            ``torch.float32``, ``torch.float16``, ``torch.bfloat16``,
            ``torch.int32``, or ``torch.int64``.

    Returns:
        List of :class:`~torch.testing._internal.opinfo.core.OpInfo` objects.
    """
    _xfail_map: Dict[torch.dtype, FrozenSet[str]] = {
        torch.float32: XFAIL_OPS["default"] | XFAIL_OPS["new-tracing"],
        torch.float16: (
            XFAIL_OPS["default"]
            | XFAIL_OPS_FLOAT16["default"]
            | XFAIL_OPS["new-tracing"]
            | XFAIL_OPS_FLOAT16["new-tracing"]
        ),
        torch.bfloat16: (
            XFAIL_OPS["default"]
            | XFAIL_OPS_BFLOAT16["default"]
            | XFAIL_OPS["new-tracing"]
            | XFAIL_OPS_BFLOAT16["new-tracing"]
        ),
        torch.int32: (
            XFAIL_OPS["default"]
            | XFAIL_OPS_INT32["default"]
            | XFAIL_OPS["new-tracing"]
            | XFAIL_OPS_INT32["new-tracing"]
        ),
        torch.int64: (
            XFAIL_OPS["default"]
            | XFAIL_OPS_INT64["default"]
            | XFAIL_OPS["new-tracing"]
            | XFAIL_OPS_INT64["new-tracing"]
        ),
    }
    if dtype not in _xfail_map:
        raise ValueError(f"Unsupported dtype {dtype!r}. Supported dtypes: {list(_xfail_map)}")
    xfail = _xfail_map[dtype]

    testable = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for op in common_methods_invocations.op_db:
            if dtype not in op.dtypes:
                continue
            if op.variant_test_name:
                continue
            if op.name in NON_DETERMINISTIC_OPS:
                continue
            if op.name.replace(".", "_") in NO_CONVERTER_OPS:
                continue
            if op.name.replace(".", "_") in xfail:
                continue
            samples = list(op.sample_inputs("cpu", dtype, requires_grad=False))
            if not samples:
                continue
            s = samples[0]
            if not isinstance(s.input, torch.Tensor):
                continue
            if not all(isinstance(a, torch.Tensor) for a in s.args):
                continue
            if s.kwargs:
                # Skip ops with non-trivial keyword arguments whose semantics
                # may not be fully supported by the ONNX exporter.
                continue
            testable.append(op)
    return testable


_OPS_FLOAT32 = _collect_ops(torch.float32)
_OPS_FLOAT16 = _collect_ops(torch.float16)
_OPS_BFLOAT16 = _collect_ops(torch.bfloat16)
_OPS_INT32 = _collect_ops(torch.int32)
_OPS_INT64 = _collect_ops(torch.int64)


class _OpWrapper(torch.nn.Module):
    """Wraps a single op invocation as a :class:`torch.nn.Module`.

    The module's ``forward`` receives the primary tensor input followed by
    any additional tensor positional arguments from the sample.  Keyword
    arguments are captured at construction time and forwarded as constants.
    """

    def __init__(
        self, fn: Callable, extra_args: Tuple[torch.Tensor, ...], kwargs: Dict[str, Any]
    ) -> None:
        super().__init__()
        self._fn = fn
        self._extra_args = extra_args
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._fn})"

    def forward(self, x: torch.Tensor, *args: torch.Tensor) -> torch.Tensor:
        """Calls the wrapped op with the supplied tensors.

        Returns:
            The result of applying the wrapped op to *x* and *args*.
        """
        return self._fn(x, *args, **self._kwargs)


def _make_export_test(op: Any, dtype: torch.dtype) -> Callable:
    """Creates a test method that exports *op* to ONNX with *dtype* and validates outputs.

    Args:
        op: An :class:`~torch.testing._internal.opinfo.core.OpInfo` instance.
        dtype: The :class:`torch.dtype` to use for sample inputs.

    Returns:
        A test method bound to *op* and *dtype* suitable for attaching to a
        :class:`unittest.TestCase` subclass.
    """

    @requires_torch("2.6")
    @ignore_warnings((UserWarning, FutureWarning, DeprecationWarning))
    def _test(self: ExtTestCase, _op: Any = op, _dtype: torch.dtype = dtype) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samples = list(_op.sample_inputs("cpu", _dtype, requires_grad=False))
        if not samples:
            raise unittest.SkipTest(f"no sample inputs for op {_op.name!r} dtype={_dtype}")
        s = samples[0]
        model = _OpWrapper(_op.op, s.args, s.kwargs)
        inputs = (s.input, *s.args)
        expected = _op.op(s.input, *s.args, **s.kwargs)
        if not _result_is_exportable(expected):
            raise unittest.SkipTest(
                f"op {_op.name!r} dtype={_dtype} produces a non-exportable result"
            )

        # Normalise to a sequence so single-tensor and tuple results are
        # handled uniformly below.
        expected_seq: Sequence[torch.Tensor] = (
            expected if isinstance(expected, (tuple, list)) else (expected,)
        )

        onx = to_onnx(
            model, inputs, export_options=ExportOptions(tracing=TracingMode.NEW_TRACING)
        )
        numpy_inputs = [to_numpy(t) for t in inputs]

        ref = ExtendedReferenceEvaluator(onx.proto)
        ref_feeds = dict(zip(ref.input_names, numpy_inputs))
        got_ref = ref.run(None, ref_feeds)

        # Use relaxed tolerances for reduced-precision dtypes: bfloat16 has
        # only 7 mantissa bits (vs 10 for float16) so uses the widest tolerance.
        # Some ops (e.g. std, std_mean) accumulate more rounding error and get
        # a further per-op override via _DTYPE_ATOL_OPS.
        _dtype_default_atol = _DTYPE_ATOL.get(_dtype, _ATOL_DEFAULT)
        atol = _DTYPE_ATOL_OPS.get(_dtype, {}).get(_op.name, _dtype_default_atol)

        for i, (exp_i, got_i) in enumerate(zip(expected_seq, got_ref)):
            diff = max_diff(exp_i, got_i)
            self.assertLess(
                diff["abs"],
                atol,
                msg=f"op={_op.name!r} dtype={_dtype} ref output[{i}] max abs diff={diff['abs']}",
            )

        if has_onnxruntime() and _dtype != torch.bfloat16:
            import onnxruntime

            sess = onnxruntime.InferenceSession(
                onx.proto.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            ort_feeds = dict(zip([inp.name for inp in sess.get_inputs()], numpy_inputs))
            got_ort = sess.run(None, ort_feeds)

            for i, (exp_i, got_i) in enumerate(zip(expected_seq, got_ort)):
                diff = max_diff(exp_i, got_i)
                self.assertLess(
                    diff["abs"],
                    atol,
                    msg=(
                        f"op={_op.name!r} dtype={_dtype} ort output[{i}]"
                        f" max abs diff={diff['abs']}"
                    ),
                )

    dtype_name = _DTYPE_NAMES[dtype]
    _test.__doc__ = (
        f"Exports :func:`torch.{op.name}` ({dtype_name}) to ONNX and validates numerical outputs."
    )
    return _test


class TestOnnxExportCommonMethodsNewTracing(ExtTestCase):
    """Tests :func:`yobx.torch.interpreter.to_onnx` with new-tracing against ops from op_db.

    One test method is generated automatically for every (op, dtype) pair in
    ``_OPS_FLOAT32``, ``_OPS_FLOAT16``, ``_OPS_BFLOAT16``, ``_OPS_INT32``,
    and ``_OPS_INT64`` via :func:`_make_export_test`.  Methods follow the
    naming convention ``test_export_<op_name>_<dtype>`` where dots are replaced
    by underscores and *dtype* is one of ``float32``, ``float16``,
    ``bfloat16``, ``int32``, or ``int64``.
    """

    @classmethod
    def _add_test_methods(cls) -> None:
        """Attaches one test method per (op, dtype) pair to *cls*."""
        for dtype, ops in (
            (torch.float32, _OPS_FLOAT32),
            (torch.float16, _OPS_FLOAT16),
            (torch.bfloat16, _OPS_BFLOAT16),
            (torch.int32, _OPS_INT32),
            (torch.int64, _OPS_INT64),
        ):
            dtype_name = _DTYPE_NAMES[dtype]
            for op in ops:
                # Replace dots (e.g. "special.log_ndtr") with underscores so
                # the method name is a valid Python identifier, then append
                # the dtype suffix for disambiguation.
                method_name = "test_export_" + op.name.replace(".", "_") + "_" + dtype_name
                setattr(cls, method_name, _make_export_test(op, dtype))


TestOnnxExportCommonMethodsNewTracing._add_test_methods()

if __name__ == "__main__":
    unittest.main(verbosity=2)

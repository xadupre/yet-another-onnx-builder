"""
Tests exporting torch ops from :mod:`torch.testing._internal.common_methods_invocations`
to ONNX using :func:`yobx.torch.interpreter.to_onnx`.

One test method is generated automatically for every op collected from ``op_db``,
following the naming convention ``test_export_<op_name>`` where dots in the op
name are replaced by underscores.
"""

import unittest
import warnings
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch.testing._internal import common_methods_invocations

from yobx.ext_test_case import ExtTestCase, ignore_warnings, requires_torch
from yobx.helpers import max_diff
from yobx.reference import ExtendedReferenceEvaluator
from yobx.torch.interpreter import to_onnx

# Ops that generate random or non-deterministic outputs are excluded from
# numerical validation since exported results cannot be compared to eager ones.
_NON_DETERMINISTIC_OPS = frozenset(
    {
        "cauchy",
        "empty",
        "empty_like",
        "exponential",
        "geometric",
        "log_normal",
        "normal",
        "rand",
        "rand_like",
        "randint",
        "randperm",
        "randn",
        "randn_like",
        "uniform",
    }
)


def _result_is_exportable(result: torch.Tensor) -> bool:
    """Checks whether *result* can be safely converted to NumPy.

    Tensors with the conjugate bit set require :meth:`torch.Tensor.resolve_conj`
    before calling ``.numpy()``.  Rather than patching up every comparison,
    ops that always produce such tensors are skipped here.

    Returns:
        ``True`` when *result* is a real, non-conjugate tensor.
    """
    return not result.is_conj() and not result.dtype.is_complex


def _collect_ops() -> List[Any]:
    """Collects ops from op_db that are suitable for ONNX export testing.

    Filters to ops that:

    - Support ``float32``
    - Have no variant test name
    - Are not non-deterministic (random-output) ops
    - Have at least one sample input whose ``.input`` is a ``float32`` tensor
    - Have all positional sample args that are also tensors (or none at all)
    - Have no non-trivial keyword arguments (to avoid unsupported ONNX kwargs)

    Returns:
        List of :class:`~torch.testing._internal.opinfo.core.OpInfo` objects.
    """
    testable = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for op in common_methods_invocations.op_db:
            if torch.float32 not in op.dtypes:
                continue
            if op.variant_test_name:
                continue
            if op.name in _NON_DETERMINISTIC_OPS:
                continue
            samples = list(op.sample_inputs("cpu", torch.float32, requires_grad=False))
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


_OPS = _collect_ops()


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

    def forward(self, x: torch.Tensor, *args: torch.Tensor) -> torch.Tensor:
        """Calls the wrapped op with the supplied tensors.

        Returns:
            The result of applying the wrapped op to *x* and *args*.
        """
        return self._fn(x, *args, **self._kwargs)


def _make_export_test(op: Any) -> Callable:
    """Creates a test method that exports *op* to ONNX and validates outputs.

    Returns:
        A test method bound to *op* suitable for attaching to a
        :class:`unittest.TestCase` subclass.
    """

    @requires_torch("2.6")
    @ignore_warnings((UserWarning, FutureWarning, DeprecationWarning))
    def _test(self: ExtTestCase, _op: Any = op) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samples = list(_op.sample_inputs("cpu", torch.float32, requires_grad=False))
        if not samples:
            raise unittest.SkipTest(f"no sample inputs for op {_op.name!r}")
        s = samples[0]
        model = _OpWrapper(_op.op, s.args, s.kwargs)
        inputs = (s.input, *s.args)
        expected = _op.op(s.input, *s.args, **s.kwargs)
        if not isinstance(expected, torch.Tensor) or not _result_is_exportable(expected):
            raise unittest.SkipTest(f"op {_op.name!r} produces a non-exportable result")

        onx = to_onnx(model, inputs)

        ref = ExtendedReferenceEvaluator(onx.proto)
        feeds = dict(zip(ref.input_names, [t.detach().numpy() for t in inputs]))
        got = ref.run(None, feeds)

        diff = max_diff(expected, got[0])
        # float32 arithmetic may introduce small rounding differences;
        # 1e-3 is a reasonable tolerance for single-precision ops.
        self.assertLess(diff["abs"], 1e-3, msg=f"op={_op.name!r} max abs diff={diff['abs']}")

    _test.__doc__ = f"Exports :func:`torch.{op.name}` to ONNX and validates numerical outputs."
    return _test


class TestOnnxExportCommonMethods(ExtTestCase):
    """Tests :func:`yobx.torch.interpreter.to_onnx` against ops from op_db.

    One test method is generated automatically for every op in ``_OPS`` via
    :func:`_make_export_test`.  Methods follow the naming convention
    ``test_export_<op_name>`` where dots are replaced by underscores.
    """

    @classmethod
    def _add_test_methods(cls) -> None:
        """Attaches one test method per op in ``_OPS`` to *cls*."""
        for op in _OPS:
            # Replace dots (e.g. "special.log_ndtr") with underscores so the
            # method name is a valid Python identifier.
            method_name = "test_export_" + op.name.replace(".", "_")
            setattr(cls, method_name, _make_export_test(op))


TestOnnxExportCommonMethods._add_test_methods()

if __name__ == "__main__":
    unittest.main(verbosity=2)

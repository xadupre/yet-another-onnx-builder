"""
Tests exporting torch ops from :mod:`torch.testing._internal.common_methods_invocations`
to ONNX using :func:`yobx.torch.interpreter.to_onnx`.
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


class TestOnnxExportCommonMethods(ExtTestCase):
    """Tests :func:`yobx.torch.interpreter.to_onnx` against ops from op_db."""

    @requires_torch("2.6")
    @ignore_warnings((UserWarning, FutureWarning, DeprecationWarning))
    def test_export_ops(self) -> None:
        """Exports op_db ops to ONNX and validates numerical outputs.

        Each op is run as a sub-test so that individual failures are reported
        without aborting the entire loop.
        """
        if not _OPS:
            raise unittest.SkipTest("no testable ops collected from op_db")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for op in _OPS:
                op_name = op.name
                with self.subTest(op=op_name):
                    samples = list(op.sample_inputs("cpu", torch.float32, requires_grad=False))
                    if not samples:
                        continue
                    s = samples[0]
                    model = _OpWrapper(op.op, s.args, s.kwargs)
                    inputs = (s.input, *s.args)
                    expected = op.op(s.input, *s.args, **s.kwargs)
                    if not isinstance(expected, torch.Tensor) or not _result_is_exportable(
                        expected
                    ):
                        continue

                    onx = to_onnx(model, inputs)

                    # Validate the ONNX output matches the PyTorch output.
                    ref = ExtendedReferenceEvaluator(onx.proto)
                    feeds = dict(zip(ref.input_names, [t.detach().numpy() for t in inputs]))
                    got = ref.run(None, feeds)

                    diff = max_diff(expected, got[0])
                    # float32 arithmetic may introduce small rounding differences;
                    # 1e-3 is a reasonable tolerance for single-precision ops.
                    self.assertLess(
                        diff["abs"], 1e-3, msg=f"op={op_name!r} max abs diff={diff['abs']}"
                    )

    @requires_torch("2.6")
    @ignore_warnings((UserWarning, FutureWarning, DeprecationWarning))
    def test_export_abs(self) -> None:
        """Exports :func:`torch.abs` to ONNX as a quick smoke test."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            op = next((o for o in common_methods_invocations.op_db if o.name == "abs"), None)
        if op is None:
            raise unittest.SkipTest("abs op not found in op_db")

        samples = list(op.sample_inputs("cpu", torch.float32, requires_grad=False))
        s = samples[0]
        model = _OpWrapper(op.op, s.args, s.kwargs)
        inputs = (s.input,)
        expected = op.op(s.input, *s.args, **s.kwargs)

        onx = to_onnx(model, inputs)
        self.assertIsNotNone(onx.proto)

        ref = ExtendedReferenceEvaluator(onx.proto)
        feeds = dict(zip(ref.input_names, [s.input.detach().numpy()]))
        got = ref.run(None, feeds)

        diff = max_diff(expected, got[0])
        self.assertLess(diff["abs"], 1e-5)

    @requires_torch("2.6")
    @ignore_warnings((UserWarning, FutureWarning, DeprecationWarning))
    def test_export_binary_add(self) -> None:
        """Exports :func:`torch.add` (binary, two-tensor) to ONNX as a smoke test."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            op = next(
                (
                    o
                    for o in common_methods_invocations.op_db
                    if o.name == "add" and not o.variant_test_name
                ),
                None,
            )
        if op is None:
            raise unittest.SkipTest("add op not found in op_db")

        samples = list(op.sample_inputs("cpu", torch.float32, requires_grad=False))
        s = samples[0]
        if not isinstance(s.input, torch.Tensor) or not all(
            isinstance(a, torch.Tensor) for a in s.args
        ):
            raise unittest.SkipTest("add sample inputs are not all tensors")

        model = _OpWrapper(op.op, s.args, s.kwargs)
        inputs = (s.input, *s.args)
        expected = op.op(s.input, *s.args, **s.kwargs)

        onx = to_onnx(model, inputs)
        self.assertIsNotNone(onx.proto)

        ref = ExtendedReferenceEvaluator(onx.proto)
        feeds = dict(zip(ref.input_names, [t.detach().numpy() for t in inputs]))
        got = ref.run(None, feeds)

        diff = max_diff(expected, got[0])
        self.assertLess(diff["abs"], 1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

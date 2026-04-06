"""
Tests exporting torch ops from :mod:`torch.testing._internal.common_methods_invocations`
to ONNX using :func:`yobx.torch.interpreter.to_onnx`.
"""

import unittest
import warnings
from typing import Any, Callable, Dict, List, Tuple

import torch

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
    """Returns ``True`` when *result* can be safely converted to NumPy.

    Tensors with the conjugate bit set require :meth:`torch.Tensor.resolve_conj`
    before calling ``.numpy()``.  Rather than patching up every comparison,
    ops that always produce such tensors are skipped here.
    """
    return not result.is_conj() and not result.dtype.is_complex


def _collect_ops() -> List[Any]:
    """Collects ops from op_db that are suitable for ONNX export testing.

    Returns only ops that:

    - Support ``float32``
    - Have no variant test name
    - Are not non-deterministic (random-output) ops
    - Have at least one sample input whose ``.input`` is a ``float32`` tensor
    - Have all positional sample args that are also tensors (or none at all)
    - Have no non-trivial keyword arguments (to avoid unsupported ONNX kwargs)
    - Produce a single, real, non-conjugate tensor result when called eagerly
    """
    try:
        from torch.testing._internal import common_methods_invocations
    except ImportError:
        return []

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
            try:
                samples = list(op.sample_inputs("cpu", torch.float32, requires_grad=False))
            except Exception:
                continue
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
            try:
                result = op.op(s.input, *s.args, **s.kwargs)
            except Exception:
                continue
            if not isinstance(result, torch.Tensor):
                continue
            if not _result_is_exportable(result):
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
        """Calls the wrapped op with the supplied tensors."""
        return self._fn(x, *args, **self._kwargs)


class TestOnnxExportCommonMethods(ExtTestCase):
    """Tests :func:`yobx.torch.interpreter.to_onnx` against ops from op_db."""

    @requires_torch("2.6")
    @ignore_warnings((UserWarning, FutureWarning, DeprecationWarning))
    def test_export_ops(self) -> None:
        """Exports op_db ops to ONNX and validates numerical outputs.

        Each op is run as a sub-test.  Export or inference failures are
        recorded but do not immediately raise; the test asserts that at least
        one op exported and validated successfully end-to-end.
        """
        if not _OPS:
            raise unittest.SkipTest(
                "torch.testing._internal.common_methods_invocations not available"
            )

        n_success = 0
        failures: List[str] = []

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

                    try:
                        onx = to_onnx(model, inputs)
                    except Exception as exc:
                        failures.append(f"{op_name}: {type(exc).__name__}: {exc}")
                        continue

                    # Validate the ONNX output matches the PyTorch output.
                    ref = ExtendedReferenceEvaluator(onx.proto)
                    feeds = dict(zip(ref.input_names, [t.detach().numpy() for t in inputs]))
                    try:
                        got = ref.run(None, feeds)
                    except Exception as exc:
                        failures.append(f"{op_name} (inference): {type(exc).__name__}: {exc}")
                        continue

                    diff = max_diff(expected, got[0])
                    # float32 arithmetic may introduce small rounding differences;
                    # 1e-3 is a reasonable tolerance for single-precision ops.
                    self.assertLess(
                        diff["abs"], 1e-3, msg=f"op={op_name!r} max abs diff={diff['abs']}"
                    )
                    n_success += 1

        # At least some ops must export and validate successfully.
        self.assertGreater(
            n_success,
            0,
            msg=(
                f"No op exported successfully out of {len(_OPS)} candidates. "
                "Failures:\n" + "\n".join(failures[:10])
            ),
        )

    @requires_torch("2.6")
    @ignore_warnings((UserWarning, FutureWarning, DeprecationWarning))
    def test_export_abs(self) -> None:
        """Exports :func:`torch.abs` to ONNX as a quick smoke test."""
        try:
            from torch.testing._internal import common_methods_invocations
        except ImportError:
            raise unittest.SkipTest(
                "torch.testing._internal.common_methods_invocations not available"
            )

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
        try:
            from torch.testing._internal import common_methods_invocations
        except ImportError:
            raise unittest.SkipTest(
                "torch.testing._internal.common_methods_invocations not available"
            )

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

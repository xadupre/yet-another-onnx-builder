"""
Tests exporting torch ops from :mod:`torch.testing._internal.common_methods_invocations`
to ONNX using :func:`yobx.torch.interpreter.to_onnx`.

One test method is generated automatically for every op collected from ``op_db``,
following the naming convention ``test_export_<op_name>`` where dots in the op
name are replaced by underscores.
"""

import unittest
import warnings
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
from torch.testing._internal import common_methods_invocations

from yobx.ext_test_case import ExtTestCase, has_onnxruntime, ignore_warnings, requires_torch
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

# Ops whose aten decomposition uses an aten function for which no ONNX
# converter has been implemented yet.  Each entry is the value of
# ``op.name.replace(".", "_")`` (the suffix of the generated test method).
# When a converter is added, remove the entry from this set so the test
# starts running automatically.
_NO_CONVERTER_OPS = frozenset(
    {
        "H",
        "__rsub__",
        "addcdiv",
        "addr",
        "alias_copy",
        "amin",
        "aminmax",
        "angle",
        "argwhere",
        "atan2",
        "atleast_1d",
        "atleast_2d",
        "atleast_3d",
        "bernoulli",
        "block_diag",
        "cartesian_prod",
        "clamp_max",
        "clamp_min",
        "conj_physical",
        "copysign",
        "count_nonzero",
        "cross",
        "cumulative_trapezoid",
        "deg2rad",
        "diag",
        "diag_embed",
        "diagflat",
        "diagonal",
        "diagonal_copy",
        "diagonal_scatter",
        "digamma",
        "dot",
        "erfc",
        "erfinv",
        "exp2",
        "fft_fftshift",
        "fft_ifftshift",
        "fliplr",
        "flipud",
        "float_power",
        "fmax",
        "fmin",
        "fmod",
        "frac",
        "frexp",
        "geqrf",
        "hash_tensor",
        "heaviside",
        "hypot",
        "i0",
        "igamma",
        "igammac",
        "inner",
        "isclose",
        "isfinite",
        "isneginf",
        "isposinf",
        "isreal",
        "kron",
        "ldexp",
        "lgamma",
        "linalg_cond",
        "linalg_cross",
        "linalg_diagonal",
        "linalg_householder_product",
        "linalg_inv",
        "linalg_inv_ex",
        "linalg_qr",
        "linalg_solve",
        "linalg_solve_ex",
        "linalg_svdvals",
        "linalg_vecdot",
        "log10",
        "log1p",
        "log2",
        "logaddexp",
        "logaddexp2",
        "logical_xor",
        "logit",
        "lu_solve",
        "lu_unpack",
        "mH",
        "mT",
        "masked_select",
        "matrix_exp",
        "median",
        "mode",
        "msort",
        "mv",
        "nanmean",
        "nanmedian",
        "nansum",
        "nextafter",
        "nn_functional_bilinear",
        "nn_functional_binary_cross_entropy",
        "nn_functional_binary_cross_entropy_with_logits",
        "nn_functional_celu",
        "nn_functional_logsigmoid",
        "nn_functional_mish",
        "nn_functional_multi_margin_loss",
        "nn_functional_multilabel_margin_loss",
        "nn_functional_multilabel_soft_margin_loss",
        "nn_functional_pairwise_distance",
        "nn_functional_pdist",
        "nn_functional_relu6",
        "nn_functional_soft_margin_loss",
        "pinverse",
        "positive",
        "qr",
        "rad2deg",
        "ravel",
        "real",
        "resize_as_",
        "resolve_conj",
        "resolve_neg",
        "rot90",
        "rsub",
        "sgn",
        "signbit",
        "sinc",
        "sort",
        "special_airy_ai",
        "special_bessel_j0",
        "special_bessel_j1",
        "special_bessel_y0",
        "special_bessel_y1",
        "special_chebyshev_polynomial_t",
        "special_chebyshev_polynomial_u",
        "special_chebyshev_polynomial_v",
        "special_chebyshev_polynomial_w",
        "special_entr",
        "special_erfcx",
        "special_hermite_polynomial_h",
        "special_hermite_polynomial_he",
        "special_i0e",
        "special_i1",
        "special_i1e",
        "special_laguerre_polynomial_l",
        "special_legendre_polynomial_p",
        "special_log_ndtr",
        "special_modified_bessel_i0",
        "special_modified_bessel_i1",
        "special_modified_bessel_k0",
        "special_modified_bessel_k1",
        "special_ndtr",
        "special_ndtri",
        "special_scaled_modified_bessel_k0",
        "special_scaled_modified_bessel_k1",
        "special_shifted_chebyshev_polynomial_t",
        "special_shifted_chebyshev_polynomial_u",
        "special_shifted_chebyshev_polynomial_v",
        "special_shifted_chebyshev_polynomial_w",
        "special_spherical_bessel_j0",
        "special_xlog1py",
        "special_zeta",
        "squeeze_copy",
        "std",
        "std_mean",
        "t_copy",
        "tensor_split",
        "to_sparse",
        "trace",
        "trapezoid",
        "trapz",
        "triangular_solve",
        "true_divide",
        "var",
        "var_mean",
        "vdot",
        "view_as",
        "xlogy",
        "zero_",
    }
)

# Ops that are currently exported but produce incorrect numerical results or
# raise errors not related to missing aten converters (e.g. numerical
# precision, unsupported dtype, data-dependent shapes, etc.).  Each entry is
# the value of ``op.name.replace(".", "_")`` (the suffix of the generated test
# method).  Remove an entry when the underlying issue is fixed so that the
# test starts running automatically.
_XFAIL_OPS = frozenset(
    {
        # Numerical mismatch between eager and ONNX output (AssertionError):
        "amax",
        "cholesky_inverse",
        "cholesky_solve",
        "jiterator_2inputs_2outputs",
        "jiterator_4inputs_with_extra_args",
        "jiterator_binary",
        "jiterator_binary_return_by_ref",
        "jiterator_unary",
        "linalg_det",
        "linalg_slogdet",
        "logdet",
        "logical_not",
        "nn_functional_cross_entropy",
        "nn_functional_l1_loss",
        "nn_functional_linear",
        "nn_functional_mse_loss",
        "nn_functional_smooth_l1_loss",
        # Data-dependent output shapes (DataDependentOutputException):
        "corrcoef",
        "cov",
        # Other export/runtime errors:
        "bfloat16",  # RuntimeError
        "broadcast_tensors",  # NotImplementedError
        "clamp",  # onnxruntime RuntimeException
        "nan_to_num",  # TypeError
    }
)


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


def _collect_ops() -> List[Any]:
    """Collects ops from op_db that are suitable for ONNX export testing.

    Filters to ops that:

    - Support ``float32``
    - Have no variant test name
    - Are not non-deterministic (random-output) ops
    - Are not in :data:`_NO_CONVERTER_OPS` (missing aten converter)
    - Are not in :data:`_XFAIL_OPS` (known failures for other reasons)
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
            if op.name.replace(".", "_") in _NO_CONVERTER_OPS:
                continue
            if op.name.replace(".", "_") in _XFAIL_OPS:
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
        if not _result_is_exportable(expected):
            raise unittest.SkipTest(f"op {_op.name!r} produces a non-exportable result")

        # Normalise to a sequence so single-tensor and tuple results are
        # handled uniformly below.
        expected_seq: Sequence[torch.Tensor] = (
            expected if isinstance(expected, (tuple, list)) else (expected,)
        )

        onx = to_onnx(model, inputs)
        numpy_inputs = [t.detach().numpy() for t in inputs]

        ref = ExtendedReferenceEvaluator(onx.proto)
        ref_feeds = dict(zip(ref.input_names, numpy_inputs))
        got_ref = ref.run(None, ref_feeds)

        for i, (exp_i, got_i) in enumerate(zip(expected_seq, got_ref)):
            diff = max_diff(exp_i, got_i)
            # float32 arithmetic may introduce small rounding differences;
            # 1e-3 is a reasonable tolerance for single-precision ops.
            self.assertLess(
                diff["abs"],
                1e-3,
                msg=f"op={_op.name!r} ref output[{i}] max abs diff={diff['abs']}",
            )

        if has_onnxruntime():
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
                    1e-3,
                    msg=f"op={_op.name!r} ort output[{i}] max abs diff={diff['abs']}",
                )

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

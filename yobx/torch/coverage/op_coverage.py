"""
Op-db coverage data for PyTorch-to-ONNX operator export.

Defines the sets of ops that have no converter, are known to fail, or are
excluded for specific dtypes when tested via
:mod:`unittests.torch.coverage.test_onnx_export_common_methods` and
:mod:`unittests.torch.coverage.test_onnx_export_common_methods_tracing`.

:data:`NO_CONVERTER_OPS` - ops whose aten decomposition uses a function for
which no ONNX converter has been implemented yet.

:data:`XFAIL_OPS`, :data:`XFAIL_OPS_FLOAT16`, :data:`XFAIL_OPS_BFLOAT16`,
:data:`XFAIL_OPS_INT32`, :data:`XFAIL_OPS_INT64` - dicts mapping export-path
name (``"default"`` or ``"tracing"``) to a :class:`frozenset` of op keys that
are expected to fail on that path.  The ``"default"`` entry covers the
standard export path; the ``"tracing"`` entry lists additional ops that fail
specifically under ``ExportOptions(tracing=True)``.  The dtype-specific dicts
contain only the *extra* exclusions beyond the dtype-agnostic :data:`XFAIL_OPS`.

:data:`ATOL_OPS_FLOAT32`, :data:`ATOL_OPS_FLOAT16`, :data:`ATOL_OPS_BFLOAT16` - per-op absolute
tolerance overrides for float16 and bfloat16, for ops whose reduced-precision
errors exceed the global dtype tolerance.

These sets are consumed by the op-db test modules
:mod:`unittests.torch.coverage.test_onnx_export_common_methods`,
:mod:`unittests.torch.coverage.test_onnx_export_common_methods_tracing`, and by
:func:`get_op_coverage_rst` which builds a documentation coverage table.
"""

from __future__ import annotations

from typing import Dict, FrozenSet

# ---------------------------------------------------------------------------
# Status symbols (used by get_op_coverage_rst)
# ---------------------------------------------------------------------------
_SUPPORTED = "✔"
_NO_CONVERTER = "✘ no converter"
_XFAIL = "⚠ xfail"
_NOT_APPLICABLE = "—"

# Ops that generate random or non-deterministic outputs.
NON_DETERMINISTIC_OPS: FrozenSet[str] = frozenset(
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
# converter has been implemented yet.  Each entry is
# ``op.name.replace(".", "_")``.  When a converter is added, remove the entry
# from this set so the test starts running automatically.
NO_CONVERTER_OPS: FrozenSet[str] = frozenset(
    {
        "H",
        "__rsub__",
        "addcdiv",
        "addr",
        "alias_copy",
        "argwhere",
        "bernoulli",
        "block_diag",
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
        "exp2",
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
        "nanmedian",
        "nextafter",
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
        "zero_",
    }
)

# Ops that are currently exported but produce incorrect numerical results or
# raise errors unrelated to missing aten converters.
# Each key is an export-path name: ``"default"`` for the standard path and
# ``"tracing"`` for the ``ExportOptions(tracing=True)`` path.  The ``"tracing"``
# entry lists *additional* ops that fail beyond those in ``"default"``.
XFAIL_OPS: Dict[str, FrozenSet[str]] = {
    "default": frozenset(
        {
            # Numerical mismatch between eager and ONNX output (AssertionError):
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
    ),
    # Ops that fail specifically under ``ExportOptions(tracing=True)``,
    # beyond those already listed under ``"default"``.
    "tracing": frozenset(
        {
            "T",  # no attr
            "__radd__",  # not implemented
            "__rand__",  # not implemented
            "__rdiv__",  # not implemented
            "__rmatmul__",  # not implemented
            "__rmod__",  # not implemented
            "__rmul__",  # not implemented
            "__ror__",  # not implemented
            "__rpow__",  # not implemented
            "__rsub__",  # not implemented
            "__rxor__",  # not implemented
            "aminmax",  # list index out of range
            "argsort",
            "bitwise_and",
            "bitwise_or",
            "byte",
            "char",
            "conj",
            "double",
            "erfinv",
            "flatten",
            "half",
            "int",
            "logical_and",
            "logical_or",
            "logical_xor",
            "long",
            "nn_functional_bilinear",
            "nn_functional_hardsigmoid",
            "nn_functional_softsign",
            "nn_functional_tanhshrink",
            "reshape_as",
            "short",
            "std",
            "std_mean",
        }
    ),
}

# Extra exclusions specific to torch.float16.
# ``"default"`` contains float16-specific failures on the standard path;
# ``"tracing"`` lists additional failures specific to the tracing path.
XFAIL_OPS_FLOAT16: Dict[str, FrozenSet[str]] = {
    "default": frozenset(
        {
            # Numerical mismatch too large for float16 tolerance (1e-2):
            "addcmul",  # ref_diff=0.03125
            "expm1",  # ref_diff=4
            # Incorrect results (ordering) for float16:
            "argsort",  # ref_diff=7944
        }
    ),
    "tracing": frozenset(),
}

# Extra exclusions specific to torch.bfloat16.
# ``"default"`` contains bfloat16-specific failures on the standard path;
# ``"tracing"`` lists additional failures specific to the tracing path.
XFAIL_OPS_BFLOAT16: Dict[str, FrozenSet[str]] = {
    "default": frozenset(
        {
            # Numerical mismatch too large for bfloat16 tolerance (2e-2):
            "addcmul",  # ref_diff > 2e-2
            "bmm",  # matmul numerical error exceeds bfloat16 tolerance
            "erfinv",  # bfloat16 precision insufficient for erfinv computation
            "expm1",  # reduced-precision exponential error
            "log10",  # bfloat16 precision loss in Log
            "log1p",  # bfloat16 precision loss in Log
            "log2",  # bfloat16 precision loss in Log
            "logaddexp",  # bfloat16 precision loss in Log/Exp
            "logaddexp2",  # bfloat16 precision loss in Log/Exp
            # Incorrect results (ordering) for bfloat16:
            "argsort",  # ordering instability due to reduced precision
        }
    ),
    "tracing": frozenset(),
}

# Extra exclusions specific to torch.int32.
# ``"default"`` contains int32-specific failures on the standard path;
# ``"tracing"`` lists additional failures specific to the tracing path.
XFAIL_OPS_INT32: Dict[str, FrozenSet[str]] = {
    "default": frozenset(
        {
            # Ops that produce float outputs from int32 inputs but type inference
            # fails or ONNX model is invalid because the op only supports float:
            "__rdiv__",  # reciprocal node type mismatch
            "__rpow__",  # negative integer powers not allowed
            "__rxor__",  # FunctionNotFoundError: bitwise_xor
            "acos",  # ONNX op only supports float dtypes
            "acosh",  # ONNX op only supports float dtypes
            "asin",  # ONNX op only supports float dtypes
            "asinh",  # ONNX op only supports float dtypes
            "angle",  # ONNX Atan not supported for integer dtypes
            "atan",  # ONNX op only supports float dtypes
            "atan2",  # ONNX Atan not supported for integer dtypes
            "atanh",  # ONNX op only supports float dtypes
            "bitwise_left_shift",  # FunctionNotFoundError
            "bitwise_right_shift",  # FunctionNotFoundError
            "bitwise_xor",  # FunctionNotFoundError
            "ceil",  # InvalidGraph: int32 not supported by Ceil
            "cos",  # ONNX op only supports float dtypes
            "cosh",  # ONNX op only supports float dtypes
            "erf",  # ONNX op only supports float dtypes
            "erfc",  # ONNX Erf only supports float dtypes
            "erfinv",  # ONNX Erf/Log/Exp only support float dtypes
            "exp",  # ONNX op only supports float dtypes
            "expm1",  # ONNX op only supports float dtypes
            "floor",  # InvalidGraph: int32 not supported by Floor
            "floor_divide",  # ref_diff=1
            "gcd",  # FunctionNotFoundError
            "isinf",  # InvalidGraph: int32 not supported by IsInf
            "isnan",  # InvalidGraph: int32 not supported by IsNaN
            "lcm",  # FunctionNotFoundError
            "log",  # ONNX op only supports float dtypes
            "log10",  # ONNX Log op only supports float dtypes
            "log1p",  # ONNX Log op only supports float dtypes
            "log2",  # ONNX Log op only supports float dtypes
            "logaddexp",  # ONNX Log/Exp ops only support float dtypes
            "logaddexp2",  # ONNX Log/Exp ops only support float dtypes
            "logit",  # ONNX Log op only supports float dtypes
            "nanmean",  # IsNaN not supported for integer dtypes
            "nansum",  # IsNaN not supported for integer dtypes
            "nn_functional_relu",  # NOT_IMPLEMENTED: Relu not supported for int32
            "nn_functional_softsign",  # type mismatch in Div
            "nn_functional_tanhshrink",  # ONNX Tanh only supports float dtypes
            "reciprocal",  # ONNX op only supports float dtypes
            "round",  # InvalidGraph: int32 not supported by Round
            "rsqrt",  # ONNX op only supports float dtypes
            "sigmoid",  # ONNX op only supports float dtypes
            "sin",  # ONNX op only supports float dtypes
            "sinh",  # ONNX op only supports float dtypes
            "sqrt",  # ONNX op only supports float dtypes
            "tan",  # ONNX op only supports float dtypes
            "tanh",  # ONNX op only supports float dtypes
            "tril",  # NOT_IMPLEMENTED: Trilu(14) not supported for int32 by onnxruntime
            "triu",  # NOT_IMPLEMENTED: Trilu(14) not supported for int32 by onnxruntime
            "trunc",  # InvalidGraph: int32 not supported by Round
            "prod",  # type mismatch: int32 input produces int64 output
            "sum",  # type mismatch: int32 input produces int64 output
            "xlogy",  # ONNX Log only supports float dtypes
        }
    ),
    "tracing": frozenset(),
}

# Extra exclusions specific to torch.int64.
# ``"default"`` contains int64-specific failures on the standard path;
# ``"tracing"`` lists additional failures specific to the tracing path.
XFAIL_OPS_INT64: Dict[str, FrozenSet[str]] = {
    "default": frozenset(
        {
            # Ops that produce float outputs from int64 inputs but type inference
            # fails or ONNX model is invalid because the op only supports float:
            "__rdiv__",  # reciprocal node type mismatch
            "__rpow__",  # negative integer powers not allowed
            "__rxor__",  # FunctionNotFoundError: bitwise_xor
            "acos",  # ONNX op only supports float dtypes
            "acosh",  # ONNX op only supports float dtypes
            "asin",  # ONNX op only supports float dtypes
            "asinh",  # ONNX op only supports float dtypes
            "angle",  # ONNX Atan not supported for integer dtypes
            "atan",  # ONNX op only supports float dtypes
            "atan2",  # ONNX Atan not supported for integer dtypes
            "atanh",  # ONNX op only supports float dtypes
            "bitwise_left_shift",  # FunctionNotFoundError
            "bitwise_right_shift",  # FunctionNotFoundError
            "bitwise_xor",  # FunctionNotFoundError
            "ceil",  # InvalidGraph: int64 not supported by Ceil
            "cos",  # ONNX op only supports float dtypes
            "cosh",  # ONNX op only supports float dtypes
            "erf",  # ONNX op only supports float dtypes
            "erfc",  # ONNX Erf only supports float dtypes
            "erfinv",  # ONNX Erf/Log/Exp only support float dtypes
            "exp",  # ONNX op only supports float dtypes
            "expm1",  # ONNX op only supports float dtypes
            "floor",  # InvalidGraph: int64 not supported by Floor
            "floor_divide",  # ref_diff=1
            "gcd",  # FunctionNotFoundError
            "isinf",  # InvalidGraph: int64 not supported by IsInf
            "isnan",  # InvalidGraph: int64 not supported by IsNaN
            "lcm",  # FunctionNotFoundError
            "log",  # ONNX op only supports float dtypes
            "log10",  # ONNX Log op only supports float dtypes
            "log1p",  # ONNX Log op only supports float dtypes
            "log2",  # ONNX Log op only supports float dtypes
            "logaddexp",  # ONNX Log/Exp ops only support float dtypes
            "logaddexp2",  # ONNX Log/Exp ops only support float dtypes
            "logit",  # ONNX Log op only supports float dtypes
            "nanmean",  # IsNaN not supported for integer dtypes
            "nansum",  # IsNaN not supported for integer dtypes
            "nn_functional_relu",  # NOT_IMPLEMENTED: Relu not supported for int64
            "nn_functional_softsign",  # type mismatch in Div
            "nn_functional_tanhshrink",  # ONNX Tanh only supports float dtypes
            "reciprocal",  # ONNX op only supports float dtypes
            "round",  # InvalidGraph: int64 not supported by Round
            "rsqrt",  # ONNX op only supports float dtypes
            "sigmoid",  # ONNX op only supports float dtypes
            "sin",  # ONNX op only supports float dtypes
            "sinh",  # ONNX op only supports float dtypes
            "sqrt",  # ONNX op only supports float dtypes
            "tan",  # ONNX op only supports float dtypes
            "tanh",  # ONNX op only supports float dtypes
            "trunc",  # InvalidGraph: int64 not supported by Round
            "xlogy",  # ONNX Log only supports float dtypes
        }
    ),
    "tracing": frozenset(),
}

# Per-op absolute tolerance overrides for torch.float16.
# Ops whose variance/std computation compounds float16 rounding errors need
# a larger tolerance than the global _ATOL_FLOAT16 = 1e-2.
ATOL_OPS_FLOAT32: Dict[str, float] = {
    "std": 3e-2,  # variance accumulates float16 rounding; sqrt amplifies
    "std_mean": 3e-2,  # same compound error as std
}
# Per-op absolute tolerance overrides for torch.float16.
# Ops whose variance/std computation compounds float16 rounding errors need
# a larger tolerance than the global _ATOL_FLOAT16 = 1e-2.
ATOL_OPS_FLOAT16: Dict[str, float] = {
    "addr": 0.02,
    "std": 1e-1,  # variance accumulates float16 rounding; sqrt amplifies
    "std_mean": 3e-1,  # same compound error as std
}

# Per-op absolute tolerance overrides for torch.bfloat16.
# bfloat16 has only 7 mantissa bits, so statistical ops need an even wider
# tolerance than the global _ATOL_BFLOAT16 = 2e-2.
ATOL_OPS_BFLOAT16: Dict[str, float] = {
    "logit": 5e-2,  # bfloat16 log precision compounds across Sub(Log(x), Log(1-x))
    "std": 2e-1,  # bfloat16 precision loss is larger than float16
    "std_mean": 2e-1,  # same compound error as std
}


def get_op_coverage_rst() -> str:
    """Returns RST tables showing op-db coverage per op and dtype.

    Queries ``torch.testing._internal.common_methods_invocations.op_db`` and
    builds two grids (one for the default export path, one for the torch tracing
    path) showing, for every op and dtype combination, whether the op is:

    * ``✔`` - in the tested set (converter exists, no known failure for that dtype),
    * ``⚠ xfail`` - converter exists but the test is a known failure,
    * ``✘ no converter`` - no ONNX converter has been implemented yet, or
    * ``—`` - the op does not support that dtype.

    Returns:
        RST source string with two ``list-table`` directives (default path and
        tracing path) ready to be printed inside a ``.. runpython::`` block with
        ``:rst:`` enabled.
    """
    import warnings

    import torch
    from torch.testing._internal import common_methods_invocations

    dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64]
    dtype_names = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.int32: "int32",
        torch.int64: "int64",
    }
    # Xfail sets for the default export path.
    xfail_map = {
        torch.float32: XFAIL_OPS["default"],
        torch.float16: XFAIL_OPS["default"] | XFAIL_OPS_FLOAT16["default"],
        torch.bfloat16: XFAIL_OPS["default"] | XFAIL_OPS_BFLOAT16["default"],
        torch.int32: XFAIL_OPS["default"] | XFAIL_OPS_INT32["default"],
        torch.int64: XFAIL_OPS["default"] | XFAIL_OPS_INT64["default"],
    }
    # Xfail sets for the torch tracing export path.
    xfail_map_tracing = {
        torch.float32: XFAIL_OPS["default"] | XFAIL_OPS["tracing"],
        torch.float16: (
            XFAIL_OPS["default"]
            | XFAIL_OPS_FLOAT16["default"]
            | XFAIL_OPS["tracing"]
            | XFAIL_OPS_FLOAT16["tracing"]
        ),
        torch.bfloat16: (
            XFAIL_OPS["default"]
            | XFAIL_OPS_BFLOAT16["default"]
            | XFAIL_OPS["tracing"]
            | XFAIL_OPS_BFLOAT16["tracing"]
        ),
        torch.int32: (
            XFAIL_OPS["default"]
            | XFAIL_OPS_INT32["default"]
            | XFAIL_OPS["tracing"]
            | XFAIL_OPS_INT32["tracing"]
        ),
        torch.int64: (
            XFAIL_OPS["default"]
            | XFAIL_OPS_INT64["default"]
            | XFAIL_OPS["tracing"]
            | XFAIL_OPS_INT64["tracing"]
        ),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops = [
            op
            for op in common_methods_invocations.op_db
            if not op.variant_test_name and op.name not in NON_DETERMINISTIC_OPS
        ]
    ops.sort(key=lambda o: o.name)

    def _make_table(title: str, xmap: dict) -> str:
        """Builds an RST list-table for *xmap* (dtype → xfail set).

        Args:
            title: Section title placed above the table.
            xmap: Mapping from :class:`torch.dtype` to the set of xfail op keys.

        Returns:
            RST string containing a rubric directive followed by a list-table.
        """
        header = ["Op"]
        for dt in dtypes:
            header.append(dtype_names[dt])

        rows = []
        for op in ops:
            key = op.name.replace(".", "_")
            row = [f"``{op.name}``"]
            for dt in dtypes:
                if dt not in op.dtypes:
                    row.append(_NOT_APPLICABLE)
                elif key in NO_CONVERTER_OPS:
                    row.append(_NO_CONVERTER)
                elif key in xmap[dt]:
                    row.append(_XFAIL)
                else:
                    row.append(_SUPPORTED)
            rows.append(row)

        n_cols = len(header)
        widths = " ".join(["20"] + ["16"] * (n_cols - 1))
        lines = [
            f".. rubric:: {title}",
            "",
            ".. list-table::",
            "    :header-rows: 1",
            f"    :widths: {widths}",
            "",
        ]
        lines.append("    * - " + header[0])
        for h in header[1:]:
            lines.append("      - " + h)
        for row in rows:
            lines.append("    * - " + row[0])
            for cell in row[1:]:
                lines.append("      - " + cell)
        lines.append("")
        return "\n".join(lines)

    default_table = _make_table("Default export path", xfail_map)
    tracing_table = _make_table("Torch tracing export path (``tracing=True``)", xfail_map_tracing)
    return default_table + "\n" + tracing_table

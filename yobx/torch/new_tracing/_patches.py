"""
Private context managers used by :class:`~yobx.torch.new_tracing.tracer.GraphTracer`
during tracing to temporarily replace or patch torch internals.

Each context manager restores the original state on exit so that patches are
limited to the duration of a single :meth:`~yobx.torch.new_tracing.tracer.GraphTracer.trace`
call.
"""

import contextlib
from typing import Any, Callable, Generator, List, Optional, Sequence, Tuple, Union
import torch

# Capture the real ``torch.cond`` at import time so that it can always be
# used as the canonical FX node target even when ``torch.cond`` has been
# temporarily replaced by the tracing shim (nested context managers would
# otherwise make the inner context see the outer shim as "original").
_ORIGINAL_TORCH_COND: Callable = torch.cond  # type: ignore[attr-defined]

# Capture the real ``torch._check`` at import time so that it can be
# restored after tracing.  ``torch._check`` may not exist on older versions.
_ORIGINAL_TORCH_CHECK: Optional[Callable] = getattr(torch, "_check", None)

# Capture the real ``torch.ops.higher_order.scan`` at import time so that it
# can always be used as the canonical FX node target.  The op may not exist on
# older PyTorch versions.
_ORIGINAL_TORCH_SCAN: Optional[Callable] = getattr(
    getattr(torch.ops, "higher_order", None), "scan", None
)

# Capture the real ``torch.arange`` at import time so calls using a symbolic
# ``TracingInt`` as the *end* (or *start*/*step*) argument can be redirected
# during tracing.
_ORIGINAL_TORCH_ARANGE: Callable = torch.arange

# Capture the real ``torch.full`` at import time so shape-only constructors
# can be redirected during tracing when they receive TracingInt sizes.
_ORIGINAL_TORCH_FULL: Callable = torch.full

# Capture the real ``torch.zeros`` and ``torch.ones`` at import time so they
# can be redirected during tracing when they receive TracingInt sizes.
_ORIGINAL_TORCH_ZEROS: Callable = torch.zeros
_ORIGINAL_TORCH_ONES: Callable = torch.ones

# Capture the real ``torch.tensor_split`` at import time so that calls with
# a :class:`~yobx.torch.new_tracing.tensor.TracingTensor` as
# ``indices_or_sections`` can be intercepted: the native C++ kernel tries to
# read concrete values from the indices tensor before dispatching, which fails
# for :class:`~yobx.torch.new_tracing.tensor.TracingTensor` instances that
# carry no backing storage.
_ORIGINAL_TORCH_TENSOR_SPLIT: Callable = torch.tensor_split

# Capture the real ``torch._higher_order_ops.while_loop`` and
# ``torch.ops.higher_order.while_loop`` at import time so that calls during
# new-tracing are intercepted and recorded as FX nodes.  Both attributes may
# refer to the same callable on some PyTorch versions; we patch whichever
# locations exist.
_ORIGINAL_TORCH_WHILE_LOOP: Optional[Callable] = getattr(
    getattr(torch, "_higher_order_ops", None), "while_loop", None
)
_ORIGINAL_TORCH_WHILE_LOOP_OP: Optional[Callable] = getattr(
    getattr(torch.ops, "higher_order", None), "while_loop", None
)

# Capture the real ``torch.compiler.is_exporting`` at import time so that it
# can be temporarily replaced during tracing.  Models that call
# ``torch.compiler.is_exporting()`` to select a scan-based export path need
# the function to return ``True`` while the new tracer is running.
_ORIGINAL_IS_EXPORTING: Optional[Callable] = getattr(
    getattr(torch, "compiler", None), "is_exporting", None
)


@contextlib.contextmanager
def _cond_replacement_ctx(tracer: "GraphTracer") -> Generator:  # type: ignore[name-defined]  # noqa: F821
    """
    Temporarily replaces ``torch.cond`` with a tracing-aware handler so that
    ``torch.cond`` calls encountered during
    :meth:`~yobx.torch.new_tracing.tracer.GraphTracer.trace` are captured as
    FX ``call_function`` nodes instead of being executed eagerly (or routed
    through dynamo/compile).

    :param tracer: The :class:`~yobx.torch.new_tracing.tracer.GraphTracer`
        whose :meth:`_handle_cond` should be used as the replacement.
    """

    def _cond_handler(
        pred: Any, true_fn: Callable, false_fn: Callable, operands: Union[List, Tuple] = ()
    ) -> Any:
        return tracer._handle_cond(pred, true_fn, false_fn, operands)

    torch.cond = _cond_handler  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.cond = _ORIGINAL_TORCH_COND  # type: ignore[assignment]


@contextlib.contextmanager
def _check_replacement_ctx(tracer: "GraphTracer") -> Generator:  # type: ignore[name-defined]  # noqa: F821
    """
    Temporarily replaces ``torch._check`` with a tracing-aware handler so that
    ``torch._check`` calls encountered during
    :meth:`~yobx.torch.new_tracing.tracer.GraphTracer.trace` register symbolic
    conditions instead of crashing when the condition is a
    :class:`~yobx.torch.new_tracing.shape.TracingBool` or a concrete ``False``
    produced by a symbolic dimension that is stored as ``0`` in the underlying
    tensor.

    Registered conditions are stored in the module-level
    :data:`~yobx.torch.new_tracing.shape._known_true_conditions` set and are
    consulted by :meth:`~yobx.torch.new_tracing.shape.TracingBool.__bool__` to
    resolve comparisons that would otherwise raise :exc:`ValueError`.

    :param tracer: The :class:`~yobx.torch.new_tracing.tracer.GraphTracer`
        whose :meth:`_handle_check` should be used as the replacement.
    """
    if _ORIGINAL_TORCH_CHECK is None:
        yield
        return

    def _check_handler(cond: Any, msg: Any = None) -> None:
        tracer._handle_check(cond, msg)

    torch._check = _check_handler  # type: ignore[assignment]
    try:
        yield
    finally:
        torch._check = _ORIGINAL_TORCH_CHECK  # type: ignore[assignment]


@contextlib.contextmanager
def _scan_replacement_ctx(tracer: "GraphTracer") -> Generator:  # type: ignore[name-defined]  # noqa: F821
    """
    A context manager that temporarily replaces ``torch.ops.higher_order.scan``
    with a tracing-aware handler so that scan calls encountered during
    :meth:`~yobx.torch.new_tracing.tracer.GraphTracer.trace` are captured as
    FX ``call_function`` nodes instead of being executed eagerly.

    :param tracer: The :class:`~yobx.torch.new_tracing.tracer.GraphTracer`
        whose :meth:`_handle_scan` should be used as the replacement.
    """
    if _ORIGINAL_TORCH_SCAN is None:
        yield
        return

    def _scan_handler(
        f: Callable,
        init_states: List,
        scan_inputs: List,
        additional_inputs: Optional[List] = None,
        dim: int = 0,
        reverse: bool = False,
    ) -> Any:
        return tracer._handle_scan(f, init_states, scan_inputs, additional_inputs, dim, reverse)

    torch.ops.higher_order.scan = _scan_handler  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.ops.higher_order.scan = _ORIGINAL_TORCH_SCAN  # type: ignore[assignment]


@contextlib.contextmanager
def _full_replacement_ctx(tracer: "GraphTracer") -> Generator:  # type: ignore[name-defined]  # noqa: F821
    """
    Temporarily replaces ``torch.full`` with a tracing-aware handler so calls
    using symbolic ``TracingInt`` sizes are captured as FX nodes.

    The handler distinguishes two call sites:

    * **User model code** — ``tracer._fake_mode`` has not yet been entered
      (its ``enter_stack`` is empty); the handler delegates to
      :meth:`~yobx.torch.new_tracing.tracer.GraphTracer._handle_full` which
      emits an FX node and returns a
      :class:`~yobx.torch.new_tracing.tensor.TracingTensor`.
    * **FakeTensor kernel implementations** — these arise during
      :meth:`~yobx.torch.new_tracing.tracer.GraphTracer.dispatch`'s
      ``with self._fake_mode:`` block; at that point
      ``tracer._fake_mode.enter_stack`` is non-empty and the handler delegates
      to the original ``torch.full`` so that FakeTensor kernels get the
      correct FakeTensor results.

    :param tracer: The :class:`~yobx.torch.new_tracing.tracer.GraphTracer`
        whose :meth:`_handle_full` should be used as the replacement.

    Returns:
        A context manager that yields control while ``torch.full``
        is temporarily replaced, then restores the original implementation on
        exit.
    """

    def _full_handler(size: Any, fill_value: Any, **kwargs: Any) -> Any:
        # When tracer._fake_mode has been entered (enter_stack is non-empty) we
        # are inside dispatch()'s `with self._fake_mode:` block — the call
        # originates from a FakeTensor kernel implementation.  Delegate to the
        # real torch.full so the kernel gets a proper FakeTensor result.
        #
        # Note: we cannot use _get_current_dispatch_mode_stack() here because
        # FakeTensorMode temporarily pops itself from the stack during its own
        # __torch_dispatch__ invocation to prevent infinite recursion.
        if tracer._fake_mode.enter_stack:
            return _ORIGINAL_TORCH_FULL(size, fill_value, **kwargs)
        return tracer._handle_full(size, fill_value, **kwargs)

    torch.full = _full_handler  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.full = _ORIGINAL_TORCH_FULL  # type: ignore[assignment]


@contextlib.contextmanager
def _zeros_replacement_ctx(tracer: "GraphTracer") -> Generator:  # type: ignore[name-defined]  # noqa: F821
    """
    Temporarily replaces ``torch.zeros`` with a tracing-aware handler so calls
    using symbolic ``TracingInt`` sizes are captured as FX nodes.

    The handler distinguishes two call sites:

    * **User model code** — ``tracer._fake_mode`` has not yet been entered
      (its ``enter_stack`` is empty); the handler delegates to
      :meth:`~yobx.torch.new_tracing.tracer.GraphTracer._handle_zeros` which
      emits an FX node and returns a
      :class:`~yobx.torch.new_tracing.tensor.TracingTensor`.
    * **FakeTensor kernel implementations** — these arise during
      :meth:`~yobx.torch.new_tracing.tracer.GraphTracer.dispatch`'s
      ``with self._fake_mode:`` block; at that point
      ``tracer._fake_mode.enter_stack`` is non-empty and the handler delegates
      to the original ``torch.zeros`` so that FakeTensor kernels get the
      correct FakeTensor results.

    :param tracer: The :class:`~yobx.torch.new_tracing.tracer.GraphTracer`
        whose :meth:`_handle_zeros` should be used as the replacement.

    Returns:
        Returns a context manager that yields control while ``torch.zeros``
        is temporarily replaced, then restores the original implementation on
        exit.
    """

    def _zeros_handler(size: Any, **kwargs: Any) -> Any:
        if tracer._fake_mode.enter_stack:
            return _ORIGINAL_TORCH_ZEROS(size, **kwargs)
        return tracer._handle_zeros(size, **kwargs)

    torch.zeros = _zeros_handler  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.zeros = _ORIGINAL_TORCH_ZEROS  # type: ignore[assignment]


@contextlib.contextmanager
def _ones_replacement_ctx(tracer: "GraphTracer") -> Generator:  # type: ignore[name-defined]  # noqa: F821
    """
    Temporarily replaces ``torch.ones`` with a tracing-aware handler so calls
    using symbolic ``TracingInt`` sizes are captured as FX nodes.

    The handler distinguishes two call sites:

    * **User model code** — ``tracer._fake_mode`` has not yet been entered
      (its ``enter_stack`` is empty); the handler delegates to
      :meth:`~yobx.torch.new_tracing.tracer.GraphTracer._handle_ones` which
      emits an FX node and returns a
      :class:`~yobx.torch.new_tracing.tensor.TracingTensor`.
    * **FakeTensor kernel implementations** — these arise during
      :meth:`~yobx.torch.new_tracing.tracer.GraphTracer.dispatch`'s
      ``with self._fake_mode:`` block; at that point
      ``tracer._fake_mode.enter_stack`` is non-empty and the handler delegates
      to the original ``torch.ones`` so that FakeTensor kernels get the
      correct FakeTensor results.

    :param tracer: The :class:`~yobx.torch.new_tracing.tracer.GraphTracer`
        whose :meth:`_handle_ones` should be used as the replacement.

    Returns:
        Returns a context manager that yields control while ``torch.ones``
        is temporarily replaced, then restores the original implementation on
        exit.
    """

    def _ones_handler(size: Any, **kwargs: Any) -> Any:
        if tracer._fake_mode.enter_stack:
            return _ORIGINAL_TORCH_ONES(size, **kwargs)
        return tracer._handle_ones(size, **kwargs)

    torch.ones = _ones_handler  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.ones = _ORIGINAL_TORCH_ONES  # type: ignore[assignment]


@contextlib.contextmanager
def _arange_replacement_ctx(
    tracer: "GraphTracer",  # type: ignore[name-defined]  # noqa: F821
) -> Generator:
    """
    Temporarily replaces ``torch.arange`` with a tracing-aware handler so calls
    using symbolic ``TracingInt`` values as *start*, *end*, or *step* arguments
    are captured as FX nodes rather than failing when
    :meth:`~yobx.torch.new_tracing.shape.TracingInt.__int__` is called on a
    symbolic dimension.

    :param tracer: The :class:`~yobx.torch.new_tracing.tracer.GraphTracer`
        whose :meth:`_handle_arange` should be used as the replacement.

    Returns:
        Yields control while ``torch.arange`` is temporarily replaced, then
        restores the original implementation on exit.
    """

    def _arange_handler(*args: Any, **kwargs: Any) -> Any:
        return tracer._handle_arange(*args, **kwargs)

    torch.arange = _arange_handler  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.arange = _ORIGINAL_TORCH_ARANGE  # type: ignore[assignment]


@contextlib.contextmanager
def _tensor_split_replacement_ctx(
    tracer: "GraphTracer",  # type: ignore[name-defined]  # noqa: F821
) -> Generator:
    """
    Temporarily replaces ``torch.tensor_split`` with a tracing-aware handler.

    The native C++ kernel for ``aten::tensor_split`` attempts to read the
    *concrete* values from the ``indices_or_sections`` tensor in order to
    decide (a) the number of output chunks and (b) their sizes along *dim*.
    This fails for :class:`~yobx.torch.new_tracing.tensor.TracingTensor`
    instances because they carry no backing storage.

    The replacement intercepts the call and delegates to
    :meth:`~yobx.torch.new_tracing.tracer.GraphTracer._handle_tensor_split`
    which performs shape inference via concrete surrogate tensors and emits
    the appropriate FX ``call_function`` node.

    :param tracer: The :class:`~yobx.torch.new_tracing.tracer.GraphTracer`
        whose :meth:`_handle_tensor_split` should be used as the replacement.
    """

    def _tensor_split_handler(input: Any, indices_or_sections: Any, dim: int = 0) -> Any:
        return tracer._handle_tensor_split(input, indices_or_sections, dim)

    torch.tensor_split = _tensor_split_handler  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.tensor_split = _ORIGINAL_TORCH_TENSOR_SPLIT  # type: ignore[assignment]


@contextlib.contextmanager
def _roll_dynamic_shape_ctx() -> Generator:
    """
    Temporarily replaces the ``aten.roll.default`` decomposition with a
    version that handles dynamic shapes.

    The standard ``torch._refs.roll`` decomposition calls ``a.numel() == 0``
    to short-circuit on empty tensors.  When the tensor has dynamic (symbolic)
    dimensions this comparison raises
    :exc:`torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode`
    because the shape environment cannot statically evaluate whether the
    element count is zero.

    Replaces the decomposition with a patched variant that catches that error
    and proceeds to the normal roll logic (which is safe for dynamic shapes).
    The original decomposition is restored on exit.
    """
    from collections.abc import Iterable as _Iterable

    import torch._prims_common as _prims_common
    import torch.fx.experimental.symbolic_shapes as _sym_shapes
    from torch._decomp import decomposition_table as _decomp_table

    _roll_op = torch.ops.aten.roll.default
    _orig_decomp = _decomp_table.get(_roll_op)

    def _roll_with_dynamic_shapes(
        a: torch.Tensor, shifts: Sequence, dims: Sequence = ()
    ) -> torch.Tensor:
        """Reimplements torch._refs.roll, handling dynamic-shape numel check."""
        dims = _prims_common.canonicalize_dims(a.ndim, dims)
        if not isinstance(shifts, _Iterable):
            shifts = (shifts,)
        if not isinstance(dims, _Iterable):
            dims = (dims,)

        # Avoid modulo by zero — handle dynamic shapes gracefully.
        try:
            if a.numel() == 0:
                return a.clone()
        except _sym_shapes.GuardOnDataDependentSymNode:
            pass  # Cannot determine at trace time; assume non-empty and proceed.

        if a.dim() == 0 and len(dims) > 0:
            raise IndexError(f"Dimension specified as {dims[0]} but tensor has no dimensions")

        len_shifts = len(shifts)
        len_dims = len(dims)
        if len_shifts != 1 or len_dims != 1:
            if len_shifts == 0:
                raise RuntimeError("`shifts` required")
            if len_dims == 0 and len_shifts == 1:
                return torch.roll(torch.flatten(a), shifts, 0).view(a.shape)
            if len_shifts != len_dims:
                raise RuntimeError(
                    f"shifts and dimensions must align. "
                    f"shifts: {len_shifts}, dims: {len_dims}"
                )
            if len_dims <= 1:
                raise AssertionError(f"Expected len_dims > 1, got {len_dims}")
            tail_shifts = shifts[1:]
            tail_dims = dims[1:]
            first_dim_rolled = torch.roll(a, (shifts[0],), dims[0])
            return torch.roll(first_dim_rolled, tail_shifts, tail_dims)

        dim = dims[0]
        size = a.shape[dim]
        start = (size - shifts[0]) % size
        idx = torch.arange(size, device=a.device)
        return a.index_select(dim, torch.fmod(start + idx, size))

    if _orig_decomp is not None:
        _decomp_table[_roll_op] = _roll_with_dynamic_shapes
    try:
        yield
    finally:
        if _orig_decomp is not None:
            _decomp_table[_roll_op] = _orig_decomp


@contextlib.contextmanager
def _while_loop_replacement_ctx(
    tracer: "GraphTracer",  # type: ignore[name-defined]  # noqa: F821
) -> Generator:
    """
    Temporarily replaces ``torch._higher_order_ops.while_loop`` (and
    ``torch.ops.higher_order.while_loop`` when it is a distinct object) with a
    tracing-aware handler so that while-loop calls encountered during
    :meth:`~yobx.torch.new_tracing.tracer.GraphTracer.trace` are captured as
    FX ``call_function`` nodes instead of being executed eagerly.

    :param tracer: The :class:`~yobx.torch.new_tracing.tracer.GraphTracer`
        whose :meth:`_handle_while_loop` should be used as the replacement.

    Yields:
        Control while the while_loop callable is temporarily replaced, then
        restores all original callables on exit.
    """
    if _ORIGINAL_TORCH_WHILE_LOOP is None and _ORIGINAL_TORCH_WHILE_LOOP_OP is None:
        yield
        return

    def _while_loop_handler(
        cond_fn: Callable,
        body_fn: Callable,
        carried_inputs: Any,
        additional_inputs: Optional[List] = None,
    ) -> Any:
        return tracer._handle_while_loop(cond_fn, body_fn, carried_inputs, additional_inputs)

    _higher_order_ops = getattr(torch, "_higher_order_ops", None)
    _higher_order = getattr(torch.ops, "higher_order", None)

    if _ORIGINAL_TORCH_WHILE_LOOP is not None and _higher_order_ops is not None:
        _higher_order_ops.while_loop = _while_loop_handler  # type: ignore[assignment]
    if (
        _ORIGINAL_TORCH_WHILE_LOOP_OP is not None
        and _higher_order is not None
        and _ORIGINAL_TORCH_WHILE_LOOP_OP is not _ORIGINAL_TORCH_WHILE_LOOP
    ):
        _higher_order.while_loop = _while_loop_handler  # type: ignore[assignment]
    try:
        yield
    finally:
        if _ORIGINAL_TORCH_WHILE_LOOP is not None and _higher_order_ops is not None:
            _higher_order_ops.while_loop = _ORIGINAL_TORCH_WHILE_LOOP  # type: ignore[assignment]
        if (
            _ORIGINAL_TORCH_WHILE_LOOP_OP is not None
            and _higher_order is not None
            and _ORIGINAL_TORCH_WHILE_LOOP_OP is not _ORIGINAL_TORCH_WHILE_LOOP
        ):
            _higher_order.while_loop = _ORIGINAL_TORCH_WHILE_LOOP_OP  # type: ignore[assignment]


@contextlib.contextmanager
def _is_exporting_replacement_ctx() -> Generator:
    """
    Temporarily replaces ``torch.compiler.is_exporting`` with a function that
    always returns ``True`` so that model code guarded by
    ``torch.compiler.is_exporting()`` takes the export branch (e.g. choosing a
    scan-based implementation) while the new tracer is active.

    The original implementation is restored unconditionally on exit.
    """
    _compiler = getattr(torch, "compiler", None)
    if _ORIGINAL_IS_EXPORTING is None or _compiler is None:
        yield
        return

    _compiler.is_exporting = lambda: True  # type: ignore[assignment]
    try:
        yield
    finally:
        _compiler.is_exporting = _ORIGINAL_IS_EXPORTING  # type: ignore[assignment]


@contextlib.contextmanager
def _trace_replacement_ctx(tracer: "GraphTracer") -> Generator:  # type: ignore[name-defined]  # noqa: F821
    """
    Applies all tracing-time torch replacement context managers at once.

    :param tracer: The :class:`~yobx.torch.new_tracing.tracer.GraphTracer`
        using these temporary runtime patches.

    Returns:
        A context manager that enters all replacement contexts and restores all
        original torch functions/decompositions on exit.
    """
    with (
        _cond_replacement_ctx(tracer),
        _check_replacement_ctx(tracer),
        _full_replacement_ctx(tracer),
        _zeros_replacement_ctx(tracer),
        _ones_replacement_ctx(tracer),
        _arange_replacement_ctx(tracer),
        _roll_dynamic_shape_ctx(),
        _scan_replacement_ctx(tracer),
        _tensor_split_replacement_ctx(tracer),
        _while_loop_replacement_ctx(tracer),
        _is_exporting_replacement_ctx(),
    ):
        yield

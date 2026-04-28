import inspect
from functools import reduce
from typing import Any, List, Sequence
import torch
from ...helpers.patch_helper import PatchInfo


def _get_DynamicDimConstraintPrinter():
    import torch.fx.experimental.symbolic_shapes

    return torch.fx.experimental.symbolic_shapes.DynamicDimConstraintPrinter


class patched_DynamicDimConstraintPrinter:
    """
    Patches
    ``torch.tx.experimental.symbolic_shapes.DynamicDimConstraintPrinter._print_Symbol``.
    Valid for ``torch>=2.10``.
    """

    __CLASS__ = _get_DynamicDimConstraintPrinter()
    __METHODS__ = ["_print_Symbol"]

    def _print_Symbol(self, expr: "sympy.Symbol") -> str:  # noqa: F821
        import sympy

        assert isinstance(expr, sympy.Symbol), str(type(expr))
        if self.symbol_to_source.get(expr):  # type: ignore
            return self.symbol_to_source[expr][0].name  # type: ignore
        return str(expr)


def patched_infer_size(a, b):
    """
    Patches ``torch._subclasses.fake_impls.infer_size``.
    This patch is needed to export
    :class:`yobx.torch.tiny_models.TinyBroadcastAddModel`.
    """
    dimsA = len(a)
    dimsB = len(b)
    ndim = max(dimsA, dimsB)
    expandedSizes = [0] * ndim
    for i in range(ndim - 1, -1, -1):
        offset = ndim - 1 - i
        dimA = dimsA - 1 - offset
        dimB = dimsB - 1 - offset
        sizeA = a[dimA] if dimA >= 0 else 1
        sizeB = b[dimB] if dimB >= 0 else 1

        # NB: It is very important to test for broadcasting, before testing
        # sizeA == sizeB.  This is because the broadcasting tests are likely
        # to be statically known (in particular, if sizeA/sizeB is unbacked
        # but size-like, we will unsoundly assume they never equal 1), but
        # the sizeA == sizeB test may not be statically known.  However, once
        # we have established that no broadcasting is happening, the
        # sizeA == sizeB is now expect_true and we can defer it as a runtime
        # assert (this works because Python will return the terminal
        # expression of an or statement as-is, without bool()'ing it; if this
        # were not the case, we'd need to write this using torch.sym_or() or
        # something like that).
        try:
            b1 = torch.fx.experimental.symbolic_shapes.guard_or_false(sizeA == 1)
        except torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
            b1 = False
        try:
            b2 = torch.fx.experimental.symbolic_shapes.guard_or_false(sizeB == 1)
        except torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
            b2 = False
        try:
            b3 = torch.fx.experimental.symbolic_shapes.guard_or_false(sizeA == sizeB)
        except torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
            b3 = False
        if b1 or b2 or b3:
            expandedSizes[i] = (
                sizeB
                if torch.fx.experimental.symbolic_shapes.guard_or_false(sizeA == 1)
                else sizeA
            )
        else:
            # PATCHED: generic case, the dimension is known, no need to assert
            expandedSizes[i] = torch.sym_max(sizeA, sizeB)  # type: ignore
    return tuple(expandedSizes)


def patched__broadcast_shapes(*_shapes):
    """
    Patches ``torch._refs._broadcast_shapes``.
    This patch is needed to export
    :class:`yobx.torch.tiny_models.TinyBroadcastAddModel`.
    """
    from torch._prims_common import IntLike

    shapes = tuple(
        (x,) if isinstance(x, IntLike) else x for x in filter(lambda x: x is not None, _shapes)
    )

    # Short-circuits on no input
    if len(shapes) == 0:
        return None

    for shape in shapes:
        if not isinstance(shape, Sequence):
            raise RuntimeError(
                "Input shapes should be of type ints, a tuple of ints, "
                "or a list of ints, got ",
                shape,
            )

    # Computes common shape
    common_shape = [1] * reduce(max, (len(shape) for shape in shapes))
    for _arg_idx, shape in enumerate(shapes):
        for idx in range(-1, -1 - len(shape), -1):
            if torch.fx.experimental.symbolic_shapes.is_nested_int(shape[idx]):
                # Broadcasting is allowed for (j0, 1) or (j0, j0);
                # not (j0, j1), (j0, 5), etc.
                if torch.fx.experimental.symbolic_shapes.is_nested_int(
                    common_shape[idx]
                ) and torch.fx.experimental.symbolic_shapes.guard_or_false(
                    shape[idx] == common_shape[idx]
                ):
                    continue
            else:
                if torch.fx.experimental.symbolic_shapes.guard_or_false(
                    shape[idx] == common_shape[idx]
                ):
                    continue
            # PATCHED: two cases, if == for sure, no broadcast,
            # otherwise maybe broadcast with max(dimensions)
            if torch.fx.experimental.symbolic_shapes.guard_or_false(common_shape[idx] != 1):
                pass
            elif torch.fx.experimental.symbolic_shapes.guard_or_false(
                common_shape[idx] == 1
            ) or torch.fx.experimental.symbolic_shapes.guard_or_false(shape[idx] != 1):
                if shape[idx] < 0:
                    raise ValueError("Attempting to broadcast a dimension with negative length!")
                common_shape[idx] = shape[idx]  # type: ignore
            else:
                common_shape[idx] = torch.sym_max(common_shape[idx], shape[idx])  # type: ignore

    return common_shape


def _combine_args(f, args, kwargs, preserve_order: bool = False) -> dict[str, Any]:
    # combine args and kwargs following the signature of f, as it happens
    # in the body of f when called with *args, **kwargs
    # the exporter needs to preserve the original order of the arguments
    # to match the dynamic shapes.
    if isinstance(f, torch.export.ExportedProgram):
        f = f.module()

    signature = (
        inspect.signature(f.forward) if isinstance(f, torch.nn.Module) else inspect.signature(f)
    )
    kwargs = kwargs if kwargs is not None else {}
    combined_args = signature.bind(*args, **kwargs).arguments
    if not preserve_order:
        return combined_args

    # If this condition is not met, torch.export.export fails.
    # But does InputObserver this should fail?
    # positional_only = sum(
    #    1
    #    for p in signature.parameters.values()
    #    if p.default == inspect.Parameter.empty
    #    and p.kind in (
    #       inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD
    #    )
    # )
    # if len(args) < positional_only:
    #    raise RuntimeError(
    #        f"There are {positional_only} mandatory positional arguments "
    #        f"and they must be given as positional arguments "
    #        f"but there only {len(args)} are given that way."
    #    )

    var_position_parameters = [
        name
        for name, p in signature.parameters.items()
        if p.kind == inspect.Parameter.VAR_POSITIONAL
    ]
    if var_position_parameters:
        n_positional_only = max(
            [
                i
                for i, p in enumerate(signature.parameters.values())
                if p.kind == inspect.Parameter.VAR_POSITIONAL
            ]
        )
        combined_args_traced_order = dict(zip(signature.parameters, args[:n_positional_only]))
        combined_args_traced_order[var_position_parameters[0]] = tuple(args[n_positional_only:])
    else:
        combined_args_traced_order = dict(zip(signature.parameters, args))
    for arg in kwargs:
        if arg in combined_args:
            combined_args_traced_order[arg] = combined_args[arg]
    for key in combined_args:
        if key not in combined_args_traced_order:
            combined_args_traced_order[key] = combined_args[key]
    return combined_args_traced_order


def patched__get_range_constraints(
    mod: torch.nn.Module,
    export_artifact: "torch.export._trace.ExportArtifact",  # noqa: F821
    args,
    kwargs,
    dynamic_shapes,
):
    """
    Patches ``torch.export._trace._get_range_constraints``.
    See PR `#174593 <https://github.com/pytorch/pytorch/pull/174593>`_.
    """
    gm: torch.fx.GraphModule = export_artifact.aten.gm
    export_graph_signature: torch.export.graph_signature.ExportGraphSignature = (
        export_artifact.aten.sig
    )
    fake_mode: "FakeTensorMode" = export_artifact.fake_mode  # noqa: F821
    num_lifted = next(
        (
            i
            for i, s in enumerate(export_graph_signature.input_specs)
            if s.kind == torch.export.graph_signature.InputKind.USER_INPUT
        ),
        len(export_graph_signature.input_specs),
    )

    # preserve_order=True:
    # this is because we trace based on the kwargs passed in from user
    # not based on the signature. I feel it would be better to just enforce
    # one ordering at the start of tracing to avoid confusions, but that is
    # bigger refactor, so do this to unblock for now.
    combined_args = _combine_args(mod, args, kwargs, preserve_order=True)

    range_constraints = torch._export.non_strict_utils.make_constraints(
        fake_mode, gm, combined_args, dynamic_shapes, num_lifted
    )
    return range_constraints


def patched__maybe_broadcast(*args, preserve_cpu_scalar_tensors=True):
    """
    Patches ``torch._refs._maybe_broadcast``.
    This patch is needed to export
    :class:`yobx.torch.tiny_models.TinyBroadcastAddModel`.
    """
    from torch._prims_common import ShapeType, TensorLike, Number

    # Computes common shape
    common_shape = patched__broadcast_shapes(
        *(t.shape if isinstance(t, TensorLike) else None for t in args)
    )

    def should_expand(a: ShapeType, b: ShapeType) -> bool:
        from torch.fx.experimental.symbolic_shapes import guard_or_false, sym_and, sym_or

        if len(a) != len(b):
            return True

        for x, y in zip(a, b):
            if guard_or_false(x != y):
                # We know they are not the same.
                return True

            # They are the same or we do not know if they are the same or not.
            # 1==1 no-broadcast
            # u0==1 and 1==u0 cases. We broadcast!
            if guard_or_false(sym_and(x == 1, y == 1)):
                pass
            elif guard_or_false(sym_or(x == 1, y == 1)):
                # assume broadcasting.
                return True

            # u0==u1 assume the same, no broadcasting!
            # PATCHED: avoid errors
            return True  # guard_or_true(x != y)
            # torch._check(
            #    x == y,
            #    lambda x=x, y=y: (
            #        f"sizes assumed to be the same due to unbacked "
            #        f"broadcasting semantics x={x!r}, y={y!r}"
            #    ),
            # )

        return False

    def __maybe_broadcast(x, shape):
        if x is None:
            return None
        elif isinstance(x, Number):
            return x
        elif isinstance(x, TensorLike):
            if preserve_cpu_scalar_tensors and torch._prims_common.is_cpu_scalar_tensor(x):  # type: ignore
                return x

            if should_expand(x.shape, common_shape):  # type: ignore
                return x.expand(common_shape)  # type: ignore

            return x
        else:
            raise RuntimeError(f"Unexpected type when broadcasting: {str(type(x))}!")

    return tuple(__maybe_broadcast(x, common_shape) for x in args)


def get_patches() -> List[PatchInfo]:
    import torch.export._trace

    return [
        PatchInfo.make(
            patched_DynamicDimConstraintPrinter._print_Symbol,
            torch.fx.experimental.symbolic_shapes.DynamicDimConstraintPrinter,
            "_print_Symbol",
            family="torch",
        ),
        PatchInfo.make(
            patched_infer_size, torch._subclasses.fake_impls, "infer_size", family="torch"
        ),
        PatchInfo.make(
            patched__broadcast_shapes, torch._refs, "_broadcast_shapes", family="torch"
        ),
        PatchInfo.make(
            patched__get_range_constraints,
            torch.export._trace,
            "_get_range_constraints",
            family="torch",
        ),
        PatchInfo.make(patched__maybe_broadcast, torch._refs, "_maybe_broadcast", family="torch"),
    ]

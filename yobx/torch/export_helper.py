from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import torch
from ..helpers import flatten_object
from .input_observer import InputCandidate


def _guard_or(a: "BoolLikeType", default: bool) -> bool:  # type: ignore # noqa: F821
    import torch.fx.experimental.symbolic_shapes as _tds

    if not isinstance(a, _tds.SymBool):
        assert isinstance(a, bool)
        return a

    result = _tds._static_eval_sym_bool(a)
    return result if result is not None else default


def torch_export(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Mapping[str, Any]] = None,
    *,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any, ...], List[Any]]] = None,
    strict: bool = False,
    preserve_module_call_signature: Tuple[str, ...] = (),
    # prefer_deferred_runtime_asserts_over_guards: bool = False,  # torch==2.9
    backed_size_oblivious: Union[bool, str] = False,
    prefer_deferred_runtime_asserts_over_guards: bool = False,
    verbose: int = 0,
    **other_kwargs,
):
    """
    Wrapper around :func:`torch.export.export`.
    ``backed_size_oblivious`` can be boolean then it calls
    ``torch.fx.experimental._config.patch(backed_size_oblivious=True)``
    or not. It can be ``'auto'`` to let select automatically the best
    mode. It can be ``'half'`` to disable some non oblivious exceptions.
    """
    export_kwargs: Dict[str, Any] = {}
    if prefer_deferred_runtime_asserts_over_guards:
        export_kwargs["prefer_deferred_runtime_asserts_over_guards"] = (
            prefer_deferred_runtime_asserts_over_guards
        )
    if preserve_module_call_signature:
        export_kwargs["preserve_module_call_signature"] = preserve_module_call_signature

    export_kwargs.update(other_kwargs)
    if backed_size_oblivious == "half":
        if verbose:
            print(f"[torch_export] backed_size_oblivious={backed_size_oblivious!r}")
        import torch.fx.experimental.symbolic_shapes as _tds

        value = _tds.ShapeEnv._init.__kwdefaults__["specialize_zero_one"]  # type: ignore
        _tds.ShapeEnv._init.__kwdefaults__["specialize_zero_one"] = False  # type: ignore
        _tds._guard_or = _guard_or

        with torch.fx.experimental._config.patch(backed_size_oblivious=True):  # type: ignore
            ep = torch.export.export(
                mod, args, kwargs, dynamic_shapes=dynamic_shapes, strict=strict, **export_kwargs
            )

        _tds._guard_or = _tds._torch_guard_or  # type: ignore
        _tds.ShapeEnv._init.__kwdefaults__["specialize_zero_one"] = value  # type: ignore
        return ep

    if backed_size_oblivious == "auto":
        if verbose:
            print(f"[torch_export] backed_size_oblivious={backed_size_oblivious!r}")

        if not dynamic_shapes:
            # Unable to predict, calling the second recursively
            # to let the stacktrace keep a trace of it.
            if verbose:
                print("[torch_export] no dynamic shapes, back to default behaviour")
            return torch_export(
                mod,
                args,
                kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=strict,
                backed_size_oblivious="auto",
                verbose=verbose,
                **export_kwargs,
            )

        if isinstance(dynamic_shapes, tuple):
            if not args:
                # Unable to predict, calling the second recursively
                # to let the stacktrace keep a trace of it.
                if verbose:
                    print(
                        f"[torch_export] dynamic_shapes={dynamic_shapes}, "
                        f"args is empty, back to default behaviour"
                    )
                return torch_export(
                    mod,
                    args,
                    kwargs,
                    dynamic_shapes=dynamic_shapes,
                    strict=strict,
                    backed_size_oblivious=False,
                    verbose=verbose,
                    **export_kwargs,
                )
            assert not kwargs, (
                f"args and kwargs are specified for this call and dynamic_shapes "
                f"are {dynamic_shapes}, this is not implemented yet."
            )

        aags, kws, ds = args, kwargs, dynamic_shapes

        if (
            aags
            and isinstance(ds, tuple)
            and len(ds) == 1
            and len(ds[0]) == len(aags)
            and isinstance(ds[0], tuple)
        ):
            ds = ds[0]

        if not ds or (args and None in aags):
            backed_size_oblivious = False
        else:
            from torch._subclasses.fake_tensor import FakeTensor

            if not any(
                isinstance(f, FakeTensor) for f in flatten_object([aags, kws], drop_keys=True)
            ):
                cand = InputCandidate(aags or (), kws or {}, clone=False, cst_kwargs={})  # type: ignore
                backed_size_oblivious = cand.needs_backed_size_oblivious(ds)
        if verbose:
            print(f"[torch_export] inferred backed_size_oblivious={backed_size_oblivious!r}")

    if backed_size_oblivious:
        if verbose:
            print(
                f"[torch_export] export starts with backed_size_oblivious={backed_size_oblivious}"
            )
        with torch.fx.experimental._config.patch(backed_size_oblivious=True):  # type: ignore
            ep = torch.export.export(
                mod, args, kwargs, dynamic_shapes=dynamic_shapes, strict=strict, **export_kwargs  # type: ignore
            )
        ep._computed_backed_size_oblivious = backed_size_oblivious  # type: ignore
        return ep

    if verbose:
        print(f"[torch_export] export starts with backed_size_oblivious={backed_size_oblivious}")
    if strict:
        return torch.export.export(
            mod, args, kwargs, dynamic_shapes=dynamic_shapes, strict=strict, **export_kwargs  # type: ignore
        )
    try:
        return torch.export.export(
            mod, args, kwargs, dynamic_shapes=dynamic_shapes, strict=strict, **export_kwargs  # type: ignore
        )
    except RuntimeError as e:
        # This happens when tensor.data_ptr() is accessed.
        if "Cannot access data pointer of Tensor (e.g. FakeTensor, FunctionalTensor)" in str(e):
            return torch.export.export(
                mod, args, kwargs, dynamic_shapes=dynamic_shapes, strict=True, **export_kwargs  # type: ignore
            )
        raise

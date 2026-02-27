"""
Implements :class:`ExportOptions` to configure how to export a model into a graph.

Adapted from
`experimental_experiment/torch_interpreter/export_options.py
<https://github.com/sdpython/experimental-experiment/blob/main/experimental_experiment/torch_interpreter/export_options.py>`_.
"""

import os
import pprint
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from ..helpers.helper import string_sig, get_sig_kwargs, string_type

# Type alias for torch operator overload
# (forward reference avoids importing torch at module level)
TorchOpOverload = Any


class ExportOptions:
    """
    Gathers altogether all the options defining the way to export a model into a graph
    (not onnx).

    :param strict: strict export or not, it only applies
        if :func:`torch.export.export` is called
    :param fallback: fallback to jit
    :param decomposition_table: decomposition_table, a string such as ``'default'``
        or ``'all'``, or a custom decomposition dict
    :param dynamo: to use ``torch._dynamo.export`` instead of :func:`torch.export.export`
    :param jit: use jit to get a graph then converts it into a fx graph
    :param strategy: to overwrite all the previous parameters with just a value
    :param remove_inplace: remove inplace nodes
    :param aten_as_function: keeps aten function as local function to keep a faithful
        translation of the fx graph; can be a set of function names or a bool
    :param allow_untyped_output: allows output with no shape and/or no type
    :param save_ep: to save the exported program; if True, will save the graph as text;
        can be a tuple ``(str, int)`` to avoid saving a model bigger than the desired size
    :param validate_ep: validates the exported program with the given inputs;
        by default the tolerance is ``1e-5``; use a float to change that value
    :param backed_size_oblivious: use
        ``torch.fx.experimental._config.patch(backed_size_oblivious=True)``
        to allow dynamic dimension equal to 1
    :param prefer_deferred_runtime_asserts_over_guards:
        see :func:`torch.export.export`
    :param fake: use fake tensors as inputs
    """

    _allowed = {
        None: {},
        "none": {},
        "strict": {"strict": True},
        "strict-dec": {"strict": True, "decomposition_table": "default"},
        "strict-decall": {"strict": True, "decomposition_table": "all"},
        "nostrict": {"strict": False},
        "nostrict-dec": {"strict": False, "decomposition_table": "default"},
        "nostrict-decall": {"strict": False, "decomposition_table": "all"},
        "jit": {"jit": True},
        "jit-dec": {"jit": True, "decomposition_table": "default"},
        "jit-decall": {"jit": True, "decomposition_table": "all"},
        "fallback": {"fallback": True},
        "fallback-dec": {"fallback": True, "decomposition_table": "default"},
        "fallback-decall": {"fallback": True, "decomposition_table": "all"},
        "dec": {"decomposition_table": "default"},
        "decall": {"decomposition_table": "all"},
        "fake": {"fake": True},
    }

    def __init__(
        self,
        strict: bool = False,
        fallback: bool = False,
        jit: bool = False,
        decomposition_table: Optional[
            Union[str, Dict[TorchOpOverload, Callable[..., Any]]]
        ] = None,
        strategy: Optional[str] = None,
        dynamo: bool = False,
        aten_as_function: Optional[Union[bool, Set[Any]]] = None,
        remove_inplace: bool = True,
        allow_untyped_output: bool = False,
        save_ep: Optional[Union[Tuple[str, int], str]] = None,
        validate_ep: Union[float, bool] = False,
        backed_size_oblivious: Union[bool, str] = "auto",
        prefer_deferred_runtime_asserts_over_guards: bool = True,
        fake: bool = False,
    ):
        self.strict = strict
        self.fallback = fallback
        self.save_ep = save_ep
        self.decomposition_table = (
            None if decomposition_table in ("none", None) else decomposition_table
        )
        self.dynamo = dynamo
        self.strategy = strategy
        self.jit = jit
        self.aten_as_function = aten_as_function
        self.remove_inplace = remove_inplace
        self.allow_untyped_output = allow_untyped_output
        self.validate_ep = validate_ep
        self.backed_size_oblivious = backed_size_oblivious
        self.prefer_deferred_runtime_asserts_over_guards = (
            prefer_deferred_runtime_asserts_over_guards
        )
        self.fake = fake

        if strategy is not None:
            assert strategy in self._allowed, (
                f"Unexpected value for strategy={strategy!r}, "
                f"it should be in {sorted(k for k in self._allowed if k is not None)}"
            )
            kwargs = self._allowed[strategy]
            for k, v in kwargs.items():
                setattr(self, k, v)

        assert not self.dynamo or not self.jit, "jit and dynamo cannot be true at the same time"

    def __repr__(self) -> str:
        return string_sig(self)

    def clone(self, **kwargs) -> "ExportOptions":
        """Makes a copy and updates some of the values."""
        kw = get_sig_kwargs(self)
        kw.update(kwargs)
        return ExportOptions(**kw)

    def get_decomposition_table(
        self,
    ) -> Optional[Dict[TorchOpOverload, Callable[..., Any]]]:
        """Returns the decomposition table.

        For ``decomposition_table='all'``, returns ``None`` because ``'all'`` triggers
        :func:`torch.export.ExportedProgram.run_decompositions` with no arguments
        (handled by :func:`apply_decompositions`).
        """
        if self.decomposition_table is None:
            return None
        if self.decomposition_table == "all":
            return None
        if isinstance(self.decomposition_table, str):
            return _get_decomposition_table_by_name(self.decomposition_table)
        assert isinstance(
            self.decomposition_table, dict
        ), f"Unexpected type {type(self.decomposition_table)} for decomposition_table"
        return self.decomposition_table

    def get_fallback_options(self, kind: Optional[str] = None) -> List["ExportOptions"]:
        """Returns the fallback scenario."""
        if kind is None or kind in ("fallback", "fallback-dec", "fallback-decall"):
            other_dec = None if self.decomposition_table else "default"
            return [
                self.clone(strict=True, decomposition_table=self.decomposition_table),
                self.clone(strict=False, decomposition_table=self.decomposition_table),
                self.clone(strict=True, decomposition_table=other_dec),
                self.clone(strict=False, decomposition_table=other_dec),
                self.clone(dynamo=True, decomposition_table=self.decomposition_table),
                self.clone(dynamo=True, decomposition_table=other_dec),
                self.clone(jit=True, decomposition_table=self.decomposition_table),
            ]
        if kind == "strict":
            return [self.clone(strict=True), self.clone(strict=False)]
        if kind == "nostrict":
            return [self.clone(strict=False), self.clone(strict=True)]
        if kind == "jit":
            return [
                self.clone(strict=True),
                self.clone(jit=True, decomposition_table=self.decomposition_table),
            ]
        raise AssertionError(f"Unable to return fallback strategy with kind={kind!r}")

    def post_process_exported_program(
        self,
        exported_program: "torch.export.ExportedProgram",  # noqa: F821
        verbose: int = 0,
        print_exported_program: bool = False,
    ) -> "torch.export.ExportedProgram":  # noqa: F821
        """
        Run decompositions, remove inplace operations.
        The graph is modified inplace.
        """
        if verbose:
            print(
                f"[ExportOptions.export] post_process_exported_program "
                f"with decomposition_table={self.decomposition_table}"
            )
        if self.decomposition_table:
            if verbose:
                begin = time.perf_counter()
                print(f"[ExportOptions.export] run decomposition {self.decomposition_table!r}")
            exported_program = apply_decompositions(
                exported_program, self.decomposition_table, self.backed_size_oblivious
            )
            if verbose:
                print(
                    f"[ExportOptions.export] done after decomposition "
                    f"in {time.perf_counter() - begin}"
                )
            if print_exported_program:
                print("-- EXPORTED PROGRAM AFTER DECOMPOSITION -- ")
                print(exported_program)
                print("-- DONE -- ")

        if self.remove_inplace:
            if verbose:
                begin = time.perf_counter()
                print("[ExportOptions.export] remove inplace nodes")
            modified = _remove_inplace_nodes(exported_program.graph)
            if verbose:
                print(
                    f"[ExportOptions.export] done remove inplace in "
                    f"{time.perf_counter() - begin}, modified={modified}"
                )
            need_dec = not self.decomposition_table and _has_inplace_nodes(exported_program.graph)
            if need_dec or modified < 0:
                if verbose:
                    begin = time.perf_counter()
                    print(
                        "[ExportOptions.export] use decomposition to remove inplace nodes left "
                        f"[modified={modified}, need_dec={need_dec}]"
                    )
                exported_program = exported_program.run_decompositions({})
                if verbose:
                    print(
                        f"[ExportOptions.export] done in {time.perf_counter() - begin}, "
                        f"modified={modified}"
                    )
            if print_exported_program:
                print("-- EXPORTED PROGRAM AFTER REMOVING INPLACE -- ")
                print(exported_program)
                print("-- DONE -- ")
        return exported_program

    def use_str_not_dyn(self, dynamic_shapes: Any, default_value: Any = None) -> Any:
        """Replaces dynamic shape objects with string placeholders."""
        if not hasattr(self, "_c_use_str_not_dyn"):
            self._c_use_str_not_dyn = 0
        if isinstance(dynamic_shapes, list):
            return [self.use_str_not_dyn(a, default_value=default_value) for a in dynamic_shapes]
        if isinstance(dynamic_shapes, tuple):
            return tuple(
                self.use_str_not_dyn(a, default_value=default_value) for a in dynamic_shapes
            )
        if isinstance(dynamic_shapes, dict):
            return {
                k: self.use_str_not_dyn(v, default_value=default_value)
                for k, v in dynamic_shapes.items()
            }
        if isinstance(dynamic_shapes, set):
            return {self.use_str_not_dyn(a, default_value=default_value) for a in dynamic_shapes}
        if not isinstance(dynamic_shapes, (int, str)) and dynamic_shapes is not None:
            self._c_use_str_not_dyn += 1
            return f"udim{self._c_use_str_not_dyn}"
        return dynamic_shapes

    def _export(
        self,
        mod: Any,
        args: Optional[Tuple[Any, ...]],
        kwargs: Optional[Dict[str, Any]],
        dynamic_shapes: Any,
        input_names: Optional[List[str]],
        exc: bool,
        verbose: int,
        backed_size_oblivious: Union[bool, str] = False,
        prefer_deferred_runtime_asserts_over_guards: bool = False,
    ) -> "torch.export.ExportedProgram":  # noqa: F821
        import torch

        try:
            from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str

            dyn_shapes = use_dyn_not_str(dynamic_shapes)
        except ImportError:
            dyn_shapes = dynamic_shapes

        export_kwargs: Dict[str, Any] = {
            "strict": self.strict,
        }
        if prefer_deferred_runtime_asserts_over_guards:
            export_kwargs["prefer_deferred_runtime_asserts_over_guards"] = True
        if backed_size_oblivious is True:
            with torch.fx.experimental._config.patch(backed_size_oblivious=True):
                return torch.export.export(
                    mod,
                    args or (),
                    kwargs=kwargs or {},
                    dynamic_shapes=dyn_shapes,
                    **export_kwargs,
                )
        return torch.export.export(
            mod,
            args or (),
            kwargs=kwargs or {},
            dynamic_shapes=dyn_shapes,
            **export_kwargs,
        )

    def export(
        self,
        mod: Any,
        args: Optional[Tuple[Any, ...]],
        kwargs: Optional[Dict[str, Any]],
        tracing_mode: bool,
        dynamic_shapes: Any,
        same_signature: bool,
        input_names: Optional[List[str]] = None,
        exc: bool = True,
        verbose: int = 0,
    ) -> Union["torch.export.ExportedProgram", "torch.fx.GraphModule"]:  # noqa: F821
        """Exports the model into an exported program."""
        import torch
        from .torch_helper import torch_deepcopy

        print_exported_program = os.environ.get("PRINT_EXPORTED_PROGRAM", "0") in (1, "1")

        if self.fake:
            assert not (
                args and kwargs
            ), "Option with fake tensors is not available if both args and kwargs are specified"
            try:
                from onnx_diagnostic.export.shape_helper import make_fake_with_dynamic_dimensions
                from onnx_diagnostic.helpers import string_type as _string_type
            except ImportError:
                raise ImportError(
                    "onnx_diagnostic is required for fake=True export option. "
                    "Install it with: pip install onnx-diagnostic"
                ) from None
            dynamic_shapes_str = self.use_str_not_dyn(dynamic_shapes)
            if verbose:
                print(f"[ExportOptions.export] dynamic_shapes={dynamic_shapes}")
                print(f"[ExportOptions.export] dynamic_shapes_str={dynamic_shapes_str}")
            if kwargs:
                if verbose:
                    print(
                        f"[ExportOptions.export] true kwargs="
                        f"{_string_type(kwargs, with_shape=True)}"
                    )
                kwargs = torch_deepcopy(kwargs)
                kwargs, _ = make_fake_with_dynamic_dimensions(
                    kwargs, dynamic_shapes=dynamic_shapes_str
                )
                if verbose:
                    print(
                        f"[ExportOptions.export] fake kwargs="
                        f"{_string_type(kwargs, with_shape=True)}"
                    )
            else:
                if verbose:
                    print(
                        f"[ExportOptions.export] true args="
                        f"{_string_type(args, with_shape=True)}"
                    )
                args = torch_deepcopy(args)
                args, _ = make_fake_with_dynamic_dimensions(
                    args, dynamic_shapes=dynamic_shapes_str
                )
                if verbose:
                    print(
                        f"[ExportOptions.export] fake args="
                        f"{_string_type(args, with_shape=True)}"
                    )

        if self.fallback or self.strategy in {
            "fallback",
            "fallback-dec",
            "fallback-decomposition",
        }:
            self._last_working = None
            if verbose:
                print("[ExportOptions.export] fallback")
            tries = self.get_fallback_options(self.strategy)
            excs = []
            for ion, opt in enumerate(tries):
                if verbose:
                    print(f"[ExportOptions.export] tries {ion + 1}/{len(tries)}: {opt}")
                try:
                    res = opt.export(
                        mod,
                        args,
                        kwargs,
                        tracing_mode=tracing_mode,
                        dynamic_shapes=dynamic_shapes,
                        same_signature=same_signature,
                        input_names=input_names,
                        exc=False,
                        verbose=max(verbose - 1, 0),
                    )
                except Exception as e:
                    excs.append((opt, e))
                    if verbose:
                        se = str(e).split("\n", maxsplit=1)[0]
                        print(f"[ExportOptions.export] fails due to {se}")
                    continue

                if isinstance(res, torch.export.ExportedProgram):
                    inplace_nodes = _inplace_nodes(res.graph)
                    if inplace_nodes:
                        excs.append(
                            (
                                opt,
                                f"Probable inplace modifications, "
                                f"there are nodes with no users: {inplace_nodes}.",
                            )
                        )
                        if verbose:
                            print(f"[ExportOptions.export] fails due to {excs[-1][-1]}")

                        if not opt.decomposition_table:
                            if verbose:
                                print(
                                    f"[ExportOptions.export] current decomposition_table="
                                    f"{opt.decomposition_table}, let's try with 'default'"
                                )
                            res = apply_decompositions(res, "default", self.backed_size_oblivious)
                            inplace_nodes = _inplace_nodes(res.graph)
                            if inplace_nodes:
                                excs.append(
                                    (
                                        opt,
                                        f"Probable inplace modifications, "
                                        f"even after decomposition. "
                                        f"there are nodes with no users: {inplace_nodes}.",
                                    )
                                )
                                if verbose:
                                    print(
                                        f"[ExportOptions.export] fails again with "
                                        f"{excs[-1][-1]}"
                                    )
                                continue
                            opt.decomposition_table = "default"
                        else:
                            continue

                if verbose:
                    print(f"[ExportOptions.export] winning options {opt}")
                self._last_working = opt
                return res

            if exc:
                raise RuntimeError(
                    f"None of the following options {tries} worked, args="
                    f"{string_type(args, limit=20)}, kwargs={string_type(kwargs, limit=20)}, "
                    f"exception=\n-----\n{pprint.pformat(excs)}"
                )
            return None

        if verbose:
            print(
                f"[ExportOptions.export] {self!r} - export {type(mod).__name__!r}"
            )
            begin = time.perf_counter()

        if self.dynamo:
            if verbose:
                print("[ExportOptions.export] torch._dynamo.export")
            res = torch._dynamo.export(
                mod,
                aten_graph=True,
                tracing_mode=tracing_mode,
                dynamic_shapes=dynamic_shapes,
                same_signature=same_signature,
                decomposition_table=self.get_decomposition_table(),
                assume_static_by_default=dynamic_shapes is None,
            )(*(args or tuple()), **(kwargs or {}))
            if self.save_ep:
                save_ep = self.save_ep[0] if isinstance(self.save_ep, tuple) else self.save_ep
                with open(f"{save_ep}.old_dynamo", "w") as f:
                    f.write(str(res))
                torch.export.save(res, f"{save_ep}.old_dynamo.pt2")
            if verbose:
                print(f"[ExportOptions.export] done in {time.perf_counter() - begin}")
            return res

        if self.jit:
            if verbose:
                print("[ExportOptions.export] torch.jit.trace")
            from torch._export.converter import TS2EPConverter

            jit_model = torch.jit.trace(mod, example_inputs=args, check_trace=False, strict=False)
            res = TS2EPConverter(jit_model, args, kwargs).convert()
            if self.save_ep:
                save_ep = self.save_ep[0] if isinstance(self.save_ep, tuple) else self.save_ep
                with open(f"{save_ep}.jit", "w") as f:
                    f.write(str(res))
                torch.export.save(res, f"{save_ep}.jit.pt2")
            dec = apply_decompositions(res, self.decomposition_table, self.backed_size_oblivious)
            if self.save_ep:
                save_ep = self.save_ep[0] if isinstance(self.save_ep, tuple) else self.save_ep
                with open(f"{save_ep}.jit.decomposed", "w") as f:
                    f.write(str(dec))
                torch.export.save(dec, f"{save_ep}.jit.decomposed.pt2")
            if verbose:
                print(f"[ExportOptions.export] done in {time.perf_counter() - begin}")
            return dec

        if verbose:
            print(f"[ExportOptions.export] torch.export.export strict={self.strict}")
            print(f"[ExportOptions.export] dynamic_shapes={dynamic_shapes}")
            print(f"[ExportOptions.export] args={string_type(args, limit=20)}")
            print(f"[ExportOptions.export] kwargs={string_type(kwargs, limit=20)}")

        if self.strict:
            args0, kwargs0 = args, kwargs
            args = torch_deepcopy(args)
            kwargs = torch_deepcopy(kwargs)

        begin = time.perf_counter()
        exported_program = self._export(
            mod,
            args,
            kwargs,
            dynamic_shapes,
            input_names,
            exc,
            verbose,
            backed_size_oblivious=self.backed_size_oblivious,
            prefer_deferred_runtime_asserts_over_guards=(
                self.prefer_deferred_runtime_asserts_over_guards
            ),
        )
        self._stat_time_torch_export_export_oblivious = time.perf_counter() - begin

        if self.strict:
            args, kwargs = args0, kwargs0

        if exported_program is None:
            if verbose:
                print(f"[ExportOptions.export] done in {time.perf_counter() - begin}")
            return exported_program

        if print_exported_program:
            print("-- EXPORTED PROGRAM AFTER EXPORT -- ")
            print(exported_program)
            print("-- DONE -- ")

        if self.save_ep:
            save_ep, threshold = (
                self.save_ep if isinstance(self.save_ep, tuple) else (self.save_ep, 2**22)
            )

            def _model_size(model: Any) -> int:
                size = 0
                for param in model.parameters():
                    dtype = param.data.dtype
                    if dtype.is_floating_point or dtype.is_complex:
                        bits = torch.finfo(dtype).bits
                    else:
                        bits = torch.iinfo(dtype).bits
                    size += param.numel() * bits // 8
                return size

            with open(f"{save_ep}.ep", "w") as f:
                f.write(str(exported_program))
            with open(f"{save_ep}.ep.graph", "w") as f:
                f.write(str(exported_program.graph))
            size = _model_size(mod)
            if verbose:
                print(f"[ExportOptions.export] model size {size / 2**20} Mb")
            if size < threshold:
                begin = time.perf_counter()
                torch.save({"args": args, "kwargs": kwargs}, f"{save_ep}.input.pt")
                torch.export.save(exported_program, f"{save_ep}.ep.pt2")
                self._stat_time_torch_export_save = time.perf_counter() - begin

        if isinstance(self.validate_ep, float) or self.validate_ep:
            begin = time.perf_counter()
            self.validate_exported_program(mod, exported_program, args, kwargs, verbose=verbose)
            self._stat_time_validate_exported_program = time.perf_counter() - begin

        begin = time.perf_counter()
        exported_program = self.post_process_exported_program(
            exported_program, verbose=verbose, print_exported_program=print_exported_program
        )
        self._stat_time_post_process_exported_program = time.perf_counter() - begin
        return exported_program

    def validate_exported_program(
        self,
        model: Any,
        exported_program: "torch.export.ExportedProgram",  # noqa: F821
        args: Optional[Tuple[Any, ...]],
        kwargs: Optional[Dict[str, Any]],
        verbose: int = 0,
    ):
        """Validates the exported program by running the model."""
        from .torch_helper import torch_deepcopy

        try:
            from onnx_diagnostic.helpers import max_diff, string_diff
            from onnx_diagnostic.helpers import string_type as _string_type
        except ImportError:
            raise ImportError(
                "onnx_diagnostic is required for validate_ep=True export option. "
                "Install it with: pip install onnx-diagnostic"
            ) from None

        ar, kws = torch_deepcopy((args, kwargs))
        if verbose:
            print(
                f"[ExportOptions.validate_exported_program] run model with "
                f"args={_string_type(args, with_shape=True)} and "
                f"kwargs={_string_type(kwargs, with_shape=True)}"
            )
        expected = model(*(ar or []), **(kws or {}))
        ar, kws = torch_deepcopy((args, kwargs))
        got = exported_program.module()(*(ar or []), **(kws or {}))
        diff = max_diff(expected, got)
        if verbose:
            print(f"[ExportOptions.validate_exported_program] discrepancies: {string_diff(diff)}")
        atol = self.validate_ep if isinstance(self.validate_ep, float) else 1e-5
        assert diff["abs"] <= atol, (
            f"Discrepancies observed between the model and the exported program "
            f"(atol={atol}) diff={string_diff(diff)}"
        )


def _get_decomposition_table_by_name(
    name: str,
) -> Dict[Any, Callable[..., Any]]:
    """Returns the decomposition table by name (only ``'default'`` is supported here;
    ``'all'`` is handled as a special case in :func:`apply_decompositions`)."""
    import torch

    if name == "default":
        return torch.export.default_decompositions()
    raise ValueError(
        f"Unknown decomposition table name {name!r}. "
        f"Expected 'default'; use apply_decompositions for 'all'."
    )


def _inplace_nodes(graph: "torch.fx.Graph") -> List[Any]:  # noqa: F821
    """Returns nodes with no users (candidates for inplace operations)."""
    nodes = []
    for node in graph.nodes:
        if node.op in ("call_function", "call_method") and len(node.users) == 0:
            name = getattr(node.target, "__name__", None) or str(node.target)
            if name.endswith("_") or "_." in name:
                nodes.append(node)
    return nodes


def _has_inplace_nodes(graph: "torch.fx.Graph") -> bool:  # noqa: F821
    """Returns True if the graph has probable inplace nodes."""
    return bool(_inplace_nodes(graph))


def _remove_inplace_nodes(
    graph: "torch.fx.Graph",  # noqa: F821
    verbose: int = 0,
) -> int:
    """
    Removes inplace nodes where possible (nodes with inplace-style names and no users).

    :param graph: graph to modify
    :param verbose: verbosity
    :return: number of nodes removed, or -1 if there are inplace nodes that could not
        be removed (e.g., they still have users)
    """
    to_erase = []
    has_remaining = False
    for node in graph.nodes:
        if node.op in ("call_function", "call_method"):
            name = getattr(node.target, "__name__", None) or str(node.target)
            if name.endswith("_") or "_." in name:
                if len(node.users) == 0:
                    to_erase.append(node)
                else:
                    has_remaining = True

    modified = 0
    for node in to_erase:
        if len(node.users) == 0:
            if verbose:
                print(f"[_remove_inplace_nodes] erasing node {node}")
            graph.erase_node(node)
            modified += 1

    return -1 if has_remaining else modified


def apply_decompositions(
    exported_program: "torch.export.ExportedProgram",  # noqa: F821
    decomposition_table: Optional[Union[str, Dict[Any, Callable[..., Any]]]],
    backed_size_oblivious: Union[bool, str],
) -> "torch.export.ExportedProgram":  # noqa: F821
    """
    Applies decompositions to an exported program.

    :param exported_program: the exported program to decompose
    :param decomposition_table: a string (``'default'`` or ``'all'``) or a dict
    :param backed_size_oblivious: whether to use ``backed_size_oblivious=True``
    :return: the decomposed exported program
    """
    import torch

    use_oblivious = (
        getattr(exported_program, "_computed_backed_size_oblivious", False)
        or backed_size_oblivious is True
    )

    if decomposition_table == "all":
        exported_program = insert_contiguous_between_transpose_and_view(exported_program)
        if use_oblivious:
            with torch.fx.experimental._config.patch(backed_size_oblivious=True):
                exported_program = exported_program.run_decompositions()
        else:
            exported_program = exported_program.run_decompositions()
        return exported_program

    if isinstance(decomposition_table, str):
        decomposition_table = _get_decomposition_table_by_name(decomposition_table)

    if decomposition_table is not None:
        exported_program = insert_contiguous_between_transpose_and_view(exported_program)
        if use_oblivious:
            with torch.fx.experimental._config.patch(backed_size_oblivious=True):
                exported_program = exported_program.run_decompositions(decomposition_table)
        else:
            exported_program = exported_program.run_decompositions(decomposition_table)

    return exported_program


def insert_contiguous_between_transpose_and_view(
    exported_program: "torch.export.ExportedProgram",  # noqa: F821
) -> "torch.export.ExportedProgram":  # noqa: F821
    """
    Modifies the module inplace to insert a ``contiguous`` node between a
    ``transpose`` node followed by a ``view`` node.

    See https://github.com/pytorch/pytorch/issues/136543.
    """
    modified = False
    graph = exported_program.graph_module.graph
    for node in graph.nodes:
        if (node.op != "call_method" or node.target != "transpose") and (
            node.op != "call_function"
            or not hasattr(node.target, "name")
            or node.target.name() != "aten::transpose.int"
        ):
            continue
        insert = False
        for user in node.users:
            if (user.op == "call_method" and user.target == "view") or (
                user.op == "call_function"
                and hasattr(node.target, "name")
                and user.target.name() == "aten::view"
            ):
                insert = True
                break
        if not insert:
            continue

        modified = True
        with graph.inserting_after(node):
            new_node = graph.call_method("contiguous", args=(node,))
            node.replace_all_uses_with(new_node)
            new_node.update_arg(0, node)
            node.users = {new_node: None}

    if not modified:
        return exported_program

    graph.lint()
    return exported_program

import inspect
import os
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import torch
from torch.fx._lazy_graph_module import _make_graph_module
from ..helpers import max_diff, string_diff, string_type
from ..helpers.helper import string_sig, get_sig_kwargs
from .torch_helper import torch_deepcopy
from .input_observer import InputCandidate

# Type alias for torch operator overload
# (forward reference avoids importing torch at module level)
TorchOpOverload = Any


class TracingMode(str, Enum):
    """
    Defines the tracing mode for :class:`ExportOptions`.

    :cvar DEFAULT: no symbolic tracing, use :func:`torch.export.export`
    :cvar TRACING: use symbolic tracing via :class:`~yobx.torch.tracing.CustomTracer`
    :cvar NEW_TRACING: use dispatch-based tracing via
        :class:`~yobx.torch.new_tracing.tracer.GraphTracer`
    :cvar ONNXSCRIPT: Delegates to :func:`torch.onnx.export` with ``dynamo=True``
        (implies :attr:`ConvertingLibrary.ONNXSCRIPT`)
    """

    DEFAULT = "default"
    TRACING = "tracing"
    NEW_TRACING = "new-tracing"
    ONNXSCRIPT = "onnxscript"


class ConvertingLibrary(str, Enum):
    """
    Specifies which library performs the conversion to ONNX.

    :cvar DEFAULT: uses yobx's own conversion pipeline (the default)
    :cvar ONNXSCRIPT: Delegates to :func:`torch.onnx.export` with ``dynamo=True``
        (the onnxscript-based exporter)
    """

    DEFAULT = "default"
    ONNXSCRIPT = "onnxscript"


class ExportOptions:
    """
    Gathers altogether all the options defining the way to export a model into a graph
    (not onnx).

    :param strict: strict export or not, it only applies
        if :func:`torch.export.export` is called
    :param decomposition_table: decomposition_table, a string such as ``'default'``
        or ``'all'``, or a custom decomposition dict, see
        :func:`get_decomposition_table
        <ybox.torch.export_options.get_decomposition_table>`,
        it can ``'all'``, ``'default'`` or a decomposition list
    :param dynamo: to use ``torch._dynamo.export`` instead of :func:`torch.export.export`
    :param tracing: use symbolic tracing; accepts a :class:`TracingMode` value,
        the string ``'tracing'``, ``'new-tracing'``, or ``'default'``, or a boolean
        (``True`` is equivalent to ``TracingMode.TRACING``, ``False`` is equivalent
        to ``TracingMode.DEFAULT``); ``TracingMode.NEW_TRACING`` uses the
        dispatch-based :class:`~yobx.torch.new_tracing.tracer.GraphTracer`
    :param jit: use jit to get a graph then converts it into a fx graph
    :param strategy: to overwrite all the previous parameters with just a value
    :param remove_inplace: remove inplace nodes
    :param aten_as_function: keeps aten function as local function to keep a faithful
        translation of the fx graph, it can also be a set of function name the export
        should export as local function such as
        ``torch.ops.aten.scaled_dot_product_attention``, the default value
        :func:`get_default_aten_as_function
        <ybox.torch.interpreter.onnx_export.get_default_aten_as_function>`
        returns a default list of functions to keep as function depending on this opset,
        if no value is specified, this defaults to the whatever the function mentioned above
        returns
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
    :param tracing_module_leaves: this option is used when the module is traced
        (``tracing=TracingMode.TRACING``), it specifies
        which modules should remain a *call_module*,
        see :class:`yobx.torch.tracing.CustomTracer`.
    :param converting_library: selects which library converts the model to ONNX;
        accepts a :class:`ConvertingLibrary` value or the strings ``'default'``
        or ``'onnxscript'``; ``ConvertingLibrary.DEFAULT`` (the default) uses
        yobx's own pipeline while ``ConvertingLibrary.ONNXSCRIPT`` delegates to
        :func:`torch.onnx.export` with ``dynamo=True``
    """

    _allowed = {
        None: {},
        "none": {},
        "strict": {"strict": True},
        "strict-dec": {"strict": True, "decomposition_table": "default"},
        "strict-decall": {"strict": True, "decomposition_table": "all"},
        "tracing": {"tracing": TracingMode.TRACING},
        "new-tracing": {"tracing": TracingMode.NEW_TRACING},
        "nostrict": {"strict": False},
        "nostrict-dec": {"strict": False, "decomposition_table": "default"},
        "nostrict-decall": {"strict": False, "decomposition_table": "all"},
        "jit": {"jit": True},
        "jit-dec": {"jit": True, "decomposition_table": "default"},
        "jit-decall": {"jit": True, "decomposition_table": "all"},
        "dec": {"decomposition_table": "default"},
        "decall": {"decomposition_table": "all"},
        "fake": {"fake": True},
        "onnxscript": {"tracing": TracingMode.ONNXSCRIPT},
    }

    def __init__(
        self,
        strict: bool = False,
        tracing: Union[bool, "TracingMode"] = TracingMode.DEFAULT,
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
        tracing_module_leaves: Optional[
            Dict[type, Callable[["torch.nn.Module", str], bool]]  # noqa: F821
        ] = None,
        converting_library: Union[str, "ConvertingLibrary"] = ConvertingLibrary.DEFAULT,
    ):
        self.strict = strict
        self.tracing = tracing
        self.tracing_module_leaves = tracing_module_leaves
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
        self.converting_library = converting_library
        if aten_as_function is None:
            from .interpreter.onnx_export import get_default_aten_as_function

            aten_as_function = get_default_aten_as_function()  # type: ignore
        self.aten_as_function = aten_as_function

        if strategy is not None:
            assert strategy in self._allowed, (
                f"Unexpected value for strategy={strategy!r}, "
                f"it should be in {sorted(k for k in self._allowed if k is not None)}"
            )
            kwargs = self._allowed[strategy]
            assert isinstance(kwargs, dict)  # type checking
            for k, v in kwargs.items():
                setattr(self, k, v)

        # Normalize self.tracing to a TracingMode value (supports bool for backward compat)
        if isinstance(self.tracing, bool):
            self.tracing = TracingMode.TRACING if self.tracing else TracingMode.DEFAULT
        elif isinstance(self.tracing, str) and not isinstance(self.tracing, TracingMode):
            valid = [m.value for m in TracingMode]
            if self.tracing not in valid:
                raise ValueError(
                    f"Invalid value for tracing={self.tracing!r}, "
                    f"expected one of {valid} or a TracingMode enum value."
                )
            self.tracing = TracingMode(self.tracing)

        # TracingMode.ONNXSCRIPT implies ConvertingLibrary.ONNXSCRIPT.
        if self.tracing == TracingMode.ONNXSCRIPT:
            self.converting_library = ConvertingLibrary.ONNXSCRIPT

        # Normalize self.converting_library to a ConvertingLibrary value
        if isinstance(self.converting_library, str) and not isinstance(
            self.converting_library, ConvertingLibrary
        ):
            valid_lib = [m.value for m in ConvertingLibrary]
            if self.converting_library not in valid_lib:
                raise ValueError(
                    f"Invalid value for converting_library={self.converting_library!r}, "
                    f"expected one of {valid_lib} or a ConvertingLibrary enum value."
                )
            self.converting_library = ConvertingLibrary(self.converting_library)

        assert not self.dynamo or not self.jit, "jit and dynamo cannot be true at the same time"
        assert (
            self.tracing not in (TracingMode.TRACING, TracingMode.NEW_TRACING) or not self.dynamo
        ), f"Both tracing and dynamo are incompatible options in {self!r}"

    def __repr__(self) -> str:
        return string_sig(self)  # type: ignore[arg-type]

    def clone(self, **kwargs) -> "ExportOptions":
        """Makes a copy and updates some of the values."""
        kw = get_sig_kwargs(self)
        kw.update(kwargs)
        return ExportOptions(**kw)

    def get_decomposition_table(self) -> Optional[Dict[TorchOpOverload, Callable[..., Any]]]:
        """
        Returns the decomposition table.

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
        begin = time.perf_counter()  # to avoid many warnings from pyrefly
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
                print("[ExportOptions.export] remove inplace nodes (2)")
            modified = self.remove_inplace_nodes(
                exported_program.graph, exported_program=exported_program, verbose=verbose
            )
            if verbose:
                print(
                    f"[ExportOptions.export] done remove inplace in "
                    f"{time.perf_counter() - begin}, modified={modified}"
                )
            need_dec, need_dec_all = (
                self.need_run_decompositions(exported_program)
                if not self.decomposition_table
                else (False, False)
            )
            if need_dec or need_dec_all or modified <= -1:
                # We need to run decomposition to fully remove all inplace operations.
                if verbose:
                    begin = time.perf_counter()
                    print(
                        "[ExportOptions.export] use decomposition to remove inplace nodes left"
                        f"[modified={modified}, need_dec={need_dec}]"
                    )
                exported_program = (
                    exported_program.run_decompositions()
                    if need_dec_all
                    else exported_program.run_decompositions({})
                )
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

    def need_run_decompositions(self, exported_program) -> Tuple[bool, bool]:
        """Final check to see if we need to run decompositions."""
        from .tracing import CustomTracer

        ret = False
        for node in exported_program.graph.nodes:
            target_name = CustomTracer.get_node_target_name(node, exc=False)
            if target_name in {"aten::index_copy_"}:
                ret = True
                continue
            if target_name in {"aten:relu_", "aten::mul_.Tensor"}:
                ret = len(node.users) == 0
                continue
            if target_name in {"aten::lstm.input"}:
                return True, True
            if target_name in {
                "torch._functorch.predispatch._add_batch_dim",
                "torch._functorch.predispatch._remove_batch_dim",
            }:
                ret = len(node.users) == 0
                continue
        return ret, False

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
        from . import use_dyn_not_str

        dyn_shapes = use_dyn_not_str(dynamic_shapes)

        export_kwargs: Dict[str, Any] = {"strict": self.strict}
        if prefer_deferred_runtime_asserts_over_guards:
            export_kwargs["prefer_deferred_runtime_asserts_over_guards"] = True
        if backed_size_oblivious == "auto":
            cand = InputCandidate(args or (), kwargs or {}, clone=False, cst_kwargs={})
            backed_size_oblivious = cand.needs_backed_size_oblivious(dynamic_shapes)
        if backed_size_oblivious is True:
            with torch.fx.experimental._config.patch(backed_size_oblivious=True):  # type: ignore[attr-defined]
                ep = torch.export.export(
                    mod,
                    args or (),
                    kwargs=kwargs or {},
                    dynamic_shapes=dyn_shapes,
                    **export_kwargs,
                )
                ep._computed_backed_size_oblivious = True  # type: ignore
                return ep
        assert backed_size_oblivious is False, f"{backed_size_oblivious=}, unexpected"
        return torch.export.export(
            mod, args or (), kwargs=kwargs or {}, dynamic_shapes=dyn_shapes, **export_kwargs
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
        from .tracing import CustomTracer

        print_exported_program = os.environ.get("PRINT_EXPORTED_PROGRAM", "0") in (1, "1")
        begin = time.perf_counter()  # to avoid many warnings from pyrefly

        if self.fake:
            assert not (
                args and kwargs
            ), "Option with fake tensors is not available if both args and kwargs are specified"
            from ..helpers import string_type as _string_type
            from .fake_tensor_helper import make_fake_with_dynamic_dimensions

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

        if verbose:
            print(f"[ExportOptions.export] {self!r} - export {type(mod).__name__!r}")
            begin = time.perf_counter()

        if self.dynamo:
            if verbose:
                print("[ExportOptions.export] torch._dynamo.export")
            res = torch._dynamo.export(
                mod,
                aten_graph=True,
                tracing_mode=tracing_mode,  # type: ignore
                dynamic_shapes=dynamic_shapes,
                same_signature=same_signature,
                decomposition_table=self.get_decomposition_table(),
                assume_static_by_default=dynamic_shapes is None,
            )(*(args or tuple()), **(kwargs or {}))
            assert not self.save_ep, f"Unable to save this type {type(res)}"
            if verbose:
                print(f"[ExportOptions.export] done in {time.perf_counter() - begin}")
            return res  # type: ignore

        if self.jit:
            if verbose:
                print("[ExportOptions.export] torch.jit.trace")
            from torch._export.converter import TS2EPConverter

            jit_model = torch.jit.trace(mod, example_inputs=args, check_trace=False, strict=False)
            res = TS2EPConverter(jit_model, args, kwargs).convert()  # type: ignore
            if self.save_ep:
                save_ep = self.save_ep[0] if isinstance(self.save_ep, tuple) else self.save_ep
                with open(f"{save_ep}.jit", "w") as f:
                    f.write(str(res))
                torch.export.save(res, f"{save_ep}.jit.pt2")  # type: ignore
            dec = apply_decompositions(res, self.decomposition_table, self.backed_size_oblivious)  # type: ignore
            if self.save_ep:
                save_ep = self.save_ep[0] if isinstance(self.save_ep, tuple) else self.save_ep
                with open(f"{save_ep}.jit.decomposed", "w") as f:
                    f.write(str(dec))
                torch.export.save(dec, f"{save_ep}.jit.decomposed.pt2")
            if verbose:
                print(f"[ExportOptions.export] done in {time.perf_counter() - begin}")
            return dec

        if self.tracing == TracingMode.TRACING:
            from .tracing import CustomTracer

            concrete_args = kwargs.copy() if kwargs else {}
            trace_dynamic_shapes = (
                None
                if dynamic_shapes is None
                else (dynamic_shapes.copy() if isinstance(dynamic_shapes, dict) else {})
            )
            if args:
                sig = inspect.signature(mod.forward)
                for ip, (p, a) in enumerate(zip(sig.parameters, args)):
                    if a is not None and p not in concrete_args:
                        if isinstance(a, int):
                            # not traceable otherwise
                            concrete_args[p] = torch.tensor(a, dtype=torch.int64)
                        elif isinstance(a, float):
                            # not traceable otherwise
                            concrete_args[p] = torch.tensor(a, dtype=torch.float32)
                        else:
                            concrete_args[p] = a
                    if trace_dynamic_shapes is not None and not isinstance(dynamic_shapes, dict):
                        trace_dynamic_shapes[p] = dynamic_shapes[ip]

            if verbose:
                print(f"[ExportOptions.export] CustomTracer().trace, verbose={verbose}")
                print(f"[ExportOptions.export] {self.tracing_module_leaves=}")
                print(f"[ExportOptions.export] dynamic_shapes={dynamic_shapes}")
                print(
                    f"[ExportOptions.export] args={string_type(args, with_shape=True, limit=20)}"
                )
                print(
                    f"[ExportOptions.export] kwargs="
                    f"{string_type(kwargs, with_shape=True, limit=20)}"
                )
                print(
                    f"[ExportOptions.export] concrete_args="
                    f"{string_type(concrete_args, limit=20)}"
                )

            tracer = CustomTracer(module_leaves=self.tracing_module_leaves)
            graph = tracer.trace(
                mod,
                concrete_args=concrete_args,
                verbose=verbose,
                dynamic_shapes=trace_dynamic_shapes,
            )
            if self.remove_inplace:
                if verbose:
                    print("[ExportOptions.export] remove_inplace_nodes (1)")
                modified = self.remove_inplace_nodes(graph, verbose=verbose)
                if verbose:
                    print(f"[ExportOptions.export] done, modified={modified}")
            if self.save_ep:
                save_ep = self.save_ep[0] if isinstance(self.save_ep, tuple) else self.save_ep
                with open(f"{save_ep}.tracing", "w") as f:
                    f.write(str(graph))
            gm = _make_graph_module(tracer.root, graph, mod.__class__.__name__)

            # from torch.fx.passes.shape_prop import ShapeProp
            # ShapeProp(gp).propagate(**concrete_args)
            # gm = torch.fx.GraphModule(getattr(tracer, "traced_model", None) or mod, graph)
            return gm

        if self.tracing == TracingMode.NEW_TRACING:
            from .new_tracing import trace_model

            if verbose:
                print(f"[ExportOptions.export] trace_model (new_tracing), verbose={verbose}")
                print(f"[ExportOptions.export] {self.tracing_module_leaves=}")
                print(f"[ExportOptions.export] dynamic_shapes={dynamic_shapes}")
                print(
                    f"[ExportOptions.export] args={string_type(args, with_shape=True, limit=20)}"
                )
                print(
                    f"[ExportOptions.export] kwargs="
                    f"{string_type(kwargs, with_shape=True, limit=20)}"
                )

            graph = trace_model(
                mod,
                args if args else tuple(),
                kwargs=kwargs,
                dynamic_shapes=dynamic_shapes,
                verbose=verbose,
                module_leaves=self.tracing_module_leaves,
            )

            from .tracing import CustomTracer

            CustomTracer.remove_inplace(graph, verbose=verbose)

            if self.save_ep:
                save_ep = self.save_ep[0] if isinstance(self.save_ep, tuple) else self.save_ep
                with open(f"{save_ep}.new_tracing", "w") as f:
                    f.write(str(graph))

            gm = _make_graph_module(mod, graph, mod.__class__.__name__)
            return gm

        if verbose:
            print(f"[ExportOptions.export] torch.export.export strict={self.strict}")
            print(f"[ExportOptions.export] dynamic_shapes={dynamic_shapes}")
            print(f"[ExportOptions.export] args={string_type(args, limit=20)}")
            print(f"[ExportOptions.export] kwargs={string_type(kwargs, limit=20)}")

        if self.strict:
            # torch.export.export may turn Tensor into FakeTensor.
            # We need to make a copy to avoid getting FakeTensor instead
            args0, kwargs0 = args, kwargs
            args = torch_deepcopy(args)
            kwargs = torch_deepcopy(kwargs)
        t0 = time.perf_counter()
        if verbose:
            print(
                f"[ExportOptions.export] export start with strict={self.strict} "
                f"backed_size_oblivious={self.backed_size_oblivious}"
            )
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

        if verbose:
            print(f"[ExportOptions.export] export done in {time.perf_counter() - t0}")

        if self.strict:
            # torch.export.export may turn Tensor into FakeTensor.
            # We need to make a copy to avoid getting FakeTensor instead
            args, kwargs = args0, kwargs0  # pyrefly: ignore[unbound-name]

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
        if verbose:
            print(
                f"[ExportOptions.export] done with no decomposition "
                f"in {time.perf_counter() - begin}"
            )
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
        ar, kws = torch_deepcopy((args, kwargs))
        if verbose:
            print(
                f"[ExportOptions.validate_exported_program] run model with "
                f"args={string_type(args, with_shape=True)} and "
                f"kwargs={string_type(kwargs, with_shape=True)}"
            )
        expected = model(*(ar or []), **(kws or {}))
        ar, kws = torch_deepcopy((args, kwargs))
        if verbose:
            print(
                f"[ExportOptions.validate_exported_program] run exported_program with "
                f"args={string_type(args, with_shape=True)} and "
                f"kwargs={string_type(kwargs, with_shape=True)}"
            )
        got = exported_program.module()(*(ar or []), **(kws or {}))
        diff = max_diff(expected, got)
        if verbose:
            print(f"[ExportOptions.validate_exported_program] discrepancies: {string_diff(diff)}")
        atol = self.validate_ep if isinstance(self.validate_ep, float) else 1e-5
        assert isinstance(diff["abs"], float) and diff["abs"] <= atol, (
            f"Discrepancies observed between the model and the exported program "
            f"(atol={atol}) diff={string_diff(diff)}"
        )

    def export_as_aten_function(self, aten_name: Any) -> bool:
        if not self.aten_as_function:
            return False
        if isinstance(self.aten_as_function, bool):
            return self.aten_as_function
        if isinstance(aten_name, str):
            return aten_name in self.aten_as_function
        return aten_name in self.aten_as_function or str(aten_name) in self.aten_as_function

    def remove_inplace_nodes(
        self,
        graph: "torch.fx.Graph",  # noqa: F821
        exported_program: Optional["torch.export.ExportedProgram"] = None,  # noqa: F821
        verbose: int = 0,
    ) -> int:
        """
        Post-processing to remove inplace nodes.

        :param graph: graph to modify
        :param exported_program: if available, it is used in the error message
            to make it easier to trace the code source
        :param verbose: verbosity
        :return: number of inplace nodes removed or -1 if there are any remaining inplace nodes
        """
        from .tracing import CustomTracer

        autocast_fixed = CustomTracer.fix_autocast_subgraph_dtypes(graph, verbose=verbose)
        if autocast_fixed and verbose:
            print(
                f"[ExportOptions.export] autocast: "
                f"{autocast_fixed} body subgraph(s) had dtypes fixed"
            )
        removed = CustomTracer.remove_unnecessary_slices(graph)
        if removed:
            if verbose:
                print(f"[ExportOptions.export] slices: {removed} slices nodes were removed")
            graph.lint()
        batch_dim_removed = CustomTracer.remove_batch_dim_nodes(graph, verbose=verbose)
        if batch_dim_removed:
            if verbose:
                print(
                    f"[ExportOptions.export] batch_dim: "
                    f"{batch_dim_removed} batch dim nodes were removed"
                )
            graph.lint()
        modified = CustomTracer.remove_inplace(
            graph, exported_program=exported_program, verbose=verbose, exc=False
        )
        if modified < 0:
            return modified
        if modified:
            if verbose:
                print(f"[ExportOptions.export] inplaces: {modified} inplaced nodes were removed")
            graph.lint()
        return modified

    def use_str_not_dyn(self, dynamic_shapes: Any, default_value=None) -> Any:
        if not hasattr(self, "_c_use_str_not_dyn"):
            self._c_use_str_not_dyn = 0
        if isinstance(dynamic_shapes, (tuple, list, set)):
            return dynamic_shapes.__class__(
                self.use_str_not_dyn(a, default_value=default_value) for a in dynamic_shapes
            )
        if isinstance(dynamic_shapes, dict):
            return {
                k: self.use_str_not_dyn(v, default_value=default_value)
                for k, v in dynamic_shapes.items()
            }
        if not isinstance(dynamic_shapes, (int, str)) and dynamic_shapes is not None:
            self._c_use_str_not_dyn += 1
            return f"udim{self._c_use_str_not_dyn}"
        return dynamic_shapes


def _get_decomposition_table_by_name(name: str) -> Dict[Any, Callable[..., Any]]:
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
            with torch.fx.experimental._config.patch(backed_size_oblivious=True):  # type: ignore[attr-defined]
                exported_program = exported_program.run_decompositions()
        else:
            exported_program = exported_program.run_decompositions()
        return exported_program

    if isinstance(decomposition_table, str):
        decomposition_table = _get_decomposition_table_by_name(decomposition_table)

    if decomposition_table is not None:
        exported_program = insert_contiguous_between_transpose_and_view(exported_program)
        if use_oblivious:
            with torch.fx.experimental._config.patch(backed_size_oblivious=True):  # type: ignore[attr-defined]
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
    The modification takes place inplace.
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
                and user.target.name() == "aten::view"  # pyrefly: ignore[missing-attribute]
            ):
                insert = True
                break
        if not insert:
            continue

        modified = True
        with graph.inserting_after(node):
            new_node = graph.call_method("contiguous", args=(node,))
            node.replace_all_uses_with(new_node)
            # new_node is replaced as well so we manually revert the replacement
            new_node.update_arg(0, node)
            node.users = {new_node: None}

    if not modified:
        # no rewrite was done.
        return exported_program

    graph.lint()
    return exported_program

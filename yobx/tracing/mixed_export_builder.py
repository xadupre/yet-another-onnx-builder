import copy
import math
from types import ModuleType
from typing import Any, Callable
import torch
from .fake_tensor import make_fake_with_dynamic_dimensions


class MixedExportTracer(torch.fx.Tracer):
    """
    Defines a custom tracer to trace the execution of a model
    and converts it into a fx graph.

    ::
        from onnx_pipe.tracing.mixed_export_builder import MixedExportTracer

        graph = MixedExportTracer().trace(model)

    :param autowrap_modules: defaults to `(math, )`,
        Python modules whose functions should be wrapped automatically
        without needing to use fx.wrap().
    :param autowrap_functions: defaults to `()`,
        Python functions that should be wrapped automatically without
        needing to use fx.wrap().
    :param param_shapes_constant: When this flag is set, calls to shape,
        size and a few other shape like attributes of a module's parameter
        will be evaluated directly, rather than returning a new Proxy value
        for an attribute access.
    :param module_leaves: modules to be considered as leaves,
        mapped to a callable ``f(module, module_qualified_name) -> bool``
        that decides whether a specific module instance is a leaf;
        the tracer does not trace into leaf modules and emits
        ``call_module`` nodes for them instead
    """

    def __init__(
        self,
        autowrap_modules: tuple[ModuleType] = (math,),  # noqa: F821
        autowrap_functions: tuple[Callable, ...] = (),
        param_shapes_constant: bool = False,
        module_leaves: dict[type, Callable[[torch.nn.Module], bool]] | None = None,
    ):
        super().__init__(
            autowrap_modules=autowrap_modules,
            autowrap_functions=autowrap_functions,
            param_shapes_constant=param_shapes_constant,
        )
        self._callables = {}
        self.module_leaves = module_leaves

    @torch.fx._compatibility.compatibility(is_backward_compatible=True)
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.

        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.

        Args:

            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        """
        is_leave = super().is_leaf_module(m, module_qualified_name)
        if is_leave:
            return is_leave
        if self.module_leaves and type(m) in self.module_leaves:
            f = self.module_leaves[type(m)]
            return f(m, module_qualified_name=module_qualified_name)
        return False

    @classmethod
    def make_wrapped_model(cls, root, concrete_args):
        raise NotImplementedError(
            "The model needs to be wrapped to get a flat list of tensors before tracing."
        )

    def trace(
        self,
        root: torch.nn.Module | Callable[..., Any],
        concrete_args: dict[str, Any] | None = None,
        dynamic_shapes: Any | None = None,
        verbose: int = 0,
    ) -> torch.fx.Graph:
        """
        Trace ``root`` and return the corresponding FX ``Graph`` representation. ``root``
        can either be an ``nn.Module`` instance or a Python callable.

        Note that after this call, ``self.root`` may be different from the ``root`` passed
        in here. For example, when a free function is passed to ``trace()``, we will
        create an ``nn.Module`` instance to use as the root and add embedded constants to.

        :param root: Either a ``Module`` or a function to be
            traced through. Backwards-compatibility for this parameter is
            guaranteed.
        :param concrete_args: Concrete arguments that should
            not be treated as Proxies. This parameter is experimental and
            its backwards-compatibility is *NOT* guaranteed.
        :param dynamic_shapes: dynamic shapes
        :param verbose: verbosity
        :return: A ``Graph`` representing the semantics of the passed-in ``root``

        If the model had to wrapped before being traced, attribute ``traced_model``
        is added to the tracer.
        """
        assert concrete_args is None or isinstance(concrete_args, dict), (
            f"Unexpected type for concrete_args {type(concrete_args)=}"
        )
        traced_model = None
        if concrete_args:
            if dynamic_shapes is None:
                dynamic_shapes = (
                    ({},) * len(concrete_args)
                    if isinstance(concrete_args, tuple)
                    else {k: {} for k in concrete_args}
                )

            flat_args = (
                concrete_args.values()
                if isinstance(concrete_args, dict)
                else concrete_args
            )

            if any(type(a) in torch.utils._pytree.SUPPORTED_NODES for a in flat_args):
                # tracing does not know the input type so we need to flatten everything.
                new_model, new_names = self.make_wrapped_model(root, concrete_args)
                traced_concrete_args, _ = make_fake_with_dynamic_dimensions(
                    copy.deepcopy(concrete_args), dynamic_shapes
                )
                self._traced_concrete_args, _ = torch.utils._pytree.tree_flatten(
                    traced_concrete_args
                )
                traced_model = new_model
            else:
                new_names = None
                self._traced_concrete_args, _ = make_fake_with_dynamic_dimensions(
                    concrete_args, dynamic_shapes
                )
                new_model = root
            graph = super().trace(new_model)

        else:
            self._traced_concrete_args = None
            new_names = None

            graph = super().trace(root)  # , concrete_args)

        if concrete_args:
            if new_names:
                flat_concrete_args, _spec = torch.utils._pytree.tree_flatten(
                    concrete_args
                )
                flat_traced_concrete_args, _spec = torch.utils._pytree.tree_flatten(
                    self._traced_concrete_args
                )
            mapped = set()
            for node in graph.nodes:
                if node.op == "placeholder":
                    if not new_names and node.name in concrete_args:
                        ti = concrete_args[node.name]
                        tif = self._traced_concrete_args[node.name]
                        node.meta["example_value"] = ti
                        node.meta["val"] = tif
                        mapped.add(node.name)
                    elif new_names and node.name in new_names:
                        ii = new_names.index(node.name)
                        ti = flat_concrete_args[ii]
                        tif = flat_traced_concrete_args[ii]
                        node.meta["example_value"] = ti
                        node.meta["val"] = tif
                        mapped.add(node.name)
            assert new_names or set(mapped) == set(concrete_args), (
                f"Missing mapped inputs, set(concrete_args)={set(concrete_args)}, "
                f"mapped={mapped}\n{graph}\nroot={root}"
            )
            assert not new_names or len(new_names) == len(flat_concrete_args), (
                f"Missing mapped inputs, new_names={new_names}, "
                f"mapped={mapped}\n{graph}\nroot={root}"
            )

        graph.lint()
        self.traced_model = traced_model
        return graph

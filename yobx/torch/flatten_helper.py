import contextlib
import pprint
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import optree
import torch
from ..helpers import string_type

PATCH_OF_PATCHES: Set[Any] = set()


def _lower_name_with_(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def make_flattening_function_for_dataclass(
    cls: type, supported_classes: Set[type]
) -> Tuple[Callable, Callable, Callable]:
    """
    Automatically creates flattening functions for a class decorated with
    ``dataclasses.dataclass``.

    :param cls: the dataclass type
    :param supported_classes: set to register the class into
    :return: tuple of (flatten, flatten_with_keys, unflatten) callables
    """

    def flatten_cls(obj: cls) -> Tuple[List[Any], torch.utils._pytree.Context]:  # type: ignore[valid-type]
        """Serializes a ``%s`` with python objects."""
        return list(obj.values()), list(obj.keys())

    def flatten_with_keys_cls(
        obj: cls,  # type: ignore[valid-type]
    ) -> Tuple[List[Tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
        """Serializes a ``%s`` with python objects with keys."""
        values, context = list(obj.values()), list(obj.keys())
        return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context

    def unflatten_cls(
        values: List[Any], context: torch.utils._pytree.Context, output_type=None
    ) -> cls:  # type: ignore[valid-type]
        """Restores an instance of ``%s`` from python objects."""
        return cls(**dict(zip(context, values)))

    name = _lower_name_with_(cls.__name__)
    flatten_cls.__name__ = f"flatten_{name}"
    flatten_with_keys_cls.__name__ = f"flatten_with_keys_{name}"
    unflatten_cls.__name__ = f"unflatten_{name}"
    flatten_cls.__doc__ = flatten_cls.__doc__ % cls.__name__
    flatten_with_keys_cls.__doc__ = flatten_with_keys_cls.__doc__ % cls.__name__
    unflatten_cls.__doc__ = unflatten_cls.__doc__ % cls.__name__
    supported_classes.add(cls)
    return flatten_cls, flatten_with_keys_cls, unflatten_cls


def register_class_flattening(
    cls,
    f_flatten: Callable,
    f_unflatten: Callable,
    f_flatten_with_keys: Callable,
    f_check: Optional[Callable] = None,
    verbose: int = 0,
) -> bool:
    """
    Registers a class.
    It can be undone with
    :func:`yobx.torch.flatten_helper.unregister_class_flattening`.

    :param cls: class to register
    :param f_flatten: see ``torch.utils._pytree.register_pytree_node``
    :param f_unflatten: see ``torch.utils._pytree.register_pytree_node``
    :param f_flatten_with_keys: see ``torch.utils._pytree.register_pytree_node``
    :param f_check: called to check the registration was successful
    :param verbose: verbosity
    :return: registered or not
    """
    if cls is not None and cls in torch.utils._pytree.SUPPORTED_NODES:
        if verbose and cls is not None:
            print(f"[register_class_flattening] already registered {cls.__name__}")
        return False

    if verbose:
        print(f"[register_class_flattening] ---------- register {cls.__name__}")
    torch.utils._pytree.register_pytree_node(
        cls,
        f_flatten,
        f_unflatten,
        serialized_type_name=f"{cls.__module__}.{cls.__name__}",
        flatten_with_keys_fn=f_flatten_with_keys,
    )

    # check
    if f_check:
        inst = f_check()
        values, spec = torch.utils._pytree.tree_flatten(inst)
        restored = torch.utils._pytree.tree_unflatten(values, spec)
        assert string_type(inst, with_shape=True) == string_type(restored, with_shape=True), (
            f"Issue with registration of class {cls} "
            f"inst={string_type(inst, with_shape=True)}, "
            f"restored={string_type(restored, with_shape=True)}"
        )
    return True


def flattening_functions(
    patch_transformers: bool = False, patch_diffusers: bool = False, verbose: int = 0
) -> Dict[type, Callable[[int], bool]]:
    """Returns the list of flattening functions."""

    supported_classes: Set[type] = set()
    classes: Dict[type, Callable[[int], bool]] = {}
    all_functions: Dict[type, Optional[str]] = {}

    if patch_transformers:
        from .transformers.flatten_class import (
            __dict__ as dtr,
            SUPPORTED_DATACLASSES,
            TRANSFORMERS_CLASSES,
        )

        all_functions.update(dtr)
        supported_classes |= SUPPORTED_DATACLASSES
        classes.update(TRANSFORMERS_CLASSES)

    for cls in supported_classes:
        lname = _lower_name_with_(cls.__name__)
        assert (
            f"flatten_{lname}" in all_functions
        ), f"Unable to find function 'flatten_{lname}' in {list(all_functions)}"
        classes[cls] = (
            lambda verbose=verbose, _ln=lname, cls=cls, _al=all_functions: register_class_flattening(  # noqa: E501
                cls,
                _al[f"flatten_{_ln}"],
                _al[f"unflatten_{_ln}"],
                _al[f"flatten_with_keys_{_ln}"],
                verbose=verbose,
            )
        )
    return classes


def replacement_before_exporting(args: Any) -> Any:
    """Does replacements on the given inputs if needed."""
    if args is None:
        return None
    if isinstance(args, (int, float)):
        return args
    if type(args) not in {dict, tuple, list}:
        # BaseModelOutput is a dict
        return args
    if isinstance(args, dict):
        return {k: replacement_before_exporting(v) for k, v in args.items()}
    if isinstance(args, tuple):
        return tuple(replacement_before_exporting(v) for v in args)
    if isinstance(args, list):
        return [replacement_before_exporting(v) for v in args]
    return args


def register_cache_flattening(
    patch_transformers: bool = False, verbose: int = 0
) -> Dict[str, bool]:
    """
    Registers many classes with
    :func:`yobx.torch.flatten_helper.register_class_flattening`.
    Returns information needed to undo the registration.

    :param patch_transformers: add flattening function for
        :epkg:`transformers` package
    :param patch_diffusers: add flattening function for
        :epkg:`diffusers` package
    :param verbosity: verbosity level
    :return: information to unpatch
    """
    import packaging.version as pv

    wrong: Dict[type, Optional[str]] = {}
    if patch_transformers:
        import transformers
        from .transformers.flatten_class import WRONG_REGISTRATIONS

        wrong |= WRONG_REGISTRATIONS
        transformers_version = pv.Version(transformers.__version__)

    registration_functions = flattening_functions(
        patch_transformers=patch_transformers, verbose=verbose
    )

    # DynamicCache flattening is different in transformers and does not
    # play way with torch.export.export.
    # see test test_export_dynamic_cache_cat with NOBYPASS=1
    # :: NOBYBASS=1 python _unittests/ut_torch_export_patches/test_dynamic_class.py -k e_c
    # This is caused by this line:
    # torch.fx._pytree.register_pytree_flatten_spec(
    #           DynamicCache, _flatten_dynamic_cache_for_fx)
    # so we remove it anyway
    # BaseModelOutput flattening is incomplete.
    # It does not include dynamic shapes mapping.
    for cls, version in wrong.items():
        if (
            cls in torch.utils._pytree.SUPPORTED_NODES
            and cls not in PATCH_OF_PATCHES
            # and pv.Version(torch.__version__) < pv.Version("2.7")
            and (version is None or transformers_version >= pv.Version(version))
        ):
            assert cls in registration_functions, (
                f"{cls} has no registration functions mapped to it, "
                f"available options are {list(registration_functions)}"
            )
            if verbose:
                print(f"[_fix_registration] {cls.__name__} is unregistered and registered first")
            unregister_class_flattening(cls)
            registration_functions[cls]()  # type: ignore[arg-type, call-arg]
            if verbose:
                print(f"[_fix_registration] {cls.__name__} done.")
            # To avoid doing it multiple times.
            PATCH_OF_PATCHES.add(cls)

    # classes with no registration at all.
    done = {}
    for k, v in registration_functions.items():
        done[k] = v()  # type: ignore[arg-type, call-arg]
    return done


def unregister_class_flattening(cls: type, verbose: int = 0):
    """Undo the registration for a class."""
    # torch.utils._pytree._deregister_pytree_flatten_spec(cls)
    if cls in torch.fx._pytree.SUPPORTED_NODES:
        del torch.fx._pytree.SUPPORTED_NODES[cls]
    if cls in torch.fx._pytree.SUPPORTED_NODES_EXACT_MATCH:
        del torch.fx._pytree.SUPPORTED_NODES_EXACT_MATCH[cls]
    if hasattr(torch.utils._pytree, "_deregister_pytree_node"):
        # torch >= 2.7
        torch.utils._pytree._deregister_pytree_node(cls)
    else:
        if cls in torch.utils._pytree.SUPPORTED_NODES:
            del torch.utils._pytree.SUPPORTED_NODES[cls]
    optree.unregister_pytree_node(cls, namespace="torch")
    if cls in torch.utils._pytree.SUPPORTED_NODES:
        import packaging.version as pv

        if pv.Version(torch.__version__) < pv.Version("2.7.0"):
            del torch.utils._pytree.SUPPORTED_NODES[cls]  # pragma: no cover
    assert cls not in torch.utils._pytree.SUPPORTED_NODES, (
        f"{cls} was not successful unregistered "
        f"from torch.utils._pytree.SUPPORTED_NODES="
        f"{pprint.pformat(list(torch.utils._pytree.SUPPORTED_NODES))}"
    )
    if verbose:
        print(f"[unregister_cache_flattening] unregistered {cls.__name__}")


def unregister_cache_flattening(undo: Dict[str, bool], verbose: int = 0):
    """
    Undo the registration made by
    :func:`yobx.torch.flatten_helper.register_cache_flattening`.
    """
    cls_ensemble = set(undo)
    for cls in cls_ensemble:
        if undo.get(cls.__name__, False):
            unregister_class_flattening(cls, verbose)


@contextlib.contextmanager
def register_flattening_functions(patch_transformers: bool = False, verbose: int = 0) -> Callable:
    """The necessary modifications to run the fx Graph."""
    fct_callable = replacement_before_exporting if patch_transformers else (lambda x: x)
    done = register_cache_flattening(patch_transformers=patch_transformers, verbose=verbose)
    try:
        yield fct_callable
    finally:
        unregister_cache_flattening(done, verbose=verbose)

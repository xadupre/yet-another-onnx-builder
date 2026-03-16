"""
Registry for direct ONNX converters of Hugging Face *transformers* modules.

This module provides a lightweight, class-keyed registry analogous to
:mod:`yobx.sklearn.register`.  Each entry maps a *transformers* module
class to a callable that appends the corresponding ONNX nodes to a
:class:`~yobx.xbuilder.GraphBuilder`.

The registry is populated lazily via
:func:`~yobx.torch.in_transformers.register_transformer_converters`, which
is idempotent — calling it multiple times has no side-effect.

Registering a custom converter
------------------------------

Use the :func:`register_transformer_converter` decorator.  Pass a single
class or a tuple of classes as the first argument::

    from yobx.torch.in_transformers.register import register_transformer_converter
    from yobx.typing import GraphBuilderExtendedProtocol

    @register_transformer_converter(MyModule)
    def my_module_to_onnx(
        g: GraphBuilderExtendedProtocol,
        module: MyModule,
        hidden_states: str,
        name: str = "my_module",
    ) -> str:
        ...

The decorator raises :class:`TypeError` if a converter is already
registered for the same class, preventing accidental double-registration.

Looking up a converter
----------------------

:func:`get_transformer_converter` takes a class and returns the registered
callable, raising :class:`ValueError` if none is found::

    from yobx.torch.in_transformers.register import get_transformer_converter
    from transformers.models.llama.modeling_llama import LlamaAttention

    converter = get_transformer_converter(LlamaAttention)
    # converter is llama_attention_to_onnx

Currently registered converters
--------------------------------

.. list-table::
    :header-rows: 1

    * - Module class
      - Converter function
    * - :class:`transformers.models.llama.modeling_llama.LlamaAttention`
      - :func:`~yobx.torch.in_transformers.classes.llama_attention.llama_attention_to_onnx`
"""

from typing import Dict, Callable, Tuple, Union

TRANSFORMER_CONVERTERS: Dict[type, Callable] = {}
"""Module-level registry mapping *transformers* module classes to their ONNX converter functions."""


def register_transformer_converter(cls: Union[type, Tuple[type, ...]]):
    """Decorator that registers a converter function for one or more *transformers* classes.

    Args:
        cls: A single class or a tuple of classes for which the decorated function
            should be registered as the ONNX converter.

    Returns:
        A decorator that registers the wrapped function and returns it unchanged,
        so the function can still be called directly.

    Raises:
        TypeError: If a converter is already registered for any of the supplied
            classes.

    Example::

        from yobx.torch.in_transformers.register import register_transformer_converter
        from yobx.typing import GraphBuilderExtendedProtocol

        @register_transformer_converter(MyModule)
        def my_module_to_onnx(
            g: GraphBuilderExtendedProtocol,
            module: MyModule,
            hidden_states: str,
            name: str = "my_module",
        ) -> str:
            ...
    """

    def decorator(fct: Callable):
        if isinstance(cls, tuple):
            for c in cls:
                if c in TRANSFORMER_CONVERTERS:
                    raise TypeError(f"A converter is already registered for {c}.")
                TRANSFORMER_CONVERTERS[c] = fct
        else:
            if cls in TRANSFORMER_CONVERTERS:
                raise TypeError(f"A converter is already registered for {cls}.")
            TRANSFORMER_CONVERTERS[cls] = fct
        return fct

    return decorator


def get_transformer_converter(cls: type) -> Callable:
    """Returns the ONNX converter registered for *cls*.

    Args:
        cls: The *transformers* module class to look up.

    Returns:
        The converter callable registered for *cls*.

    Raises:
        ValueError: If no converter has been registered for *cls*.

    Example::

        from yobx.torch.in_transformers import register_transformer_converters
        from yobx.torch.in_transformers.register import get_transformer_converter
        from transformers.models.llama.modeling_llama import LlamaAttention

        register_transformer_converters()
        converter = get_transformer_converter(LlamaAttention)
        # converter is llama_attention_to_onnx
    """
    if cls in TRANSFORMER_CONVERTERS:
        return TRANSFORMER_CONVERTERS[cls]
    raise ValueError(f"Unable to find a converter for type {cls}.")


def get_transformer_converters() -> Dict[type, Callable]:
    """Returns a snapshot of all registered converters.

    Returns:
        A new dictionary mapping each registered *transformers* module class
        to its ONNX converter function.  Mutating the returned dict does not
        affect the registry.

    Example::

        from yobx.torch.in_transformers import register_transformer_converters
        from yobx.torch.in_transformers.register import get_transformer_converters

        register_transformer_converters()
        for cls, fn in get_transformer_converters().items():
            print(cls.__name__, "->", fn.__name__)
    """
    return dict(TRANSFORMER_CONVERTERS)

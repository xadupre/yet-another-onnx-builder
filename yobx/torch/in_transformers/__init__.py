from .register import (
    register_transformer_converter,
    get_transformer_converter,
    get_transformer_converters,
)


def register_transformer_converters():
    """Registers all built-in *transformers*-to-ONNX converters in this package.

    This function is **idempotent** — calling it multiple times is safe and has
    no additional side-effect after the first call.

    After this call, :func:`~yobx.torch.in_transformers.register.get_transformer_converter`
    can look up converters by class, for example::

        from yobx.torch.in_transformers import (
            register_transformer_converters,
            get_transformer_converter,
        )
        from transformers.models.llama.modeling_llama import LlamaAttention

        register_transformer_converters()
        converter = get_transformer_converter(LlamaAttention)
        # converter is llama_attention_to_onnx
    """
    from . import classes  # noqa: F401 - triggers auto-registration as side effect

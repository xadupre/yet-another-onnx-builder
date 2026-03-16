from .register import get_transformer_converter, get_transformer_converters


def register_transformer_converters():
    """
    Registers all built-in *transformers*-to-ONNX converters in this package.

    This function is **idempotent** — calling it multiple times is safe and has
    no additional side-effect after the first call.
    """
    from .classes import register

    register()

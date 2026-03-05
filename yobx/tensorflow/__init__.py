from .convert import to_onnx


def register_tensorflow_converters():
    """Registers all converters implemented in this package."""
    from .register import TENSORFLOW_CONVERTERS

    if TENSORFLOW_CONVERTERS:
        # already done
        return
    from .layers import register as register_layers
    from .sequential import register as register_sequential

    register_layers()
    register_sequential()

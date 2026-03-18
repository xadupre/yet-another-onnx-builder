from .convert import to_onnx
from .tensorflow_helper import jax_to_concrete_function

_CONVERTERS_REGISTERED = False


def register_tensorflow_converters():
    """Registers all built-in TF op converters implemented in this package."""
    global _CONVERTERS_REGISTERED
    if _CONVERTERS_REGISTERED:
        return
    _CONVERTERS_REGISTERED = True
    from .ops import register as register_ops

    register_ops()

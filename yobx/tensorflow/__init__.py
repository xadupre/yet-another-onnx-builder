from .convert import to_onnx
from .tensorflow_helper import jax_to_concrete_function


def register_tensorflow_converters():
    """Registers all built-in TF op converters implemented in this package."""
    from .register import TF_OP_CONVERTERS

    if TF_OP_CONVERTERS:
        # already done
        return
    from .ops import register as register_ops

    register_ops()

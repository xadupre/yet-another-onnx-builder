from .convert import to_onnx


def register_tensorflow_converters():
    """Registers all built-in TF op converters implemented in this package."""
    from .register import TF_OP_CONVERTERS

    if TF_OP_CONVERTERS:
        # already done
        return
    from .ops import register as register_ops

    register_ops()

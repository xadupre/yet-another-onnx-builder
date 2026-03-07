from typing import Callable, Dict, Optional, Tuple, Union

# Maps TF op-type string (e.g. "MatMul", "Relu") to a converter function.
TF_OP_CONVERTERS: Dict[str, Callable] = {}


def register_tf_op_converter(op_type: Union[str, Tuple[str, ...]]):
    """Decorator that registers a converter for one or more TF op-type strings."""

    def decorator(fct: Callable):
        global TF_OP_CONVERTERS
        types = (op_type,) if isinstance(op_type, str) else op_type
        for t in types:
            if t in TF_OP_CONVERTERS:
                raise TypeError(f"A converter is already registered for op type {t!r}.")
            TF_OP_CONVERTERS[t] = fct
        return fct

    return decorator


def get_tf_op_converter(op_type: str) -> Optional[Callable]:
    """Returns the converter for a TF op type, or ``None`` if not found."""
    global TF_OP_CONVERTERS
    return TF_OP_CONVERTERS.get(op_type)


def get_tf_op_converters() -> Dict[str, Callable]:
    """Returns all registered TF op converters."""
    global TF_OP_CONVERTERS
    return dict(TF_OP_CONVERTERS)

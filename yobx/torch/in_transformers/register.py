from typing import Dict, Callable, Tuple, Union

TRANSFORMER_CONVERTERS: Dict[type, Callable] = {}


def register_transformer_converter(cls: Union[type, Tuple[type, ...]]):
    def decorator(fct: Callable):
        """Registers a function to convert a transformer module to ONNX."""
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


def get_transformer_converter(cls: type):
    """Returns the converter for a specific type."""
    if cls in TRANSFORMER_CONVERTERS:
        return TRANSFORMER_CONVERTERS[cls]
    raise ValueError(f"Unable to find a converter for type {cls}.")


def get_transformer_converters():
    """Returns all registered converters as a mapping from type to converter function."""
    return dict(TRANSFORMER_CONVERTERS)

from typing import Dict, Callable, Tuple, Union

SKLEARN_CONVERTERS: Dict[type, Callable] = {}


def register_sklearn_converter(cls: Union[type, Tuple[type]]):
    def decorator(fct: Callable):
        """Registers a function to converts a model."""
        global SKLEARN_CONVERTERS
        if isinstance(cls, tuple):
            for c in cls:
                if c in SKLEARN_CONVERTERS:
                    raise TypeError(f"A converter is already registered for {c}.")
                SKLEARN_CONVERTERS[c] = fct
        else:
            SKLEARN_CONVERTERS[cls] = fct
        return fct

    return decorator


def get_sklearn_converter(cls: type):
    """Returns the converter for a specific type."""
    global SKLEARN_CONVERTERS
    if cls in SKLEARN_CONVERTERS:
        return SKLEARN_CONVERTERS[cls]
    raise ValueError(f"Unable to find a converter for type {cls}.")

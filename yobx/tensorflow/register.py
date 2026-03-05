from typing import Dict, Callable, Tuple, Union

TENSORFLOW_CONVERTERS: Dict[type, Callable] = {}


def register_tensorflow_converter(cls: Union[type, Tuple[type, ...]]):
    def decorator(fct: Callable):
        """Registers a function to convert a TensorFlow/Keras layer or model."""
        global TENSORFLOW_CONVERTERS
        if isinstance(cls, tuple):
            for c in cls:
                if c in TENSORFLOW_CONVERTERS:
                    raise TypeError(f"A converter is already registered for {c}.")
                TENSORFLOW_CONVERTERS[c] = fct
        else:
            if cls in TENSORFLOW_CONVERTERS:
                raise TypeError(f"A converter is already registered for {cls}.")
            TENSORFLOW_CONVERTERS[cls] = fct
        return fct

    return decorator


def get_tensorflow_converter(cls: type):
    """Returns the converter for a specific type."""
    global TENSORFLOW_CONVERTERS
    if cls in TENSORFLOW_CONVERTERS:
        return TENSORFLOW_CONVERTERS[cls]
    raise ValueError(f"Unable to find a converter for type {cls}.")


def get_tensorflow_converters():
    """Returns all registered converters as a mapping from type to converter function."""
    global TENSORFLOW_CONVERTERS
    return dict(TENSORFLOW_CONVERTERS)

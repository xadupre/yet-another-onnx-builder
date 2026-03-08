from typing import Sequence


def get_output_names(model) -> Sequence[str]:
    """
    Returns output names for a Keras model or layer.

    .. note::
        This POC implementation always returns a single output named ``"output"``.
        Multi-output models are not yet supported.
    """
    return ["output"]


def tf_dtype_to_np_dtype(tf_dtype):
    """Converts a TensorFlow dtype to a numpy dtype."""
    import numpy as np

    mapping = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "bool": np.bool_,
    }
    name = tf_dtype.name if hasattr(tf_dtype, "name") else str(tf_dtype)
    if name not in mapping:
        raise ValueError(f"Unsupported TensorFlow dtype: {tf_dtype!r}")
    return mapping[name]

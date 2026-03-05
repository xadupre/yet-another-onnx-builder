import re
from typing import Sequence


def get_output_names(model) -> Sequence[str]:
    """Returns output names for a Keras model or layer.

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


def sanitize_name(tf_name: str) -> str:
    """Converts a TF tensor or op name to a valid ONNX tensor name.

    TF names like ``"dense/MatMul:0"`` are converted to ``"dense_MatMul"``
    (the ``":N"`` output-index suffix is stripped, ``"/"`` is replaced with
    ``"_"``, and any other non-alphanumeric characters are replaced with
    ``"_"``).
    """
    # Strip output-index suffix (":0", ":1", ...)
    name = tf_name.split(":")[0] if ":" in tf_name else tf_name
    # Replace path separators and spaces with underscores
    name = re.sub(r"[/\s]", "_", name)
    # Remove any remaining invalid characters
    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
    return name

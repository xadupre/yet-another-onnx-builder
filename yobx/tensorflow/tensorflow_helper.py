from typing import Any, Dict, List, Optional, Sequence, Tuple


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
    if name == "bfloat16":
        import ml_dtypes

        return ml_dtypes.bfloat16
    if name not in mapping:
        raise ValueError(f"Unsupported TensorFlow dtype: {tf_dtype!r}")
    return mapping[name]


def jax_to_concrete_function(
    jax_fn: Any,
    args: Tuple[Any, ...],
    input_names: Optional[Sequence[str]] = None,
    dynamic_shapes: Optional[Tuple[Dict[int, str], ...]] = None,
):
    """
    Converts a :epkg:`JAX` function into a :class:`tensorflow.ConcreteFunction`.

    Uses :func:`jax.experimental.jax2tf.convert` to wrap the JAX function as a
    TensorFlow function, then traces it with :func:`tensorflow.function` to produce
    a concrete computation graph.  The resulting
    :class:`~tensorflow.ConcreteFunction` can be passed directly to
    :func:`yobx.tensorflow.to_onnx` for ONNX export.

    :param jax_fn: a callable JAX function (or :mod:`flax`/:mod:`equinox` model
        wrapped in a plain Python function) whose outputs are JAX arrays.
    :param args: dummy inputs as :class:`numpy.ndarray` objects; used to infer
        the dtype and static shape of each input tensor.
    :param input_names: optional list of names for the ONNX input tensors.
        When *None*, inputs are named ``"X"`` (single input) or
        ``"X0"``, ``"X1"``, … (multiple inputs).
    :param dynamic_shapes: optional per-input axis-to-dim-name mappings.
        Example: ``({0: "batch"},)`` marks axis 0 of the first input as a
        dynamic (variable-length) dimension.
        When *None*, axis 0 of every input is made dynamic by default.
    :return: a :class:`tensorflow.ConcreteFunction` ready for ONNX export.

    Example::

        import numpy as np
        import jax.numpy as jnp
        from yobx.tensorflow import to_onnx
        from yobx.tensorflow.tensorflow_helper import jax_to_concrete_function

        def jax_fn(x):
            return jnp.sin(x)

        x = np.random.rand(4, 3).astype(np.float32)
        cf = jax_to_concrete_function(jax_fn, (x,), dynamic_shapes=({0: "batch"},))
        onx = to_onnx(cf, (x,), dynamic_shapes=({0: "batch"},))
    """
    import numpy as np
    import tensorflow as tf
    from jax.experimental import jax2tf  # type: ignore[import]

    if input_names is None:
        input_names = ["X"] if len(args) == 1 else [f"X{i}" for i in range(len(args))]

    if len(input_names) != len(args):
        raise ValueError(f"Length mismatch: {len(args)} args but {len(input_names)} input_names")

    specs = []
    polymorphic_shapes: List[Optional[str]] = []
    for i, (name, arg) in enumerate(zip(input_names, args)):
        arr = np.asarray(arg)
        shape = list(arr.shape)
        n = arr.ndim

        if dynamic_shapes and i < len(dynamic_shapes):
            dyn_axes = dynamic_shapes[i]
        else:
            dyn_axes = {}

        for axis in dyn_axes:
            # TODO: replace by a dynamic new name
            shape[axis] = None  # type: ignore
        specs.append(tf.TensorSpec(shape=shape, dtype=tf.as_dtype(arr.dtype), name=name))

        # Build the polymorphic_shapes entry required by jax2tf.convert():
        # static axes keep their integer size; dynamic axes use their symbolic name.
        if n == 0 or not dyn_axes:
            polymorphic_shapes.append(None)
        else:
            dims = [
                str(dyn_axes[ax]) if ax in dyn_axes else str(arr.shape[ax]) for ax in range(n)
            ]
            polymorphic_shapes.append(f"({', '.join(dims)})")

    tf_fn = jax2tf.convert(
        jax_fn,
        polymorphic_shapes=polymorphic_shapes,
        native_serialization=False,
        enable_xla=False,
        with_gradient=False,
    )
    cf = tf.function(tf_fn, autograph=False).get_concrete_function(*specs)
    return cf

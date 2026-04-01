"""
yet-another-onnx-builder converts models from any kind to ONNX format.
"""

import os
from typing import Any, Dict, Optional, Sequence, Tuple, Union

__version__ = "0.1.0"

DEFAULT_TARGET_OPSET = 21


def to_onnx(
    model: Any,
    args: Any = None,
    **kwargs: Any,
) -> Any:
    """
    Convert any supported model type to ONNX.

    This is the unified top-level entry point.  It inspects *model* and
    *args* to decide which backend converter to call:

    * **torch** — when *model* is a :class:`torch.nn.Module` or
      :class:`torch.fx.GraphModule`.  Delegates to
      :func:`yobx.torch.to_onnx`.
    * **scikit-learn** — when *model* is a
      :class:`sklearn.base.BaseEstimator`.  Delegates to
      :func:`yobx.sklearn.to_onnx`.
    * **TensorFlow / Keras** — when *model* is a
      :class:`tensorflow.Module` (including Keras models and layers).
      Delegates to :func:`yobx.tensorflow.to_onnx`.
    * **LiteRT / TFLite** — when *model* is :class:`bytes` (raw flatbuffer)
      or a :class:`str` / :class:`os.PathLike` whose path ends with
      ``".tflite"``.  Delegates to :func:`yobx.litert.to_onnx`.
    * **SQL / DataFrame / NumPy callable** — when *model* is a plain
      :class:`str` (SQL query), a Python :func:`callable`, or a
      :class:`polars.LazyFrame`.  Delegates to :func:`yobx.sql.to_onnx`.

    :param model: the model to convert; see dispatch rules above.
    :param args: input arguments passed to the selected converter.
        For torch this is a sequence of :class:`torch.Tensor` objects.
        For sklearn / tensorflow / litert this is a tuple of
        :class:`numpy.ndarray` objects.
        For SQL / DataFrame this is a ``{column: dtype}`` mapping or a
        :class:`numpy.ndarray` (numpy-function tracing).
    :param kwargs: additional keyword arguments forwarded verbatim to the
        selected converter.  Consult the documentation of each backend for
        the full list of accepted parameters.
    :return: :class:`~yobx.container.ExportArtifact` wrapping the exported
        ONNX proto together with an :class:`~yobx.container.ExportReport`.

    Example — scikit-learn::

        import numpy as np
        from sklearn.linear_model import LinearRegression
        from yobx import to_onnx

        X = np.random.randn(20, 4).astype(np.float32)
        y = X[:, 0] + X[:, 1]
        reg = LinearRegression().fit(X, y)
        artifact = to_onnx(reg, (X,))

    Example — PyTorch::

        import torch
        from yobx import to_onnx

        class Neuron(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 2)
            def forward(self, x):
                return torch.relu(self.linear(x))

        model = Neuron()
        x = torch.randn(3, 4)
        artifact = to_onnx(model, (x,))

    Example — SQL::

        import numpy as np
        from yobx import to_onnx

        artifact = to_onnx(
            "SELECT a + b AS total FROM t WHERE a > 0",
            {"a": np.float32, "b": np.float32},
        )
    """
    # ------------------------------------------------------------------ #
    # 1. torch.nn.Module / torch.fx.GraphModule                          #
    # ------------------------------------------------------------------ #
    try:
        import torch  # noqa: PLC0415

        if isinstance(model, (torch.nn.Module, torch.fx.GraphModule)):
            from .torch import to_onnx as torch_to_onnx  # noqa: PLC0415

            return torch_to_onnx(model, args, **kwargs)
    except ImportError:
        pass

    # ------------------------------------------------------------------ #
    # 2. scikit-learn BaseEstimator                                       #
    # ------------------------------------------------------------------ #
    try:
        from sklearn.base import BaseEstimator  # noqa: PLC0415

        if isinstance(model, BaseEstimator):
            from .sklearn import to_onnx as sklearn_to_onnx  # noqa: PLC0415

            return sklearn_to_onnx(model, args, **kwargs)
    except ImportError:
        pass

    # ------------------------------------------------------------------ #
    # 3. TensorFlow / Keras module                                        #
    # ------------------------------------------------------------------ #
    try:
        import tensorflow as tf  # noqa: PLC0415

        if isinstance(model, tf.Module):
            from .tensorflow import to_onnx as tf_to_onnx  # noqa: PLC0415

            return tf_to_onnx(model, args, **kwargs)
    except ImportError:
        pass

    # ------------------------------------------------------------------ #
    # 4. LiteRT / TFLite: bytes or path ending with ".tflite"            #
    # ------------------------------------------------------------------ #
    if isinstance(model, bytes):
        from .litert import to_onnx as litert_to_onnx  # noqa: PLC0415

        return litert_to_onnx(model, args if args is not None else (), **kwargs)

    if isinstance(model, os.PathLike) and str(model).endswith(".tflite"):
        from .litert import to_onnx as litert_to_onnx  # noqa: PLC0415

        return litert_to_onnx(model, args if args is not None else (), **kwargs)

    if (
        isinstance(model, str)
        and model.endswith(".tflite")
        and os.path.isfile(model)
    ):
        from .litert import to_onnx as litert_to_onnx  # noqa: PLC0415

        return litert_to_onnx(model, args if args is not None else (), **kwargs)

    # ------------------------------------------------------------------ #
    # 5. SQL string, callable, or polars LazyFrame                       #
    # ------------------------------------------------------------------ #
    if isinstance(model, (str, os.PathLike)) or callable(model):
        from .sql import to_onnx as sql_to_onnx  # noqa: PLC0415

        return sql_to_onnx(model, args, **kwargs)

    # Try polars LazyFrame (duck-typed to avoid hard dependency)
    _module_name = getattr(type(model), "__module__", "") or ""
    if _module_name == "polars" or _module_name.startswith("polars."):
        from .sql import to_onnx as sql_to_onnx  # noqa: PLC0415

        return sql_to_onnx(model, args, **kwargs)

    raise TypeError(
        f"to_onnx: unsupported model type {type(model)!r}. "
        "Supported types: torch.nn.Module, sklearn.base.BaseEstimator, "
        "tensorflow.Module, bytes (TFLite flatbuffer), "
        "str / os.PathLike (SQL query or .tflite path), "
        "callable (DataFrame-tracing function or numpy function), "
        "polars.LazyFrame."
    )

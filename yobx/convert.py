"""
Top-level ``to_onnx`` dispatcher.

:func:`to_onnx` is the single entry point for all :mod:`yobx` converters.
It inspects the type of *model* (the first positional argument) and
delegates to the correct sub-package converter:

+------------------------------------------+----------------------------------+
| Input type                               | Converter called                 |
+==========================================+==================================+
| :class:`sklearn.base.BaseEstimator`      | :func:`yobx.sklearn.to_onnx`     |
+------------------------------------------+----------------------------------+
| TensorFlow / Keras model                 | :func:`yobx.tensorflow.to_onnx`  |
+------------------------------------------+----------------------------------+
| :class:`torch.nn.Module` /               | :func:`yobx.torch.interpreter.   |
| :class:`torch.fx.GraphModule`            | to_onnx`                         |
+------------------------------------------+----------------------------------+
| :class:`str` ending in ``.tflite``,      | :func:`yobx.litert.to_onnx`      |
| :class:`os.PathLike`, or :class:`bytes`  |                                  |
+------------------------------------------+----------------------------------+
| SQL :class:`str`, callable, or           | :func:`yobx.sql.to_onnx`         |
| ``polars.LazyFrame``                     |                                  |
+------------------------------------------+----------------------------------+

All keyword arguments are forwarded unchanged to the selected converter, so
every option supported by the individual converters is still accessible.

Example — sklearn::

    import numpy as np
    from sklearn.linear_model import LinearRegression
    from yobx import to_onnx

    X = np.random.randn(10, 3).astype(np.float32)
    y = X @ np.array([1.0, 2.0, 3.0], dtype=np.float32)
    reg = LinearRegression().fit(X, y)
    artifact = to_onnx(reg, (X,))

Example — SQL::

    import numpy as np
    from yobx import to_onnx

    artifact = to_onnx(
        "SELECT a + b AS total FROM t WHERE a > 0",
        {"a": np.float32, "b": np.float32},
    )
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

from .container import ExportArtifact


def _has_package(name: str) -> bool:
    """Return *True* if *name* is importable (without actually importing it)."""
    return importlib.util.find_spec(name) is not None


# ---------------------------------------------------------------------------
# Private per-framework helpers
# ---------------------------------------------------------------------------


def _sklearn_to_onnx(model: Any, args: Any, **kwargs: Any) -> ExportArtifact:
    """Dispatch to :func:`yobx.sklearn.to_onnx`."""
    from sklearn.base import BaseEstimator
    from .sklearn import to_onnx as _to_onnx

    if isinstance(model, BaseEstimator):
        return _to_onnx(model, args, **kwargs)
    return NotImplemented  # type: ignore[return-value]


def _tensorflow_to_onnx(model: Any, args: Any, **kwargs: Any) -> ExportArtifact:
    """Dispatch to :func:`yobx.tensorflow.to_onnx`."""
    import tensorflow as tf
    from .tensorflow import to_onnx as _to_onnx

    if isinstance(
        model,
        (
            tf.Module,
            tf.keras.layers.Layer,
            tf.types.experimental.ConcreteFunction,
        ),
    ):
        return _to_onnx(model, args, **kwargs)
    return NotImplemented  # type: ignore[return-value]


def _torch_to_onnx(model: Any, args: Any, **kwargs: Any) -> ExportArtifact:
    """Dispatch to :func:`yobx.torch.interpreter.to_onnx`."""
    import torch
    from .torch.interpreter import to_onnx as _to_onnx

    if isinstance(model, (torch.nn.Module, torch.fx.GraphModule)):
        return _to_onnx(model, args, **kwargs)
    return NotImplemented  # type: ignore[return-value]


def _litert_to_onnx(model: Any, args: Any, **kwargs: Any) -> ExportArtifact:
    """Dispatch to :func:`yobx.litert.to_onnx` for TFLite paths or bytes."""
    from .litert import to_onnx as _to_onnx

    if isinstance(model, bytes):
        return _to_onnx(model, args, **kwargs)
    if isinstance(model, (str, os.PathLike)) and os.fspath(model).endswith(".tflite"):
        return _to_onnx(model, args, **kwargs)
    return NotImplemented  # type: ignore[return-value]


def _sql_to_onnx(model: Any, args: Any, **kwargs: Any) -> ExportArtifact:
    """Dispatch to :func:`yobx.sql.to_onnx` for SQL strings, callables, and LazyFrames."""
    from .sql import to_onnx as _to_onnx

    # Plain SQL string (str not ending in .tflite has already been handled above).
    if isinstance(model, str):
        return _to_onnx(model, args, **kwargs)
    # Callable (DataFrame-tracing function).
    if callable(model):
        return _to_onnx(model, args, **kwargs)
    # polars LazyFrame — duck-typed to avoid a hard polars dependency.
    if hasattr(model, "explain") and hasattr(model, "collect"):
        return _to_onnx(model, args, **kwargs)
    return NotImplemented  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def to_onnx(model: Any, args: Any = None, **kwargs: Any) -> ExportArtifact:
    """Convert *model* to ONNX, automatically selecting the right converter.

    The converter is selected by inspecting the type of *model*:

    * **scikit-learn** :class:`~sklearn.base.BaseEstimator` →
      :func:`yobx.sklearn.to_onnx`
    * **TensorFlow / Keras** model or layer →
      :func:`yobx.tensorflow.to_onnx`
    * **PyTorch** :class:`~torch.nn.Module` or
      :class:`~torch.fx.GraphModule` →
      :func:`yobx.torch.interpreter.to_onnx`
    * **TFLite / LiteRT** model — a :class:`str` / :class:`os.PathLike`
      path whose name ends with ``.tflite``, **or** raw :class:`bytes` →
      :func:`yobx.litert.to_onnx`
    * **SQL string**, DataFrame-tracing callable, or
      ``polars.LazyFrame`` → :func:`yobx.sql.to_onnx`

    :param model: the model (or query) to convert; its type determines
        which sub-package converter is invoked.
    :param args: positional second argument forwarded to the selected
        converter (e.g. a tuple of dummy numpy arrays for sklearn/torch/TF,
        or a column-dtype mapping dict for SQL).
    :param kwargs: additional keyword arguments forwarded unchanged to the
        selected converter.
    :return: :class:`~yobx.container.ExportArtifact` wrapping the exported
        ONNX proto and an :class:`~yobx.container.ExportReport`.
    :raises TypeError: if *model* does not match any of the known input
        types.

    Example — scikit-learn estimator::

        import numpy as np
        from sklearn.linear_model import LinearRegression
        from yobx import to_onnx

        X = np.random.randn(10, 3).astype(np.float32)
        y = X @ np.array([1.0, 2.0, 3.0], dtype=np.float32)
        reg = LinearRegression().fit(X, y)
        artifact = to_onnx(reg, (X,))
        proto = artifact.proto

    Example — SQL query::

        import numpy as np
        from yobx import to_onnx

        artifact = to_onnx(
            "SELECT a + b AS total FROM t WHERE a > 0",
            {"a": np.float32, "b": np.float32},
        )

    Example — PyTorch module::

        import torch
        from yobx import to_onnx

        class Neuron(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 2)
            def forward(self, x):
                return torch.relu(self.linear(x))

        x = torch.randn(3, 4)
        artifact = to_onnx(Neuron(), (x,))

    Example — TFLite model::

        import numpy as np
        from yobx import to_onnx

        X = np.random.rand(1, 4).astype(np.float32)
        artifact = to_onnx("model.tflite", (X,))
    """
    # Each private helper returns NotImplemented when the model type does not match.
    # Framework availability is checked with importlib.util.find_spec before calling
    # the helper, so the imports inside the helpers are always expected to succeed.

    # 1. scikit-learn BaseEstimator
    if _has_package("sklearn"):
        result = _sklearn_to_onnx(model, args, **kwargs)
        if result is not NotImplemented:
            return result

    # 2. TensorFlow / Keras model
    if _has_package("tensorflow"):
        result = _tensorflow_to_onnx(model, args, **kwargs)
        if result is not NotImplemented:
            return result

    # 3. PyTorch nn.Module / fx.GraphModule
    if _has_package("torch"):
        result = _torch_to_onnx(model, args, **kwargs)
        if result is not NotImplemented:
            return result

    # 4. TFLite / LiteRT — file path ending in .tflite or raw bytes
    result = _litert_to_onnx(model, args, **kwargs)
    if result is not NotImplemented:
        return result

    # 5. SQL string, DataFrame-tracing callable, or polars LazyFrame
    #
    # Any plain str that did not match the .tflite check above is treated as
    # a SQL query.  File paths to other model formats (e.g. '.pt', '.onnx')
    # are not supported via this dispatcher and will produce a SQL parse error.
    result = _sql_to_onnx(model, args, **kwargs)
    if result is not NotImplemented:
        return result

    raise TypeError(
        f"to_onnx: cannot determine the converter to use for model type "
        f"{type(model)!r}.  Supported types are: scikit-learn BaseEstimator, "
        f"TensorFlow/Keras model, PyTorch nn.Module/fx.GraphModule, "
        f"TFLite path (str ending in '.tflite') or bytes, "
        f"SQL string, DataFrame-tracing callable, or polars LazyFrame."
    )

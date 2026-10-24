"""
Top-level dispatcher that routes a model to the appropriate backend converter.
"""

import os
import inspect
import runpy
import sys
from typing import Any, Dict, Optional, Sequence, Union
from onnx_light.onnx.defs import onnx_opset_version

#: Default ONNX opset version targeted by all converters.
DEFAULT_TARGET_OPSET = onnx_opset_version()


def guess_export(
    filename: Union[str, os.PathLike],
    model_name: Optional[str] = None,
    args_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Guesses arguments for :func:`to_onnx` from a trusted Python file.

    Executes the file with :func:`runpy.run_path` (not as ``__main__``).
    Only module-level objects are inspected; model classes and factory
    functions are never instantiated or called. The file must therefore
    create its model and example inputs outside a ``__main__`` guard.
    The file is executed outside a package context: explicit relative
    imports such as ``from .foo import bar`` are not supported. Use absolute
    imports resolvable in the current Python environment instead.

    Prefers a variable named ``model``. Otherwise selects the single
    supported framework instance or locally defined callable, ignoring
    private names, imported functions, and classes. SQL strings and TFLite
    paths/bytes require ``model`` or an explicit *model_name*.

    Looks for inputs named ``args``, ``example_inputs``, or ``inputs``, in
    that order, then a single ``X`` or ``x``. A single input is wrapped in
    a tuple; sequences and mappings are preserved. Module-level
    ``input_names`` and ``dynamic_shapes`` are also included when present.

    .. warning::

        This function executes arbitrary Python code. Only use trusted files.
        It is not a sandbox, and execution errors propagate unchanged.
        It is not thread-safe: ``runpy.run_path`` temporarily modifies
        process-global state, including ``sys.argv[0]`` and import-related
        state such as ``sys.path[0]``. Call it sequentially, not concurrently
        with other executions or changes to import state.

    :param filename: path to a Python file.
    :param model_name: explicit name of the model variable.
    :param args_name: explicit name of the example-input variable.
    :raises ValueError: if a model or inputs cannot be identified unambiguously.
    :raises TypeError: if the selected model is not supported.

    Returns:
        A dictionary usable as ``to_onnx(**guess_export(filename))``.
        Backend selection remains the responsibility of :func:`to_onnx`.
    """
    namespace = runpy.run_path(os.fspath(filename), run_name="__yobx_guess_export__")

    def is_model(value: Any, explicit: bool = False) -> bool:
        if inspect.isclass(value):
            return False
        if explicit and isinstance(value, (str, bytes, os.PathLike)):
            return True
        for module_name, class_name in (
            ("torch.nn", "Module"),
            ("sklearn.base", "BaseEstimator"),
            ("tensorflow", "Module"),
            ("polars", "LazyFrame"),
        ):
            module = sys.modules.get(module_name)
            if module is not None and isinstance(value, getattr(module, class_name, ())):
                return True
        return callable(value) and (
            explicit or getattr(value, "__module__", None) == namespace["__name__"]
        )

    if model_name is None:
        if "model" in namespace:
            model_name = "model"
        else:
            candidates = {}
            for name, value in namespace.items():
                if not name.startswith("_") and is_model(value):
                    candidates.setdefault(id(value), name)
            if len(candidates) != 1:
                raise ValueError(
                    f"Unable to identify one model in {filename!r}: "
                    f"candidates={list(candidates.values())!r}. Specify model_name."
                )
            model_name = next(iter(candidates.values()))
    if model_name not in namespace:
        raise ValueError(f"Model variable {model_name!r} was not found in {filename!r}.")
    model = namespace[model_name]
    if not is_model(model, explicit=True):
        raise TypeError(f"Variable {model_name!r} has unsupported model type {type(model)!r}.")

    if args_name is None:
        args_name = next(
            (name for name in ("args", "example_inputs", "inputs") if name in namespace), None
        )
        if args_name is None:
            candidates = [name for name in ("X", "x") if name in namespace]
            if len(candidates) != 1:
                raise ValueError(
                    f"Unable to identify example inputs in {filename!r}: "
                    f"candidates={candidates!r}. Specify args_name."
                )
            args_name = candidates[0]
    if args_name not in namespace:
        raise ValueError(f"Input variable {args_name!r} was not found in {filename!r}.")
    args = namespace[args_name]
    if args is None:
        raise ValueError(
            f"Input variable {args_name!r} is None; use () for a model with no inputs."
        )
    if not isinstance(args, (tuple, list, dict)):
        args = (args,)

    # Early validation: mapping-style args are only supported for the SQL/DataFrame-style
    # backends (SQL string, polars LazyFrame, or a tracing callable). For framework
    # instances (torch/sklearn/tensorflow) or LiteRT/TFLite models, require tuple/list.
    if isinstance(args, dict):
        allow_mapping_args = False

        # Tracing callable backends can accept mapping-style args.
        if callable(model):
            allow_mapping_args = True

        # polars.LazyFrame backend can accept mapping-style args.
        module = sys.modules.get("polars")
        if module is not None and isinstance(model, getattr(module, "LazyFrame", ())):
            allow_mapping_args = True

        # SQL string backend can accept mapping-style args.
        if isinstance(model, str) and not model.endswith(".tflite"):
            allow_mapping_args = True

        # Disallow mapping-style args for framework models and TFLite/LiteRT artifacts.
        is_framework_model = False
        for module_name, class_name in (
            ("torch.nn", "Module"),
            ("sklearn.base", "BaseEstimator"),
            ("tensorflow", "Module"),
        ):
            module = sys.modules.get(module_name)
            if module is not None and isinstance(model, getattr(module, class_name, ())):
                is_framework_model = True
                break

        is_tflite_artifact = isinstance(model, bytes) or (
            isinstance(model, (str, os.PathLike)) and str(model).endswith(".tflite")
        )

        if (is_framework_model or is_tflite_artifact) and not allow_mapping_args:
            raise ValueError(
                f"Input variable {args_name!r} is a dict, but the selected model backend "
                f"does not support mapping-style inputs. Use a tuple/list of example inputs "
                f"instead (e.g., args=(x,) or args=[x]). If you intended to use a SQL/polars/"
                f"callable backend, ensure 'model' refers to that object."
            )

    result = dict(model=model, args=args)
    for name in ("input_names", "dynamic_shapes"):
        if name in namespace:
            result[name] = namespace[name]
    return result


def to_onnx(
    model: Any,
    args: Any = None,
    input_names: Optional[Sequence[str]] = None,
    dynamic_shapes: Optional[Any] = None,
    target_opset: Union[int, Dict[str, int]] = DEFAULT_TARGET_OPSET,
    verbose: int = 0,
    large_model: bool = False,
    external_threshold: int = 1024,
    filename: Optional[str] = None,
    return_optimize_report: bool = False,
    graph_backend: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Convert any supported model type to ONNX.

    This is the unified top-level entry point.  It inspects *model* and
    *args* to decide which backend converter to call:

    * **torch** — when *model* is a :class:`torch.nn.Module` or
      :class:`torch.fx.GraphModule`.  Delegates to
      :func:`yobx.torch.to_onnx`.  Any extra *kwargs* are forwarded
      verbatim; see that function for the full list of accepted parameters
      (``as_function``, ``options``, ``dispatcher``, ``dynamic_shapes``,
      ``export_options``, ``function_options``, …).
    * **scikit-learn** — when *model* is a
      :class:`sklearn.base.BaseEstimator`.  Delegates to
      :func:`yobx.sklearn.to_onnx`.  Extra *kwargs* include
      ``builder_cls``, ``extra_converters``, ``function_options``,
      ``convert_options``, …
    * **TensorFlow / Keras** — when *model* is a
      :class:`tensorflow.Module` (including Keras models and layers).
      Delegates to :func:`yobx.tensorflow.to_onnx`.  Extra *kwargs* include
      ``builder_cls``, ``extra_converters``, …
    * **LiteRT / TFLite** — when *model* is :class:`bytes` (raw flatbuffer)
      or a :class:`str` / :class:`os.PathLike` whose path ends with
      ``".tflite"``.  Delegates to :func:`yobx.litert.to_onnx`.  Extra
      *kwargs* include ``builder_cls``, ``extra_converters``,
      ``subgraph_index``, …  Note: *filename* is not supported by this
      backend and is silently ignored.
    * **SQL / DataFrame / NumPy callable** — when *model* is a plain
      :class:`str` (SQL query), a Python :func:`callable`, or a
      :class:`polars.LazyFrame`.  Delegates to :func:`yobx.sql.to_onnx`.
      Extra *kwargs* include ``custom_functions``, ``builder_cls``, …

    :param model: the model to convert; see dispatch rules above.
    :param args: input arguments passed to the selected converter.
        For torch this is a sequence of :class:`torch.Tensor` objects.
        For sklearn / tensorflow / litert this is a tuple of
        :class:`numpy.ndarray` objects.
        For SQL / DataFrame this is a ``{column: dtype}`` mapping or a
        :class:`numpy.ndarray` (numpy-function tracing).
    :param input_names: optional list of names for the ONNX graph input
        tensors.  Supported by all backends.
    :param dynamic_shapes: optional specification of dynamic (symbolic)
        dimensions.  The exact format and default behaviour **differ by
        backend**:

        * **torch** — follows :func:`torch.export.export` conventions: a
          dict mapping input names to per-axis ``torch.export.Dim``
          objects, or a tuple with one entry per input.  ``None`` means
          *no* dynamic dimensions (all shapes are fixed).
        * **sklearn / tensorflow / litert** — a tuple of
          ``{axis: dim_name}`` dicts, one per input.  ``None`` means axis
          **0 is treated as dynamic** (batch dimension) for every input,
          while all other axes remain static.
        * **sql (numpy callable)** — same ``{axis: dim_name}`` tuple
          format.  ``None`` lets the backend apply its own default
          (typically axis 0 dynamic).  Ignored for SQL strings and
          DataFrame-tracing callables.

    :param target_opset: ONNX opset version to target.  Either an integer
        for the default domain (``""``), or a ``Dict[str, int]`` mapping
        domain names to opset versions (e.g.
        ``{"": 21, "ai.onnx.ml": 5}``).  Defaults to
        :data:`~yobx.DEFAULT_TARGET_OPSET`.
    :param verbose: verbosity level (0 = silent).  Supported by all backends.
    :param large_model: if *True* the returned
        :class:`~yobx.container.ExportArtifact` stores tensors as external
        data rather than embedding them inside the proto.  Supported by all
        backends.
    :param external_threshold: when *large_model* is *True*, tensors whose
        element count exceeds this threshold are stored externally.  Supported
        by all backends.
    :param filename: if set, the exported ONNX model is saved to this path.
        Supported by torch, sklearn, tensorflow, and sql backends.
        Not supported by the LiteRT backend (ignored when *model* is a
        ``.tflite`` file or raw flatbuffer bytes).
    :param return_optimize_report: if True, the returned
        :class:`~yobx.container.ExportArtifact` has its
        :attr:`~yobx.container.ExportArtifact.report` attribute populated with
        per-pattern optimization statistics.  Supported by all backends.
    :param graph_backend: ``None`` uses the converter's default native builder.
        ``"onnx-light"`` explicitly selects native graph construction, pattern
        optimization, and symbolic shape inference from the onnx-light wheel.
        Exported artifacts retain native protobufs. PyTorch models use
        ``torch.export`` followed by native ONNX lowering. Cannot be combined
        with ``builder_cls``.
    :param kwargs: additional backend-specific keyword arguments forwarded
        verbatim to the selected converter.  See the backend-specific
        ``to_onnx`` functions listed above for their full parameter lists.
    :return: :class:`~yobx.container.ExportArtifact` wrapping the exported
        ONNX proto together with an :class:`~yobx.container.ExportReport`.

    Example — scikit-learn (batch dimension fixed)::

        import numpy as np
        from sklearn.linear_model import LinearRegression
        from yobx import to_onnx

        X = np.random.randn(20, 4).astype(np.float32)
        y = X[:, 0] + X[:, 1]
        reg = LinearRegression().fit(X, y)
        artifact = to_onnx(reg, (X,))

    Example — scikit-learn (explicit dynamic batch dimension)::

        import numpy as np
        from sklearn.linear_model import LinearRegression
        from yobx import to_onnx

        X = np.random.randn(20, 4).astype(np.float32)
        y = X[:, 0] + X[:, 1]
        reg = LinearRegression().fit(X, y)
        # Mark axis 0 (batch) as dynamic for input 0:
        artifact = to_onnx(reg, (X,), dynamic_shapes=({0: "batch"},))

    Example — PyTorch (dynamic batch dimension via torch.export.Dim)::

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
        batch = torch.export.Dim("batch", min=1, max=256)
        artifact = to_onnx(model, (x,), dynamic_shapes={"x": {0: batch}})

    Example — SQL::

        import numpy as np
        from yobx import to_onnx

        artifact = to_onnx(
            "SELECT a + b AS total FROM t WHERE a > 0",
            {"a": np.float32, "b": np.float32},
        )
    """
    if graph_backend not in (None, "onnx-light"):
        raise ValueError(
            f"Unknown graph_backend={graph_backend!r}; expected 'onnx-light' or None."
        )
    if graph_backend is not None and "builder_cls" in kwargs:
        raise ValueError("graph_backend and builder_cls cannot be specified together.")

    # Build a dict of the common named arguments so they can be passed to
    # each backend without duplicating keyword logic.
    from .ext_test_case import has_litert, has_sklearn, has_tensorflow, has_torch  # noqa: PLC0415

    common: Dict[str, Any] = dict(
        input_names=input_names,
        dynamic_shapes=dynamic_shapes,
        target_opset=target_opset,
        verbose=verbose,
        large_model=large_model,
        external_threshold=external_threshold,
        filename=filename,
        return_optimize_report=return_optimize_report,
    )

    # ------------------------------------------------------------------ #
    # 1. torch.nn.Module / torch.fx.GraphModule                          #
    # ------------------------------------------------------------------ #
    if has_torch():
        import torch  # noqa: PLC0415

        if isinstance(model, (torch.nn.Module, torch.fx.GraphModule)):
            from .torch import to_onnx as torch_to_onnx  # noqa: PLC0415

            return torch_to_onnx(model, args, **common, **kwargs)

    if graph_backend == "onnx-light":
        from .builder.onnxlight import OnnxLightGraphBuilder

        kwargs["builder_cls"] = OnnxLightGraphBuilder

    # ------------------------------------------------------------------ #
    # 2. scikit-learn BaseEstimator                                       #
    # ------------------------------------------------------------------ #
    if has_sklearn():
        from sklearn.base import BaseEstimator  # noqa: PLC0415

        if isinstance(model, BaseEstimator):
            from .sklearn import (  # noqa: PLC0415
                SklearnOnnxLightGraphBuilder,
                to_onnx as sklearn_to_onnx,
            )

            if graph_backend == "onnx-light":
                kwargs["builder_cls"] = SklearnOnnxLightGraphBuilder
            return sklearn_to_onnx(model, args, **common, **kwargs)

    # ------------------------------------------------------------------ #
    # 3. TensorFlow / Keras module                                        #
    # ------------------------------------------------------------------ #
    if has_tensorflow():
        import tensorflow as tf  # noqa: PLC0415

        if isinstance(model, tf.Module):
            from .tensorflow import to_onnx as tf_to_onnx  # noqa: PLC0415

            return tf_to_onnx(model, args, **common, **kwargs)

    # ------------------------------------------------------------------ #
    # 4. LiteRT / TFLite: bytes or path ending with ".tflite"            #
    # ------------------------------------------------------------------ #
    # LiteRT does not support filename; remove it before forwarding.
    litert_common = {k: v for k, v in common.items() if k != "filename"}

    _is_tflite_path = (isinstance(model, os.PathLike) and str(model).endswith(".tflite")) or (
        isinstance(model, str) and model.endswith(".tflite") and os.path.isfile(model)
    )
    if has_litert() and (isinstance(model, bytes) or _is_tflite_path):
        from .litert import to_onnx as litert_to_onnx  # noqa: PLC0415

        return litert_to_onnx(model, args if args is not None else (), **litert_common, **kwargs)

    # ------------------------------------------------------------------ #
    # 5. SQL string, callable, or polars LazyFrame                       #
    # ------------------------------------------------------------------ #
    if isinstance(model, (str, os.PathLike)) or callable(model):
        from .sql import to_onnx as sql_to_onnx  # noqa: PLC0415

        return sql_to_onnx(model, args, **common, **kwargs)

    # Try polars LazyFrame (duck-typed to avoid hard dependency)
    _module_name = getattr(type(model), "__module__", "") or ""
    if _module_name == "polars" or _module_name.startswith("polars."):
        from .sql import to_onnx as sql_to_onnx  # noqa: PLC0415

        return sql_to_onnx(model, args, **common, **kwargs)

    raise TypeError(
        f"to_onnx: unsupported model type {type(model)!r}. "
        "Supported types: torch.nn.Module, sklearn.base.BaseEstimator, "
        "tensorflow.Module, bytes (TFLite flatbuffer), "
        "str / os.PathLike (SQL query or .tflite path), "
        "callable (DataFrame-tracing function or numpy function), "
        "polars.LazyFrame."
    )

from typing import Dict, Callable, Tuple, Union

SKLEARN_CONVERTERS: Dict[type, Callable] = {}


def register_sklearn_converter(cls: Union[type, Tuple[type, ...]]):
    def decorator(fct: Callable):
        """Registers a function to converts a model."""
        global SKLEARN_CONVERTERS
        if isinstance(cls, tuple):
            for c in cls:
                if c in SKLEARN_CONVERTERS:
                    raise TypeError(f"A converter is already registered for {c}.")
                SKLEARN_CONVERTERS[c] = fct
        else:
            if cls in SKLEARN_CONVERTERS:
                raise TypeError(f"A converter is already registered for {cls}.")
            SKLEARN_CONVERTERS[cls] = fct
        return fct

    return decorator


def get_sklearn_converter(cls: type):
    """Returns the converter for a specific type."""
    global SKLEARN_CONVERTERS
    if cls in SKLEARN_CONVERTERS:
        return SKLEARN_CONVERTERS[cls]
    raise ValueError(f"Unable to find a converter for type {cls}.")


def get_sklearn_converters():
    """Returns all registered converters as a mapping from type to converter function."""
    global SKLEARN_CONVERTERS
    return dict(SKLEARN_CONVERTERS)


def get_sklearn_estimator_coverage():
    """Return a coverage report for scikit-learn estimators.

    Enumerates every estimator/transformer exposed by
    :func:`sklearn.utils.all_estimators` and reports which ones already have
    a converter registered in :mod:`yobx.sklearn` and which ones are supported
    by :epkg:`sklearn-onnx` (``skl2onnx``), when that package is installed.

    Returns
    -------
    list[dict]
        Each entry is a dict with the following keys:

        ``"name"``
            Estimator class name (``str``).
        ``"cls"``
            The estimator class itself.
        ``"module"``
            Public sklearn module path (private submodules stripped).
        ``"yobx"``
            ``True`` if a converter is registered in :mod:`yobx.sklearn`.
        ``"skl2onnx"``
            ``True`` if ``skl2onnx`` is installed and supports this class,
            ``None`` if ``skl2onnx`` is not available.
    """
    from sklearn.utils import all_estimators
    import importlib.util

    skl2onnx_available = importlib.util.find_spec("skl2onnx") is not None
    if skl2onnx_available:
        from skl2onnx._supported_operators import sklearn_operator_name_map

        skl2onnx_classes = set(sklearn_operator_name_map.keys())
    else:
        skl2onnx_classes = set()

    def _public_module(cls):
        parts = cls.__module__.split(".")
        return ".".join(p for p in parts if not p.startswith("_"))

    # Enumerate all sklearn estimators, explicitly including trainable transforms.
    # Then add any yobx-registered converters not captured by the type filter.
    all_pairs = dict(
        all_estimators(
            type_filter=["classifier", "regressor", "cluster", "transformer"]
        )
    )
    for cls in SKLEARN_CONVERTERS:
        if cls.__name__ not in all_pairs:
            all_pairs[cls.__name__] = cls

    rows = []
    for name, cls in sorted(all_pairs.items(), key=lambda x: x[0]):
        rows.append(
            {
                "name": cls.__name__,
                "cls": cls,
                "module": _public_module(cls),
                "yobx": cls in SKLEARN_CONVERTERS,
                "skl2onnx": (
                    cls in skl2onnx_classes if skl2onnx_available else None
                ),
            }
        )
    return rows

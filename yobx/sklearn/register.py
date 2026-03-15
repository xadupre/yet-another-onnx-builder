from typing import Dict, Callable, List, Tuple, Union

SKLEARN_CONVERTERS: Dict[type, Callable] = {}

# Static catalogue of converters for external libraries that use the
# scikit-learn estimator API (fit/predict/transform).  Each entry describes
# one supported class so that documentation can be generated even when the
# optional dependency is not installed in the build environment.
_EXTERNAL_LIBRARY_CONVERTERS: List[dict] = [
    {
        "library": "category_encoders",
        "class_name": "QuantileEncoder",
        "class_module": "category_encoders",
        "converter_name": "category_encoders_quantile_encoder",
        "converter_module": "yobx.sklearn.category_encoders.quantile_encoder",
    },
    {
        "library": "lightgbm",
        "class_name": "LGBMClassifier",
        "class_module": "lightgbm",
        "converter_name": "sklearn_lgbm_classifier",
        "converter_module": "yobx.sklearn.lightgbm.lgbm",
    },
    {
        "library": "lightgbm",
        "class_name": "LGBMRegressor",
        "class_module": "lightgbm",
        "converter_name": "sklearn_lgbm_regressor",
        "converter_module": "yobx.sklearn.lightgbm.lgbm",
    },
    {
        "library": "xgboost",
        "class_name": "XGBClassifier",
        "class_module": "xgboost",
        "converter_name": "sklearn_xgb_classifier",
        "converter_module": "yobx.sklearn.xgboost.xgb",
    },
    {
        "library": "xgboost",
        "class_name": "XGBRegressor",
        "class_module": "xgboost",
        "converter_name": "sklearn_xgb_regressor",
        "converter_module": "yobx.sklearn.xgboost.xgb",
    },
]


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

        ``"category"``
            Estimator class name (``str``).
        ``"name"``
            Estimator class name (``str``).
        ``"cls"``
            The estimator class itself.
        ``"module"``
            Public sklearn module path (private submodules stripped).
        ``"yobx"``
            the converting function if a converter is registered in :mod:`yobx.sklearn`.
    """
    from sklearn.utils import all_estimators

    def _public_module(cls):
        parts = cls.__module__.split(".")
        return ".".join(p for p in parts if not p.startswith("_"))

    # Enumerates all sklearn estimators, explicitly including *predictable* transforms.
    # Then add any yobx-registered converters not captured by the type filter.
    all_pairs = dict(all_estimators())
    for cls in SKLEARN_CONVERTERS:
        if cls.__name__ not in all_pairs:
            all_pairs[cls.__name__] = cls

    rows = []
    for _name, cls in sorted(all_pairs.items(), key=lambda x: x[0]):
        rows.append(
            {
                "category": cls.__module__.split(".")[-2].strip("_"),
                "name": cls.__name__,
                "predictable": hasattr(cls, "transform") or hasattr(cls, "predict"),
                "cls": cls,
                "module": _public_module(cls),
                "yobx": SKLEARN_CONVERTERS.get(cls, None),
            }
        )
    return rows


def get_external_library_converters() -> List[dict]:
    """Return the static catalogue of converters for sklearn-like external libraries.

    Unlike :func:`get_sklearn_estimator_coverage`, this function does *not*
    require the optional dependencies (lightgbm, xgboost, category_encoders) to
    be installed — it reads from a hardcoded registry so that documentation can
    be generated in environments where those packages are absent.

    Returns
    -------
    list[dict]
        Each entry has the following keys:

        ``"library"``
            Name of the external package (e.g. ``"lightgbm"``).
        ``"class_name"``
            Name of the estimator class (e.g. ``"LGBMClassifier"``).
        ``"class_module"``
            Top-level module of the estimator class (e.g. ``"lightgbm"``).
        ``"converter_name"``
            Name of the yobx converter function
            (e.g. ``"sklearn_lgbm_classifier"``).
        ``"converter_module"``
            Fully-qualified module containing the converter
            (e.g. ``"yobx.sklearn.lightgbm.lgbm"``).
    """
    return list(_EXTERNAL_LIBRARY_CONVERTERS)

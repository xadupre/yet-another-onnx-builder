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


def sklearn_exportable_methods() -> Tuple[str, ...]:
    """
    Returns the methods which can be exported into ONNX.

    .. runpython::
        :showcode:

        from yobx.sklearn.register import sklearn_exportable_methods

        print(sklearn_exportable_methods())
    """
    return "transform", "predict", "predict_proba", "mahalanobis", "score_samples"


def get_sklearn_estimator_coverage(
    rst: bool = False, libraries: Union[str, Tuple[str, ...]] = "all"
):
    """
    Returns a coverage report for scikit-learn estimators.

    Enumerates every estimator/transformer exposed by
    :func:`sklearn.utils.all_estimators`  but also by other libraries
    following the same design and supported by this package.
    It reports which ones already have
    a converter registered in :mod:`yobx.sklearn`,
    when that package is installed.

    Args
    ----
        rst:
            returns the information a RST text
        libraries:
            `'all'` to include all available modules,
            or a list of libraries to include such as
            ``('sklearn', 'lightgbm', ...)``

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
    if not rst:

        def _public_module(cls):
            parts = cls.__module__.split(".")
            return ".".join(p for p in parts if not p.startswith("_"))

        if libraries == "all":
            libraries = "category_encoders", "imblearn", "lightgbm", "sklearn", "xgboost"

        all_pairs = {}
        for lib in libraries:
            if lib == "sklearn":
                from sklearn.utils import all_estimators

                all_pairs.update(dict(all_estimators()))
            elif lib == "xgboost":
                from .xgboost import all_estimators

                all_pairs.update(dict(all_estimators()))
            elif lib == "lightgbm":
                from .lightgbm import all_estimators

                all_pairs.update(dict(all_estimators()))
            elif lib == "category_encoders":
                from .category_encoders import all_estimators

                all_pairs.update(dict(all_estimators()))
            elif lib == "imblearn":
                from .imblearn import all_estimators

                all_pairs.update(dict(all_estimators()))
            else:
                raise ValueError(f"Unknown libraries {lib!r}")

        for cls in SKLEARN_CONVERTERS:
            if cls.__name__ not in all_pairs:
                all_pairs[cls.__name__] = cls

        methods = sklearn_exportable_methods()
        rows = []

        for _name, cls in sorted(all_pairs.items(), key=lambda x: x[0]):
            rows.append(
                {
                    "category": cls.__module__.split(".")[-2].strip("_"),
                    "name": cls.__name__,
                    "predictable": any(hasattr(cls, m) for m in methods),
                    "cls": cls,
                    "module": _public_module(cls),
                    "yobx": SKLEARN_CONVERTERS.get(cls, None),
                }
            )
        return rows

    rows = get_sklearn_estimator_coverage(libraries=libraries)
    rows = sorted(rows, key=lambda x: (x["category"], x["name"]))

    # Header
    rst_rows = [
        ".. list-table::",
        "    :header-rows: 1",
        "",
        "    * - category",
        "      - estimator",
        "      - predictable",
        "      - yobx",
        "      - converter",
    ]

    n_possible = 0
    n_done = 0
    for row in rows:
        cat = row["category"]
        fct = row["yobx"]
        if fct:
            yobx_mark = "✓"
            cvt = f":func:`{fct.__name__} <{fct.__module__}.{fct.__name__}>`"
        else:
            yobx_mark = ""
            cvt = ""
        predictable = "✓" if row["predictable"] else ""
        clss = f":class:`{row['name']} <{row['module']}.{row['name']}>`"
        rst_rows.extend(
            [
                f"    * - {cat}",
                f"      - {clss}",
                f"      - {predictable}",
                f"      - {yobx_mark}",
                f"      - {cvt}",
            ]
        )
        if yobx_mark:
            n_done += 1
        if predictable:
            n_possible += 1
    coverage_pct = n_done / n_possible * 100 if n_possible > 0 else 0.0
    rst_rows.extend(["", "", f"**Coverage**: {n_done}/{n_possible} ~ {coverage_pct:1.1f}%"])
    return "\n".join(rst_rows)

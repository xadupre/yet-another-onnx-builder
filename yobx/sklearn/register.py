from typing import Dict, Callable, List, Optional, Tuple, Union

SKLEARN_CONVERTERS: Dict[type, Callable] = {}
VERSION_CONVERTERS: Dict[type, Dict[str, str]] = {}


def register_sklearn_converter(
    cls: Union[type, Tuple[type, ...]],
    sklearn_version: Optional[str] = None,
    other_versions: Optional[Dict[str, str]] = None,
) -> Callable:
    """
    Registers a converter for a specific class following :epkg:`scikit-learn` API.
    If *version* is defined, the converter is register only if the version of
    `scikit-learn` is equal or more recent to this one.

    :param cls: class to register
    :param sklearn_version: first version of scikit-learn it can work
    :param other_versions: same for any particular package the class comes from,
        example: ``{'xgboost': '3.4'}``
    :return: Callable
    """
    assert not other_versions, f"{other_versions=} and this is not implemented yet for {cls=}"
    enabled = True
    if sklearn_version:
        global VERSION_CONVERTERS
        from sklearn import __version__
        from ..pv_version import PvVersion

        enabled = PvVersion(__version__) >= PvVersion(sklearn_version)
        if isinstance(cls, tuple):
            for c in cls:
                VERSION_CONVERTERS[c] = {"sklearn": sklearn_version}
        else:
            VERSION_CONVERTERS[cls] = {"sklearn": sklearn_version}

    def decorator(fct: Callable):
        """Registers a function to converts a model."""
        if enabled:
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


def has_sklearn_converter(cls: type):
    """Returns if the model has a converter."""
    global SKLEARN_CONVERTERS
    return cls in SKLEARN_CONVERTERS


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


def check_converter_output_alignment() -> List:
    """Check that every registered converter's return-type annotation is
    consistent with the number of outputs expected for its estimator class.

    Uses :func:`~yobx.sklearn.sklearn_helper.n_outputs_for_class` to
    determine how many outputs a class always produces:

    * ``1`` → the converter must **never** return a :class:`tuple`; its
      return annotation must be ``str``.
    * ``2`` → the converter must **always** return two values; its return
      annotation must be exactly ``Tuple[str, str]``.  A ``Union[str,
      Tuple[str, str]]`` annotation is **rejected** because it allows the
      converter to return only one output (the plain-``str`` arm), which
      conflicts with the guarantee that this class always produces two outputs.
    * ``None`` → the number of outputs depends on the instance (e.g.
      classifiers, pipelines, meta-estimators); *any* return annotation is
      accepted.

    Returns a :class:`list` of
    ``(cls, converter_function, expected_n_outputs, return_annotation)``
    tuples for every converter whose annotation is considered misaligned.
    An empty list means all converters are aligned.
    """
    import typing

    from .sklearn_helper import n_outputs_for_class

    misaligned: List = []

    for cls, fct in SKLEARN_CONVERTERS.items():
        expected = n_outputs_for_class(cls)
        if expected is None:
            # Variable — we cannot enforce a single annotation.
            continue

        hints = typing.get_type_hints(fct)
        ret = hints.get("return", None)
        if ret is None:
            # No annotation — cannot check.
            continue

        origin = getattr(ret, "__origin__", None)

        if expected == 1:
            # The return type must be exactly `str` — a Tuple or a Union
            # that includes a Tuple would indicate more outputs than expected.
            is_tuple = origin is tuple
            is_union_with_tuple = origin is typing.Union and any(
                getattr(a, "__origin__", None) is tuple for a in ret.__args__
            )
            if is_tuple or is_union_with_tuple:
                misaligned.append((cls, fct, expected, ret))

        elif expected == 2:
            # The return type must be exactly `Tuple[str, str]`.
            is_plain_str = ret is str
            is_tuple = origin is tuple
            is_union = origin is typing.Union
            if is_plain_str:
                # Annotation says "always 1 output", but 2 are expected.
                misaligned.append((cls, fct, expected, ret))
            elif is_union:
                has_plain_str = str in ret.__args__
                has_tuple = any(
                    getattr(a, "__origin__", None) is tuple for a in ret.__args__
                )
                if not has_tuple:
                    # No Tuple arm — annotation can never produce 2 outputs.
                    misaligned.append((cls, fct, expected, ret))
                elif has_plain_str:
                    # Union includes a plain-str arm — the converter may
                    # sometimes return only 1 output, violating the guarantee
                    # of always producing 2 outputs for this class.
                    misaligned.append((cls, fct, expected, ret))
            elif not is_tuple:
                # Some other non-Tuple type — misaligned.
                misaligned.append((cls, fct, expected, ret))

    return misaligned

def get_sklearn_estimator_coverage(
    libraries: Union[str, Tuple[str, ...]] = "all", rst: bool = False
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
        libraries:
            `'all'` to include all available modules,
            or a list of libraries to include such as
            ``('sklearn', 'lightgbm', ...)``
        rst:
            returns the information a RST text

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
    if libraries == "all":
        libraries = (
            "category_encoders",
            "imblearn",
            "lightgbm",
            "sklearn",
            "sksurv",
            "statsmodels",
            "xgboost",
        )
    if isinstance(libraries, str):
        libraries = (libraries,)

    if not rst:

        def _public_module(cls):
            if cls.__name__ in {"GammaRegressor", "PoissonRegressor", "TweedieRegressor"}:
                return "sklearn.linear_model"
            if cls.__name__ in {"HDBSCAN"}:
                return "sklearn.cluster"
            if cls.__name__ in {
                "HistGradientBoostingClassifier",
                "HistGradientBoostingRegressor",
            }:
                return "sklearn.ensemble"
            if cls.__name__.startswith("LGBM"):
                return "lightgbm"
            if cls.__name__.startswith("XGB"):
                return "xgboost"
            parts = cls.__module__.split(".")
            if cls.__module__.startswith("yobx.sklearn.statsmodels"):
                return f"statsmodels.{'.'.join(parts[3:])}"
            return ".".join(p for p in parts if not p.startswith("_"))

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
            elif lib == "sksurv":
                from .sksurv import all_estimators

                all_pairs.update(dict(all_estimators()))
            elif lib == "statsmodels":
                from .statsmodels import all_estimators

                all_pairs.update(dict(all_estimators()))
            else:
                raise ValueError(f"Unknown libraries {lib!r}")

        methods = sklearn_exportable_methods()
        rows = []

        for _name, cls in sorted(all_pairs.items(), key=lambda x: x[0]):
            obs = {
                "category": cls.__module__.split(".")[-2].strip("_"),
                "name": cls.__name__,
                "predictable": any(hasattr(cls, m) for m in methods),
                "cls": cls,
                "module": _public_module(cls),
                "yobx": SKLEARN_CONVERTERS.get(cls, None),
            }
            if cls in VERSION_CONVERTERS:
                since = VERSION_CONVERTERS[cls]
                if "sklearn" in since:
                    obs["scikit-learn"] = f"{since['sklearn']}+"
            rows.append(obs)
        return rows

    rows = get_sklearn_estimator_coverage(libraries=libraries, rst=False)
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
        "      - since",
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
        since = row.get("scikit-learn", "")
        rst_rows.extend(
            [
                f"    * - {cat}",
                f"      - {clss}",
                f"      - {predictable}",
                f"      - {yobx_mark}",
                f"      - {cvt}",
                f"      - {since}",
            ]
        )
        if yobx_mark:
            n_done += 1
        if predictable:
            n_possible += 1
    coverage_pct = n_done / n_possible * 100 if n_possible > 0 else 0.0
    rst_rows.extend(["", "", f"**Coverage**: {n_done}/{n_possible} ~ {coverage_pct:1.1f}%"])
    return "\n".join(rst_rows)

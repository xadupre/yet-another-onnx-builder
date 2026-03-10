.. _l-design-sklearn-supported-converters:

====================
Supported Converters
====================

The following :epkg:`scikit-learn` estimators and transformers have a
registered converter in :mod:`yobx.sklearn`.  The list is generated
programmatically from the live converter registry.

The following list can be extended by levering existing converters,
example :ref:`l-plot-sklearn-with-sklearn-onnx` shows how to plug
:epkg:`sklearn-onnx`.

.. runpython::
    :showcode:
    :rst:

    from yobx.sklearn import register_sklearn_converters
    register_sklearn_converters()
    from yobx.sklearn.register import get_sklearn_converters

    CATEGORY_TITLES = {
        "cluster": "Cluster Models",
        "preprocessing": "Preprocessing",
        "linear_model": "Linear Models",
        "tree": "Tree Models",
        "pipeline": "Pipeline",
        "neural_network": "Neural Networks",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
    }

    def public_module(cls):
        """Return the public sklearn module path (strip private submodules)."""
        parts = cls.__module__.split(".")
        return ".".join(p for p in parts if not p.startswith("_"))

    converters = get_sklearn_converters()
    groups = {}
    for cls, fct in converters.items():
        category = fct.__module__.split(".")[2]
        groups.setdefault(category, []).append((cls, fct))

    for cat in sorted(groups.keys()):
        title = CATEGORY_TITLES.get(cat, cat.replace("_", " ").title())
        print(title)
        print("=" * len(title))
        print()
        for cls, fct in sorted(groups[cat], key=lambda x: x[0].__name__):
            mod = public_module(cls)
            print(
                f"* :class:`{cls.__name__} <{mod}.{cls.__name__}>` → "
                f":func:`{fct.__name__} <{fct.__module__}.{fct.__name__}>`"
            )
        print()

Coverage Table
==============

The table below lists all :epkg:`scikit-learn` estimators and transformers,
showing which ones have a native converter in :mod:`yobx.sklearn` and which
ones are supported by :epkg:`sklearn-onnx` (``skl2onnx``).  Columns:

* **yobx** — ✓ if a converter is registered in :mod:`yobx.sklearn`.
* **skl2onnx** — ✓ if :epkg:`sklearn-onnx` is installed and supports the
  class; ``n/a`` if ``skl2onnx`` is not available in this build.

.. runpython::
    :showcode:
    :rst:

    from yobx.sklearn import register_sklearn_converters
    register_sklearn_converters()
    from yobx.sklearn.register import get_sklearn_estimator_coverage

    rows = get_sklearn_estimator_coverage()

    skl2onnx_available = rows[0]["skl2onnx"] is not None if rows else False
    skl2onnx_col = "skl2onnx" if skl2onnx_available else "skl2onnx (n/a)"

    # Header
    print(".. list-table::")
    print("    :header-rows: 1")
    print()
    print("    * - Estimator")
    print("      - yobx")
    print(f"      - {skl2onnx_col}")
    for row in rows:
        yobx_mark = "✓" if row["yobx"] else ""
        if row["skl2onnx"] is None:
            skl2_mark = "n/a"
        elif row["skl2onnx"]:
            skl2_mark = "✓"
        else:
            skl2_mark = ""
        print(f"    * - :class:`{row['name']} <{row['module']}.{row['name']}>`")
        print(f"      - {yobx_mark}")
        print(f"      - {skl2_mark}")

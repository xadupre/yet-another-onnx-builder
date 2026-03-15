.. _l-design-sklearn-supported-converters:

====================
Supported Converters
====================

The following :epkg:`scikit-learn` estimators and transformers have a
registered converter in :mod:`yobx.sklearn`.  The list is generated
programmatically from the live converter registry.  External-library
estimators from :epkg:`lightgbm`, :epkg:`xgboost`, and
:epkg:`category_encoders` are always listed; see
:ref:`l-design-sklearn-like-converters` for architecture details.

Coverage Table
==============

The table below lists all :epkg:`scikit-learn` estimators and transformers,
showing which ones have a native converter in :mod:`yobx.sklearn` among those
which can (*predictable*).  External-library estimators are appended at the
end.

* **yobx** — ✓ if a converter is registered in :mod:`yobx.sklearn`.

.. runpython::
    :showcode:
    :rst:

    from yobx.sklearn import register_sklearn_converters
    from yobx.sklearn.register import get_sklearn_estimator_coverage

    register_sklearn_converters()

    rows = get_sklearn_estimator_coverage()

    # External library converters: always include even if the optional
    # package was not installed at doc-build time.
    _EXTERNAL = [
        ("category_encoders", "QuantileEncoder", "category_encoders",
         "category_encoders_quantile_encoder",
         "yobx.sklearn.category_encoders.quantile_encoder"),
        ("lightgbm", "LGBMClassifier", "lightgbm",
         "sklearn_lgbm_classifier", "yobx.sklearn.lightgbm.lgbm"),
        ("lightgbm", "LGBMRegressor", "lightgbm",
         "sklearn_lgbm_regressor", "yobx.sklearn.lightgbm.lgbm"),
        ("xgboost", "XGBClassifier", "xgboost",
         "sklearn_xgb_classifier", "yobx.sklearn.xgboost.xgb"),
        ("xgboost", "XGBRegressor", "xgboost",
         "sklearn_xgb_regressor", "yobx.sklearn.xgboost.xgb"),
    ]
    covered = {r["name"] for r in rows}
    for cat, name, module, cvt_name, cvt_module in _EXTERNAL:
        if name not in covered:
            rows.append({
                "category": cat, "name": name, "predictable": True,
                "module": module, "yobx": (cvt_name, cvt_module),
            })

    rows = sorted(rows, key=lambda x: (x["category"], x["name"]))

    # Header
    print(".. list-table::")
    print("    :header-rows: 1")
    print()
    print("    * - category")
    print("      - estimator")
    print("      - predictable")
    print("      - yobx")
    print("      - converter")

    n_possible = 0
    n_done = 0
    for row in rows:
        cat = row["category"]
        fct = row["yobx"]
        if isinstance(fct, tuple):
            cvt_name, cvt_module = fct
            yobx_mark = "✓"
            cvt = f":func:`{cvt_name} <{cvt_module}.{cvt_name}>`"
        elif fct:
            yobx_mark = "✓"
            cvt = f":func:`{fct.__name__} <{fct.__module__}.{fct.__name__}>`"
        else:
            yobx_mark = ""
            cvt = ""
        predictable = "✓" if row["predictable"] else ""
        cls = f":class:`{row['name']} <{row['module']}.{row['name']}>`"
        print(f"    * - {cat}")
        print(f"      - {cls}")
        print(f"      - {predictable}")
        print(f"      - {yobx_mark}")
        print(f"      - {cvt}")
        if yobx_mark:
            n_done += 1
        if predictable:
            n_possible += 1
    print()
    print()
    print(f"**Coverage**: {n_done}/{n_possible} ~ {n_done/n_possible * 100:1.1f}%")


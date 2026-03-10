.. _l-design-sklearn-supported-converters:

====================
Supported Converters
====================

The following :epkg:`scikit-learn` estimators and transformers have a
registered converter in :mod:`yobx.sklearn`.  The list is generated
programmatically from the live converter registry.

Coverage Table
==============

The table below lists all :epkg:`scikit-learn` estimators and transformers,
showing which ones have a native converter in :mod:`yobx.sklearn` among those
which can (*predictable*).

* **yobx** — ✓ if a converter is registered in :mod:`yobx.sklearn`.

.. runpython::
    :showcode:
    :rst:

    from yobx.sklearn import register_sklearn_converters
    from yobx.sklearn.register import get_sklearn_estimator_coverage

    register_sklearn_converters()

    rows = get_sklearn_estimator_coverage()

    # Header
    print(".. list-table::")
    print("    :header-rows: 1")
    print()
    print("    * - category")
    print("      - estimator")
    print("      - predictable")
    print("      - yobx converter")

    n_possible = 0
    n_done = 0
    for row in rows:
        cat = row["category"]
        fct = row["yobx"]
        yobx_mark = "✓" if fct else ""
        trainable = "✓" if row["trainable"] else ""
        cls = f":class:`{row['name']} <{row['module']}.{row['name']}>`"
        cvt = f":func:`{fct.__name__} <{fct.__module__}.{fct.__name__}>`" if fct else ""
        print(f"    * - {cat}")
        print(f"      - {cls}")
        print(f"      - {trainable}")
        print(f"      - {yobx_mark}")
        print(f"      - {cvt}")
        if yobx_mark:
            n_done += 1
        if trainable:
            n_possible += 1
    print()
    print()
    print(f"**Coverage**: {n_done}/{n_possible} ~ {n_done/n_possible * 100:1.1f}%")

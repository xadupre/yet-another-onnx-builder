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
        "preprocessing": "Preprocessing",
        "linear_model": "Linear Models",
        "tree": "Tree Models",
        "pipeline": "Pipeline",
        "neural_network": "Neural Networks",
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

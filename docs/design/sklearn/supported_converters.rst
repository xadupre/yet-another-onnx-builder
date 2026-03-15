.. _l-design-sklearn-supported-converters:

====================
Supported Converters
====================

The following :epkg:`scikit-learn` estimators and transformers have a
registered converter in :mod:`yobx.sklearn`.  The list is generated
programmatically from the live converter registry.  External-library
estimators from :epkg:`lightgbm`, :epkg:`xgboost`,
:epkg:`category_encoders`, and :epkg:`imbalanced-learn` are listed when
the corresponding optional dependencies are installed; see
:ref:`l-design-sklearn-like-converters` for architecture details.

Coverage Table
==============

The table below lists all :epkg:`scikit-learn` estimators and transformers,
showing which ones have a native converter in :mod:`yobx.sklearn` among those
which can (*predictable*).  External-library estimators are appended at the
end.

scikit-learn
------------

.. runpython::
    :showcode:
    :rst:

    from yobx.sklearn import register_sklearn_converters
    from yobx.sklearn.register import get_sklearn_estimator_coverage

    register_sklearn_converters()

    print(get_sklearn_estimator_coverage(libraries=("sklearn",), rst=True))

category_encoders
-----------------

.. runpython::
    :showcode:
    :rst:

    from yobx.sklearn import register_sklearn_converters
    from yobx.sklearn.register import get_sklearn_estimator_coverage

    register_sklearn_converters()

    print(get_sklearn_estimator_coverage(libraries=("category_encoders",), rst=True))

lightgbm
--------

.. runpython::
    :showcode:
    :rst:

    from yobx.sklearn import register_sklearn_converters
    from yobx.sklearn.register import get_sklearn_estimator_coverage

    register_sklearn_converters()

    print(get_sklearn_estimator_coverage(libraries=("lightgbm",), rst=True))

xgboost
-------

.. runpython::
    :showcode:
    :rst:

    from yobx.sklearn import register_sklearn_converters
    from yobx.sklearn.register import get_sklearn_estimator_coverage

    register_sklearn_converters()

    print(get_sklearn_estimator_coverage(libraries=("xgboost",), rst=True))

imbalanced-learn
----------------

.. runpython::
    :showcode:
    :rst:

    from yobx.sklearn import register_sklearn_converters
    from yobx.sklearn.register import get_sklearn_estimator_coverage

    register_sklearn_converters()

    print(get_sklearn_estimator_coverage(libraries=("imblearn",), rst=True))

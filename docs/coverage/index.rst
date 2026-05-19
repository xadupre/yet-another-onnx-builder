.. _l-coverage:

Coverage
========

This page collects links to all coverage and supported-operator pages
across the different conversion backends.

SQL / DataFrame / Polars
------------------------

:ref:`l-design-sql-coverage` summarises which SQL constructs, DataFrame
operations, and polars ``LazyFrame`` operations are currently supported
when converting tabular data manipulations to ONNX.

scikit-learn and compatible libraries
--------------------------------------

:ref:`l-design-sklearn-supported-converters` lists all
:epkg:`scikit-learn` estimators and transformers, along with estimators
from :epkg:`category_encoders`, :epkg:`imbalanced-learn`,
:epkg:`lightgbm`, :epkg:`scikit-survival`, :epkg:`statsmodels`, and
:epkg:`xgboost`, showing which ones have a registered converter in
:mod:`yobx.sklearn`.

TensorFlow / JAX
-----------------

:ref:`l-design-tensorflow-supported-ops` lists every TensorFlow op that
has a converter to ONNX.

:ref:`l-design-tensorflow-supported-jax-ops` lists the JAX / StableHLO
ops that are handled during JAX-to-ONNX conversion.

LiteRT
------

:ref:`l-design-litert-converter` describes the overall LiteRT/TFLite to
ONNX conversion workflow, including dynamic shapes and custom op converters.

:ref:`l-design-litert-supported-ops` lists every LiteRT (TFLite) op that
has a converter to ONNX.

PyTorch
-------

:ref:`l-design-torch-supported-aten-functions` enumerates every ATen
function and its mapping to an ONNX operator.

:ref:`l-torch-converter` also contains an overview of exportability
(:ref:`l-design-torch-case-coverage`) that runs a broad set of model cases
through multiple exporters and reports which ones succeed.

:ref:`l-design-torch-op-coverage` shows which ``op_db`` ops and data types
are covered by the op-db export tests, distinguishing between ops with a
working converter, known failures, and ops missing a converter.

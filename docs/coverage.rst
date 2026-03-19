.. _l-coverage:

Coverage
========

This page collects links to all coverage and supported-operator pages
across the different conversion backends.

scikit-learn and compatible libraries
--------------------------------------

:ref:`l-design-sklearn-supported-converters` lists all
:epkg:`scikit-learn` estimators and transformers, along with estimators
from :epkg:`category_encoders`, :epkg:`imbalanced-learn`,
:epkg:`lightgbm`, :epkg:`scikit-survival`, and :epkg:`xgboost`, showing
which ones have a registered converter in :mod:`yobx.sklearn`.

PyTorch
-------

:ref:`l-design-torch-supported-aten-functions` enumerates every ATen
function and its mapping to an ONNX operator.

:ref:`l-torch-converter` also contains an overview of exportability
(:ref:`l-design-torch-case-coverage`) that runs a broad set of model cases
through multiple exporters and reports which ones succeed.

TensorFlow / JAX
-----------------

:ref:`l-design-tensorflow-supported-ops` lists every TensorFlow op that
has a converter to ONNX.

:ref:`l-design-tensorflow-supported-jax-ops` lists the JAX / StableHLO
ops that are handled during JAX-to-ONNX conversion.

LiteRT
------

:ref:`l-design-litert-supported-ops` lists every LiteRT (TFLite) op that
has a converter to ONNX.

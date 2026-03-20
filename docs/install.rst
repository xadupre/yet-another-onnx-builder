.. _l-install:

Installation
============

Requirements
------------

**yet-another-onnx-builder** (``yobx``) requires Python 3.10 or later and the
following core dependencies (installed automatically):

* `numpy <https://numpy.org/>`_ ≥ 2.1
* `onnx <https://onnx.ai/onnx/>`_ ≥ 1.14
* `onnxruntime <https://onnxruntime.ai/>`_ ≥ 1.24
* `ml_dtypes <https://github.com/jax-ml/ml_dtypes>`_
* `scipy <https://scipy.org/>`_

Basic installation
------------------

Install the base package from `PyPI <https://pypi.org/project/yet-another-onnx-builder/>`_:

.. code-block:: bash

    pip install yet-another-onnx-builder

Optional dependencies
---------------------

Framework-specific extras are declared in ``pyproject.toml``.
Install one or more of them alongside the base package:

**scikit-learn** (and compatible libraries)

.. code-block:: bash

    # scikit-learn models
    pip install "yet-another-onnx-builder[sklearn]"

    # category-encoders support
    pip install "yet-another-onnx-builder[category_encoders]"

    # imbalanced-learn support
    pip install "yet-another-onnx-builder[imblearn]"

    # LightGBM support
    pip install "yet-another-onnx-builder[lightgbm]"

    # XGBoost support
    pip install "yet-another-onnx-builder[xgboost]"

    # scikit-survival support
    pip install "yet-another-onnx-builder[sksurv]"

**Deep learning**

.. code-block:: bash

    # PyTorch export
    pip install "yet-another-onnx-builder[torch]"

    # TensorFlow / JAX export
    pip install "yet-another-onnx-builder[tensorflow]"

    # JAX-only export
    pip install "yet-another-onnx-builder[jax]"

    # LiteRT / TFLite export
    pip install "yet-another-onnx-builder[litert]"

**ONNX graph building**

.. code-block:: bash

    # spox back-end for GraphBuilder
    pip install "yet-another-onnx-builder[spox]"

**Multiple extras at once**

.. code-block:: bash

    pip install "yet-another-onnx-builder[sklearn,torch]"

Development installation
------------------------

To contribute or run the test suite, clone the repository and install the
``dev`` extras:

.. code-block:: bash

    git clone https://github.com/xadupre/yet-another-onnx-builder.git
    cd yet-another-onnx-builder
    pip install -e ".[dev]"

To build the documentation locally, also install the ``docs`` extras
in addition to all the packages the library supports.

.. code-block:: bash

    pip install -e ".[docs]"
    cd docs
    bash make_doc

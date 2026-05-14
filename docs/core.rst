.. _l-core:

Core
====

This section documents the architecture and design of **yet-another-onnx-builder** (*yobx*),
a toolkit for converting machine learning models from multiple frameworks to ONNX format.
It covers the conversion pipelines for scikit-learn, PyTorch, and TensorFlow models;
the core ``GraphBuilder`` and graph-optimization infrastructure that powers all conversions;
miscellaneous utilities; and the repository structure and CI workflows
that maintain code across the project.

.. toctree::
   :maxdepth: 1

   design/shape/index
   design/builder/index
   design/xtracing/index
   design/technical_details/index
   design/misc/index
   design/ci
   design/env_variables

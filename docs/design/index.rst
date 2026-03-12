Elements of Design
==================

This section documents the architecture and design of **yet-another-onnx-builder** (*yobx*),
a toolkit for converting machine learning models from multiple frameworks to ONNX format.
It covers the conversion pipelines for scikit-learn, PyTorch, and TensorFlow models;
the core ``GraphBuilder`` and graph-optimization infrastructure that powers all conversions;
miscellaneous utilities; and the repository structure and CI workflows
that maintain code quality across the project.

.. toctree::
   :maxdepth: 1
   :capture: Libraries

   sklearn/index
   tensorflow/index
   torch/index

.. toctree::
   :maxdepth: 1
   :capture: GraphBuilder and other classes

   builder/index
   misc/index
   ci

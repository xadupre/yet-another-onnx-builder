.. _l-converters:

Converters
==========

This section lists all available converters.
They all relies on a function ``to_onnx(model, args)`` which walks through
the model, converts piece by piece to onnx and connect them.
They all returns an instance of :class:`~yobx.container.ExportArtifact`.

.. toctree::
   :maxdepth: 1
   :caption: Common Pieces

   design/export_artifact
   coverage
   debug
   main_to_onnx

.. toctree::
   :maxdepth: 1
   :caption: Converters

   design/to_onnx
   design/litert/index
   design/sklearn/index
   design/sql/index
   design/tensorflow/index
   design/torch/index

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

   design/to_onnx
   design/export_artifact
   coverage
   debug

Converters all share the same way to create and optimize onnx models,
but they all trace the model to convert in different ways unqiue
to each library the model is implemented with.

.. toctree::
   :maxdepth: 1
   :caption: Converters

   design/litert/index
   design/sklearn/index
   design/sql/index
   design/tensorflow/index
   design/torch/index

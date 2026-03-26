.. _l-design-sklearn-debug-env-vars:

====================================
Debugging with Environment Variables
====================================

All converters for the same API and can be debugged the same way.
Logging usually provides quite some information difficult
to digest to diagnose an issue. At the very end, the error
either come with a result name, or the final onnx has an issue
caused by a specific result name. An optimization did not trigger
or :epkg:`onnxruntime` raises an issue giving a result name
causing the exception. Either way, the key for debugging
is the ONNX result name.

Called with the default :class:`yobx.xbuilder.GraphBuilder`,
the library can be debugged with environment variables.
The conversion stops as soon as a result name is issued or
or a pattern is triggered or not triggered.

* :ref:`l-design-xshape-debugging`
* :ref:`l-design-pattern-optimizer-debugging`
* :ref:`l-graphbuilder-debugging-env`
* :ref:`l-design-env-variables` — complete reference of all environment variables

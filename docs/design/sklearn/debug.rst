.. _l-design-sklearn-debug-env-vars:

====================================
Debugging with Environment Variables
====================================

Debugging depends on the builder used to build the ONNX model.
The default option is :class:`~yobx.xbuilder.GraphBuilder`
and offers debugging options through environment variables.
See :ref:`l-graphbuilder-debugging-env`.

The converters for `scikit-learn` are written in an efficient
way and there is not much pattern optimization can bring to it
except maybe optimizations specific to :epkg:`onnxruntime`.
In any way, if anything goes wrong, the following page gives
some tips to find out the cause of any failure, usually a pattern
not triggering when it should: :ref:`l-debug-patterns`.

.. _l-design-shape:

Native symbolic shape inference
===============================

The public :class:`~yobx.xshape.NativeShapeInference` adapter owns a persistent
``onnx_light.onnx_core.shape_inference.ShapesContext``. Native
``compute_shape_model`` and ``compute_shape_node`` are the only shape/type
inference entry points used by this adapter. Native ``SymTensor`` descriptors
contain element types, dimensions, and optional shape-tensor values.

The adapter normalizes native names to Python strings and dimensions to tuples.
Read-only ``_known_types``, ``_known_shapes``, and ``_known_ranks`` snapshots
support diagnostic callers; they are not independent inference caches.
Anonymous dimensions remain unknown. The wheel currently cannot distinguish an
unknown rank from a scalar descriptor, so shape/cost mode rejects graph inputs
without a declared rank.

Constraints and shape-tensor arithmetic stay in the native context.
``register_constraint_dimension`` delegates to its equality API; custom
operators can register callbacks on ``context``. Native inference failures
propagate rather than selecting a second shape engine.

``to_onnx`` copies the original native model through serialization, then uses
``apply_inferred_shapes_to_model``. This preserves model metadata and avoids
modifying the input model merely to inspect its shapes.

``ShapeBuilder`` and the low-level runtime/helper modules remain available for
legacy Torch builder consumers. Their Python shape implementations are not
used by ``NativeShapeInference``; inherited convenience methods only support
formatting and comparison of already inferred descriptors.

For usage, see :ref:`l-howto-shape-inference`. For arithmetic cost formulas,
see :ref:`l-design-cost`.

.. _l-design-xshape-debugging:

Diagnostics
-----------

``get_debug_msg()`` displays the native shapes and element types currently
available. Detailed inference events are available by setting
``inference.context.events_enabled = True`` before inference and reading
``inference.context.events()`` afterwards. Unsupported operators and invalid
constraints raise the native exception directly.

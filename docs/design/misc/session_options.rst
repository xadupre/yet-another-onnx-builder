.. _l-design-session-options:

================================
onnxruntime.SessionOptions Guide
================================

:class:`onnxruntime.SessionOptions` controls how
:class:`onnxruntime.InferenceSession` loads and runs a model.  This page
lists every property, method, and related enum together with a short
description, its default value, and — where applicable — the matching
``yobx`` wrapper parameter.

.. rubric:: yobx wrappers

The following classes accept individual ``SessionOptions`` fields as
keyword arguments so that callers rarely need to build a
:class:`onnxruntime.SessionOptions` object by hand:

* :class:`~yobx.reference._inference_session._InferenceSession`
* :class:`~yobx.reference._inference_session_numpy.InferenceSessionForNumpy`
* :class:`~yobx.reference._inference_session_torch.InferenceSessionForTorch`
* :class:`~yobx.reference.onnxruntime_evaluator.OnnxruntimeEvaluator`

Pass a fully-configured :class:`onnxruntime.SessionOptions` object via the
``session_options`` argument to bypass all individual keyword arguments.

Properties
==========

.. list-table::
   :widths: 30 12 58
   :header-rows: 1

   * - Property
     - Default
     - Description
   * - ``enable_cpu_mem_arena``
     - ``True``
     - When ``True``, enables the CPU memory arena.  The arena pre-allocates
       a large block of memory and serves subsequent allocations from it,
       which reduces allocation overhead for many small tensors.  Set to
       ``False`` to disable the arena and use the system allocator directly
       (useful when memory is tight).
   * - ``enable_mem_pattern``
     - ``True``
     - When ``True``, ONNX Runtime analyses the graph to determine a static
       memory layout that can be reused across runs.  Disabling this can
       slightly reduce peak memory at the cost of some per-run overhead.
   * - ``enable_mem_reuse``
     - ``True``
     - When ``True``, output buffers are reused across inference calls where
       possible.  Set to ``False`` to always allocate fresh output buffers
       (useful for debugging memory issues).
   * - ``enable_profiling``
     - ``False``
     - When ``True``, ONNX Runtime collects per-node timing data during
       inference.  The profile is written to a JSON file whose name is
       derived from ``profile_file_prefix``.  Corresponds to the
       ``enable_profiling`` parameter of the yobx wrappers.
   * - ``execution_mode``
     - ``ORT_SEQUENTIAL``
     - Controls whether operators are executed sequentially or in parallel.
       See :ref:`l-ort-execution-mode` for the available values.
       Use ``ORT_PARALLEL`` together with ``inter_op_num_threads`` to
       exploit multi-node parallelism.
   * - ``execution_order``
     - ``DEFAULT``
     - Determines the order in which nodes are scheduled for execution.
       See :ref:`l-ort-execution-order` for the available values.
   * - ``graph_optimization_level``
     - ``ORT_ENABLE_ALL``
     - Sets the level of graph optimizations applied before execution.
       See :ref:`l-ort-graph-optimization-level` for the available levels.
       Corresponds to the ``graph_optimization_level`` parameter of the
       yobx wrappers (also accepts a plain ``bool``).
   * - ``inter_op_num_threads``
     - ``0`` (auto)
     - Number of threads used for parallelism *between* independent graph
       nodes when ``execution_mode`` is ``ORT_PARALLEL``.  ``0`` lets
       ONNX Runtime choose automatically.
   * - ``intra_op_num_threads``
     - ``0`` (auto)
     - Number of threads used for parallelism *within* a single operator
       (e.g., matrix multiplication).  ``0`` lets ONNX Runtime choose
       automatically.
   * - ``log_severity_level``
     - ``-1`` (default)
     - Severity threshold for session-level log messages.  Messages below
       this level are suppressed.  Common values: ``0`` = VERBOSE,
       ``1`` = INFO, ``2`` = WARNING, ``3`` = ERROR, ``4`` = FATAL.
       Corresponds to the ``log_severity_level`` parameter of the yobx
       wrappers.
   * - ``log_verbosity_level``
     - ``0``
     - Verbosity sub-level for VERBOSE messages
       (``log_severity_level == 0``).  Higher values produce more output.
       Corresponds to the ``log_verbosity_level`` parameter of the yobx
       wrappers.
   * - ``logid``
     - ``""``
     - String tag prepended to log messages emitted by this session.
       Useful when multiple sessions run concurrently.
   * - ``optimized_model_filepath``
     - ``""``
     - Path where the optimized ONNX model is saved after graph
       optimization.  Leave empty to skip saving.  When set, a companion
       data file is also configured automatically by the yobx wrappers.
       Corresponds to the ``optimized_model_filepath`` parameter of the
       yobx wrappers.
   * - ``profile_file_prefix``
     - ``"onnxruntime_profile_"``
     - Prefix of the JSON file written when ``enable_profiling`` is
       ``True``.  ONNX Runtime appends a timestamp and ``.json`` suffix.
   * - ``use_deterministic_compute``
     - ``False``
     - When ``True``, forces ONNX Runtime to use deterministic algorithms
       everywhere, at the cost of potentially lower performance.  Useful
       for reproducible debugging.
   * - ``use_per_session_threads``
     - ``True``
     - When ``True``, each session owns its own thread pool.  Set to
       ``False`` to share a global thread pool across sessions, which
       reduces thread-creation overhead when many short-lived sessions are
       created.

Methods
=======

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Method
     - Description
   * - ``add_session_config_entry(key, value)``
     - Sets a single session configuration entry as a key/value string pair.
       This is the primary way to pass advanced options that are not exposed
       as named properties.  See :ref:`l-ort-session-config-entries` for a
       selection of commonly used keys.  Used by the yobx wrappers to set
       ``session.disable_aot_function_inlining`` and the external-data
       file name for ``optimized_model_filepath``.
   * - ``get_session_config_entry(key)``
     - Returns the string value previously set with
       ``add_session_config_entry``.
   * - ``add_free_dimension_override_by_name(dim_name, value)``
     - Binds the symbolic input dimension named ``dim_name`` to a concrete
       integer ``value`` for this session.  Allows ONNX Runtime to
       specialize and optimize the model for a fixed shape without
       re-exporting.
   * - ``add_free_dimension_override_by_denotation(denotation, value)``
     - Like ``add_free_dimension_override_by_name`` but identifies the
       dimension by its ONNX *denotation* string (e.g.
       ``"DATA_BATCH"``).
   * - ``add_initializer(name, ort_value)``
     - Shares a pre-allocated :class:`onnxruntime.OrtValue` as a named
       initializer with the session.  Avoids copying large weight tensors
       into the session at load time.
   * - ``add_external_initializers(names, ort_values)``
     - Supplies a list of external initializers (by name and
       ``OrtValue``) that override the initializers stored in the ONNX
       model.  Useful for sharing weights across sessions.
   * - ``add_external_initializers_from_files_in_memory(filenames, buffers, lengths)``
     - Like ``add_external_initializers`` but reads the initializer data
       from in-memory byte buffers that correspond to external data files.
   * - ``add_provider(provider_name, options)``
     - Adds an explicit execution provider with a string options mapping.
       Prefer passing the ``providers`` list to
       :class:`onnxruntime.InferenceSession` directly; this method is
       useful when building options programmatically.
   * - ``add_provider_for_devices(ort_ep_devices, options)``
     - Like ``add_provider`` but identifies the provider through a sequence
       of ``OrtEpDevice`` descriptors returned by the device-selection API.
   * - ``has_providers()``
     - Returns ``True`` if the ``SessionOptions`` object already has
       execution providers, ``OrtEpDevices``, or policies configured.
   * - ``register_custom_ops_library(path)``
     - Registers a shared library (``*.so`` / ``*.dll``) that implements
       custom ONNX operator kernels required by the model.  Must be called
       before the :class:`onnxruntime.InferenceSession` is created.
   * - ``set_load_cancellation_flag(cancel)``
     - When ``cancel=True``, requests that an in-progress session load be
       aborted.  Useful for implementing load timeouts in long-running
       services.
   * - ``set_provider_selection_policy(policy)``
     - Sets an automatic EP selection policy
       (``OrtExecutionProviderDevicePolicy``) that ONNX Runtime uses to
       pick the best execution provider at runtime.
   * - ``set_provider_selection_policy_delegate(callable)``
     - Provides a Python callable that ONNX Runtime calls to choose an
       execution provider.  The callable receives the candidate
       ``OrtEpDevice`` list and must return the chosen provider name and
       options.

.. _l-ort-graph-optimization-level:

GraphOptimizationLevel
======================

``onnxruntime.GraphOptimizationLevel`` is an enum that controls which
optimization passes run before model execution.

.. list-table::
   :widths: 30 10 60
   :header-rows: 1

   * - Name
     - Value
     - Description
   * - ``ORT_DISABLE_ALL``
     - ``0``
     - Disables all graph optimizations.  The model is executed exactly as
       exported.  Use this when diagnosing incorrect results caused by
       graph rewrites, or when benchmarking the unoptimized graph.
   * - ``ORT_ENABLE_BASIC``
     - ``1``
     - Enables constant folding, redundant node elimination, and other
       cheap algebraic simplifications that are always safe.
   * - ``ORT_ENABLE_EXTENDED``
     - ``2``
     - Adds more aggressive fusions on top of ``ORT_ENABLE_BASIC`` such as
       GELU, attention, and layer-normalization fusions.
   * - ``ORT_ENABLE_LAYOUT``
     - ``3``
     - Adds layout-transformation optimizations (e.g. NCHWc) on top of
       ``ORT_ENABLE_EXTENDED``.
   * - ``ORT_ENABLE_ALL``
     - ``99``
     - Enables all optimizations, including the ones that depend on the
       selected execution provider.  This is the default.

In the yobx wrappers the ``graph_optimization_level`` parameter also accepts
a plain ``bool``: ``True`` maps to ``ORT_ENABLE_ALL`` and ``False`` maps to
``ORT_DISABLE_ALL``.

.. _l-ort-execution-mode:

ExecutionMode
=============

``onnxruntime.ExecutionMode`` controls whether independent graph nodes are
run sequentially or concurrently.

.. list-table::
   :widths: 30 10 60
   :header-rows: 1

   * - Name
     - Value
     - Description
   * - ``ORT_SEQUENTIAL``
     - ``0``
     - Nodes are executed one after another in topological order.  This is
       the default and is usually fastest for single-batch inference because
       it avoids thread-synchronization overhead.
   * - ``ORT_PARALLEL``
     - ``1``
     - Independent nodes may run concurrently on different threads.  Use
       together with ``inter_op_num_threads`` to set the thread count.
       Most beneficial when the graph contains wide parallelism (many
       independent branches) and the hardware has many cores.

.. _l-ort-execution-order:

ExecutionOrder
==============

``onnxruntime.ExecutionOrder`` determines the scheduling order for nodes
in the execution plan.

.. list-table::
   :widths: 30 10 60
   :header-rows: 1

   * - Name
     - Value
     - Description
   * - ``DEFAULT``
     - ``0``
     - Nodes are scheduled in the default topological order computed by
       ONNX Runtime.
   * - ``PRIORITY_BASED``
     - ``1``
     - Nodes are scheduled according to a priority that ONNX Runtime
       assigns to minimize peak memory usage while maximizing throughput.
   * - ``MEMORY_EFFICIENT``
     - ``2``
     - Nodes are scheduled to minimize peak memory consumption, at the
       potential cost of some throughput.

.. _l-ort-session-config-entries:

Session Configuration Entries
==============================

``add_session_config_entry(key, value)`` accepts string key/value pairs for
advanced options.  The following table lists a selection of commonly used
keys.

.. list-table::
   :widths: 45 15 40
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``session.disable_aot_function_inlining``
     - ``"0"``
     - Set to ``"1"`` to prevent ONNX Runtime from inlining ONNX
       functions ahead-of-time.  Useful when the original function
       boundaries are needed for debugging or profiling.  Corresponds to
       the ``disable_aot_function_inlining`` parameter of the yobx
       wrappers.
   * - ``session.optimized_model_external_initializers_file_name``
     - ``""``
     - When ``optimized_model_filepath`` is set, this entry names the
       companion ``.data`` file that stores large initializers externally.
       Set automatically by the yobx wrappers.
   * - ``session.use_ort_model_bytes_directly``
     - ``"0"``
     - Set to ``"1"`` to use the raw model bytes in-place instead of
       copying them into internal storage.  Reduces memory when the caller
       controls the lifetime of the buffer.
   * - ``session.use_ort_model_bytes_for_initializers``
     - ``"0"``
     - Set to ``"1"`` to keep initializer data in the original model
       bytes rather than copying them.  Reduces peak memory during session
       creation for large models.
   * - ``session.load_model_format``
     - ``""``
     - Force the model format: ``"ORT"`` for the ORT flatbuffer format,
       ``"ONNX"`` for the protobuf format.  Leave empty for automatic
       detection.
   * - ``session.save_model_format``
     - ``""``
     - Controls the format used when saving the optimized model
       (``optimized_model_filepath``).  ``"ORT"`` saves as a flatbuffer,
       ``"ONNX"`` saves as a protobuf.

Example
=======

The following snippet shows how to configure a session that disables all
graph optimizations, enables profiling, and restricts execution to two
intra-op threads:

.. code-block:: python

    import onnxruntime

    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.enable_profiling = True
    opts.profile_file_prefix = "/tmp/my_model_profile_"
    opts.intra_op_num_threads = 2

    sess = onnxruntime.InferenceSession(
        "model.onnx",
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )

The same can be achieved with the yobx wrappers by passing individual
keyword arguments:

.. code-block:: python

    from yobx.reference import OnnxruntimeEvaluator

    evaluator = OnnxruntimeEvaluator(
        "model.onnx",
        graph_optimization_level=False,   # False → ORT_DISABLE_ALL
        enable_profiling=True,
    )

.. seealso::

    :ref:`l-design-evaluator` — overview of the three evaluators provided
    by ``yobx`` and when to use each one.

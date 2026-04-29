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
advanced session options that are not exposed as named properties.  The full
set of recognised keys is defined in the ONNX Runtime source file
``include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h``.
The tables below enumerate all keys, grouped by prefix.

``session.*`` keys
------------------

.. list-table::
   :widths: 55 10 35
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``session.disable_prepacking``
     - ``"0"``
     - Set to ``"1"`` to disable pre-packing of constant initializers.
       Pre-packing rearranges weights at load time for faster kernels;
       disabling it reduces session-creation time at the cost of slower
       inference.
   * - ``session.use_env_allocators``
     - ``"0"``
     - Set to ``"1"`` to use allocators registered in the
       ``OrtEnv`` instead of per-session allocators.  Allows allocator
       sharing across sessions.
   * - ``session.load_model_format``
     - ``""``
     - Force the model format: ``"ORT"`` for the ORT flatbuffer format,
       ``"ONNX"`` for the protobuf format.  Leave empty for automatic
       detection based on file extension or byte signature.
   * - ``session.save_model_format``
     - ``""``
     - Controls the format used when saving the optimized model
       (``optimized_model_filepath``).  ``"ORT"`` saves as a flatbuffer,
       ``"ONNX"`` saves as a protobuf.  Leave empty for automatic detection
       based on the file extension.
   * - ``session.set_denormal_as_zero``
     - ``"0"``
     - Set to ``"1"`` to enable flush-to-zero and denormal-as-zero in
       floating-point arithmetic for all threads in the session thread
       pool.  Can improve performance on some hardware but may reduce
       accuracy for models that rely on denormal values.
   * - ``session.disable_quant_qdq``
     - ``"0"``
     - Set to ``"1"`` to disable QDQ (QuantizeLinear/DequantizeLinear)
       fusion optimizations.  Defaults to ``"1"`` automatically when the
       DirectML EP is registered.
   * - ``session.disable_qdq_constant_folding``
     - ``"0"``
     - Set to ``"1"`` to prevent DequantizeLinear nodes from being
       individually constant-folded, even when
       ``session.disable_quant_qdq`` is ``"1"``.  Useful for EPs such
       as WebNN that disable QDQ fusion but still need the original DQ/Q
       nodes.
   * - ``session.disable_double_qdq_remover``
     - ``"0"``
     - Set to ``"1"`` to keep the middle two nodes in
       ``Q→(DQ→Q)→DQ`` patterns instead of removing them.
   * - ``session.enable_quant_qdq_cleanup``
     - ``"0"``
     - Set to ``"1"`` to remove residual ``Q→DQ`` pairs after all QDQ
       handling is complete.  Can improve performance but may affect
       accuracy; test carefully.  Available since ORT 1.11.
   * - ``session.disable_aot_function_inlining``
     - ``"0"``
     - Set to ``"1"`` to prevent ONNX Runtime from inlining ONNX
       functions ahead-of-time.  Useful when function boundaries are
       needed for debugging or profiling.  Corresponds to the
       ``disable_aot_function_inlining`` parameter of the yobx wrappers.
   * - ``session.graph_optimizations_loop_level``
     - ``"1"``
     - Controls whether graph optimizations run in a feedback loop.
       ``"0"`` = single pass; ``"1"`` = loop re-runs if Level 4
       optimizations were applied; ``"2"`` = loop re-runs if any
       Level 2+ optimization was applied.
   * - ``session.use_device_allocator_for_initializers``
     - ``"0"``
     - Set to ``"1"`` to allocate initialized tensor memory through the
       device allocator (i.e., ``malloc``/``new``) instead of the
       session arena.
   * - ``session.inter_op.allow_spinning``
     - ``"1"``
     - Set to ``"0"`` to make inter-op threads block immediately when
       idle instead of spinning.  Reduces CPU utilization at the cost of
       potentially higher latency.  Defaults to ``"0"`` in client/on-device
       builds (``ORT_CLIENT_PACKAGE_BUILD``).
   * - ``session.intra_op.allow_spinning``
     - ``"1"``
     - Same as ``session.inter_op.allow_spinning`` but for intra-op
       threads.
   * - ``session.intra_op.spin_duration_us``
     - *(iteration-based)*
     - Duration in microseconds that intra-op threads spin before
       blocking.  Requires ``session.intra_op.allow_spinning`` to be
       enabled.  Typical range: ``500``–``2000``.
   * - ``session.inter_op.spin_duration_us``
     - *(iteration-based)*
     - Same as ``session.intra_op.spin_duration_us`` but for inter-op
       threads.
   * - ``session.intra_op.spin_backoff_max``
     - ``"1"``
     - Maximum exponential-backoff cap for the intra-op spin loop.
       Values ≥ 2 reduce CPU load during spinning.  Clamped to 64.
   * - ``session.inter_op.spin_backoff_max``
     - ``"1"``
     - Same as ``session.intra_op.spin_backoff_max`` but for inter-op
       threads.
   * - ``session.use_ort_model_bytes_directly``
     - ``"0"``
     - Set to ``"1"`` to use the raw in-memory model bytes without
       copying them.  The caller must keep the buffer alive for the
       lifetime of the session.
   * - ``session.use_ort_model_bytes_for_initializers``
     - ``"0"``
     - Set to ``"1"`` to read initializer data directly from the
       flatbuffer bytes (requires
       ``session.use_ort_model_bytes_directly``).  Reduces peak memory
       during loading.
   * - ``session.qdqisint8allowed``
     - ``"0"``
     - Set to ``"1"`` when exporting an ORT format model for use on ARM
       platforms (enables INT8 QDQ).  Available since ORT 1.11.
   * - ``session.x64quantprecision``
     - ``"0"``
     - Set to ``"1"`` to use U8U8 (instead of U8S8) matrix multiplication
       on x64 platforms with AVX2/AVX512 to avoid overflow.  Slower but
       more numerically correct.
   * - ``session.dynamic_block_base``
     - *(disabled)*
     - Set to a positive integer (e.g. ``"4"``) to enable dynamic
       block-sizing for the thread pool.  Helps reduce E2E latency
       variance on wide-parallelism graphs.  Available since ORT 1.11.
   * - ``session.force_spinning_stop``
     - ``"0"``
     - Set to ``"1"`` to force thread-pool threads to stop spinning
       immediately when the last concurrent ``Run()`` call returns.
       Reduces idle CPU usage between infrequent requests.
   * - ``session.strict_shape_type_inference``
     - ``"0"``
     - Set to ``"1"`` to turn shape/type inference inconsistencies into
       hard failures instead of logged warnings.
   * - ``session.allow_released_opsets_only``
     - ``"0"``
     - Set to ``"1"`` to reject models using opsets newer than the latest
       released version.  Useful to catch accidental use of pre-release
       opsets in production.
   * - ``session.node_partition_config_file``
     - ``""``
     - Path to a file that specifies which nodes are assigned to which
       execution providers (logic-stream partitioning).
   * - ``session.intra_op_thread_affinities``
     - ``""``
     - Semicolon-separated CPU affinity specification for intra-op
       threads.  Example: ``"1,2,3;4,5"`` pins thread 0 to CPUs 1–3 and
       thread 1 to CPUs 4–5.  The number of entries must equal
       ``intra_op_num_threads - 1``.
   * - ``session.debug_layout_transformation``
     - ``"0"``
     - Set to ``"1"`` to dump intermediate ONNX models during layout
       transformation (e.g. NHWC conversion for NNAPI/XNNPACK/QNN EPs).
       Intended for developer debugging only.
   * - ``session.disable_cpu_ep_fallback``
     - ``"0"``
     - Set to ``"1"`` to prevent unsupported nodes from falling back to
       the CPU EP.  Session creation fails if the selected EP cannot
       handle all nodes.  Incompatible with explicitly adding the CPU EP.
   * - ``session.optimized_model_external_initializers_file_name``
     - ``""``
     - When ``optimized_model_filepath`` is set, this entry names the
       companion ``.data`` file for large initializers stored externally.
       Set automatically by the yobx wrappers.
   * - ``session.optimized_model_external_initializers_min_size_in_bytes``
     - ``"1024"``
     - Minimum initializer size in bytes above which initializers are
       placed in the external data file during serialization.
   * - ``session.model_external_initializers_file_folder_path``
     - ``""``
     - Folder path for external data files when loading a model from a
       memory buffer.  All external data files must reside in this folder.
   * - ``session.save_external_prepacked_constant_initializers``
     - ``"0"``
     - Set to ``"1"`` to write pre-packed constant initializers to an
       external data file.  Allows memory-mapping them on load, reducing
       heap usage for large models.
   * - ``session.collect_node_memory_stats_to_file``
     - *(not set)*
     - Full path to a CSV file where per-node memory statistics
       (initializer size, dynamic output sizes, temp allocations) are
       written.  Useful for estimating runtime memory requirements.
   * - ``session.resource_cuda_partitioning_settings``
     - ``""``
     - Composite ``"memory_limit_kb,stats_file"`` string enabling
       capacity-aware partitioning for the CUDA EP.
   * - ``session.layer_assignment_settings``
     - ``""``
     - Semicolon-separated per-device annotation strings that guide node
       assignment during partitioning, matched against node metadata
       ``layer_ann`` entries.
   * - ``session.qdq_matmulnbits_accuracy_level``
     - ``"4"``
     - Accuracy level used when converting ``DQ + MatMul`` to
       ``MatMulNBits``.  See the ``MatMulNBits`` op schema for allowed
       values.
   * - ``session.qdq_matmulnbits_block_size``
     - ``"0"`` (→ 32)
     - Block size for the ``DQ + MatMul → MatMulNBits`` conversion.
       ``"0"`` uses the default of 32; ``"-1"`` picks the largest
       power-of-2 ≤ min(K, 256) that minimizes padding.
   * - ``session.enable_dq_matmulnbits_fusion``
     - ``"0"``
     - Set to ``"1"`` to enable the ``DQ → MatMulNBits`` fusion graph
       transformer.  Typically enabled automatically by the
       NvTensorRTRTX EP.
   * - ``session.disable_model_compile``
     - ``"0"``
     - Set to ``"1"`` to fail session creation if any EP needs to compile
       the model (i.e. require a pre-compiled EPContext model).
   * - ``session.fail_on_suboptimal_compiled_model``
     - ``"0"``
     - Set to ``"1"`` to fail session creation when the compiled model
       compatibility is ``SUPPORTED_PREFER_RECOMPILATION`` (suboptimal).
   * - ``session.record_ep_graph_assignment_info``
     - ``"0"``
     - Set to ``"1"`` to record which nodes were assigned to which EPs.
       Retrieve the information via ``Session_GetEpGraphAssignmentInfo()``.

``optimization.*`` keys
-----------------------

.. list-table::
   :widths: 55 10 35
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``optimization.enable_gelu_approximation``
     - ``"0"``
     - Set to ``"1"`` to enable the fast GELU approximation in graph
       optimization.  May change inference results slightly.
   * - ``optimization.enable_cast_chain_elimination``
     - ``"0"``
     - Set to ``"1"`` to enable elimination of chains of Cast nodes.
       May change inference results in edge cases.
   * - ``optimization.disable_specified_optimizers``
     - ``""``
     - Comma-separated list of optimizer names to skip (e.g.
       ``"ConstantFolding,MatMulAddFusion"``).  Useful when a specific
       optimizer causes incorrect results or excessive load time.  Not
       available in minimal builds.
   * - ``optimization.minimal_build_optimizations``
     - ``""``
     - Controls how minimal-build optimizations are applied in a full
       build: ``"save"`` saves them when exporting an ORT model;
       ``"apply"`` only applies optimizations available in a minimal
       build; leave empty for all full-build optimizations.  Available
       since ORT 1.11.
   * - ``optimization.memory_optimizer_config``
     - ``""``
     - *(Training only)* Path to a JSON file describing memory
       optimization configurations (recompute subgraph patterns) for
       ``onnxruntime-training``.
   * - ``optimization.enable_memory_probe_recompute_config``
     - ``"0:0"``
     - *(Training only)* Integer pair controlling subgraph detection for
       memory-footprint reduction via recompute.

``ep.*`` keys
-------------

.. list-table::
   :widths: 55 10 35
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``ep.nnapi.partitioning_stop_ops``
     - *(default set)*
     - Comma-separated list of op types at which the NNAPI EP stops
       graph partitioning.  Set to ``""`` to disable stop-op exclusion
       entirely.
   * - ``ep.context_enable``
     - ``"0"``
     - Set to ``"1"`` to enable the EPContext feature: after session
       creation the partitioned graph (with compiled EP context blobs)
       is saved to an ONNX file for reuse in future inference sessions,
       avoiding repeated compile overhead.
   * - ``ep.context_file_path``
     - *(original name + ``_ctx.onnx``)*
     - Path for the EPContext ONNX file written when
       ``ep.context_enable`` is ``"1"``.  Must be a file path, not a
       directory.
   * - ``ep.context_embed_mode``
     - ``"0"``
     - ``"0"`` stores the EP context blob in a separate file (path kept
       in the ONNX model); ``"1"`` embeds the blob directly inside the
       ONNX model.
   * - ``ep.context_node_name_prefix``
     - ``""``
     - Prefix added to EPContext node names to make them unique when
       multiple EPContext graphs are merged into one model.
   * - ``ep.share_ep_contexts``
     - ``"0"``
     - Set to ``"1"`` to share EP resources (e.g. compiled binaries)
       across sessions.
   * - ``ep.stop_share_ep_contexts``
     - ``"0"``
     - Set to ``"1"`` to stop sharing EP resources from this point on.
   * - ``ep.context_model_external_initializers_file_name``
     - ``""``
     - When generating an EPContext model and some nodes fall back to
       the CPU EP, this entry names the external data file into which
       all initializers are placed in the generated ONNX file.
   * - ``ep.enable_weightless_ep_context_nodes``
     - ``"0"``
     - Set to ``"1"`` to request that EPs create EPContext nodes without
       embedded weights (weights are provided as explicit inputs).
       Requires ``ep.context_enable`` to be ``"1"``.

``mlas.*`` keys
---------------

.. list-table::
   :widths: 55 10 35
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``mlas.enable_gemm_fastmath_arm64_bfloat16``
     - ``"0"``
     - Set to ``"1"`` to enable BFloat16-accelerated GEMM on ARM64
       (fastmath mode).
   * - ``mlas.use_lut_gemm``
     - ``"0"``
     - Set to ``"1"`` to use lookup-table-based GEMM kernels for
       quantized models when available.
   * - ``mlas.disable_kleidiai``
     - ``"0"``
     - Set to ``"1"`` to disable KleidiAI kernels even if the
       platform supports them.

Dynamic EP options (``ep.dynamic.*``)
--------------------------------------

These keys are intended for use with ``SetEpDynamicOptions`` and may be
changed at any time, not just at session creation.

.. list-table::
   :widths: 55 10 35
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``ep.dynamic.workload_type``
     - ``"Default"``
     - Scheduling-priority hint for the session workload.  ``"Default"``
       lets the OS choose; ``"Efficient"`` signals an
       efficiency-oriented, low-priority workload.
   * - ``ep.dynamic.qnn_htp_performance_mode``
     - ``"default"``
     - QNN HTP performance mode.  Allowed values: ``"burst"``,
       ``"balanced"``, ``"default"``, ``"high_performance"``,
       ``"high_power_saver"``, ``"low_balanced"``,
       ``"extreme_power_saver"``, ``"low_power_saver"``,
       ``"power_saver"``, ``"sustained_high_performance"``.

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

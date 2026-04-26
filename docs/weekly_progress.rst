.. _l-weekly-progress:

=================================
Weekly Progress Since the Start
=================================

This page summarises the major achievements in **yet-another-onnx-builder** (``yobx``)
week by week since the project was created on February 25, 2026.

.. contents::
   :local:
   :depth: 1

----

Week 1 — Feb 25 – Mar 1 (~199 commits)
=======================================

The project was bootstrapped from scratch.
The first day alone saw 41 commits establishing the repository layout, CI pipelines
(GitHub Actions for core, sklearn, torch, tensorflow, docs, style, spelling),
Codecov integration, black/ruff enforcement, and a README with badges.

The central focus was the ``yobx.xshape`` shape-inference engine.
Operators received dedicated implementations: ``TopK``, ``Resize``, ``Pad``,
``Gather``, ``Einsum``, ``NonZero``, ``GridSample``, ``LogSoftmax``,
``Softmax``, ``LpNormalization``, ``InstanceNormalization``, window functions
(``BlackmanWindow``, ``HannWindow``, ``HammingWindow``), and
``Squeeze``/``Unsqueeze`` with both attribute-based and tensor-based axes.
Every new path was immediately covered by unit tests.

Before the week ended, several higher-level building blocks were already in place:
the ``MiniOnnxBuilder`` fluent graph-builder, the ``ExtendedReferenceEvaluator``
(combining the ONNX reference implementation with OnnxRuntime),
the ``yobx.translate`` module that converts an ONNX model back to Python source
code, the ``LightGraphBuilder``, and the ``ExtendedModelContainer`` container
that wraps a model proto together with per-run statistics.
The ``xoptim`` graph-optimisation framework — with its pattern-matching engine —
was seeded by the end of this week.

----

Week 2 — Mar 2 – Mar 8 (~170 commits)
======================================

The week opened with a major refactor of the ``GraphBuilder`` class:
``FunctionOptions``, ``InferShapesOptions``, ``WrapDim``, ``WrapSym``, and
``InitializerInfo`` were extracted into dedicated modules.
An exhaustive unit-test suite was written for every public method of
``GraphBuilder`` (constants, shapes, types, sequences, dimension helpers,
subgraph inlining, …).

The first substantial **scikit-learn → ONNX** converter landed (``yobx.sklearn``),
initially covering ``DecisionTreeClassifier``/``Regressor`` and later extended to
``RandomForestClassifier``/``Regressor``, ``HistGradientBoosting*``,
``PCA``, ``KMeans``, and the full family of linear models.
A dedicated sklearn CI job was added with two scikit-learn versions (1.4 and 1.8).

Two alternative ``GraphBuilder`` back-ends were contributed:
``OnnxScriptGraphBuilder`` (bridging to the *onnxscript* IR) and
``SpoxGraphBuilder`` (bridging to *spox*), both implementing the new typed
``GraphBuilderExtendedProtocol``.

A proof-of-concept **TensorFlow/Keras → ONNX** converter appeared as well,
together with a first sphinx-gallery example.
The Sphinx theme was switched from *furo* to *piccolo_theme* and documentation
cross-references were switched from viewcode to GitHub source links.

----

Week 3 — Mar 9 – Mar 15 (~148 commits)
=======================================

The sklearn converter library reached near-complete coverage of the
scikit-learn API in a single sprint:
``SVM`` (SVC, SVR, LinearSVC, LinearSVR, NuSVC, NuSVR),
``NaiveBayes`` (Gaussian, Bernoulli, Multinomial, Complement, Categorical),
``VotingClassifier``/``Regressor``,
``StackingClassifier``/``Regressor``,
``ExtraTreesClassifier``/``Regressor``,
``BaggingClassifier``/``Regressor``,
``AdaBoostClassifier``/``Regressor``,
``GaussianProcessClassifier``/``Regressor``,
``QuantileTransformer``, ``PowerTransformer``, ``KBinsDiscretizer``,
``OneHotEncoder``, ``RobustScaler``, ``SplineTransformer``,
``GaussianMixture``, ``MultiOutputClassifier``/``Regressor``,
``FeatureUnion``, ``PLSRegression``, ``GradientBoostingClassifier``/``Regressor``,
``TruncatedSVD``, ``MaxAbsScaler``, ``KNNImputer``, ``IsotonicRegression``,
and many more.
**XGBoost** and **LightGBM** converters were also added.

On the torch side the aten-function interpreter matured, covering pool ops,
``addmm``, logical operators, ``argmin``/``argmax``, and trigonometric ops.
``LlamaAttention`` received a direct ONNX converter.

The ``validate_model`` function (and ``python -m yobx validate`` CLI command)
was introduced to give a structured report on the quality of an exported model.

The **SQL-to-ONNX converter** (``yobx.sql``) made its initial appearance,
translating SQL ``SELECT`` statements into ONNX graphs.

----

Week 4 — Mar 16 – Mar 22 (~116 commits)
========================================

The ``ExportArtifact`` / ``ExportReport`` pair became the unified return type of
every ``to_onnx`` function, replacing the raw ``ModelProto``.
The report can be serialised to a string or saved as a multi-sheet Excel workbook.

All ``to_onnx`` functions grew ``filename`` and ``verbose`` parameters to
save the model and report to disk automatically.

The ``yobx.sql`` converter expanded significantly: ``GROUP BY`` with true
per-group aggregation, subqueries in ``FROM``, multi-column ``JOIN`` keys,
and support for custom Python functions via numpy tracing.
A dedicated SQL gallery was added.

``lazyframe_to_onnx`` was implemented, letting a **Polars LazyFrame** execution
plan be compiled directly to ONNX.
``jax_to_concrete_function`` bridged the JAX → TF ``ConcreteFunction`` → ONNX
pipeline and enabled more JAX gallery examples.

``yobx.litert`` appeared as a pure-Python **LiteRT/TFLite → ONNX** converter.

The ``DataFrameTransformer`` (a traceable sklearn transformer with built-in ONNX
export) and ``xtracing`` (numpy-tracing-based ONNX export for
``FunctionTransformer``) rounded off the new data-manipulation backends.

The Sphinx theme was upgraded a final time to ``pydata_sphinx_theme``.

----

Week 5 — Mar 23 – Mar 29 (~113 commits)
========================================

The top-level ``yobx.to_onnx`` dispatcher was implemented, giving users a single
entry point that automatically routes to the right backend by inspecting the
input type at runtime.

``pivot_table`` export via the DataFrame tracer was contributed, together with
binary arithmetic operators between traced DataFrames and multi-DataFrame inputs.
The ``TracedDataFrame`` column dictionary was migrated to ``Dict[ColumnRef, TracedSeries]``
for stronger typing.

Cost analysis infrastructure landed: the ``ModelStatistics`` class,
the ``cost_inference`` module (per-operator FLOPs formulas), a ``stats`` CLI
sub-command, a FLOPs sheet in the Excel report, and a gallery example computing
the symbolic FLOPs of an attention block before and after optimisation.

**imbalanced-learn** converters were added
(``BalancedBaggingClassifier``, ``BalancedRandomForestClassifier``, ``RUSBoostClassifier``,
``EasyEnsembleClassifier``).
``category_encoders`` converters grew
(``OrdinalEncoder``, ``BinaryEncoder``, ``PolynomialEncoder``, ``TargetEncoder``).
**scikit-survival** converters appeared (``IPCRidge``, ``RandomSurvivalForest``).

``while_loop`` support was added to the torch ONNX converter, along with numerous
inplace-op fixes for the tracing exporter (``InplaceAdd``, ``InplaceSetItemMask``,
``InplaceSetItemSquare``).

A CI job for the litert and sql sub-packages was created.

----

Week 6 — Mar 30 – Apr 5 (~42 commits)
======================================

A lighter week in terms of commits but important for depth.
The brand-new **dispatch-level tracer** (``yobx/torch/new_tracing``) was seeded —
a ``__torch_dispatch__``-based approach building an FX graph at the ATen-operator
level, giving finer-grained control over dynamic shapes than the existing
``CustomTracer``.
``node.meta["stack_trace"]`` was populated during tracing to aid debugging.

On the existing tracing path, several hard edge-cases were fixed:
``ControlFlowCondNestedModule``, ``ControlFlowScanCDist2``,
``AtenAsStrided`` with dynamic shapes, ``ExportWithDimension0``.

JAX progress: ``sigmoid`` test enabled, dynamic-batch export,
``export_to_onnx_dynamic_shapes`` test.
LiteRT improved: all subgraphs are now merged into one ONNX model.
SQL: multiple DataFrames as direct outputs, ``copy()`` on ``TracedDataFrame``.

The ``TracingMode`` enum replaced the bare ``tracing: bool`` flag in
``ExportOptions``, making the API forward-compatible.

----

Week 7 — Apr 6 – Apr 12 (~67 commits)
======================================

A milestone week for the new-tracing path.
``CustomProxyShape``, ``CustomProxyBool``, and ``CustomProxyInt`` replaced
the ad-hoc ``_SafeShape`` / ``_SymGuardProxy`` helpers, giving first-class
proxy objects that participate in the FX graph without triggering
``GuardOnDataDependentSymNode`` errors.
``torch._check`` assertions are now registered as known conditions so that
symbolic shape guards inside the model body are resolved correctly.

The ``ConvertingLibrary`` enum (choosing between yobx, onnxscript, and spox
back-ends) and ``TracingMode.NEW_TRACING`` were exposed through ``ExportOptions``.

A comprehensive op-db–driven test suite was introduced, running
``common_methods_invocations`` against all supported dtype × operator combinations
(float32, float16, int32, int64, bfloat16) and publishing a per-operator
coverage page in the docs.

The ``return_optimize_report`` parameter was added to all ``to_onnx`` functions.
CI now shows a durations page plotting how long each CI workflow takes over time,
with per-workflow charts and a rolling average.

``ControlFlowCond`` with ``torch.cond`` was made codegen-safe by using callable
``get_attr`` nodes in the new-tracing path.

----

Week 8 — Apr 13 – Apr 19 (~27 commits)
========================================

A focused, quality-oriented week.
**TensorFlow CI** was repaired for JAX 0.10+: MLIR Python bindings are now parsed
correctly, ``ir.Module`` return types are handled, and the MLIR context is always
active during ``call_inlining``.

Over **30 new aten-function ONNX converters** were shipped:
``amin``, ``aminmax``, ``angle``, ``atan2``, ``std``/``std_mean``,
``bilinear``, ``xlogy``, ``logit``, ``nanmean``/``nansum``,
``log10``, ``log1p``, ``log2``, ``logaddexp``/``logaddexp2``,
``erfc``/``erfinv``, ``atleast_1d``/``2d``/``3d``,
``isclose``, ``isfinite``, ``isneginf``/``isposinf``,
``clamp_min_Tensor``/``clamp_max_Tensor``,
``count_nonzero``, ``mse_loss``, ``logical_xor``, ``ravel``, ``cartesian_prod``,
``aten_alias_copy``, ``addr``, ``dot``, ``exp2``,
and FFT-shift operators (``fft.fftshift``, ``fft.ifftshift``).

``DynamoInterpreter`` was renamed to ``FxGraphInterpreter`` to better reflect its
role.  A float32 cross-path comparison section was added to the op-coverage page.

The CI duration cache was restructured to use per-workflow CSV files for more
granular persistence.
``torch.autocast(enabled=True)`` was supported in ONNX export.

----

Week 9 — Apr 20 – Apr 25 (~36 commits, ongoing)
=================================================

The new-tracing path closed out a long backlog of failing evaluation cases:
``DynamicCacheInput``, ``ControlFlowNumelZero2``, ``ControlFlowShapeCheck``,
``ControlFlowScan2Carried``, ``SignatureShapeAsIndex``,
``InplaceSetItemEllipsis_1``/``2``, ``InplaceSetItemSquare``,
``CreateFromShape``, and ``ControlFlowCondNonZero`` are all now passing.

The ``XlaLayer`` class replaced bare dicts for StableHLO layer representation
in the XlaCallModule converter, making attribute access safer.
``TracingInt`` was integrated into ``GraphBuilder`` to handle symbolic integer
dimensions coming from the new-tracing path.

Two new fusion patterns landed for ``com.microsoft`` contrib ops:
**BiasSplitGelu** and **RelativePositionBias**.

Additional aten converters were contributed:
``tensor_split``, ``fliplr``/``flipud``, ``geqrf``,
``frac``, ``frexp``,
``linalg.det``, ``linalg.slogdet``, ``linalg.cross``, ``linalg.vecdot``,
``fmax``, ``fmin``, ``fmod``,
``mT``/``mH`` (matrix transpose/conjugate-transpose),
``heaviside``, ``signbit``, ``diag``/``diag_embed``,
``float_power``, ``true_divide``.

The CI was reorganised: targeted subfolder tests now run first on pull requests
and block the full suite on failure, and a reusable workflow detects which
``yobx/`` subfolder a PR touches.
A new workflow posts a comment listing impacted subfolders.

``validate_model`` was enriched with per-node-type statistics and a discrepancy
sheet in the Excel output.
Documentation now tracks and displays the five slowest Sphinx pages to build.

----

What Copilot Could Not Do (or Struggled With)
==============================================

The vast majority of the work in this project was driven by **GitHub Copilot**
as a coding agent.  This section records the cases where it fell short — either
opening a pull request but producing no working code, taking a wrong approach
that had to be discarded, or needing several attempts before landing a correct
solution.  The goal is transparency: understanding where an AI agent struggles
helps calibrate expectations and guides where human review is most valuable.

Instantly-Abandoned WIP Pull Requests
--------------------------------------

On three occasions Copilot opened a pull request but committed nothing beyond
the boilerplate template message.  These are the clearest sign that it lacked
enough context or that the problem was too open-ended to start without explicit
guidance:

* `PR #2009 <https://github.com/xadupre/yet-another-onnx-builder/pull/2009>`_ —
  *[WIP] Support ai.onnx.ml>=5 when computing statistics on trees* (Apr 25).
  The PR body was empty (just the issue template); no code was written and the
  PR was closed the same day it was opened.  After the issue was reformulated
  and restricted in scope, Copilot succeeded in
  `PR #2012 <https://github.com/xadupre/yet-another-onnx-builder/pull/2012>`_.

* `PR #1801 <https://github.com/xadupre/yet-another-onnx-builder/pull/1801>`_ —
  *[WIP] Fix std and std_mean for test_onnx_export_common_methods* (Apr 7).
  Same pattern: Copilot opened the draft with no commits.  A narrower issue was
  later resolved as part of the Week 8 batch of aten converters.

* `PR #1493 <https://github.com/xadupre/yet-another-onnx-builder/pull/1493>`_ —
  *[WIP] Fix CropLastDimensionWithTensorContent for the torch converter* (Mar 25).
  Again no code; the PR was discarded immediately.  The root fix required
  understanding the interplay between dynamic-shape tracing and tensor-content
  slicing — context that Copilot could not assemble from the issue text alone.

Wrong Approaches That Had to Be Discarded
------------------------------------------

A larger set of pull requests contained real code, but the design or algorithm
was incorrect and the PR had to be closed in favour of a fresh attempt:

* `PR #1962 <https://github.com/xadupre/yet-another-onnx-builder/pull/1962>`_ —
  *fix ControlFlowShapeCheck for the default exporter* (Apr 23).
  The patch covered only 6 % of the lines it touched and the test it added did
  not actually exercise the fixed code paths.  The issue was later resolved
  correctly in the new-tracing path.

* `PR #1798 <https://github.com/xadupre/yet-another-onnx-builder/pull/1798>`_ —
  *Fix AtenAsStrided export for new-tracing with dynamic shapes* (Apr 6–7).
  Copilot tried to patch ``TracingShape.from_existing_shape`` and
  ``DynamoInterpreter._get_tensor_shape``, but the multi-file interaction was
  too subtle; the draft was discarded and the fix was absorbed into a later
  refactor.

* `PR #1709 <https://github.com/xadupre/yet-another-onnx-builder/pull/1709>`_ —
  *Propagate symbolic TracingShape through operations in new dispatch tracer*
  (Apr 3).  An ambitious draft adding ``is_symbolic`` properties and
  ``_infer_output_tracing_shape`` helpers.  The approach added fragile
  instance-level state that conflicted with subsequent design choices; the
  PR was dropped and the problem was solved differently.

* `PR #1690 <https://github.com/xadupre/yet-another-onnx-builder/pull/1690>`_ —
  *Extend GraphBuilderTorchProtocol with missing make_* methods* (Apr 2).
  The protocol extension itself was correct but was not the right abstraction
  layer; the PR was closed without merge.

* `PR #1682 <https://github.com/xadupre/yet-another-onnx-builder/pull/1682>`_ —
  *CustomProxyInt: statically resolve comparisons between offset-derived
  proxies* (Apr 1).  Root/offset tracking was added to ``CustomProxyInt``,
  but the approach introduced edge cases that broke other comparison paths.
  The issue was later fixed differently.

* `PR #1673 <https://github.com/xadupre/yet-another-onnx-builder/pull/1673>`_ —
  *Fix ControlFlowNumelZero2 tracing* (Apr 1).  Copilot added
  ``can_be_null``/``only_positive`` flags to ``CustomProxy.numel()`` but the
  flags were not threaded correctly through the shape-guard machinery; the PR
  had 0 % patch coverage and was discarded.  ``ControlFlowNumelZero2`` was
  eventually fixed in
  `PR #1933 <https://github.com/xadupre/yet-another-onnx-builder/pull/1933>`_
  using a completely different strategy.

Features That Needed Multiple Attempts
---------------------------------------

Some features required two or three separate PRs before a mergeable
implementation was produced:

* **DataFrame gallery example** — three attempts:
  `PR #1590 <https://github.com/xadupre/yet-another-onnx-builder/pull/1590>`_,
  `PR #1609 <https://github.com/xadupre/yet-another-onnx-builder/pull/1609>`_,
  `PR #1632 <https://github.com/xadupre/yet-another-onnx-builder/pull/1632>`_
  (Mar 26–29).  Each attempt produced a runnable example, but the result was
  either incomplete, structurally inconsistent with the existing gallery, or a
  near-duplicate of a prior attempt.

* **Top-level ``to_onnx`` dispatcher** — two attempts in rapid succession
  (`PR #1394 <https://github.com/xadupre/yet-another-onnx-builder/pull/1394>`_
  and the merged implementation in Week 5).  The first attempt put too much
  logic in the dispatcher itself and duplicated code from framework-specific
  modules.

* **``ControlFlowNumelZero2``** — two failed PRs (#1673, and an earlier
  exploratory one) before the fix landed in PR #1933.

Graph-Optimisation Patterns Abandoned Mid-Implementation
---------------------------------------------------------

Several ``xoptim`` pattern proposals were opened as pull requests but closed
without merging because the correctness conditions were too hard to specify
precisely:

* `PR #1659 <https://github.com/xadupre/yet-another-onnx-builder/pull/1659>`_ —
  *Add ConstantOfShapeIdentityPattern* to fold ``x + zeros_like(x)`` → ``x``.
  The pattern is mathematically sound but required shape-equality guards that
  the proposed implementation checked only approximately.

* `PR #1645 <https://github.com/xadupre/yet-another-onnx-builder/pull/1645>`_ —
  *Add CustomTracer.remove_tests()* to prune dead comparison nodes.  The method
  worked for the motivating test case but was too aggressive: it removed nodes
  whose results were indirectly used through boolean short-circuit chains.

* `PR #1448 <https://github.com/xadupre/yet-another-onnx-builder/pull/1448>`_ —
  *Add* ``op_types`` *static attribute to* ``PatternOptimization``.  The
  implementation had 100 % patch coverage and no test failures, but was closed
  because the design did not account for patterns that match on computed
  node types (e.g. custom ops) rather than fixed strings.

* `PR #1263 <https://github.com/xadupre/yet-another-onnx-builder/pull/1263>`_ —
  *BinaryUnsqueezeExpandPattern*.  Correct for the common case but produced
  wrong shapes when broadcasting rules applied across more than one axis.

Common Themes
-------------

Reviewing these cases, a few recurring difficulties stand out:

* **Multi-file interactions with tight invariants** — Bugs at the boundary
  between ``new_tracing``, ``GraphBuilder``, and the ONNX interpreter require
  holding a large amount of invariant context simultaneously.  Copilot tends to
  fix the symptom in one file while missing a side-effect in another.

* **Correctness conditions that are easy to state but hard to check** —
  "``x + zeros_like(x)`` is always ``x``" sounds simple, but verifying that
  the shapes are genuinely compatible requires traversing the ONNX graph in ways
  that Copilot did not fully implement.

* **Iterative discovery tasks** — When the issue asked Copilot to *find all
  cases* of a problem and fix them, it often found the most obvious one and
  missed the rest.  These tasks benefited most from explicit guidance (e.g. a
  list of failing test cases to use as a checklist).

* **Open-ended documentation / gallery examples** — Asking Copilot to write a
  "good example" for a feature produced structurally correct but pedagogically
  weak examples.  Multiple rounds of feedback were needed to arrive at examples
  that fit naturally into the existing gallery style.

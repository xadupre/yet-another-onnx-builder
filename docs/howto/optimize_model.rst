.. _l-howto-optimize-model:

Optimize an existing ONNX model
===============================

This page answers common *"how do I…"* questions for optimizing an
existing :class:`onnx.ModelProto` with the pattern-based optimizer
provided by :mod:`yobx.xoptim`.

The optimizer searches for sequences of nodes matching predefined
patterns and rewrites them into equivalent — but more efficient — ones
(constant folding, operator fusion, transpose simplification, …)
without changing the model inputs or outputs.

----

How to optimize a model with the default patterns
--------------------------------------------------

Load the model into a :class:`yobx.xbuilder.GraphBuilder` and call
:meth:`to_onnx <yobx.xbuilder.GraphBuilder.to_onnx>` with
``optimize=True``. The default list of patterns is applied automatically.

.. runpython::
    :showcode:

    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.xbuilder import GraphBuilder
    from yobx.doc import demo_mlp_model

    onx = demo_mlp_model("temp_doc_optimize_mlp.onnx")
    print("--- before optimization ---")
    print(pretty_onnx(onx))

    gr = GraphBuilder(onx, infer_shapes_options=True)
    opt_onx = gr.to_onnx(optimize=True)

    print("--- after optimization ---")
    print(pretty_onnx(opt_onx))

The two ``MatMul + Add`` sequences are fused into ``Gemm`` nodes by the
default patterns.

----

How to choose which patterns to apply
-------------------------------------

Use :class:`yobx.xbuilder.OptimizationOptions` to enable or disable
patterns. Patterns can be passed as a list of names separated by
commas, or as a list of pattern instances.

There exist a few predefined lists:

* ``default`` — patterns using only standard ONNX operators.
* ``onnxruntime`` — patterns specific to :epkg:`onnxruntime`,
  the resulting model may use ``com.microsoft`` operators and may
  only run with :epkg:`onnxruntime`.
* ``default+onnxruntime`` — both lists combined.

.. runpython::
    :showcode:

    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.xbuilder import GraphBuilder, OptimizationOptions
    from yobx.doc import demo_mlp_model

    onx = demo_mlp_model("temp_doc_optimize_mlp.onnx")

    gr = GraphBuilder(
        onx,
        infer_shapes_options=True,
        optimization_options=OptimizationOptions(
            patterns="MatMulAdd,GemmTranspose", verbose=0
        ),
    )
    opt_onx = gr.to_onnx(optimize=True)
    print(pretty_onnx(opt_onx))

The full list of available patterns is documented at
:ref:`l-design-pattern-optimizer-patterns`.

----

How to inspect what the optimizer did
-------------------------------------

Calling :meth:`optimize <yobx.xoptim.GraphBuilderPatternOptimization.optimize>`
on the builder returns one row per applied rewriting, with timings and
the number of nodes added or removed. The rows can be aggregated with
:epkg:`pandas` to get a per-pattern summary.

.. runpython::
    :showcode:

    import pandas
    from yobx.xbuilder import GraphBuilder, OptimizationOptions
    from yobx.doc import demo_mlp_model

    onx = demo_mlp_model("temp_doc_optimize_mlp.onnx")

    gr = GraphBuilder(
        onx,
        infer_shapes_options=True,
        optimization_options=OptimizationOptions(patterns="default"),
    )
    stat = gr.optimize()

    df = pandas.DataFrame(stat)
    for c in ["added", "removed"]:
        df[c] = df[c].fillna(0).astype(int)
    agg = df.groupby("pattern")[["added", "removed", "time_in"]].sum()
    print(agg[(agg["added"] > 0) | (agg["removed"] > 0)])

Setting ``verbose=1`` (or higher) on
:class:`OptimizationOptions <yobx.xbuilder.OptimizationOptions>` prints
the same information while the optimization runs.

----

.. _l-howto-optimize-model-vs-onnxscript:

Comparison with onnxscript rewriter
-----------------------------------

The :epkg:`onnxscript` package ships its own pattern-based rewriter
(`Pattern-based Rewrite Using Rules With onnxscript
<https://microsoft.github.io/onnxscript/tutorial/rewriter/rewrite_patterns.html>`_).
Both tools serve the same purpose — rewriting an ONNX graph by matching
sub-graphs and replacing them — but they differ in API and scope.

A typical onnxscript rewriter looks like this:

.. code-block:: python

    import onnx
    from onnxscript.rewriter import pattern, rewrite

    op = pattern.onnxop


    def matmul_add_pattern(op, x, w, b):
        t = op.MatMul(x, w)
        return op.Add(t, b)


    def gemm_replacement(op, x, w, b):
        return op.Gemm(x, w, b)


    rule = pattern.RewriteRule(matmul_add_pattern, gemm_replacement)

    onx = onnx.load("model.onnx")
    new_onx = rewrite(onx, pattern_rewrite_rules=[rule])

The equivalent with ``yobx`` reuses the built-in ``MatMulAdd`` pattern
already shipped with the optimizer:

.. code-block:: python

    import onnx
    from yobx.xbuilder import GraphBuilder, OptimizationOptions

    onx = onnx.load("model.onnx")
    gr = GraphBuilder(
        onx,
        infer_shapes_options=True,
        optimization_options=OptimizationOptions(patterns="MatMulAdd"),
    )
    new_onx = gr.to_onnx(optimize=True)

A user-defined rewrite is written as a subclass of
:class:`EasyPatternOptimization <yobx.xoptim.EasyPatternOptimization>`
(declarative match + apply) or
:class:`PatternOptimization <yobx.xoptim.PatternOptimization>` (manual
``match`` / ``apply``). The example below re-implements the same
``MatMul + Add → Gemm`` rewrite with the declarative API and plugs it
into the optimizer alongside the default patterns:

.. runpython::
    :showcode:

    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.xbuilder import GraphBuilder, OptimizationOptions
    from yobx.xoptim import EasyPatternOptimization
    from yobx.doc import demo_mlp_model


    class MatMulAddToGemmPattern(EasyPatternOptimization):
        """Fuses ``Add(MatMul(x, w), b)`` into ``Gemm(x, w, b)``."""

        def match_pattern(self, g: "GraphBuilder", x, w, b):
            t = g.op.MatMul(x, w)
            return g.op.Add(t, b)

        def apply_pattern(self, g: "GraphBuilder", x, w, b):
            return g.op.Gemm(x, w, b)


    onx = demo_mlp_model("temp_doc_optimize_mlp.onnx")

    gr = GraphBuilder(
        onx,
        infer_shapes_options=True,
        optimization_options=OptimizationOptions(
            patterns=[MatMulAddToGemmPattern()],
        ),
    )
    opt_onx = gr.to_onnx(optimize=True)
    print(pretty_onnx(opt_onx))

The ``patterns`` argument of
:class:`OptimizationOptions <yobx.xbuilder.OptimizationOptions>`
accepts a list mixing predefined names and user-defined instances —
e.g. ``patterns=["default", MatMulAddToGemmPattern()]`` to combine a
custom rewrite with the built-in catalogue. For rewrites whose
applicability depends on shapes, dtypes or attributes (a typical
*"custom op"* fusion that targets ``com.microsoft`` or another
domain), subclass
:class:`PatternOptimization <yobx.xoptim.PatternOptimization>`
directly: see the worked *NotNot* example in
:ref:`l-design-pattern-optimizer`.

Main differences:

* **Out-of-the-box catalog** — ``yobx`` ships a curated list of patterns
  enabled by ``patterns="default"`` (constant folding, transpose
  simplification, MatMul/Gemm fusions, …). Equivalent rules with
  :epkg:`onnxscript` must be assembled by the user from the rewriter
  tutorial. The list of patterns shipped with ``yobx`` is in
  :ref:`l-design-pattern-optimizer-patterns`.
* **Granularity of the API** — :epkg:`onnxscript` rewrite rules are
  expressed as two ONNX functions (match + replacement). ``yobx``
  also supports this style through
  :class:`OnnxEasyPatternOptimization
  <yobx.xoptim.patterns_api.OnnxEasyPatternOptimization>`, but it
  additionally exposes a more imperative API based on
  :class:`PatternOptimization <yobx.xoptim.PatternOptimization>` where
  ``match`` and ``apply`` are arbitrary Python methods. This is
  convenient when the rewriting condition depends on shapes, dtypes or
  attributes, which is harder to express purely structurally.
* **Shape and type information** — the ``yobx`` matcher runs after
  shape inference and can therefore filter matches on tensor ranks,
  shapes and dtypes through the ``g`` argument of ``match``. The
  :epkg:`onnxscript` rewriter relies on a ``check`` callback for
  similar purposes.
* **Diagnostics** — every iteration records statistics
  (``added`` / ``removed`` nodes, ``time_in`` per pattern), which makes
  it easy to spot which patterns actually fire and which ones are
  expensive. See *How to inspect what the optimizer did* above.
* **Integration with conversion** — ``yobx`` runs the same optimizer
  automatically at the end of every ``to_onnx`` call (for example
  :func:`yobx.torch.to_onnx`, :func:`yobx.sklearn.to_onnx`,
  :func:`yobx.sql.to_onnx`). Optimizing a pre-existing ONNX file is
  exactly the same code path, just starting from an
  :class:`onnx.ModelProto` instead of a framework model.

----

A note on performance
---------------------

The matching algorithm is roughly :math:`O(N \cdot P \cdot I)` in the
number of nodes ``N``, the number of patterns ``P`` and the number of
iterations ``I`` (see :ref:`l-design-pattern-optimizer`). Two design
choices keep the constant factor small in practice:

* Each pattern can declare its entry node operator type via
  :meth:`fast_op_type <yobx.xoptim.PatternOptimization.fast_op_type>`.
  When set, the optimizer indexes the graph once per iteration and
  only feeds the relevant nodes to the pattern, instead of iterating
  over the whole graph.
* The optimizer is incremental: at each iteration only nodes that
  could be affected by previously applied rewritings are revisited,
  and the loop stops as soon as no pattern fires.

In practice, optimizing a typical transformer-sized model with the
``default`` list of patterns runs in seconds on a single CPU core. The
``time_in`` column produced by :meth:`optimize
<yobx.xoptim.GraphBuilderPatternOptimization.optimize>` (see above)
gives a per-pattern budget that can be used to drop patterns whose
cost exceeds their benefit on a given model — this is exactly the
purpose of the ``DROPPATTERN`` environment variable documented in
:class:`GraphBuilderPatternOptimization
<yobx.xoptim.GraphBuilderPatternOptimization>`.

.. seealso::

    * :ref:`l-design-pattern-optimizer` — design and algorithm of the
      pattern optimizer.
    * :ref:`l-design-pattern-optimizer-patterns` — full list of
      patterns shipped with ``yobx``.
    * :epkg:`Pattern-based Rewrite Using Rules With onnxscript` — the
      onnxscript rewriter tutorial.

.. _l-design-pattern-optimizer:

=================
Pattern Optimizer
=================

The pattern optimizer is implemented by class :class:`GraphBuilderPatternOptimization
<yobx.xoptim.GraphBuilderPatternOptimization>`.
It searches for a specific sequence of nodes in the graph and
replaces it by another one without changing the inputs or the outputs
of the graph. The goal of the optimizer is to make the whole computation
graph more efficient. The goal of this implementation is to make this
optimization as fast as possible.
Assuming the nodes in an onnx graph are ordered in a way every input of a
node was created by previous nodes, the optimizer must not require
any global reordering. The cost should be in :math:`O(N P I)` in the worst
case where *N* is the number of nodes, *P* is the number of patterns,
*I* is the number of iterations.

It is difficult to foresee what a pattern needs in order to rewrite a part
of the graph. This API tries to give as much freedom as it can without
leaving too much to do to the developer which tries to add a new pattern.

Patterns
========

Patterns must inherit from :class:`PatternOptimization
<yobx.xoptim.PatternOptimization>`. This class defines two methods.

PatternOptimization.match
+++++++++++++++++++++++++

::

    def match(
        self,
        g: "GraphBuilderPatternOptimization",
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:

* ``g`` is a :class:`GraphBuilderPatternOptimization
  <yobx.xoptim.GraphBuilderPatternOptimization>`,
  it holds all the existing nodes, is able to return any information
  about type, shape, the node before, the node after another one.
* ``node``: the matching must determine if some nodes around this one
  are part of set of nodes this pattern optimizer can rewrite.
  From there, the function explores wherever it needs,
  checking any condition it needs.
* ``matched``: usually unused, it contains the list of nodes already matching
  a pattern

The method must not modify the graph.
The method returns None if no match is found or an instance of class :class:`MatchResult
<yobx.xoptim.MatchResult>`. It must contain:

* a list of nodes involved in the rewriting. It does not mean all of them will be
  removed but all of them are needed to do the rewriting and must
  not be impacted by other pattern optimizer.
* A function doing the rewriting (usually method *apply* of the pattern class).
* An existing node where the rewritten nodes can be inserted.
  Knowing it makes it faster to rewrite. If not specified, the optimizer
  will automatically determine the position of the new nodes.

*Debugging: method none*

::

    def none(
        self,
        node: Optional[NodeProto] = None,
        lineno: Optional[int] = None,
        msg: Optional[Union[Callable[[], str], str]] = None,
    ):

It may be useful to know the reason why a pattern matching failed.
Instead of returning None, method *match* can return the following
expression:

.. code-block:: python

    return self.none(node, inspect.currentframe().f_lineno)

By setting the verbosity (see next Section), the user may then know
which lines in the code returned None and which condition failed.
The last parameter is used to print a more comprehensive message about the
reason why the match failed.

PatternOptimization.apply
+++++++++++++++++++++++++

.. code-block:: python

    @classmethod
    def apply(
        cls, g: "GraphBuilder", *nodes: Sequence[NodeProto]
    ) -> List[NodeProto]:

The method does the rewriting. It assumes it can happen.
It takes a list of nodes impacted by the rewriting. It assumes no other
pattern optimizer modified them or will modify them.
It receives the list of nodes
returned by method *match*. Since it is a list of arguments, method
*match* can include None values. The method returns the new nodes.
The optimizer considers that any node given to this function is removed
from the graph, and any node returned by it are added.
If a received node must be kept, it must be added to the list of returned nodes.

PatternOptimization.fast_op_type
+++++++++++++++++++++++++++++++++

::

    @classmethod
    def fast_op_type(cls) -> Set[str]:

The base class returns an empty set. Overriding this method is an optional
performance hint: when the returned set contains **exactly one** ``op_type``
string, the optimizer builds an op-type → nodes index over the graph once per
matching step and restricts :meth:`enumerate_matches
<yobx.xoptim.PatternOptimization.enumerate_matches>` to only the nodes of
that type. This avoids iterating over the entire graph for patterns whose
entry point is always a specific operator.

When the method returns an empty set (the default) or a set with more than one
element, the full node list is used and no pre-filtering takes place.

.. code-block:: python

    from yobx.xoptim import PatternOptimization

    class ReshapePattern(PatternOptimization):
        """Base class for patterns whose entry node is always a Reshape."""

        @classmethod
        def fast_op_type(cls):
            return {"Reshape"}

Subclasses that always start matching from the same inherited entry point do
**not** need to override ``fast_op_type``; the inherited implementation is
already correct.

Optimization Algorithm
======================

It is implemented in method :meth:`optimize
<yobx.xoptim.GraphBuilderPatternOptimization.optimize>`

.. code-block:: python

    def optimize(
        self, max_iter=-1, remove_identity: bool = True
    ) -> List[Dict[str, Any]]:


The algorithm runs multiple iterations until the graph is not evolving
or `max_iter` is reached. By default, it is equal to the number of nodes.
An iteration is:

::

    matches = []

    builds all successors and predecessors

    # Step 1: match

    build op_type → nodes index (fast_nodes)

    for all patterns P:

        nodes_to_visit = fast_nodes[P.fast_op_type()]  # pre-filtered
                         if len(P.fast_op_type()) == 1
                         else all nodes

        for all nodes n in nodes_to_visit:

            r = p.match(n)
            if r:
                if no node already scheduled to be rewritten by another match:
                    matches.append(r)
    # Step 2: apply

    for all matches r:
        apply the match r

    # Step 3: clean

    remove unused nodes
    remove identity nodes

This algorithm may apply more than one rewriting at each iteration
but it guarantees the local structure when applying the rewriting was
not altered by another one.

Adding a pattern
================

Simple API
++++++++++

We consider the following simple model:

.. runpython::
    :showcode:
    :process:

    import torch
    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.xbuilder import OptimizationOptions
    from yobx.torch import to_onnx


    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(10, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.layers(x)


    x = torch.rand(3, 10)
    onx = to_onnx(
        MLP(), (x,), input_names=["x"], options=OptimizationOptions(patterns=None)
    )
    with open("temp_doc_mlp.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    print(pretty_onnx(onx))

Which we can render as follows:

.. gdot::
    :script: DOT-SECTION

    import onnx
    from yobx.doc import to_dot, demo_mlp_model

    onx = demo_mlp_model("temp_doc_mlp.onnx")

    print("DOT-SECTION", to_dot(onx))

We then apply the optimizations by writing the following code:

.. runpython::
    :showcode:

    import onnx
    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.xbuilder import GraphBuilder
    from yobx.doc import demo_mlp_model

    onx = demo_mlp_model("temp_doc_mlp.onnx")

    # The model is placed in a GraphBuilder.
    # It creates dictionaries to store shapes, ranks, types
    # to make it easier to the optimizers to find the information
    # they need. It still uses NodeProto to store nodes
    gr = GraphBuilder(onx, infer_shapes_options=True)

    # Let's optimize.
    opt_onx = gr.to_onnx(optimize=True)
    with open("temp_doc_mlp_opt.onnx", "wb") as f:
        f.write(opt_onx.SerializeToString())
    print(pretty_onnx(opt_onx))

Which renders as follows:

.. gdot::
    :script: DOT-SECTION

    import onnx
    from yobx.doc import to_dot

    onx = onnx.load("temp_doc_mlp_opt.onnx")

    print("DOT-SECTION", to_dot(onx))

Verbosity
+++++++++

.. runpython::
    :showcode:

    import onnx
    from yobx.xbuilder import GraphBuilder
    from yobx.doc import demo_mlp_model

    onx = demo_mlp_model("temp_doc_mlp.onnx")

    gr = GraphBuilder(onx, infer_shapes_options=True, verbose=1)
    opt_onx = gr.to_onnx(optimize=True)

With more verbosity:

.. runpython::
    :showcode:

    import onnx
    from yobx.xbuilder import GraphBuilder
    from yobx.doc import demo_mlp_model

    onx = demo_mlp_model("temp_doc_mlp.onnx")

    gr = GraphBuilder(onx, infer_shapes_options=True, verbose=11)
    opt_onx = gr.to_onnx(optimize=True)

Select the pattern to use
+++++++++++++++++++++++++

Class :class:`OptimizationOptions <yobx.xbuilder.OptimizationOptions>`
is used to enable or disable patterns.

.. runpython::
    :showcode:

    import onnx
    from yobx.xbuilder import GraphBuilder, OptimizationOptions
    from yobx.doc import demo_mlp_model

    onx = demo_mlp_model("temp_doc_mlp.onnx")

    gr = GraphBuilder(
        onx,
        infer_shapes_options=True,
        optimization_options=OptimizationOptions(
            patterns="TransposeTranspose,TransposeMatMul", verbose=1
        ),
    )
    opt_onx = gr.to_onnx(optimize=True)

There exists some predefined lists of patterns:

* ``default``: includes all patterns using only standard onnx patterns.
* ``onnxruntime``: patterns specific to :epkg:`onnxruntime`, the final model
  may be executed by onnxruntime and possibly only onnxruntime as it may
  introduce patterns from :epkg:`Supported Operators and Data Types`.

.. runpython::
    :showcode:

    import onnx
    from yobx.xbuilder import GraphBuilder, OptimizationOptions
    from yobx.doc import demo_mlp_model

    onx = demo_mlp_model("temp_doc_mlp.onnx")

    gr = GraphBuilder(
        onx,
        infer_shapes_options=True,
        optimization_options=OptimizationOptions(
            patterns="default+onnxruntime", verbose=1
        ),
    )
    opt_onx = gr.to_onnx(optimize=True)

Statistics
++++++++++

This can be used to see when a pattern is applied and how long it takes.

.. runpython::
    :showcode:

    import pandas
    import onnx
    from yobx.xbuilder import GraphBuilder, OptimizationOptions
    from yobx.doc import demo_mlp_model

    onx = demo_mlp_model("temp_doc_mlp.onnx")

    gr = GraphBuilder(
        onx,
        infer_shapes_options=True,
        optimization_options=OptimizationOptions(patterns="default"),
    )
    stat = gr.optimize()

    print(pandas.DataFrame(stat))

It can be aggregated:

.. runpython::
    :showcode:

    import pandas
    import onnx
    from yobx.xbuilder import GraphBuilder, OptimizationOptions
    from yobx.doc import demo_mlp_model

    onx = demo_mlp_model("temp_doc_mlp.onnx")

    gr = GraphBuilder(
        onx,
        infer_shapes_options=True,
        optimization_options=OptimizationOptions(patterns="default"),
    )
    stat = gr.optimize()

    df = pandas.DataFrame(stat)
    for c in df.columns:
        if "time" not in c and "pattern" not in c and "exit_point" not in c:
            df[c] = df[c].fillna(0).astype(int)
    aggs = {
        "time_in": "sum",
        "added": "sum",
        "removed": "sum",
        "iteration": "max",
        "match_index": "max",
        "instances": "sum",
    }
    print(df.groupby("pattern").agg(aggs))

.. _l-design-pattern-matching-algorithm:

Matching Algorithm
==================

:class:`EasyPatternOptimization <yobx.xoptim.EasyPatternOptimization>`
implements a bidirectional subgraph-matching algorithm that avoids a full
enumeration of all possible node assignments. Rather than writing a custom
``match`` method, the user only has to declare the subgraph to look for
(*match_pattern*) and the replacement (*apply_pattern*) using the same
builder API that is used to build ONNX graphs.

Pattern definition
++++++++++++++++++

Both ``match_pattern`` and ``apply_pattern`` are written as regular Python
functions that call ``g.op.<OpType>(...)`` to create nodes.
Each positional argument becomes a symbolic input to the subgraph.
The function returns the name(s) of the symbolic output(s).

.. code-block:: python

    class TransposeTransposePattern(EasyPatternOptimization):

        def match_pattern(self, g: "GraphBuilder", x):
            t1 = g.op.Transpose(x)
            return g.op.Transpose(t1)

        def apply_pattern(self, g: "GraphBuilder", x):
            return x   # two transposes cancel each other

At build time the framework converts each function into a small
:class:`GraphBuilderPatternOptimization
<yobx.xoptim.GraphBuilderPatternOptimization>` that stores the nodes in
topological order. The **last node** of the match pattern is used as the
*anchor*: the matching loop only fires when a graph node has the same
``op_type`` as that anchor.

Bidirectional matching
++++++++++++++++++++++

Given a candidate graph node with the same type as the anchor,
the algorithm expands the match iteratively with a stack-based approach:

::

    marked  = {anchor_pattern_key: (graph_node, anchor_pattern_node)}
    stacked = [anchor_pattern_key]

    while stacked:
        (graph_node, pattern_node) = pop(stacked)

        # --- backward pass ---
        # Walk up the predecessors of pattern_node.
        # For each predecessor in the pattern, find the corresponding
        # predecessor in the graph. Fail if types or arities differ.
        backward_match(graph_node, pattern_node)

        # --- forward pass ---
        # Walk down the successors of pattern_node.
        # For each successor in the pattern, find the corresponding
        # successor in the graph. Fail if types or arities differ.
        forward_match(graph_node, pattern_node)

        # New matched pairs are pushed onto stacked.

The two sub-routines are implemented in
:meth:`_match_backward <yobx.xoptim.EasyPatternOptimization._match_backward>`
and
:meth:`_match_forward <yobx.xoptim.EasyPatternOptimization._match_forward>`.

Ambiguity detection
+++++++++++++++++++

A dictionary ``pair_results_names`` maps every *pattern result name* to the
*graph result name* it has been paired with. Before recording a new pair the
algorithm checks that neither name already points to a different name
(*ambiguity*). An ambiguity means the same pattern result would have to
correspond to two different graph results simultaneously, which would be
inconsistent; the match is rejected in that case.

Validation
++++++++++

After all pattern nodes have been matched the algorithm performs two
additional checks:

* :meth:`validate_attribute_mapping
  <yobx.xoptim.EasyPatternOptimization.validate_attribute_mapping>` – verifies
  that the attributes of the matched graph nodes are consistent with those
  declared in the pattern (e.g. same ``axis`` value).

* :meth:`validate_mapping
  <yobx.xoptim.EasyPatternOptimization.validate_mapping>` – an optional hook
  for subclasses to add arbitrary semantic checks (e.g. verify that a
  constant operand has a specific numerical value).

Only when both validations succeed does the method return a
:class:`MatchResult <yobx.xoptim.MatchResult>` that schedules the matched
nodes for replacement.

Overlap prevention
++++++++++++++++++

The outer loop (see `Optimization Algorithm`_ above) maintains a *marked* set
of all node identifiers that have already been claimed by an earlier
:class:`MatchResult <yobx.xoptim.MatchResult>`. A candidate match is
discarded if any of its nodes appears in that set, so no two rewrites ever
touch the same node during the same pass.

Worked examples
+++++++++++++++

The two classes cover the same use-cases but at different levels of abstraction.
The examples below both implement a *Not + Not → Identity* fusion so that the
difference is easy to compare.

**PatternOptimization** (manual ``match`` / ``apply``)

The developer writes the matching logic by hand, navigating the graph with the
helpers provided by
:class:`GraphBuilderPatternOptimization <yobx.xoptim.GraphBuilderPatternOptimization>`.

.. code-block:: python

    import inspect
    from typing import List, Optional
    from onnx import NodeProto
    from yobx.xoptim import PatternOptimization, MatchResult


    class NotNotPattern(PatternOptimization):
        """Fuses ``Not(Not(x))`` into ``Identity(x)``."""

        def match(
            self,
            g: "GraphBuilderPatternOptimization",
            node: NodeProto,
            matched: List[MatchResult],
        ) -> Optional[MatchResult]:
            # Only consider Not nodes.
            if node.op_type != "Not" or node.domain != "":
                return self.none()

            # Walk one step backward: the producer of node's input must also be Not.
            not_before = g.node_before(node.input[0])
            if not_before is None or not_before.op_type != "Not" or not_before.domain != "":
                return self.none(node, inspect.currentframe().f_lineno)

            # Return both nodes as the rewrite target.
            return MatchResult(self, [not_before, node], self.apply, insert_at=node)

        def apply(
            self,
            g: "GraphBuilder",
            not_before: NodeProto,
            not_after: NodeProto,
        ) -> List[NodeProto]:
            pre_nodes = []
            # Keep the first Not if its output is consumed elsewhere.
            if g.is_used_more_than_once(not_before.output[0]):
                pre_nodes.append(not_before)
            return [
                *pre_nodes,
                g.make_node(
                    "Identity",
                    [not_before.input[0]],
                    [not_after.output[0]],
                    name=f"{self.__class__.__name__}--{not_after.name}",
                ),
            ]

**EasyPatternOptimization** (declarative ``match_pattern`` / ``apply_pattern``)

The developer declares the subgraph to look for and the replacement as builder
calls. The framework takes care of matching and result renaming automatically.

.. code-block:: python

    from typing import List, Optional
    from onnx import NodeProto
    from yobx.xoptim import EasyPatternOptimization, MatchResult


    class NotNotEasyPattern(EasyPatternOptimization):
        """Fuses ``Not(Not(x))`` into ``Identity(x)`` using the easy API."""

        def match_pattern(self, g: "GraphBuilder", x):
            t = g.op.Not(x)      # first Not
            return g.op.Not(t)   # second Not  <-- anchor node

        def apply_pattern(self, g: "GraphBuilder", x):
            return g.op.Identity(x)

Key differences
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - ``PatternOptimization``
     - ``EasyPatternOptimization``
   * - Matching logic
     - Written by hand in ``match()``. The developer calls graph-navigation
       helpers such as ``node_before``, ``next_nodes``, ``get_attribute``, …
     - Declared as a Python function ``match_pattern()`` using ``g.op.*``
       calls. The bidirectional BFS is run automatically by the framework.
   * - Replacement logic
     - Written by hand in ``apply()``. The developer calls ``g.make_node``
       and explicitly manages which nodes are kept or removed.
     - Declared as a Python function ``apply_pattern()`` using ``g.op.*``
       calls. The framework renames results and assembles the replacement
       nodes automatically.
   * - Flexibility
     - Full control: can inspect any attribute, handle optional inputs, cope
       with multi-output rewrites, or make graph-wide checks.
     - More constrained: the subgraph must have a fixed topology with no
       branching within the pattern. Attribute checks require overriding
       ``validate_mapping`` or ``validate_attribute_mapping``.
   * - Typical use-case
     - Complex rewrites (e.g. Attention fusion) where the matching involves
       many conditional checks that are hard to express as a fixed topology.
     - Simple structural fusions (e.g. double-Not, LeakyRelu decomposition,
       Gelu decomposition) where the topology is fixed and self-describing.

Shape inference
===============

The optimizers require to know the shapes to ensure they can rewrite
some nodes and avoid producing a model which does not return the
same results. If it is missing, some patterns cannot match for sure
and they will not match.

This information can be built by running shape inference
on the onnx models. That's what is done in the previous examples.
However, the best case is when this information comes from torch.

Function :func:`to_onnx <yobx.torch.interpreter.to_onnx>`
converts a torch model into ONNX. While doing so, it stores the shape
information coming from torch. There is no need to run shape inference
on the onnx model it generates before optimizing it.

Available Patterns and API
==========================

All patterns are documented in :ref:`l-design-pattern-optimizer-patterns`.

When writing a pattern, walking along the graph or checking the shape
is very common. Class :class:`GraphBuilderPatternOptimization
<yobx.xoptim.GraphBuilderPatternOptimization>`
provides the following methods.

Opsets
++++++

Patterns must rewrite using the nodes of the opset defined in the model.

* :attr:`main_opset <yobx.xoptim.GraphBuilderPatternOptimization.main_opset>`: returns the opset

Shapes, Types
+++++++++++++

* :meth:`has_type <yobx.xoptim.GraphBuilderPatternOptimization.has_type>`: tells if a result type is known
* :meth:`get_type <yobx.xoptim.GraphBuilderPatternOptimization.get_type>`: returns a result type, fails if not known
* :meth:`has_shape <yobx.xoptim.GraphBuilderPatternOptimization.has_shape>`: tells if a result shape is known
* :meth:`get_shape <yobx.xoptim.GraphBuilderPatternOptimization.get_shape>`: returns a result shape, fails if not known
* :meth:`has_rank <yobx.xoptim.GraphBuilderPatternOptimization.has_rank>`: tells if a result rank is known
* :meth:`get_rank <yobx.xoptim.GraphBuilderPatternOptimization.get_rank>`: returns a result rank, fails if not known
* :meth:`try_infer_type <yobx.xoptim.GraphBuilderPatternOptimization.try_infer_type>`: returns a type if it can be guessed
* :meth:`try_infer_shape <yobx.xoptim.GraphBuilderPatternOptimization.try_infer_shape>`: returns a shape if it can be guessed
* :meth:`has_device <yobx.xoptim.GraphBuilderPatternOptimization.has_device>`: tells if a result device is known
* :meth:`get_device <yobx.xoptim.GraphBuilderPatternOptimization.get_device>`: returns a result device, fails if not known

Constants
+++++++++

* :meth:`is_constant <yobx.xoptim.GraphBuilderPatternOptimization.is_constant>`:
  tells if a node is a constant (it may be a constant, an initializer or any value built on other constants)
* :meth:`is_constant_scalar <yobx.xoptim.GraphBuilderPatternOptimization.is_constant_scalar>`:
  checks a constant is a scalar and compares its value to a number
* :meth:`get_computed_constant <yobx.xoptim.GraphBuilderPatternOptimization.get_computed_constant>`:
  returns the constant, computing it if it is a constant built from other constants
* :meth:`get_attribute <yobx.xoptim.GraphBuilderPatternOptimization.get_attribute>`:
  returns an attribute of a node

Graph
+++++

* :meth:`next_node <yobx.xoptim.GraphBuilderPatternOptimization.next_node>`:
  returns the next node only if there is only one
* :meth:`next_nodes <yobx.xoptim.GraphBuilderPatternOptimization.next_nodes>`:
  returns the node consuming this result
* :meth:`node_before <yobx.xoptim.GraphBuilderPatternOptimization.node_before>`:
  returns the node producing the result
* :meth:`is_output <yobx.xoptim.GraphBuilderPatternOptimization.is_output>`:
  tells if a result is an output
* :meth:`is_used_by_subgraph <yobx.xoptim.GraphBuilderPatternOptimization.is_used_by_subgraph>`:
  tells if a result is used by a subgraph
* :meth:`is_used_more_than_once <yobx.xoptim.GraphBuilderPatternOptimization.is_used_more_than_once>`:
  tells if a result is used more than once
* :meth:`is_used_only_by <yobx.xoptim.GraphBuilderPatternOptimization.is_used_only_by>`:
  tells if a result is only used by specific nodes

Nodes
+++++

* :meth:`make_node <yobx.xoptim.GraphBuilderPatternOptimization.make_node>`:
  creates a node without adding it to the graph
* :meth:`make_node_check_opset <yobx.xoptim.GraphBuilderPatternOptimization.make_node_check_opset>`:
  creates a node without adding it to the graph, deals with some constraints
  related to opset version

.. _l-design-pattern-optimizer-debugging:

Debugging Optimization with Environment Variables
=================================================

Several environment variables can be set to help debug the pattern optimizer.

* ``LOG_PATTERN_OPTIMIZE``: sets the verbosity level for all patterns.
  Setting it to ``10`` produces the most detailed output. Example::

      LOG_PATTERN_OPTIMIZE=10 python my_script.py

* ``PATTERN``: increases the verbosity to ``10`` for one or more specific patterns
  (comma-separated class names or class names with the ``Pattern`` suffix removed).
  This is useful to focus on a single pattern without flooding the output with
  information from all the others. Example::

      PATTERN=ReshapeReshapePattern python my_script.py

* ``<ClassName>``: setting an environment variable whose name matches the class name
  of a pattern (e.g. ``ReshapeReshapePattern=10``) sets the verbosity for that
  individual pattern. This is equivalent to using ``PATTERN`` but more explicit.

* ``DROPPATTERN``: comma-separated list of pattern class names to exclude from the
  optimizer. Useful to bisect which pattern is causing a wrong result or an
  unexpected error. Example::

      DROPPATTERN=ReshapeReshapePattern,CastPattern python my_script.py

* ``DUMPPATTERNS``: when set to a folder path, the optimizer writes the matched
  nodes and their replacements to that folder for every successful pattern
  application. Useful for inspecting what the optimizer is actually doing.
  Example::

      DUMPPATTERNS=/tmp/dump_patterns python my_script.py

* ``PATTERNNOREMOVE``: when set to a result name, the optimizer raises an exception
  if an optimization step removes that name from the graph. Useful to track down
  which pattern is eliminating a particular node or result. Example::

      PATTERNNOREMOVE=output_0 python my_script.py

* ``PATTERNSTEP``: when set to ``1``, ``True``, or ``true``, the optimizer runs one
  optimization step at a time, which can help narrow down which step introduces
  a problem. Example::

      PATTERNSTEP=1 python my_script.py

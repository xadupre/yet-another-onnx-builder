
.. _l-design-einsum-decomposition:

=====================================
Einsum Decomposition into ONNX Nodes
=====================================

The ONNX specification includes an ``Einsum`` operator, but several runtimes
either do not support it or only support a limited subset of equations.
:func:`decompose_einsum <yobx.helpers.einsum_helper.decompose_einsum>` replaces
a single ``Einsum`` node with a sequence of simpler, universally supported ONNX
operators: ``Transpose``, ``Reshape``, ``MatMul``, ``Mul``, ``ReduceSum``,
``Unsqueeze``, ``Squeeze``, and ``Identity``.

The implementation lives in :mod:`yobx.helpers._einsum`, a self-contained
sub-package with no external dependencies beyond NumPy and ONNX.

.. contents::
   :local:
   :depth: 2

Overview
========

An einsum equation such as ``"bij,bjk->bik"`` describes a contraction of one
or more input tensors into an output tensor.  The subscripts before ``->``
label the dimensions of each input operand; the subscript after ``->`` labels
the dimensions of the output.

The decomposition algorithm proceeds in three stages:

1. **Equation analysis** — parse the equation into a compact matrix
   representation that records, for every letter, its position in each operand
   and in the output.
2. **Graph construction** — traverse the operands left to right, emitting
   :class:`EinsumSubOp <yobx.helpers._einsum.einsum_impl_classes.EinsumSubOp>`
   nodes that align dimensions, contract pairs of operands, and reduce
   dimensions that are no longer needed.
3. **ONNX emission** — walk the
   :class:`GraphEinsumSubOp <yobx.helpers._einsum.einsum_impl_classes.GraphEinsumSubOp>`
   and lower each ``EinsumSubOp`` to one or more ONNX nodes.

Stage 1 — Equation Analysis
============================

:func:`analyse_einsum_equation <yobx.helpers._einsum.einsum_impl.analyse_einsum_equation>`
parses the equation and returns four objects:

* **letters** — sorted string of all unique letters that appear in the
  equation (e.g. ``"bcdeijk"`` for ``"bac,cd,def->ebc"``).
* **mat** — a ``(n_inputs + 1) × n_letters`` integer matrix where each entry
  ``mat[i, j]`` is the *position* of letter ``j`` in operand ``i``, or ``-1``
  if that letter does not appear in operand ``i``.  The last row encodes the
  output.
* **lengths** — list of ranks (one per operand plus one for the output).
* **duplicates** — per-operand dict of letters that appear more than once
  (used to detect diagonal / trace operations).

Example for ``"bac,cd,def->ebc"``:

.. runpython::
    :showcode:

    from yobx.helpers._einsum import analyse_einsum_equation
    letters, mat, lengths, duplicates = analyse_einsum_equation("bac,cd,def->ebc")
    print("letters :", letters)
    print("lengths :", lengths)
    print("mat     :")
    print(mat)

The matrix encodes the full algebraic structure of the equation in a form that
is easy to manipulate programmatically.

Stage 2 — Graph Construction
=============================

:func:`decompose_einsum_equation <yobx.helpers._einsum.einsum_impl.decompose_einsum_equation>`
builds a directed acyclic graph of
:class:`EinsumSubOp <yobx.helpers._einsum.einsum_impl_classes.EinsumSubOp>`
nodes.  Each node represents one primitive tensor operation.

The available node types are:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Node type
     - Meaning
   * - ``id``
     - Identity / input placeholder — references one of the original input
       operands.
   * - ``expand_dims``
     - Inserts size-1 axes so that all operands share the same set of
       dimension positions (analogous to ``numpy.expand_dims``).
   * - ``transpose``
     - Permutes axes to bring them into the canonical alphabetical order.
   * - ``diagonal``
     - Extracts the diagonal when a letter appears twice in the same operand
       (e.g. ``"ii->i"``).
   * - ``reduce_sum``
     - Sums out dimensions that are no longer needed after the current
       contraction step.
   * - ``reduce_sum_mm``
     - Like ``reduce_sum`` but takes two inputs; only the first is reduced
       (used internally during matrix-multiplication decomposition).
   * - ``matmul``
     - Generic matrix contraction implemented with a single
       ``numpy.einsum`` call on two already-aligned operands.
   * - ``batch_dot``
     - Matrix multiplication for the ``"numpy"`` strategy: combines
       ``Transpose`` + ``Reshape`` + ``MatMul`` to avoid any remaining
       ``einsum`` call.
   * - ``mul``
     - Element-wise multiplication (used when there are no contraction axes).
   * - ``transpose_mm``
     - Like ``transpose`` but takes two inputs; only the first is permuted.
   * - ``squeeze``
     - Removes the size-1 axes that were added by ``expand_dims`` once they
       are no longer required.

The graph is built by iterating over the input operands in order.  For each
operand the algorithm:

1. Emits an ``id`` node to reference the raw input.
2. Handles any diagonal operation if the same letter appears twice.
3. Calls ``_apply_transpose_reshape`` to insert ``expand_dims`` and
   ``transpose`` nodes that bring all dimensions into the shared alphabetical
   order.
4. Emits an optional ``reduce_sum`` to eliminate dimensions that will not
   appear in any later operand or in the output.
5. If a previous partial result already exists, calls
   ``_apply_einsum_matmul`` to contract the previous result with the
   current operand, producing ``matmul`` or ``batch_dot`` (and auxiliary
   ``transpose`` / ``reduce_sum``) nodes as required.

After all operands have been processed, a final ``reduce_sum`` and
``squeeze`` / ``transpose`` step brings the accumulated result into the
shape and axis order demanded by the output subscript.

The graph produced for ``"bac,cd,def->ebc"`` looks like this:

.. gdot::
    :script: DOT-SECTION
    :process:

    from yobx.helpers._einsum import decompose_einsum_equation
    seq = decompose_einsum_equation(
        "bac,cd,def->ebc", (2, 2, 2), (2, 2), (2, 2, 2))
    print("DOT-SECTION", seq.to_dot())

Decomposition strategies
------------------------

Two strategies are available via the *strategy* parameter of
:func:`decompose_einsum_equation <yobx.helpers._einsum.einsum_impl.decompose_einsum_equation>`:

* ``"simple"`` — contractions between two aligned operands are emitted as a
  single ``matmul`` node which is still evaluated with ``numpy.einsum``
  internally.
* ``"numpy"`` — contractions are fully expanded into ``Transpose`` +
  ``Reshape`` + ``batch_dot`` nodes so that no ``numpy.einsum`` call remains.
  This is the default used by
  :func:`decompose_einsum <yobx.helpers.einsum_helper.decompose_einsum>` when
  generating ONNX models.

Printing the operation sequence
--------------------------------

The :class:`GraphEinsumSubOp <yobx.helpers._einsum.einsum_impl_classes.GraphEinsumSubOp>`
object can be iterated to inspect the full sequence:

.. runpython::
    :showcode:

    from yobx.helpers._einsum import decompose_einsum_equation
    seq = decompose_einsum_equation("bac,cd,def->ebc")
    for op in seq:
        print(op)

Stage 3 — ONNX Emission
========================

:func:`decompose_einsum <yobx.helpers.einsum_helper.decompose_einsum>` calls
:meth:`GraphEinsumSubOp.to_onnx <yobx.helpers._einsum.einsum_impl_classes.GraphEinsumSubOp.to_onnx>`
which walks the graph and converts each
:class:`EinsumSubOp <yobx.helpers._einsum.einsum_impl_classes.EinsumSubOp>` into
one or more ONNX nodes according to the following mapping:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - EinsumSubOp
     - ONNX node(s)
   * - ``id``
     - direct wire (no node emitted)
   * - ``expand_dims``
     - one ``Unsqueeze`` node per inserted axis
   * - ``transpose``
     - ``Transpose`` (omitted when the permutation is the identity)
   * - ``diagonal``
     - ``Gather`` along the diagonal axis
   * - ``reduce_sum``
     - ``ReduceSum``
   * - ``matmul``
     - ``MatMul`` (after optional ``Transpose`` + ``Reshape``)
   * - ``batch_dot``
     - ``Transpose`` + ``Reshape`` + ``MatMul`` + ``Reshape`` + ``Transpose``
   * - ``mul``
     - ``Mul``
   * - ``squeeze``
     - ``Squeeze``

The resulting model is a stand-alone :class:`onnx.ModelProto` that can be
run directly with any compliant ONNX runtime.

End-to-end example
==================

The following snippet decomposes the batched matrix multiplication
``"bij,bjk->bik"`` and validates the result numerically:

.. runpython::
    :showcode:

    import numpy as np
    import onnxruntime
    from yobx.helpers.einsum_helper import decompose_einsum

    model = decompose_einsum("bij,bjk->bik", (2, 3, 4), (2, 4, 5))
    print("ONNX node types:", [n.op_type for n in model.graph.node])

    sess = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    a = np.random.rand(2, 3, 4).astype(np.float32)
    b = np.random.rand(2, 4, 5).astype(np.float32)
    (result,) = sess.run(None, {"X0": a, "X1": b})
    expected = np.einsum("bij,bjk->bik", a, b)
    print("max |error|:", np.max(np.abs(result - expected)))

More examples (including a three-operand contraction and a chart comparing
node counts) are shown in the gallery example :ref:`l-plot-einsum-decomposition`.

.. seealso::

    * :func:`decompose_einsum <yobx.helpers.einsum_helper.decompose_einsum>`
      — public API function.
    * :func:`decompose_einsum_equation <yobx.helpers._einsum.einsum_impl.decompose_einsum_equation>`
      — internal function that builds the operation graph.
    * :class:`EinsumSubOp <yobx.helpers._einsum.einsum_impl_classes.EinsumSubOp>`
      — a single node in the decomposition graph.
    * :class:`GraphEinsumSubOp <yobx.helpers._einsum.einsum_impl_classes.GraphEinsumSubOp>`
      — the full decomposition graph.
    * :ref:`l-plot-einsum-decomposition` — gallery example with numerical
      validation and node-count comparison.

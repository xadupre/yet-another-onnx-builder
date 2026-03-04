.. _l-design-patches:

=========================
Patches (torch export)
=========================

.. note::
    This section covers functionality that is **specific to PyTorch**.
    It is only relevant when exporting :class:`torch.nn.Module` models with
    :func:`torch.export.export` and has no bearing on ONNX models built
    directly with the builder APIs.

Before exporting a :class:`torch.nn.Module` with :func:`torch.export.export`,
certain internal functions in :mod:`torch` and :mod:`transformers` must be
temporarily **replaced** with corrected versions to avoid crashes or incorrect
behaviour during tracing.  The patching infrastructure in
:mod:`yobx.helpers.patch_helper` and :mod:`yobx.torch.patch` provides a
structured, reversible way to apply, track and report such replacements.

Why patches are needed
======================

:func:`torch.export.export` symbolically traces a model, meaning that every
operation is recorded with *symbolic* dimension variables rather than concrete
sizes.  Several low-level torch and transformers helpers were not written to
cope with symbolic dimensions and raise exceptions or silently return wrong
results when they encounter them.

Control flow is the primary source of failures
-----------------------------------------------

The most fundamental issue is **Python control flow** inside traced functions.
During symbolic tracing, dimension values are :class:`torch.SymInt` objects
rather than plain integers.  Any code written as:

.. code-block:: python

    if condition_on_size:
        raise SomeError(...)

will crash the exporter because evaluating the ``if`` forces the symbolic
integer to a concrete boolean — a :class:`~torch.SymBool` — which cannot be
resolved at trace time, raising ``GuardOnDataDependentSymNode``.

The fix is to replace such guards with :func:`torch._check`:

.. code-block:: python

    torch._check(condition_on_size)

:func:`torch._check` is understood by the exporter and recorded as a
constraint on the symbolic value instead of executing a branch.  The patches
shipped with this library systematically replace ``if ... : raise`` patterns
in torch and transformers internals with ``torch._check(...)`` equivalents so
that symbolic tracing can proceed without crashing.

Other common failure modes
--------------------------

* ``GuardOnDataDependentSymNode`` — a symbolic size cannot be evaluated to a
  concrete boolean at trace time, crashing broadcasting helpers such as
  ``infer_size`` or ``_broadcast_shapes``.
* Incorrect range-constraint computation — ``_get_range_constraints`` uses the
  wrong argument order, causing dynamic shape constraints to be mis-assigned.
* Wrapped ``RotaryEmbedding.forward`` — some transformers model variants wrap
  ``RotaryEmbedding.forward`` with a decorator that introduces control-flow
  invisible to the exporter; the wrapper must be replaced with a traceable
  version.

Rather than forking torch or transformers, patches swap in corrected
implementations only for the duration of the export, then restore the originals.
On that particular topic, :class:`yobx.torch.tiny_models.TinyBroadcastAddModel`
illustrates what problem a user can run into.
The exporter makes a strong assumption when it comes to infer
shapes after a broadcast. Two dimensions are broadcastable
if they are equal or one of them is equal to 1. But sometimes,
this assumption cannot be verified in a sense the result is not known.
In that condition, it is reasonable to assume the model is correct
and the final dimension is the maximum of the two broadcasted dimensions.
The patches enables that to be possible as the example
:class:`yobx.torch.tiny_models.TinyBroadcastAddModel` demonstrates.

Core data structures
====================

PatchInfo
---------

:class:`~yobx.helpers.patch_helper.PatchInfo` describes a single patch:

* **patch** — the replacement callable.
* **do** / **undo** — lambdas that swap the function in and out of the target
  module or class.
* **family** — a free-form category string (``"torch"`` or
  ``"transformers"``) used for reporting.
* **depends_on** — a list of :class:`~yobx.helpers.patch_helper.PatchInfo`
  objects that must also be applied for this patch to work correctly.

The convenience constructor :meth:`~yobx.helpers.patch_helper.PatchInfo.make`
covers the common pattern of replacing an attribute on a module or class:

.. runpython::
    :showcode:

    import torch
    import torch._refs
    from yobx.helpers.patch_helper import PatchInfo

    def my_patched_fn(*shapes):
        """Minimal stand-in that just returns the first non-empty shape."""
        for s in shapes:
            if s:
                return list(s)
        return []

    patch = PatchInfo.make(
        my_patched_fn,
        torch._refs,
        "_broadcast_shapes",
        family="torch",
    )

    # Apply the patch.
    patch.do()
    print("active patch:", patch.name)
    print("patched function:", torch._refs._broadcast_shapes.__name__)

    # Restore the original.
    patch.undo()
    print("after undo:", torch._refs._broadcast_shapes.__name__)

:meth:`~yobx.helpers.patch_helper.PatchInfo.make_diff` produces a unified diff
between the original function and the patched replacement.  Calling this for
every active patch (or via
:meth:`~yobx.helpers.patch_helper.PatchDetails.make_report`) is important
because it gives an **exhaustive** picture of what changed — including patches
that are only reachable through a module method (e.g. a ``RotaryEmbedding``
variant whose ``forward`` is called indirectly from the top-level
``nn.Module``).  Without the full diff it is easy to overlook a patched
sub-function and miss the reason a particular graph node was introduced:

.. code-block:: python

    import torch
    import torch._refs
    from yobx.helpers.patch_helper import PatchInfo

    def my_patched_fn(*shapes):
        """Returns the first non-empty shape."""
        for s in shapes:
            if s:
                return list(s)
        return []

    patch = PatchInfo.make(
        my_patched_fn,
        torch._refs,
        "_broadcast_shapes",
        family="torch",
    )
    patch.do()
    diff = patch.make_diff()
    patch.undo()
    # Print the first few lines of the diff.
    print("\n".join(diff.splitlines()[:10]))

That gives::

    --- original
    +++ rewritten
    @@ -1,77 +1,6 @@
    -def _broadcast_shapes(*_shapes):
    -    from torch.fx.experimental.symbolic_shapes import (
    -        guard_or_false,
    -        has_hint,
    -        is_nested_int,
    -        size_hint,
    -    )  
    ...

Or with ``diff = patch.format_diff("rst")``::

    .. _patch-torch-my_patched_fn:

    torch: _broadcast_shapes -> my_patched_fn
    =========================================

    .. code-block:: diff
        :linenos:

        --- original
        +++ rewritten
        @@ -1,77 +1,6 @@
        -def _broadcast_shapes(*_shapes):
        -    from torch.fx.experimental.symbolic_shapes import (
        -        guard_or_false,
        -        has_hint,
        -        is_nested_int,
        -        size_hint,
        -    )
        -
        -    backed_so = torch.fx.experimental._config.backed_size_oblivious

    ...

PatchDetails
------------

:class:`~yobx.helpers.patch_helper.PatchDetails` is an ordered collection of
:class:`~yobx.helpers.patch_helper.PatchInfo` objects.  It is the object
yielded by the :func:`~yobx.torch.patch.apply_patches_for_model` context
manager so that callers can inspect every patch that was active during export.

Useful methods:

* :meth:`~yobx.helpers.patch_helper.PatchDetails.find` — look up a patch by
  function name.
* :meth:`~yobx.helpers.patch_helper.PatchDetails.patches_involved_in_graph`
  — identify which patches contributed nodes to a graph by
  cross-referencing ``node.meta["stack_trace"]`` with each patch's source
  location.  The method is designed for :class:`torch.fx.Graph` but accepts
  any object that provides:

  * ``graph.nodes`` — an iterable of node objects;
  * ``node.meta`` — a :class:`dict` on each node;
  * ``node.meta["stack_trace"]`` — the call-stack string captured when the
    node was created.

  Any custom graph representation that exposes these three attributes works
  equally well.
* :meth:`~yobx.helpers.patch_helper.PatchDetails.make_report` — render a
  human-readable report (raw text or RST) listing each involved patch together
  with its diff and the affected graph nodes.

Applying patches
================

The recommended entry point is the context manager
:func:`~yobx.torch.patch.apply_patches_for_model`:

.. code-block:: python

    from yobx.torch import apply_patches_for_model

    with apply_patches_for_model(
        patch_torch=True,
        patch_transformers=True,
        model=model,
        verbose=1,
    ) as details:
        ep = torch.export.export(model, (), kwargs=inputs, dynamic_shapes=ds)

    # After the `with` block every patch has been restored.
    for patch in details:
        print(patch.format_diff(format="raw"))

Parameters:

* **patch_torch** — applies the four patches in
  :mod:`yobx.torch.in_torch.patches` that fix symbolic-shape handling inside
  :mod:`torch`.
* **patch_transformers** — applies the ``RotaryEmbedding`` patches from
  :mod:`yobx.torch.in_transformers.patches`.  When *model* is provided, the
  function inspects each sub-module for wrapped ``RotaryEmbedding.forward``
  implementations and adds a model-specific patch automatically.
* **model** — the :class:`torch.nn.Module` being exported.  Supplying it
  enables automatic detection of non-standard ``RotaryEmbedding`` variants.
* **verbose** — print each patch name as it is applied or removed.

Shipped patches
===============

torch patches
-------------

The following patches are registered in :mod:`yobx.torch.in_torch.patches`
(family ``"torch"``):

+--------------------------------------------------+-----------------------------------------------+
| Patch function                                   | What it fixes                                 |
+==================================================+===============================================+
| ``patched_DynamicDimConstraintPrinter``          | ``DynamicDimConstraintPrinter._print_Symbol`` |
| ``._print_Symbol``                               | returns the source name for a symbol rather   |
|                                                  | than crashing when ``symbol_to_source`` does  |
|                                                  | not contain the symbol.                       |
+--------------------------------------------------+-----------------------------------------------+
| ``patched_infer_size``                           | ``torch._subclasses.fake_impls.infer_size`` — |
|                                                  | uses ``torch.sym_max`` in the generic case    |
|                                                  | instead of asserting equality, allowing       |
|                                                  | symbolic dimensions to broadcast correctly.   |
+--------------------------------------------------+-----------------------------------------------+
| ``patched__broadcast_shapes``                    | ``torch._refs._broadcast_shapes`` — replaces  |
|                                                  | hard equality guards with ``sym_max`` so      |
|                                                  | that shapes with unknown symbolic dimensions  |
|                                                  | can be broadcast without raising              |
|                                                  | ``GuardOnDataDependentSymNode``.              |
+--------------------------------------------------+-----------------------------------------------+
| ``patched__get_range_constraints``               | ``torch.export._trace._get_range_constraints``|
|                                                  | — passes ``preserve_order=True`` to           |
|                                                  | ``_combine_args`` so that dynamic-shape       |
|                                                  | constraints are matched to the correct        |
|                                                  | arguments (see `pytorch/pytorch#174593        |
|                                                  | <https://github.com/pytorch/pytorch/pull/     |
|                                                  | 174593>`_).                                   |
+--------------------------------------------------+-----------------------------------------------+

transformers patches
--------------------

:func:`~yobx.torch.in_transformers.patches.get_patches_for` returns patches for
the ``transformers`` library.  When a model is supplied the function walks its
sub-modules and adds a model-specific patch for every ``RotaryEmbedding`` whose
``forward`` method has been wrapped (common in recent ``transformers`` releases
where ``dynamic_rope_update`` is applied via a decorator).

The ``RotaryEmbedding`` patch replaces the wrapped ``forward`` with
``common_RotaryEmbedding.forward`` from
:mod:`yobx.torch.in_transformers._patches_model_rope_utils`, and adds a
dependency patch that replaces
``transformers.modeling_rope_utils.dynamic_rope_update`` with a traceable
version.

.. seealso::

    :ref:`l-design-flatten` — registering pytree nodes for ``DynamicCache``
    and other transformers classes, which is typically used alongside patches.

    :mod:`yobx.helpers.patch_helper` — :class:`PatchInfo` and
    :class:`PatchDetails` API reference.

    :mod:`yobx.torch.patch` — :func:`apply_patches_for_model` API reference.

    :mod:`yobx.torch.in_torch.patches` — torch patch implementations.

    :mod:`yobx.torch.in_transformers.patches` — transformers patch
    implementations.

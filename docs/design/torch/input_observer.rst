
.. _l-design-input-observer:

==============
InputObserver
==============

.. note::
    This section covers functionality that is **specific to PyTorch**.
    It is only relevant when exporting :class:`torch.nn.Module` models with
    :func:`torch.export.export` and has no bearing on ONNX models built
    directly with the builder APIs.

:func:`torch.export.export` requires callers to supply both the model inputs
(``args`` / ``kwargs``) *and* a ``dynamic_shapes`` specification describing
which tensor dimensions are symbolic at export time.  Assembling these two
artefacts by hand is tedious and error-prone, especially for large language
models whose inputs change between the *prefill* phase (first token) and the
*decode* phase (subsequent tokens).

:class:`~yobx.torch.input_observer.InputObserver` automates this task: it
temporarily **replaces** the model's ``forward`` method with a thin wrapper
that records every call, then reconstructs the export arguments and dynamic
shapes from those observations.

Why manual argument construction is hard
=========================================

A typical LLM ``forward`` signature includes optional arguments such as
``attention_mask``, ``past_key_values``, and ``pixel_values`` (for
vision-language models).  The set of arguments that is *actually* present
changes between calls:

* The prefill call includes ``pixel_values`` but no ``past_key_values``.
* Decode calls include ``past_key_values`` but no ``pixel_values``.

:func:`torch.export.export` needs a *single* set of representative inputs that
covers all paths, with ``None`` placeholders for optional arguments.  Figuring
out which arguments are optional and what their shapes look like normally
requires reading model source code.  :class:`InputObserver` infers this
automatically from real forward calls.

Basic usage
===========

Wrap the model in a ``with`` block and run one or more forward passes (or
:meth:`generate` calls for LLMs).  After the block the observer holds enough
information to build the export arguments:

.. code-block:: python

    import torch
    from yobx.torch.input_observer import InputObserver

    observer = InputObserver()
    with observer(model):
        # Run one or more forward passes with representative inputs.
        model(x1, y1)
        model(x2, y2)

    # Build export arguments from the observed inputs.
    args = observer.infer_arguments()
    dynamic_shapes = observer.infer_dynamic_shapes()

    ep = torch.export.export(model, args, dynamic_shapes=dynamic_shapes)

For LLMs the entire token-generation loop can be observed via
:meth:`~torch.nn.Module.generate`:

.. code-block:: python

    observer = InputObserver()
    with observer(model):
        model.generate(input_ids)

    ep = torch.export.export(
        model,
        (),
        kwargs=observer.infer_arguments(),
        dynamic_shapes=observer.infer_dynamic_shapes(),
    )

Handling optional arguments with ``value_if_missing``
======================================================

When an argument appears only in some observed calls (e.g. ``pixel_values``
only in the prefill pass), the observer cannot automatically fabricate a
representative empty tensor for it.  Pass ``value_if_missing`` to supply
default shapes for such arguments:

.. code-block:: python

    observer = InputObserver(
        value_if_missing=dict(
            pixel_values=torch.empty((0, 3, 896, 896), dtype=torch.float16)
        )
    )
    with observer(model):
        model.generate(input_ids)

    args = observer.infer_arguments()
    dynamic_shapes = observer.infer_dynamic_shapes()

The values in ``value_if_missing`` are **only** used to infer shapes and
argument structures; the actual tensor data is not passed to the model.

Inferring dynamic shapes
========================

:meth:`~yobx.torch.input_observer.InputObserver.infer_dynamic_shapes`
compares the shapes observed across multiple forward calls and marks any
dimension that *varies* as dynamic.  Most models have a dynamic batch
dimension but all observed inputs are run with the same batch size because
generating different batch sizes is expensive.

Use ``set_batch_dimension_for`` to mark the first axis of selected inputs as
dynamic even when all observations use the same batch size:

.. code-block:: python

    dynamic_shapes = observer.infer_dynamic_shapes(
        set_batch_dimension_for={"input_ids", "attention_mask"}
    )

Pass ``True`` to mark the first dimension of *all* inputs as dynamic:

.. code-block:: python

    dynamic_shapes = observer.infer_dynamic_shapes(set_batch_dimension_for=True)

Inspecting observations
=======================

:attr:`~yobx.torch.input_observer.InputObserver.num_obs` reports how many
forward calls were captured.  The ``info`` attribute exposes the raw
:class:`~yobx.torch.input_observer.InputObserverInfo` object, which contains
the full per-call input and output records along with the inferred argument
alignment.

How it works — internals
=========================

The observer operates in four phases.

**Phase 1 — Recording.**
When the ``with observer(model):`` block is entered, the ``forward`` method
(or any other named method) is replaced by a thin lambda that calls the
original method and stores a deep-copy of its inputs and outputs as an
:class:`~yobx.torch.input_observer.InputCandidate`.  Recording stops after
``store_n_calls`` invocations (default 3) to bound memory usage; any
subsequent calls pass through to the real method without recording.

Every argument is immediately flattened via
``torch.utils._pytree.tree_flatten`` into a flat list of tensors.
Non-tensor scalars (``int``, ``float``, ``bool``, ``str``) whose values
differ from the parameter default are stored separately as *constant kwargs*
and will be passed back as-is to :func:`torch.export.export` without dynamic
shape annotations.  All other scalars and ``None`` values are dropped.

**Phase 2 — Best-candidate selection and alignment.**
When either :meth:`infer_dynamic_shapes` or :meth:`infer_arguments` is called,
the observer first picks a *best candidate*: the recorded call that produced
the largest total number of flattened tensors.  This candidate is used as the
reference layout for all other calls.

Each other candidate is then *aligned* against the best candidate.  For every
positional or named argument slot in the best candidate the aligner checks
whether that slot is present in the other call.  If it is absent or ``None``,
it inserts a ``None`` placeholder so that every aligned flat list has the same
length.  This is what makes the observer resilient to optional arguments.

.. note::
    At least one observed call must supply *all* the arguments that appear in
    *any* other observed call.  If no single call covers the full union of
    arguments, alignment fails with ``RuntimeError: At least one call to the
    observed model must contain all the named arguments``.

**Phase 3 — Dynamic shape inference.**
With every candidate now aligned to the same flat structure, the observer
iterates over each tensor slot and collects the sequence of shapes seen across
all calls.  A dimension is marked **dynamic** (``torch.export.Dim.DYNAMIC``)
when its size differs across at least two calls.  If only one call was
recorded, no dimension varies and no axis is automatically marked dynamic;
``set_batch_dimension_for`` can override this per-input.

When ``dim_names=True`` the observer assigns *named* dynamic dimension labels
instead of ``torch.export.Dim.DYNAMIC``.  Tensor slots whose observed size
sequences are identical across every call share the same label, instructing
:func:`torch.export.export` to treat those dimensions as *constrained equal*.
Well-known transformer parameter names (``input_ids``, ``position_ids``,
``attention_mask``, ``past_key_values``, ``pixel_values``, etc.) receive
pre-defined semantic labels such as ``"batch_size"`` and
``"sequence_length"``; all other parameters receive auto-generated
``<name>_dim_<n>`` labels.

**Phase 4 — Argument inference.**
:meth:`infer_arguments` selects the first recorded call that has the same
number of positional and named arguments as the best candidate — usually the
first call.  For every ``None`` slot in the flat list (an optional argument
that was absent in that call), it manufactures an *empty tensor*:

* If the corresponding slot has no dynamic dimensions, a zero tensor with the
  same shape and dtype is used (``torch.zeros``).
* Otherwise the largest dynamic dimension is zeroed out (``torch.empty`` with
  that dimension set to 0), signalling to :func:`torch.export.export` that
  the argument is optional.

The resulting flat list is unflattened back into the original pytree structure
and returned as either a tuple (positional-only), a dict (keyword-only), or
a ``{name: tensor}`` dict that merges positional and keyword arguments.

Known failure cases
===================

The following situations cause :meth:`infer_dynamic_shapes` or
:meth:`infer_arguments` to raise an exception.  Each entry describes the
root cause and — where applicable — the fix.

**No forward calls were observed.**
Calling either inference method before the ``with`` block exits, or when the
model was never actually called inside the block, raises
``RuntimeError: No inputs were captured``.  Make sure the model is called at
least once inside the observer context.

**No single call covers all optional arguments.**
If argument ``A`` appears only in call 1 and argument ``B`` appears only in
call 2, alignment fails:

.. code-block:: text

    RuntimeError: At least one call to the observed model must contain
    all the named arguments.

Fix: include at least one *combined* call where both ``A`` and ``B`` are
present, or supply the missing argument via ``value_if_missing``.

**An argument's tensor count changes between calls.**
When an argument is a container (e.g. a list of tensors) and that container
has a different *number* of tensors in different calls, alignment raises:

.. code-block:: text

    RuntimeError: Named argument 'y' has N tensors but previously got M tensors.
    Inference is impossible in that case.

This happens, for example, when ``y=[t1]`` in one call and ``y=[t1, t2]`` in
another.  :class:`InputObserver` cannot reconcile a changing tensor count; the
model must be called with a consistent container size across all observations.

**Constant kwargs differ between calls.**
When a scalar argument (``int``, ``float``, ``bool``, or ``str``) is passed
with *different* values in different calls the observer cannot pick a single
representative:

.. code-block:: text

    RuntimeError: Two calls were made with different constant values,
    {'add': True} != {'add': False}

The export target must be a single-behaviour graph.  Observe the model with a
consistent scalar value or export two separate graphs.

**Only one call was made and no batch dimension is forced.**
With a single observation every dimension appears constant.
:meth:`infer_dynamic_shapes` returns an empty dict ``{}`` for every tensor,
meaning no axis is treated as dynamic.  This is usually incorrect.  Fix by
either running the model with several different input shapes or by passing
``set_batch_dimension_for=True`` (or a specific set of argument names/indices)
to force the first dimension to be treated as dynamic.

**Custom container types not registered as pytree nodes.**
:class:`InputObserver` uses ``torch.utils._pytree.tree_flatten`` to decompose
arguments into a flat list of tensors.  If a model argument is an instance of
a custom class (such as ``DynamicCache`` from ``transformers``) that has not
been registered as a pytree node, flattening silently treats the whole object
as a single leaf.  The miscount then causes alignment to fail with:

.. code-block:: text

    NotImplementedError: infer_dynamic_shapes is not implemented when the
    best candidate is not 'aligned'. … You need to register the flattening
    function: with register_flattening_functions(patch_transformers=True): …

Fix: call :func:`yobx.torch.flatten.register_class_flattening` (or use the
``patch_transformers=True`` shorthand) **before** the observer context so that
the custom class is known to the pytree machinery.  See :ref:`l-design-flatten`
for details.

**Missing ``value_if_missing`` for arguments absent in all recorded calls.**
If an optional argument never appears in any recorded call (for example
``pixel_values`` is only used in the prefill step but all recorded calls are
decode steps), alignment succeeds but :meth:`infer_arguments` raises:

.. code-block:: text

    RuntimeError: There is no tensor at position N in any flattened inputs.

Fix: pass the argument with its expected shape through ``value_if_missing``:

.. code-block:: python

    observer = InputObserver(
        value_if_missing=dict(
            pixel_values=torch.empty((0, 3, 896, 896), dtype=torch.float16)
        )
    )

The zero batch dimension signals that this is an *empty* placeholder; the data
is never forwarded to the model.

**``value_if_missing`` key is not in the model's signature.**
If the key provided in ``value_if_missing`` does not match any parameter in
the ``forward`` signature (and the signature does not accept ``**kwargs``), the
first observed call raises:

.. code-block:: text

    ValueError: Unexpected keyword argument 'nonexistent' provided as a
    value_if_missing input for a function that does not accept it.

Verify the spelling of argument names against the model's ``forward``
signature.

**Mixed positional and keyword calls with ``*args``.**
When the signature contains a variadic positional parameter (``*args``) and
:meth:`infer_arguments` cannot express the result as a plain tuple or dict, it
raises:

.. code-block:: text

    RuntimeError: Cannot return arguments as a single tuple or a single
    dictionary because of '*args' in the function signature.
    You need to set `as_args_kwargs=True`.

Pass ``as_args_kwargs=True`` to receive a ``(args_tuple, kwargs_dict)`` pair
instead.

.. seealso::

    :ref:`l-design-flatten` — registering pytree nodes for ``DynamicCache``
    and other transformers classes, which is typically needed alongside
    :class:`InputObserver` when working with LLMs.

    :ref:`l-design-patches` — applying patches to torch and transformers
    internals to enable symbolic tracing during :func:`torch.export.export`.

    :mod:`yobx.torch.input_observer` — API reference for
    :class:`InputObserver`, :class:`InputObserverInfo`, and
    :class:`InputCandidate`.

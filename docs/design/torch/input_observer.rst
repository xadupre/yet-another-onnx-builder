
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

.. seealso::

    :ref:`l-design-flatten` — registering pytree nodes for ``DynamicCache``
    and other transformers classes, which is typically needed alongside
    :class:`InputObserver` when working with LLMs.

    :ref:`l-design-patches` — applying patches to torch and transformers
    internals to enable symbolic tracing during :func:`torch.export.export`.

    :mod:`yobx.torch.input_observer` — API reference for
    :class:`InputObserver`, :class:`InputObserverInfo`, and
    :class:`InputCandidate`.

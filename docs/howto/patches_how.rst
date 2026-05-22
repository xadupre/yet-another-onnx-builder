.. _l-howto-patches:

Patches (PyTorch export)
========================

This page answers common *"how do I…"* questions for applying and writing
**patches** — temporary function replacements that make :func:`torch.export.export`
succeed on models that would otherwise crash or produce incorrect graphs during
symbolic tracing.

.. note::
    Patches are only relevant when exporting a :class:`torch.nn.Module` with
    :func:`torch.export.export`.  They have no effect on ONNX models built
    directly via the builder APIs or when exporting from scikit-learn,
    TensorFlow or other non-PyTorch frameworks.

For background on *why* patches are needed, see :ref:`l-design-patches`.

----

How to apply the built-in patches when exporting
-------------------------------------------------

Wrap the :func:`torch.export.export` call with
:func:`~yobx.torch.patch.apply_patches_for_model`.  Use ``patch_torch=True``
to activate the patches that fix symbolic-shape handling inside :mod:`torch`,
and ``patch_transformers=True`` for models that use 🤗 Transformers internals
(e.g. ``RotaryEmbedding``).

The example below exports :class:`~yobx.torch.tiny_models.TinyBroadcastAddModel`,
whose output has the symbolic shape ``(batch, max(d1, d2))`` due to broadcasting.
This model requires both ``patch_torch=True`` and
``torch.fx.experimental._config.patch(backed_size_oblivious=True)`` to export
successfully:

.. runpython::
    :showcode:
    :process:

    import torch
    from yobx.torch import apply_patches_for_model, use_dyn_not_str
    from yobx.torch.tiny_models import TinyBroadcastAddModel

    model = TinyBroadcastAddModel()
    inputs = TinyBroadcastAddModel._export_inputs()
    dynamic_shapes = use_dyn_not_str(TinyBroadcastAddModel._dynamic_shapes())

    with (
        torch.fx.experimental._config.patch(backed_size_oblivious=True),
        apply_patches_for_model(patch_torch=True) as details,
    ):
        ep = torch.export.export(model, (), kwargs=inputs, dynamic_shapes=dynamic_shapes)

    print(f"Applied {details.n_patches} patch(es).")
    print(ep)

The patches are automatically removed when the ``with`` block exits, leaving
the original PyTorch functions fully restored.

----

How to list the patches that were applied
-----------------------------------------

The context manager yields a :class:`~yobx.helpers.patch_helper.PatchDetails`
object.  Iterate over it to see the name and family of every
:class:`~yobx.helpers.patch_helper.PatchInfo` that was registered:

.. runpython::
    :showcode:
    :process:

    from yobx.torch import apply_patches_for_model

    with apply_patches_for_model(patch_torch=True) as details:
        for patch in details:
            print(f"[{patch.family}] {patch.name}")

----

How to view the diff for each patch
------------------------------------

:meth:`~yobx.helpers.patch_helper.PatchInfo.format_diff` returns a unified
diff that shows exactly what changed between the original PyTorch function and
the patched replacement.  This is useful for auditing what the library is
doing and for debugging unexpected behaviour.

.. runpython::
    :showcode:
    :process:

    from yobx.torch import apply_patches_for_model

    with apply_patches_for_model(patch_torch=True) as details:
        pass  # patches are removed on exit but diffs remain accessible

    # Show the first few lines of each diff.
    for patch in details:
        diff_lines = patch.format_diff(format="raw").splitlines()
        print(f"=== {patch.name} ===")
        print("\n".join(diff_lines[:8]))
        print("...")
        print()

Pass ``format="rst"`` to get a reStructuredText block with a cross-reference
anchor, which is how :ref:`patches-torch` is generated.

----

How to write a custom patch
----------------------------

Use :meth:`~yobx.helpers.patch_helper.PatchInfo.make` to create a
:class:`~yobx.helpers.patch_helper.PatchInfo` that swaps a function or method
in a module for the duration of the export:

.. runpython::
    :showcode:
    :process:

    import torch
    import torch._refs
    from yobx.helpers.patch_helper import PatchInfo

    def my_broadcast_shapes(*_shapes):
        """Minimal stand-in that returns the first non-empty shape."""
        for s in _shapes:
            if s:
                return list(s)
        return []

    patch = PatchInfo.make(
        my_broadcast_shapes,
        torch._refs,
        "_broadcast_shapes",
        family="torch",
    )

    patch.do()
    print("active patch:", patch.name)
    print("patched function:", torch._refs._broadcast_shapes.__name__)

    patch.undo()
    print("after undo:", torch._refs._broadcast_shapes.__name__)

The four arguments to :meth:`~yobx.helpers.patch_helper.PatchInfo.make` are:

* **patch** — the replacement callable.
* **module_or_class** — the module or class whose attribute is replaced.
* **method_or_function_name** — the attribute name (string) to patch.
* **family** — a free-form category label (e.g. ``"torch"`` or
  ``"transformers"``) used in diffs and reports.

To add the custom patch alongside the built-in ones, pass it via the
``extra_patches`` argument of
:func:`~yobx.torch.patch.apply_patches_for_model`:

.. runpython::
    :showcode:
    :process:

    import torch
    import torch._refs
    from yobx.helpers.patch_helper import PatchInfo
    from yobx.torch import apply_patches_for_model

    def my_broadcast_shapes(*_shapes):
        """Minimal stand-in that returns the first non-empty shape."""
        for s in _shapes:
            if s:
                return list(s)
        return []

    my_patch = PatchInfo.make(
        my_broadcast_shapes,
        torch._refs,
        "_broadcast_shapes",
        family="custom",
    )

    with apply_patches_for_model(extra_patches=[my_patch]) as details:
        print(f"Total patches: {details.n_patches}")
        for p in details:
            print(f"  [{p.family}] {p.name}")
        print("patched function:", torch._refs._broadcast_shapes.__name__)

    print("after context, function restored:", torch._refs._broadcast_shapes.__name__)

----

How to identify which patches affected the exported graph
----------------------------------------------------------

After export, call
:meth:`~yobx.helpers.patch_helper.PatchDetails.patches_involved_in_graph` with
the :class:`torch.fx.Graph` from the
:class:`~torch.export.ExportedProgram`.  The method cross-references the
``stack_trace`` metadata on each FX node with the source location of every
registered patch and returns ``(PatchInfo, [node, …])`` pairs.

.. runpython::
    :showcode:
    :process:

    import torch
    from yobx.torch import apply_patches_for_model, use_dyn_not_str
    from yobx.torch.tiny_models import TinyBroadcastAddModel

    model = TinyBroadcastAddModel()
    inputs = TinyBroadcastAddModel._export_inputs()
    dynamic_shapes = use_dyn_not_str(TinyBroadcastAddModel._dynamic_shapes())

    with (
        torch.fx.experimental._config.patch(backed_size_oblivious=True),
        apply_patches_for_model(patch_torch=True) as details,
    ):
        ep = torch.export.export(model, (), kwargs=inputs, dynamic_shapes=dynamic_shapes)

    patches = details.patches_involved_in_graph(ep.graph)
    print(f"Patches that contributed nodes: {len(patches)}")
    for patch_info, nodes in patches:
        node_names = [n.name for n in nodes]
        print(f"  {patch_info.name}: {node_names}")

Use :meth:`~yobx.helpers.patch_helper.PatchDetails.make_report` to produce a
human-readable summary of every involved patch together with its diff:

.. code-block:: python

    report = details.make_report(patches, format="raw")
    print(report)

.. seealso::

    :ref:`l-design-patches` — background on why patches are needed and how
    the :class:`~yobx.helpers.patch_helper.PatchInfo` and
    :class:`~yobx.helpers.patch_helper.PatchDetails` data structures work.

    :ref:`patches-torch` — the full list of shipped patches with unified
    diffs.

    :ref:`l-plot-patch-model-diff` — a gallery example that applies patches,
    displays the diffs, and identifies which patches were exercised when
    exporting a real Transformers model.

    :mod:`yobx.helpers.patch_helper` — API reference for
    :class:`~yobx.helpers.patch_helper.PatchInfo` and
    :class:`~yobx.helpers.patch_helper.PatchDetails`.

    :mod:`yobx.torch.patch` — API reference for
    :func:`~yobx.torch.patch.apply_patches_for_model`.

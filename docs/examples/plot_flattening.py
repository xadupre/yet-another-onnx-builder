"""
.. _l-plot-flattening:

Registering a custom class as a pytree node
============================================

:func:`torch.export.export` requires every object that appears as a model
input or output to be decomposable into a flat list of
:class:`torch.Tensor` objects.  ``torch.utils._pytree`` handles this
decomposition, but it only knows about built-in Python containers.
Custom classes — including all the cache types from :epkg:`transformers` —
must be explicitly **registered** before exporting.

This example walks through the three steps:

1. Writing *flatten* / *unflatten* / *flatten-with-keys* callables for a
   custom dict-like class.
2. Registering the class with
   :func:`register_class_flattening
   <yobx.torch.flatten_helper.register_class_flattening>`.
3. Verifying the round-trip and then cleaning up with
   :func:`unregister_class_flattening
   <yobx.torch.flatten_helper.unregister_class_flattening>`.

See :ref:`l-design-flatten` for a full description of the flattening design
including the :epkg:`transformers` cache registrations.
"""

from dataclasses import dataclass
import torch
import torch.utils._pytree
from yobx.torch.flatten_helper import (
    make_flattening_function_for_dataclass,
    register_class_flattening,
    unregister_class_flattening,
)

# %%
# 1. Define a custom dict-like container
# ----------------------------------------
#
# We create a minimal dict subclass that stores named tensors.  This pattern
# mirrors how :class:`transformers.modeling_outputs.ModelOutput` works.


class EncoderOutput(dict):
    """Holds the output tensors produced by a (mock) encoder."""


# %%
# 2. Write the three required callables
# --------------------------------------
#
# * **flatten** — extract a flat list of tensors plus a *context* (the key
#   order) that is needed to reconstruct the original object.
# * **flatten_with_keys** — same, but pair each tensor with a
#   :class:`torch.utils._pytree.MappingKey` so that :func:`torch.export.export`
#   can refer to each leaf by name.
# * **unflatten** — given the flat tensors and the context, recreate the
#   original :class:`EncoderOutput`.


def flatten_encoder_output(obj):
    keys = list(obj.keys())
    return [obj[k] for k in keys], keys


def flatten_with_keys_encoder_output(obj):
    keys = list(obj.keys())
    values = [obj[k] for k in keys]
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(keys, values)], keys


def unflatten_encoder_output(values, context, output_type=None):
    return EncoderOutput(zip(context, values))


# %%
# 3. Register the class
# ----------------------
#
# :func:`register_class_flattening
# <yobx.torch.flatten_helper.register_class_flattening>` wraps
# ``torch.utils._pytree.register_pytree_node`` and returns ``True`` when the
# registration succeeds (``False`` when the class is already registered).

registered = register_class_flattening(
    EncoderOutput,
    flatten_encoder_output,
    unflatten_encoder_output,
    flatten_with_keys_encoder_output,
)
assert EncoderOutput in torch.utils._pytree.SUPPORTED_NODES
print("registered:", registered)

# %%
# 4. Flatten a nested structure
# ------------------------------
#
# Once registered, ``torch.utils._pytree.tree_flatten`` can decompose any
# nested Python structure that contains :class:`EncoderOutput` objects.

output = EncoderOutput(t1=torch.zeros(2, 5, 8), t2=torch.ones(2, 8))

flat, spec = torch.utils._pytree.tree_flatten(output)
print("number of leaf tensors:", len(flat))
for i, t in enumerate(flat):
    print(f"  leaf[{i}]: shape={tuple(t.shape)}, dtype={t.dtype}")

# %%
# 5. Unflatten and verify the round-trip
# ---------------------------------------
#
# :func:`torch.utils._pytree.tree_unflatten` reconstructs the original
# :class:`EncoderOutput` from the flat list using the spec returned by
# ``tree_flatten``.

restored = torch.utils._pytree.tree_unflatten(flat, spec)
print("restored type :", type(restored).__name__)
print("restored keys :", list(restored.keys()))
assert torch.equal(restored["t1"], output["t1"])
assert torch.equal(restored["t2"], output["t2"])
print("round-trip OK")

# %%
# 6. Auto-generate callables with make_flattening_function_for_dataclass
# -----------------------------------------------------------------------
#
# For classes that already expose ``.keys()`` / ``.values()`` (all
# :class:`transformers.modeling_outputs.ModelOutput` subclasses do),
# :func:`make_flattening_function_for_dataclass
# <yobx.torch.flatten_helper.make_flattening_function_for_dataclass>`
# generates the three required callables automatically.


@dataclass
class EncoderOutput2:
    """Holds the output tensors produced by a (mock) encoder."""

    t1: torch.Tensor
    t2: torch.Tensor


supported = set()
flatten_fn, flatten_with_keys_fn, unflatten_fn = make_flattening_function_for_dataclass(
    EncoderOutput2, supported
)

print("auto-generated names:")
print(" ", flatten_fn.__name__)
print(" ", flatten_with_keys_fn.__name__)
print(" ", unflatten_fn.__name__)
print("supported set:", {c.__name__ for c in supported})

# %%
# Let's register.

registered = register_class_flattening(
    EncoderOutput2, flatten_fn, unflatten_fn, flatten_with_keys_fn
)
assert EncoderOutput2 in torch.utils._pytree.SUPPORTED_NODES
print("registered:", registered)


# %%
# New test.

output2 = EncoderOutput2(t1=torch.zeros(2, 5, 8), t2=torch.ones(2, 8))

flat, spec = torch.utils._pytree.tree_flatten(output)
restored = torch.utils._pytree.tree_unflatten(flat, spec)
print("restored type :", type(restored).__name__)
print("restored keys :", list(restored.keys()))
assert torch.equal(restored["t1"], output["t1"])
assert torch.equal(restored["t2"], output["t2"])
print("round-trip OK again")


# %%
# 7. Unregister to restore the original state
# --------------------------------------------
#
# After exporting (or when running inside a test) call
# :func:`unregister_class_flattening
# <yobx.torch.flatten_helper.unregister_class_flattening>` to undo the
# registration and leave ``torch.utils._pytree.SUPPORTED_NODES`` exactly as
# it was before.
assert EncoderOutput in torch.utils._pytree.SUPPORTED_NODES
assert EncoderOutput2 in torch.utils._pytree.SUPPORTED_NODES
unregister_class_flattening(EncoderOutput)
unregister_class_flattening(EncoderOutput2)
print("EncoderOutput and EncoderOutput2 unregistered")
assert EncoderOutput not in torch.utils._pytree.SUPPORTED_NODES
assert EncoderOutput2 not in torch.utils._pytree.SUPPORTED_NODES

from typing import Any, List, Optional
import torch


class CacheKeyValue:
    """
    Wraps a transformers cache object (e.g. ``DynamicCache``, ``StaticCache``) and
    exposes ``key_cache`` and ``value_cache`` as flat lists of tensors,
    one entry per layer.

    The cache object is expected to have a ``layers`` attribute where each layer
    exposes ``keys`` and ``values`` tensors (or ``None`` when uninitialized).
    """

    def __init__(self, cache: Any):
        if not hasattr(cache, "layers"):
            raise AttributeError(
                f"Cache object of type {type(cache).__name__!r} has no 'layers' attribute"
            )
        self.key_cache: List[Optional[torch.Tensor]] = []
        self.value_cache: List[Optional[torch.Tensor]] = []
        for layer in cache.layers:
            self.key_cache.append(layer.keys)
            self.value_cache.append(layer.values)


def flatten_unflatten_for_dynamic_shapes(obj: Any) -> List[Any]:
    """
    Flattens a pytree-registered object into a list of its leaf tensors.
    Used to display objects that are registered in ``torch.utils._pytree.SUPPORTED_NODES``.
    """
    flat, _ = torch.utils._pytree.tree_flatten(obj)
    return flat

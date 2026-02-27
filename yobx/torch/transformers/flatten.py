"""
Flatten and unflatten functions for transformers cache types.

Enables registration of transformers cache classes with
:mod:`torch.utils._pytree` so that :func:`torch.export.export`
can handle them as inputs and outputs.
"""

import itertools
import re
from typing import Any, Callable, List, Set, Tuple
import torch
import transformers.cache_utils
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache, StaticCache
from transformers.modeling_outputs import BaseModelOutput
from yobx.helpers.cache_helper import make_dynamic_cache, make_static_cache, CacheKeyValue

SUPPORTED_DATACLASSES: Set[type] = set()

WRONG_REGISTRATIONS = {
    DynamicCache: "4.50",
    BaseModelOutput: None,
}

SHORTEN_LAYER_NAMES = {
    "DynamicLayer": "D",
    "DynamicSlidingWindowLayer": "W",
    "StaticLayer": "S",
    "StaticSlidingWindowLayer": "X",
    "D": "DynamicLayer",
    "W": "DynamicSlidingWindowLayer",
    "S": "StaticLayer",
    "X": "StaticSlidingWindowLayer",
}

KWARGS_LAYER_NAMES = {
    "DynamicLayer": lambda layer: "",
    "DynamicSlidingWindowLayer": lambda layer: str(
        getattr(layer, "sliding_window", getattr(layer, "max_cache_len", 0))
    ),
    "StaticLayer": lambda layer: "",
    "StaticSlidingWindowLayer": lambda layer: str(
        getattr(layer, "sliding_window", getattr(layer, "max_cache_len", 0))
    ),
}

PARSE_LAYER_NAMES = {
    "DynamicLayer": lambda skw: {},
    "DynamicSlidingWindowLayer": lambda skw: dict(sliding_window=int(skw[1:])),
    "StaticLayer": lambda skw: {},
    "StaticSlidingWindowLayer": lambda skw: dict(sliding_window=int(skw[1:])),
}


def _flatten_key_value_cache(cache: Cache) -> Tuple[List[Any], torch.utils._pytree.Context]:
    ca = CacheKeyValue(cache)
    flat = list(itertools.chain.from_iterable(zip(ca.key_cache, ca.value_cache)))
    unique = set(ca.cls_layers) if ca.cls_layers else None
    if (
        cache.__class__.__name__ != "DynamicCache"
        or unique is None
        or (len(unique) == 1 and unique.pop().__name__ == "DynamicLayer")
    ):
        keys = list(
            itertools.chain.from_iterable(
                (f"key_{i}", f"value_{i}") for i in range(len(ca.key_cache))
            )
        )
        return flat, keys

    keys = []
    for i in range(len(ca.key_cache)):
        letter = SHORTEN_LAYER_NAMES[ca.cls_layers[i].__name__]
        if hasattr(cache, "layers"):
            kwargs = KWARGS_LAYER_NAMES[ca.cls_layers[i].__name__](cache.layers[i])
        else:
            kwargs = ""
        keys.extend([f"key_{letter}{kwargs}_{i}", f"value_{letter}{kwargs}_{i}"])
    return flat, keys


def _flatten_with_keys_cache(
    cache: Cache,
) -> Tuple[List[Tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
    values, context = _flatten_key_value_cache(cache)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


def _unflatten_cache(
    make_cache: Callable,
    values: List[Any],
    context: torch.utils._pytree.Context,
    output_type=None,
) -> DynamicCache:
    """Restores a cache from python objects."""
    expected = list(
        itertools.chain.from_iterable(
            (f"key_{i}", f"value_{i}") for i in range(len(values) // 2)
        )
    )
    if expected == context:
        res = make_cache(list(zip(values[::2], values[1::2])))
    else:
        cls_layer_names = [SHORTEN_LAYER_NAMES[name.split("_")[1][0]] for name in context][::2]
        cls_kwargs = [
            PARSE_LAYER_NAMES[SHORTEN_LAYER_NAMES[name.split("_")[1][0]]](name.split("_")[1])
            for name in context
        ][::2]
        cls_layers = [
            getattr(transformers.cache_utils, cls_name) for cls_name in cls_layer_names
        ]
        res = make_cache(
            list(zip(values[::2], values[1::2])), cls_layers=cls_layers, cls_kwargs=cls_kwargs
        )

    assert output_type is None or isinstance(
        res, output_type
    ), f"Type mismatch between {output_type} (expected) and {type(res)}"
    return res


##############
# DynamicCache
##############


def flatten_dynamic_cache(
    dynamic_cache: DynamicCache,
) -> Tuple[List[Any], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.DynamicCache` with python objects."""
    return _flatten_key_value_cache(dynamic_cache)


def flatten_with_keys_dynamic_cache(
    dynamic_cache: DynamicCache,
) -> Tuple[List[Tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.DynamicCache` with python objects."""
    return _flatten_with_keys_cache(dynamic_cache)


def unflatten_dynamic_cache(
    values: List[Any], context: torch.utils._pytree.Context, output_type=None
) -> DynamicCache:
    """Restores a :class:`transformers.cache_utils.DynamicCache` from python objects."""
    return _unflatten_cache(make_dynamic_cache, values, context, output_type=output_type)


#############
# StaticCache
#############


def flatten_static_cache(
    cache: StaticCache,
) -> Tuple[List[Any], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.StaticCache` with python objects."""
    ca = CacheKeyValue(cache)
    assert not ca.key_cache or cache.max_cache_len == ca.key_cache[0].shape[2], (
        f"Serialization does not work when "
        f"cache.max_cache_len={cache.max_cache_len} != "
        f"cache.key_cache[0].shape[2]={ca.key_cache[0].shape[2]}"
    )
    return _flatten_key_value_cache(cache)


def flatten_with_keys_static_cache(
    cache: StaticCache,
) -> Tuple[List[Tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.StaticCache` with python objects."""
    return _flatten_with_keys_cache(cache)


def unflatten_static_cache(
    values: List[Any], context: torch.utils._pytree.Context, output_type=None
) -> StaticCache:
    """Restores a :class:`transformers.cache_utils.StaticCache` from python objects."""
    return _unflatten_cache(
        lambda *args, **kwargs: make_static_cache(
            *args, max_cache_len=values[0].shape[2], **kwargs
        ),
        values,
        context,
        output_type=output_type,
    )


#####################
# EncoderDecoderCache
#####################


def flatten_encoder_decoder_cache(
    ec_cache: EncoderDecoderCache,
) -> Tuple[List[Any], torch.utils._pytree.Context]:
    """
    Serializes a :class:`transformers.cache_utils.EncoderDecoderCache`
    with python objects.
    """
    dictionary = {
        "self_attention_cache": ec_cache.self_attention_cache,
        "cross_attention_cache": ec_cache.cross_attention_cache,
    }
    return torch.utils._pytree._dict_flatten(dictionary)


def flatten_with_keys_encoder_decoder_cache(
    ec_cache: EncoderDecoderCache,
) -> Tuple[
    List[Tuple[torch.utils._pytree.KeyEntry, Any]],
    torch.utils._pytree.Context,
]:
    """
    Serializes a :class:`transformers.cache_utils.EncoderDecoderCache`
    with python objects.
    """
    dictionary = {
        "self_attention_cache": ec_cache.self_attention_cache,
        "cross_attention_cache": ec_cache.cross_attention_cache,
    }
    return torch.utils._pytree._dict_flatten_with_keys(dictionary)


def unflatten_encoder_decoder_cache(
    values: List[Any], context: torch.utils._pytree.Context, output_type=None
) -> EncoderDecoderCache:
    """Restores a :class:`transformers.cache_utils.EncoderDecoderCache` from python objects."""
    dictionary = torch.utils._pytree._dict_unflatten(values, context)
    return EncoderDecoderCache(
        dictionary["self_attention_cache"], dictionary["cross_attention_cache"]
    )


#############
# dataclasses
#############


def _lower_name_with_(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def make_serialization_function_for_dataclass(
    cls: type, supported_classes: Set[type]
) -> Tuple[Callable, Callable, Callable]:
    """
    Automatically creates serialization functions for a class decorated with
    ``dataclasses.dataclass``.

    :param cls: the dataclass type
    :param supported_classes: set to register the class into
    :return: tuple of (flatten, flatten_with_keys, unflatten) callables
    """

    def flatten_cls(obj: cls) -> Tuple[List[Any], torch.utils._pytree.Context]:  # type: ignore[valid-type]
        """Serializes a ``%s`` with python objects."""
        return list(obj.values()), list(obj.keys())

    def flatten_with_keys_cls(
        obj: cls,  # type: ignore[valid-type]
    ) -> Tuple[List[Tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
        """Serializes a ``%s`` with python objects with keys."""
        values, context = list(obj.values()), list(obj.keys())
        return [
            (torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)
        ], context

    def unflatten_cls(
        values: List[Any], context: torch.utils._pytree.Context, output_type=None
    ) -> cls:  # type: ignore[valid-type]
        """Restores an instance of ``%s`` from python objects."""
        return cls(**dict(zip(context, values)))

    name = _lower_name_with_(cls.__name__)
    flatten_cls.__name__ = f"flatten_{name}"
    flatten_with_keys_cls.__name__ = f"flatten_with_keys_{name}"
    unflatten_cls.__name__ = f"unflatten_{name}"
    flatten_cls.__doc__ = flatten_cls.__doc__ % cls.__name__
    flatten_with_keys_cls.__doc__ = flatten_with_keys_cls.__doc__ % cls.__name__
    unflatten_cls.__doc__ = unflatten_cls.__doc__ % cls.__name__
    supported_classes.add(cls)
    return flatten_cls, flatten_with_keys_cls, unflatten_cls


(
    flatten_base_model_output,
    flatten_with_keys_base_model_output,
    unflatten_base_model_output,
) = make_serialization_function_for_dataclass(BaseModelOutput, SUPPORTED_DATACLASSES)

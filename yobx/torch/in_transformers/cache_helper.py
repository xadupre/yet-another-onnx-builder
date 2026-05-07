from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import transformers

KWARGS_LAYER = {}
if hasattr(transformers.cache_utils, "DynamicSlidingWindowLayer"):
    KWARGS_LAYER.update(
        {
            transformers.cache_utils.DynamicSlidingWindowLayer: lambda tensor: {
                "sliding_window": tensor.shape[2]
            },
            transformers.cache_utils.StaticSlidingWindowLayer: lambda tensor: {
                "sliding_window": tensor.shape[2]
            },
        }
    )


def _preprocess_key_value_pairs(
    key_value_pairs: Union[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    if not key_value_pairs or isinstance(key_value_pairs[0], tuple):
        return key_value_pairs  # type: ignore[return-value]
    return list(zip(key_value_pairs[::2], key_value_pairs[1::2]))  # type: ignore[arg-type]


class CacheKeyValue:
    """
    Starting transformers>=4.54, the cache API has deprecated
    ``cache.key_cache`` and ``cache.value_cache``.
    This class wraps a cache independently from transformers version and enables
    attributes ``key_cache`` and ``value_cache``.

    .. code-block:: python

        capi = CacheKeyValue(cache)
        capi.key_cache
        capi.value_cache
    """

    key_cache: Optional[List[Any]]
    value_cache: Optional[List[Any]]
    cls_layers: Optional[Union[str, List[type]]]

    def __init__(
        self, cache: Optional[Any] = None, cls_layers: Optional[Union[str, List[type]]] = None
    ):
        if hasattr(cache, "layers"):
            assert cache is not None
            layers = [
                layer
                for layer in cache.layers
                if layer is not None and layer.keys is not None and layer.values is not None
            ]
            self.key_cache = [layer.keys for layer in layers]
            self.value_cache = [layer.values for layer in layers]
            assert (
                cls_layers is None
            ), f"cache is {type(cache)}, cannot specify cls_layers={cls_layers}"
            self.cls_layers = [type(lay) for lay in cache.layers]
        elif cache is not None and hasattr(cache, "key_cache"):
            self.key_cache = cache.key_cache
            self.value_cache = cache.value_cache
            self.cls_layers = cls_layers
        elif (
            cache is not None
            and isinstance(cache, list)
            and all(isinstance(t, torch.Tensor) for t in cache)
        ):
            self.key_cache = cache[::2]
            self.value_cache = cache[1::2]
            self.cls_layers = cls_layers
        elif cache is None:
            self.key_cache = None
            self.value_cache = None
            self.cls_layers = cls_layers
        else:
            raise NotImplementedError(f"type(cache)={type(cache)}")

    def make_dynamic_cache(self):
        """Does the reverse operation."""
        assert self.key_cache is not None and self.value_cache is not None
        return make_dynamic_cache(
            list(zip(self.key_cache, self.value_cache)), cls_layers=self.cls_layers  # type: ignore
        )

    @property
    def n_layers(self) -> int:
        """Returns the number of layers."""
        return len(self.key_cache) if self.key_cache else 0

    def __len__(self) -> int:
        "Returns the number of tensors."
        return len(self.key_cache or []) + len(self.value_cache or [])

    def aslist(self) -> List[torch.Tensor]:
        "Returns tensors in a list."
        res: List[torch.Tensor] = []
        if self.key_cache is None or self.value_cache is None:
            return res
        for i in range(self.n_layers):
            res.append(self.key_cache[i])
            res.append(self.value_cache[i])
        return res


def flatten_unflatten_for_dynamic_shapes(
    obj: Any,
    use_dict: bool = False,
    change_function: Optional[Callable[[torch.Tensor], Any]] = None,
) -> Any:
    """
    Returns the object in a different structure similar to what
    the definition of the dynamic shapes should use.

    :param obj: object from a custom class
    :param use_dict: closer to the original result but
        :func:`torch.export.export` only considers the values,
        the context gives the dictionary keys but it is not expressed
        in the dynamic shapes, these specifications seems to be different
        for the strict and non strict mode. It also preserves tuple.
    :param change_function: to modify the tensor in the structure itself,
        like replace them by a shape
    :return: the serialized object
    """
    if isinstance(obj, torch.Tensor):
        return change_function(obj) if change_function else obj
    flat, spec = torch.utils._pytree.tree_flatten(obj)  # pyrefly: ignore[implicit-import]
    start = 0
    end = 0
    subtrees = []
    for subspec in spec.children():
        end += subspec.num_leaves
        value = subspec.unflatten(flat[start:end])
        value = flatten_unflatten_for_dynamic_shapes(
            value, use_dict=use_dict, change_function=change_function
        )
        subtrees.append(value)
        start = end
    if use_dict:
        if spec.type is dict:
            # This is a dictionary.
            return dict(zip(spec.context, subtrees))
        if spec.type is tuple:
            return tuple(subtrees)
        if spec.type is list:
            return list(subtrees)
        if spec.type is None and not subtrees:
            return None
        if spec.context:
            # This is a custom class with attributes.
            # It is returned as a list.
            return list(subtrees)
        raise ValueError(
            f"Unable to interpret spec type {spec.type} "
            f"(type is {type(spec.type)}, context is {spec.context}), "
            f"spec={spec}, subtrees={subtrees}"
        )
    # This is a list.
    return subtrees


def make_dynamic_cache(
    key_value_pairs: Union[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]],
    cls_layers: Optional[Union[str, List[type]]] = None,
    cls_kwargs: Optional[Union[Dict[str, int], List[Dict[str, int]]]] = None,
) -> transformers.cache_utils.DynamicCache:
    """
    Creates an instance of :class:`transformers.cache_utils.DynamicCache`.
    This version is valid for ``transformers >= 4.50``.

    :param key_value_pairs: list of pairs of (key, values)
    :param cls_layers: to select the appropriate class to use on each layer,
        if specified, sliding_window is ignored, it can be a string
        if all layers are expected to follow the same class
    :param cls_kwargs: arguments used to build a specific layer,
        such as ``sliding_window`` for ``DynamicSlidingWindowLayer``
    :return: :class:`transformers.cache_utils.DynamicCache`

    Example:

    .. runpython::
        :showcode:
        :process:

        import torch
        from yobx.helpers import string_type
        from yobx.torch.in_transformers.cache_helper import make_dynamic_cache

        n_layers = 2
        bsize, nheads, slen, dim = 2, 4, 3, 7

        past_key_values = make_dynamic_cache(
            [
                (
                    torch.randn(bsize, nheads, slen, dim),
                    torch.randn(bsize, nheads, slen, dim),
                )
                for i in range(n_layers)
            ]
        )
        print(string_type(past_key_values, with_shape=True))

    The function is fully able to handle ``FakeTensor`` with dynamic dimensions if
    ``transformers>=4.56``. Before that version, only FakeTensor with static dimensions
    are supported.
    """
    key_value_pairs = _preprocess_key_value_pairs(key_value_pairs)
    if isinstance(cls_layers, str):
        assert hasattr(
            transformers.cache_utils, cls_layers
        ), f"Missing layer class {cls_layers!r}"
        cls_layers = getattr(transformers.cache_utils, cls_layers)
    if cls_layers and not isinstance(cls_layers, list):
        cls_layers = [cls_layers for _ in key_value_pairs]  # type: ignore[misc]
    if cls_layers is not None and isinstance(cls_layers, list):
        assert len(cls_layers) == len(key_value_pairs), (
            f"Length mismatch {len(key_value_pairs)} expected but "
            f"{len(cls_layers)} layer types are given."
        )
        if cls_kwargs is None:
            cls_kwargs = [{} for _kv in key_value_pairs]  # type: ignore[assignment]
        assert len(cls_layers) == len(cls_kwargs), (
            f"Length mismatch {len(cls_kwargs)} expected but "
            f"{len(cls_layers)} layer types are given, "
            f"cls_layers={cls_layers}, cls_kwargs={cls_kwargs}"
        )
        cls_layer = None
        assert (
            key_value_pairs and key_value_pairs[0]
        ), f"not implemented for type(key_value_pairs[0])={type(key_value_pairs[0])}"
        for kv, clsy, kws in zip(key_value_pairs, cls_layers, cls_kwargs):
            default_values = KWARGS_LAYER.get(clsy, lambda tensor: {})(kv[0])  # type: ignore[call-overload]
            for k, v in default_values.items():
                if k not in kws:
                    kws[k] = v  # type: ignore[index]
    else:
        assert cls_kwargs is None, "cls_layers must be a list if cls_kwargs is specified"
        assert cls_layers is None, f"cls_layers must be list or a string but it is {cls_layers}"
        cls_kwargs = {}
        cls_layer = (
            transformers.cache_utils.DynamicLayer
            if hasattr(transformers.cache_utils, "DynamicLayer")
            else None
        )

    if cls_layer is not None:
        assert isinstance(cls_kwargs, dict), (
            f"one layer = one set of arguments, cls_layer={cls_layer}, "
            f"cls_kwargs={cls_kwargs}"
        )
        cls_layers = [cls_layer for _ in key_value_pairs]
        cls_kwargs = (
            cls_kwargs  # type: ignore[assignment]
            if isinstance(cls_kwargs, list)
            else [cls_kwargs for _ in key_value_pairs]
        )
    elif cls_layers is not None:
        assert isinstance(cls_layers, list), f"Unexpected type cls_layers={cls_layers}"
        assert isinstance(cls_kwargs, list), f"Unexpected type cls_kwargs={cls_kwargs}"

    from ...pv_version import PvVersion

    if (
        key_value_pairs
        and isinstance(
            key_value_pairs[0][0], (torch._subclasses.fake_tensor.FakeTensor, torch.fx.Proxy)
        )
        and PvVersion(transformers.__version__) >= PvVersion("4.56")
    ):
        cache = transformers.cache_utils.DynamicCache()
        cache.layers.extend(
            [cls_layer(**kws) for cls_layer, kws in zip(cls_layers, cls_kwargs)]  # type: ignore[operator, arg-type]
        )
        for i, layer in enumerate(cache.layers):
            k, v = key_value_pairs[i][0], key_value_pairs[i][1]
            if not isinstance(k, torch.fx.Proxy):
                layer.dtype = k.dtype  # type: ignore
                layer.device = k.device  # type: ignore
            layer.keys = k  # type: ignore
            layer.values = v  # type: ignore
            layer.is_initialized = True  # type: ignore
        assert not hasattr(cache, "layers") or len(key_value_pairs) == len(cache.layers), (
            f"Unexpected number of layers in the cache ({len(cache.layers)}), "
            f"{len(key_value_pairs)} expected."
        )
        return finalize_cache(cache)  # type: ignore[return-value]

    cache = transformers.cache_utils.DynamicCache()
    if hasattr(cache, "layers") and (
        cls_layer is None or cls_layer != transformers.cache_utils.DynamicLayer
    ):
        assert isinstance(cls_layers, list) and isinstance(cls_kwargs, list), (
            f"Wrong type {type(cls_layers)} for cls_layers or "
            f"{type(cls_kwargs)} for cls_kwargs"
        )
        assert len(cls_kwargs) == len(cls_layers) and len(cls_kwargs) == len(key_value_pairs), (
            f"Length mismatch between len(cls_kwargs)={len(cls_kwargs)}, "
            f"len(cls_layers)={len(cls_layers)}, "
            f"len(key_value_pairs)={len(key_value_pairs)}, "
            f"cls_kwargs={cls_kwargs}, cls_layers={cls_layers}"
        )
        del cache.layers[:]
        cache.layers.extend(
            [cls_layer(**kws) for cls_layer, kws in zip(cls_layers, cls_kwargs)]  # type: ignore[operator, arg-type]
        )
        for i, layer in enumerate(cache.layers):
            layer.keys, layer.values = key_value_pairs[i][0], key_value_pairs[i][1]  # type: ignore
            layer.is_initialized = True  # type: ignore
    else:
        cache = transformers.cache_utils.DynamicCache(key_value_pairs)
        if hasattr(cache, "layers") and len(key_value_pairs) < len(cache.layers):
            # The cache constructor contains the two following lines
            # (in cache_utils.py) which append empty layers when the cache is
            # initialized. We need to remove them.
            # self.num_hidden_layers = getattr(config, "num_hidden_layers", 1)
            # self.append_new_layers(self.num_hidden_layers - 1)
            cache.layers[:] = cache.layers[-len(key_value_pairs) :]
    assert not hasattr(cache, "layers") or len(key_value_pairs) == len(cache.layers), (
        f"Unexpected number of layers in the cache ({len(cache.layers)}), "
        f"{len(key_value_pairs)} expected."
    )
    return finalize_cache(cache)  # type: ignore[return-value]


def make_dynamic_shapes_kv_cache(
    cache: transformers.cache_utils.Cache, shape_of_one: Dict[int, Any]
) -> List[Dict[int, Any]]:
    """
    Returns the dynamic shapes for key-value cache

    :param cache: a cache
    :param shape_of_one: shape of one element
    :return: dynamic shapes
    """
    return [shape_of_one for _ in range(CacheKeyValue(cache).n_layers * 2)]


def finalize_cache(cache: transformers.cache_utils.Cache) -> transformers.cache_utils.Cache:
    """
    Ensures the created cache is consistent.
    Returns the cache modified inplace.
    """
    if (
        hasattr(cache, "layer_class_to_replicate")
        and hasattr(cache, "layers")
        and cache.layers
        and not cache.layer_class_to_replicate
    ):
        # This is used to expand the cache when it does not contains enough layers.
        # This is needed since transformers>4.55.3
        cache.layer_class_to_replicate = cache.layers[0].__class__
    assert (
        not hasattr(cache, "layers") or len(cache.layers) != 1 or cache.layers[0].keys is not None  # type: ignore
    ), (
        f"Size mismatch between {len(cache.layers)=}, "
        f"first key={cache.layers[0].keys}, "  # type: ignore
        f"first value={cache.layers[0].values}"  # type: ignore
    )
    assert not hasattr(cache, "layers") or all(
        not hasattr(layer, "is_initialized") or layer.is_initialized for layer in cache.layers
    ), f"A layyer (among {len(cache.layers)}) is not initialized."
    return cache


def make_static_cache(
    key_value_pairs: Union[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]],
    max_cache_len: Optional[int] = None,
    cls_layers: Optional[Union[str, List[type]]] = None,
) -> transformers.cache_utils.DynamicCache:
    """
    Creates an instance of :class:`transformers.cache_utils.StaticCache`.
    :param key_value_pairs: list of pairs of (key, values)
    :param max_cache_len: max_cache_length or something inferred from the vector
    :return: :class:`transformers.cache_utils.StaticCache`

    Example:

    .. runpython::
        :showcode:
        :process:

        import torch
        from yobx.helpers import string_type
        from yobx.torch.in_transformers.cache_helper import make_static_cache

        n_layers = 2
        bsize, nheads, slen, dim = 2, 4, 3, 7

        past_key_values = make_static_cache(
            [
                (
                    torch.randn(bsize, nheads, slen, dim),
                    torch.randn(bsize, nheads, slen, dim),
                )
                for i in range(n_layers)
            ],
            max_cache_len=10,
        )
        print(string_type(past_key_values, with_shape=True))
    """
    assert not cls_layers or set(cls_layers) == {
        transformers.cache_utils.StaticLayer
    }, f"Not implemented when cls_layers={cls_layers!r}"
    key_value_pairs = _preprocess_key_value_pairs(key_value_pairs)

    class _config:
        def __init__(self):
            self.head_dim = key_value_pairs[0][0].shape[-1]
            self.num_attention_heads = key_value_pairs[0][0].shape[1]
            self.num_hidden_layers = len(key_value_pairs)

        def get_text_config(self, *args, **kwargs):
            return self

    assert max_cache_len is not None, (
        f"max_cache_len={max_cache_len} cannot be setup "
        f"automatically yet from shape {key_value_pairs[0][0].shape}"
    )
    torch._check(
        max_cache_len >= key_value_pairs[0][0].shape[2],
        (
            f"max_cache_len={max_cache_len} cannot be smaller "
            f"shape[2]={key_value_pairs[0][0].shape[2]} in shape "
            f"{key_value_pairs[0][0].shape}"
        ),
    )
    cache = transformers.cache_utils.StaticCache(
        config=_config(),  # type: ignore[arg-type]
        max_batch_size=key_value_pairs[0][0].shape[0],
        device=key_value_pairs[0][0].device,
        dtype=key_value_pairs[0][0].dtype,
        max_cache_len=max_cache_len,
    )
    assert hasattr(cache, "layers"), f"Missing attribute 'layers' for {cache!r}"
    # transformers>= 4.55.2, layers are empty
    for i, (key, value) in enumerate(key_value_pairs):
        cache.update(key, value, i)
    return finalize_cache(cache)  # type: ignore[return-value]


def make_encoder_decoder_cache(
    self_attention_cache: transformers.cache_utils.DynamicCache,
    cross_attention_cache: transformers.cache_utils.DynamicCache,
) -> transformers.cache_utils.EncoderDecoderCache:
    """Creates an EncoderDecoderCache."""
    return transformers.cache_utils.EncoderDecoderCache(
        # self_attention_cache=self_attention_cache,
        # cross_attention_cache=cross_attention_cache
        self_attention_cache,
        cross_attention_cache,
    )

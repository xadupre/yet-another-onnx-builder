from typing import Any, Dict, Optional, Set, Tuple
import torch


class FakeTensorContext:
    """Stores information used to reuse same dimension for the same dimension names."""

    def __init__(
        self, fake_mode: Optional["torch._subclasses.fake_tensor.FakeTensorMode"] = None
    ):
        if fake_mode is None:
            from torch._subclasses.fake_tensor import FakeTensorMode
            from torch.fx.experimental.symbolic_shapes import ShapeEnv

            shape_env = ShapeEnv()
            self.fake_mode = FakeTensorMode(shape_env=shape_env)
        else:
            self.fake_mode = fake_mode
        self._candidates = self._first_primes()
        self._unique_: Set[int] = set()
        self._mapping_int: Dict[int, Any] = {}
        self._mapping_str: Dict[str, int] = {}

    @classmethod
    def _first_primes(cls, n=1000):
        sieve = [True] * (n + 1)
        sieve[0:2] = [False, False]

        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                # Élimine les multiples de i
                sieve[i * i : n + 1 : i] = [False] * len(range(i * i, n + 1, i))

        return [i for i, prime in enumerate(sieve) if prime and i >= 13]

    def _unique(self) -> int:
        i = 0
        c = self._candidates[i]
        while c in self._unique_ or c in self._mapping_int:
            i += 1
            assert i < len(
                self._candidates
            ), f"Too many unique dimensions to generate, requested: {len(self._unique_)}"
            c = self._candidates[i]
        self._unique_.add(c)
        return c

    def from_tensor(self, x, static_shapes=False) -> "torch._subclasses.fake_tensor.FakeTensor":
        """
        Returns a fake tensor.
        ``pytorch`` returns the same name for the same dimension.
        """
        fake = self.fake_mode.from_tensor(x, static_shapes=static_shapes)
        for i, s in zip(x.shape, fake.shape):
            assert i not in self._mapping_int or self._mapping_int[i] == s, (
                f"Inconsistency between {x.shape} and {fake.shape}, "
                f"mapping has {self._mapping_int[i]} and s={s}"
            )
            self._mapping_int[i] = s
        return fake

    def fake_reshape(
        self,
        true_tensor: torch.Tensor,
        sh: Dict[int, Any],
        fake_tensor: Optional["torch._subclasses.fake_tensor.FakeTensor"] = None,
    ) -> "torch._subclasses.fake_tensor.FakeTensor":
        """
        Changes the shape of a true tensor to make it dynamic.

        :param true_tensor: true tensor
        :param sh: dynamic shape
        :param fake_tensor: fake tensor, if None, make a fake one
        :return: fake tensor
        """
        import torch

        # deal with 0/1
        for i in sh:
            if true_tensor.shape[i] <= 1:
                expanded_shape = list(true_tensor.shape)
                expanded_shape[i] = self._unique()
                true_tensor = torch.empty(
                    tuple(expanded_shape), dtype=true_tensor.dtype, device=true_tensor.device
                )

        # deal with equivalent dimension
        new_shape = list(true_tensor.shape)
        mapping = {}
        for i, s in sh.items():
            d = true_tensor.shape[i]
            if d not in mapping:
                mapping[d] = s
            elif mapping[d] != s:
                d = self._unique()
                mapping[d] = s
                new_shape[i] = d
        true_tensor = torch.empty(
            tuple(new_shape), dtype=true_tensor.dtype, device=true_tensor.device
        )

        # now switch to FakeTensor
        fake_tensor = self.from_tensor(true_tensor, static_shapes=False)
        new_shape = list(true_tensor.shape)
        for i in sh:
            new_shape[i] = fake_tensor.shape[i]

        reduced_tensor = self.from_tensor(true_tensor, static_shapes=True).sum(
            dim=tuple(sorted(sh)), keepdim=True
        )
        if len(reduced_tensor.shape) == 0 == len(new_shape):
            return fake_tensor
        return reduced_tensor.expand(*new_shape)  # type: ignore[return-value]

    def make_fake(self, x: Any) -> Optional[Any]:
        """See :func:`yobx.torch.fake_tensor_helper.make_fake`."""
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return x.__class__([self.make_fake(i) for i in x])
        if isinstance(x, dict):
            return {k: self.make_fake(v) for k, v in x.items()}
        if x.__class__.__name__ in {"DynamicCache", "StaticCache", "HybridCache"}:
            assert hasattr(x, "layers"), (
                f"A more recent version of transformers (>=4.55), "
                f"'layers' not found in class {type(x)}"
            )
            for layer in x.layers:
                assert hasattr(layer, "keys") and hasattr(layer, "values"), (
                    f"A more recent version of transformers (>=4.55), 'layers' "
                    f"not found in class {type(layer)} ({dir(layer)})"
                )
                layer.keys = self.make_fake(layer.keys)
                layer.values = self.make_fake(layer.values)
            return x
        if x.__class__.__name__ == "EncoderDecoderCache":
            self.make_fake(x.self_attention_cache)
            self.make_fake(x.cross_attention_cache)
            return x
        if hasattr(x, "shape"):
            return self.from_tensor(x, static_shapes=False)
        from ..helpers import string_type

        raise TypeError(
            f"Unexpected type {type(x)} for x, content is {string_type(x, with_shape=True)}"
        )

    def make_fake_with_dynamic_dimensions(self, x: Any, dynamic_shapes: Any) -> Any:
        """
        See
        :func:`yobx.torch.make_fake_with_dynamic_dimensions`.
        If caches are used, it requires ``transformers>=4.57``.
        """
        if x is None:
            return None
        if type(x) in (list, tuple):
            if dynamic_shapes is None:
                ds_list = [None] * len(x)
            else:
                assert len(x) == len(dynamic_shapes), (
                    f"Length mismatch between x (len={len(x)}) and "
                    f"dynamic_shapes (len={len(dynamic_shapes)}); "
                    f"dynamic_shapes must have one entry per element of x, "
                    f"or be None to use no dynamic dimensions, "
                    f"dynamic_shapes={dynamic_shapes}"
                )
                ds_list = dynamic_shapes
            return x.__class__(
                [
                    self.make_fake_with_dynamic_dimensions(i, dynamic_shapes=ds)
                    for i, ds in zip(x, ds_list)
                ]
            )
        if type(x) is dict:
            return {
                k: self.make_fake_with_dynamic_dimensions(
                    v, dynamic_shapes=dynamic_shapes[k] if dynamic_shapes else None
                )
                for k, v in x.items()
            }
        if x.__class__.__name__ in {"DynamicCache", "StaticCache", "HybridCache"}:
            assert hasattr(x, "layers"), (
                f"Une more recent version of transformers (>=4.55), "
                f"'layers' not found in class {type(x)}"
            )
            assert dynamic_shapes is None or (
                isinstance(dynamic_shapes, list)
                and (not dynamic_shapes or not isinstance(dynamic_shapes[0], list))
            ), f"Unexpected dynamic_shapes={dynamic_shapes} for a DynamicCache"
            for il, layer in enumerate(x.layers):
                assert hasattr(layer, "keys") and hasattr(layer, "values"), (
                    f"Une more recent version of transformers (>=4.55), 'layers' "
                    f"not found in class {type(layer)} ({dir(layer)})"
                )
                layer.keys = self.make_fake_with_dynamic_dimensions(
                    layer.keys, dynamic_shapes=dynamic_shapes[il * 2] if dynamic_shapes else None
                )
                layer.values = self.make_fake_with_dynamic_dimensions(
                    layer.values,
                    dynamic_shapes=dynamic_shapes[il * 2 + 1] if dynamic_shapes else None,
                )
            return x
        if x.__class__.__name__ == "EncoderDecoderCache":
            self.make_fake_with_dynamic_dimensions(
                x.self_attention_cache, dynamic_shapes=dynamic_shapes[0]
            )
            self.make_fake_with_dynamic_dimensions(
                x.cross_attention_cache, dynamic_shapes=dynamic_shapes[1]
            )
            return x
        if x.__class__.__name__ == "BaseModelOutput":
            assert list(x.keys()) == ["last_hidden_state"] and x.last_hidden_state is not None, (
                f"Field 'last_hidden_state' is empty for {type(x)} or other fields "
                f"{list(x.keys())} are used."
            )
            x.last_hidden_state = self.make_fake_with_dynamic_dimensions(
                x.last_hidden_state, dynamic_shapes=dynamic_shapes[0]
            )
            return x
        if hasattr(x, "shape"):
            assert dynamic_shapes is None or isinstance(dynamic_shapes, dict), (
                f"dynamic_shapes must be a dictionary at this stage but "
                f"dynamic_shapes={dynamic_shapes}"
            )
            # We need to overwrite the values.
            new_shape = []
            for idim, dim in enumerate(x.shape):
                if dynamic_shapes is not None and idim in dynamic_shapes:
                    s = dynamic_shapes[idim]
                    if s.__class__.__name__ == "Dim":
                        s = s.__name__
                    if isinstance(s, str):
                        if s in self._mapping_str:
                            dim = self._mapping_str[s]
                        else:
                            i = self._unique()
                            self._mapping_str[s] = i
                            dim = i
                    else:
                        # torch.export.Dim.DYNAMIC, torch.export.Dim.AUTO, and other
                        # _DimHint values are unnamed dynamic dimensions; treat each
                        # occurrence as an independent dimension with a fresh unique size.
                        dim = self._unique()
                assert isinstance(dim, int), (
                    f"Unexpected type {type(dim)}, dynamic_shapes={dynamic_shapes} "
                    f"at index {idim}, dim={dim}"
                )
                new_shape.append(dim)
            if tuple(new_shape) != x.shape:
                import torch

                x = torch.empty(tuple(new_shape), dtype=x.dtype, device=x.device)

            if dynamic_shapes is not None:
                t = self.fake_reshape(x, dynamic_shapes)  # type: ignore[arg-type]
            else:
                t = self.from_tensor(x, static_shapes=True)
            assert t.device == x.device, f"device mismatch {x.device} -> {t.device}"
            assert t.dtype == x.dtype, f"dtype mismatch {x.dtype} -> {t.dtype}"
            return t
        if isinstance(x, (int, bool, float)):
            # It is a constant, we don't change that.
            return x
        from ..helpers import string_type

        raise TypeError(
            f"Unexpected type {type(x)} for x, content is {string_type(x, with_shape=True)}"
        )

    def value_info_proto_to_torch(
        self, vip: Any
    ) -> Tuple["torch._subclasses.fake_tensor.FakeTensor", Dict[int, str]]:
        """Convert an :class:`onnx.ValueInfoProto` to a fake :class:`torch.Tensor`.

        Symbolic dimensions (those with a non-empty ``dim_param``) are assigned
        unique prime concrete sizes so that :func:`torch.export.export` sees
        distinct values.  The mapping from axis index to symbolic-dimension name
        is returned so the caller can construct a ``dynamic_shapes`` argument for
        :func:`torch.export.export`.

        A :class:`ValueError` is raised when the ``ValueInfoProto`` has no
        shape information, or when a dimension has ``dim_value <= 0`` with no
        ``dim_param`` (unknown/unset dimensions must be represented as symbolic
        dims using ``dim_param``).

        :param vip: an ONNX value-info descriptor (:class:`onnx.ValueInfoProto`)
        :return: ``(fake_tensor, dynamic_axes)`` where *dynamic_axes* maps each
            dynamic dimension index to its symbolic name (empty dict when there
            are no symbolic dimensions)
        :raises ValueError: if the shape is missing or a dimension is
            unresolvable
        """
        from onnx import TensorProto as _TensorProto
        from .torch_helper import onnx_dtype_to_torch_dtype

        tt = vip.type.tensor_type
        elem_type = tt.elem_type if tt.elem_type else _TensorProto.FLOAT
        torch_dtype = onnx_dtype_to_torch_dtype(elem_type)

        dynamic_axes: Dict[int, str] = {}
        if tt.HasField("shape"):
            shape = []
            for i, dim in enumerate(tt.shape.dim):
                if dim.dim_param:
                    # Symbolic dimension — pick (or reuse) a unique prime size.
                    name = dim.dim_param
                    if name in self._mapping_str:
                        concrete = self._mapping_str[name]
                    else:
                        concrete = self._unique()
                        self._mapping_str[name] = concrete
                    shape.append(concrete)
                    dynamic_axes[i] = name
                else:
                    value = dim.dim_value
                    if value <= 0:
                        raise ValueError(
                            f"Dimension {i} of ValueInfoProto {vip.name!r} has "
                            f"dim_value={value} with no dim_param. "
                            "Please set a positive dim_value or a symbolic dim_param."
                        )
                    shape.append(value)
        else:
            raise ValueError(
                f"ValueInfoProto {vip.name!r} has no shape information. "
                "Please set the shape field."
            )

        if shape:
            real_tensor = torch.empty(tuple(shape), dtype=torch_dtype)
            if dynamic_axes:
                fake_tensor = self.fake_reshape(real_tensor, dynamic_axes)
            else:
                fake_tensor = self.from_tensor(real_tensor, static_shapes=True)
        else:
            # Scalar tensor (0-D).
            real_tensor = torch.empty((), dtype=torch_dtype)
            fake_tensor = self.from_tensor(real_tensor, static_shapes=True)

        return fake_tensor, dynamic_axes


def make_fake(
    x: Any, context: Optional[FakeTensorContext] = None
) -> Tuple[Optional["torch._subclasses.fake_tensor.FakeTensor"], Optional[FakeTensorContext]]:
    """
    Replaces all tensors by fake tensors.
    This modification happens inplace for caches.
    This function is only implemented for cache with
    ``transformers>=4.55``.

    .. runpython::
        :showcode:
        :process:

        import pprint
        import torch
        from yobx.torch.fake_tensor_helper import make_fake
        from yobx.torch.in_transformers.cache_helper import make_dynamic_cache

        inputs, _ = make_fake(
            dict(
                input_ids=torch.randint(30360, size=(2, 3), dtype=torch.int64),
                attention_mask=torch.randint(1, size=(2, 33), dtype=torch.int64),
                position_ids=torch.randint(32, size=(2, 3), dtype=torch.int64),
                past_key_values=make_dynamic_cache(
                    [
                        (
                            torch.rand((2, 32, 30, 96), dtype=torch.float16),
                            torch.rand((2, 32, 30, 96), dtype=torch.float16),
                        ),
                        (
                            torch.rand((2, 32, 30, 96), dtype=torch.float16),
                            torch.rand((2, 32, 30, 96), dtype=torch.float16),
                        ),
                    ]
                ),
            )
        )
        pprint.pprint(inputs)
    """
    if x is None:
        return None, None
    if context is None:
        context = FakeTensorContext()
    return context.make_fake(x), context


def make_fake_with_dynamic_dimensions(
    x: Any, dynamic_shapes: Any, context: Optional[FakeTensorContext] = None
) -> Tuple[Optional[Any], Optional[FakeTensorContext]]:
    """
    Replaces all tensors by fake tensor respecting the same
    constraints as the following dynamic shapes.
    This uses function :func:`yobx.torch.fake_tensor_helper.make_fake`.
    Parameter ``existing`` is used to reuse the same object when the dynamic
    dimension is given the same name as another one.
    This function works with caches only if ``transformers>=4.57``.

    A simple tensor:

    .. runpython::
        :showcode:
        :process:

        import torch
        from yobx.torch.in_transformers.cache_helper import make_dynamic_cache
        from yobx.torch.fake_tensor_helper import make_fake_with_dynamic_dimensions

        inputs, _ = make_fake_with_dynamic_dimensions(
            torch.rand((2, 3, 4, 5), dtype=torch.float32),
            {0: "batch", 2: "cache_length"},
        )
        print(inputs)

    Two tensors:

    .. runpython::
        :showcode:
        :process:

        import torch
        from yobx.torch.in_transformers.cache_helper import make_dynamic_cache
        from yobx.torch.fake_tensor_helper import make_fake_with_dynamic_dimensions

        inputs, _ = make_fake_with_dynamic_dimensions(
            (
                torch.rand((2, 3, 4, 5), dtype=torch.float32),
                torch.rand((2, 3, 4, 5), dtype=torch.float32),
            ),
            ({0: "batch", 2: "cache_length"}, {0: "batch", 2: "cache_length"}),
        )
        print(inputs)

    With a cache:

    .. runpython::
        :showcode:
        :process:

        import pprint
        import torch
        from yobx.torch.in_transformers.cache_helper import make_dynamic_cache
        from yobx.torch.fake_tensor_helper import make_fake_with_dynamic_dimensions

        inputs, _ = make_fake_with_dynamic_dimensions(
            dict(
                input_ids=torch.randint(30360, size=(2, 3), dtype=torch.int64),
                attention_mask=torch.randint(1, size=(2, 33), dtype=torch.int64),
                position_ids=torch.randint(32, size=(2, 3), dtype=torch.int64),
                past_key_values=make_dynamic_cache(
                    [
                        (
                            torch.rand((2, 32, 30, 96), dtype=torch.float16),
                            torch.rand((2, 32, 30, 96), dtype=torch.float16),
                        ),
                        (
                            torch.rand((2, 32, 30, 96), dtype=torch.float16),
                            torch.rand((2, 32, 30, 96), dtype=torch.float16),
                        ),
                    ]
                ),
            ),
            dynamic_shapes={
                "input_ids": {0: "batch", 1: "seq_length"},
                "attention_mask": {0: "batch", 1: "cache+seq"},
                "position_ids": {0: "batch", 1: "seq_length"},
                "past_key_values": [
                    {0: "batch", 2: "cache_length"},
                    {0: "batch", 2: "cache_length"},
                    {0: "batch", 2: "cache_length"},
                    {0: "batch", 2: "cache_length"},
                ],
            },
        )
        pprint.pprint(inputs)
    """
    if x is None:
        return None, None
    if context is None:
        context = FakeTensorContext()
    return context.make_fake_with_dynamic_dimensions(x, dynamic_shapes), context

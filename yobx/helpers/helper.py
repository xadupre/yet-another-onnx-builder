import enum
import inspect
import json
from dataclasses import is_dataclass, fields
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def _string_tensor(obj, cls: str, with_shape: bool, with_device: bool) -> str:
    from ..torch.torch_helper import torch_dtype_to_onnx_dtype

    i = torch_dtype_to_onnx_dtype(obj.dtype)
    prefix = ("G" if obj.get_device() >= 0 else "C") if with_device else ""
    if not with_shape:
        return f"{prefix}{cls}{i}r{len(obj.shape)}"
    return f"{prefix}{cls}{i}s{'x'.join(map(str, obj.shape))}"


def string_type(
    obj: Any,
    with_shape: bool = False,
    with_min_max: bool = False,
    with_device: bool = False,
    ignore: bool = False,
    limit: int = 20,
) -> str:
    """
    Displays the types of an object as a string.

    :param obj: any
    :param with_shape: displays shapes as well
    :param with_min_max: displays information about the values
    :param with_device: display the device
    :param ignore: if True, just prints the type for unknown types
    :return: str

    The function displays something like the following for a tensor.

    .. code-block:: text

        T7s2x7[0.5:6:A3.56]
        ^^^+-^^----+------^
        || |       |
        || |       +-- information about the content of a tensor or array
        || |           [min,max:A<average>]
        || |
        || +-- a shape
        ||
        |+-- integer following the code defined by onnx.TensorProto,
        |    7 is onnx.TensorProto.INT64 (see onnx_dtype_name)
        |
        +-- A,T,F
            A is an array from numpy
            T is a Tensor from pytorch
            F is a FakeTensor from pytorch

    The element types for a tensor are displayed as integer to shorten the message.
    The semantic is defined by :class:`onnx.TensorProto` and can be obtained
    by :func:`yobx.helpers.onnx_helper.onnx_dtype_name`.

    .. runpython::
        :showcode:

        from yobx.helpers import string_type

        print(string_type((1, ["r", 6.6])))

    With pytorch:

    .. runpython::
        :showcode:

        import torch
        from yobx.helpers import string_type

        inputs = (
            torch.rand((3, 4), dtype=torch.float16),
            [
                torch.rand((5, 6), dtype=torch.float16),
                torch.rand((5, 6, 7), dtype=torch.float16),
            ]
        )

        # with shapes
        print(string_type(inputs, with_shape=True))

        # with min max
        print(string_type(inputs, with_shape=True, with_min_max=True))
    """
    if obj is None:
        return "None"

    # tuple
    if isinstance(obj, tuple):
        if len(obj) == 1:
            s = string_type(
                obj[0],
                with_shape=with_shape,
                with_min_max=with_min_max,
                with_device=with_device,
                ignore=ignore,
                limit=limit,
            )
            return f"({s},)"
        if len(obj) < limit:
            js = ",".join(
                string_type(
                    o,
                    with_shape=with_shape,
                    with_min_max=with_min_max,
                    with_device=with_device,
                    ignore=ignore,
                    limit=limit,
                )
                for o in obj
            )
            return f"({js})"
        tt = string_type(
            obj[0],
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            ignore=ignore,
            limit=limit,
        )
        if with_min_max and all(isinstance(_, (int, float, bool)) for _ in obj):
            mini, maxi, avg = min(obj), max(obj), sum(float(_) for _ in obj) / len(obj)
            return f"#{len(obj)}({tt},...)[{mini},{maxi}:A[{avg}]]"
        return f"#{len(obj)}({tt},...)"
    # list
    if isinstance(obj, list):
        if len(obj) < limit:
            js = ",".join(
                string_type(
                    o,
                    with_shape=with_shape,
                    with_min_max=with_min_max,
                    with_device=with_device,
                    ignore=ignore,
                    limit=limit,
                )
                for o in obj
            )
            return f"#{len(obj)}[{js}]"
        tt = string_type(
            obj[0],
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            ignore=ignore,
            limit=limit,
        )
        if with_min_max and all(isinstance(_, (int, float, bool)) for _ in obj):
            mini, maxi, avg = min(obj), max(obj), sum(float(_) for _ in obj) / len(obj)
            return f"#{len(obj)}[{tt},...][{mini},{maxi}:{avg}]"
        return f"#{len(obj)}[{tt},...]"
    # set
    if isinstance(obj, set):
        if len(obj) < 10:
            js = ",".join(
                string_type(
                    o,
                    with_shape=with_shape,
                    with_min_max=with_min_max,
                    with_device=with_device,
                    ignore=ignore,
                    limit=limit,
                )
                for o in obj
            )
            return f"{{{js}}}"
        if with_min_max and all(isinstance(_, (int, float, bool)) for _ in obj):
            mini, maxi, avg = min(obj), max(obj), sum(float(_) for _ in obj) / len(obj)
            return f"{{...}}#{len(obj)}[{mini},{maxi}:A{avg}]"
        return f"{{...}}#{len(obj)}" if with_shape else "{...}"
    # dict
    if isinstance(obj, dict) and type(obj) is dict:
        if len(obj) == 0:
            return "{}"

        try:
            import torch as _torch

            _has_torch = True
        except ImportError:
            _has_torch = False

        if (
            _has_torch
            and all(isinstance(k, int) for k in obj)
            and all(
                isinstance(
                    v,
                    (
                        str,
                        _torch.export.dynamic_shapes._Dim,
                        _torch.export.dynamic_shapes._DerivedDim,
                        _torch.export.dynamic_shapes._DimHint,
                    ),
                )
                for v in obj.values()
            )
        ):
            # This is dynamic shapes
            rows = []
            for k, v in obj.items():
                if isinstance(v, str):
                    rows.append(f"{k}:DYN({v})")
                else:
                    rows.append(f"{k}:{string_type(v)}")
            res = f"{{{','.join(rows)}}}"
            return res.replace("_DimHint(type:AUTO,_factory:bool)", "AUTO")

        kws = dict(
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            ignore=ignore,
            limit=limit,
        )
        s = ",".join(f"{kv[0]}:{string_type(kv[1],**kws)}" for kv in obj.items())  # type: ignore[arg-type]
        s = s.replace("_DimHint(type:AUTO,_factory:bool)", "AUTO")
        if all(isinstance(k, int) for k in obj):
            return f"{{{s}}}"
        return f"dict({s})"
    # array
    if isinstance(obj, np.ndarray):
        from .onnx_helper import np_dtype_to_tensor_dtype

        if with_min_max:
            s = string_type(obj, with_shape=with_shape)
            if len(obj.shape) == 0:
                return f"{s}={obj}"
            if obj.size == 0:
                return f"{s}[empty]"
            n_nan = np.isnan(obj.reshape((-1,))).astype(int).sum()
            if n_nan > 0:
                nob = obj.ravel()
                nob = nob[~np.isnan(nob)]
                if nob.size == 0:
                    return f"{s}[N{n_nan}nans]"
                return f"{s}[{nob.min()},{nob.max()}:A{nob.astype(float).mean()}N{n_nan}nans]"
            return f"{s}[{obj.min()},{obj.max()}:A{obj.astype(float).mean()}]"
        i = np_dtype_to_tensor_dtype(obj.dtype)
        if not with_shape:
            return f"A{i}r{len(obj.shape)}"
        return f"A{i}s{'x'.join(map(str, obj.shape))}"

    if isinstance(obj, bool):
        if with_min_max:
            return f"bool={obj}"
        return "bool"
    if isinstance(obj, int):
        if with_min_max:
            return f"int={obj}"
        return "int"
    if isinstance(obj, float):
        if with_min_max:
            return f"float={obj}"
        return "float"
    if isinstance(obj, str):
        return "str"
    if isinstance(obj, slice):
        return "slice"

    try:
        import torch

        has_torch = True
    except ImportError:
        has_torch = False

    has_torch = has_torch and hasattr(torch, "__version__")
    if has_torch:
        # Dim, SymInt
        if isinstance(obj, torch.export.dynamic_shapes._DerivedDim):
            return "DerivedDim"
        if isinstance(obj, torch.export.dynamic_shapes._Dim):
            return f"Dim({obj.__name__})"
        if isinstance(obj, torch.SymInt):
            return "SymInt"
        if isinstance(obj, torch.SymFloat):
            return "SymFloat"

        if isinstance(obj, torch.export.dynamic_shapes._DimHint):
            cl = (
                torch.export.dynamic_shapes._DimHintType
                if hasattr(torch.export.dynamic_shapes, "_DimHintType")
                else torch.export.Dim
            )
            if obj in (torch.export.Dim.DYNAMIC, cl.DYNAMIC):
                return "DYNAMIC"
            if obj in (torch.export.Dim.AUTO, cl.AUTO):
                return "AUTO"
            return (
                str(obj).replace("DimHint(DYNAMIC)", "DYNAMIC").replace("DimHint(AUTO)", "AUTO")
            )

        if obj.__class__.__name__ == "_DimHintType":
            if obj in (torch.export.Dim.DYNAMIC, obj.__class__.DYNAMIC):
                return "DYNAMIC"
            if obj in (torch.export.Dim.AUTO, obj.__class__.AUTO):
                return "AUTO"
            return (
                str(obj).replace("DimHint(DYNAMIC)", "DYNAMIC").replace("DimHint(AUTO)", "AUTO")
            )

        # Tensors
        if isinstance(obj, torch._subclasses.fake_tensor.FakeTensor):
            return _string_tensor(obj, "F", with_shape, with_device)

        if isinstance(obj, torch.Tensor):
            from ..torch.new_tracing.tensor import TracingTensor

            if isinstance(obj, TracingTensor):
                # M = chr(((ord(T) - 65) * 2) % 26 + 65)
                return _string_tensor(obj, "M", with_shape, with_device)

            from ..torch.torch_helper import torch_dtype_to_onnx_dtype

            if with_min_max:
                s = string_type(obj, with_shape=with_shape, with_device=with_device)
                if len(obj.shape) == 0:
                    return f"{s}={obj}"
                if obj.numel() == 0:
                    return f"{s}[empty]"
                n_nan = obj.reshape((-1,)).isnan().to(int).sum()
                if n_nan > 0:
                    nob = obj.reshape((-1,))
                    nob = nob[~nob.isnan()]
                    if obj.dtype in {torch.complex64, torch.complex128}:
                        return (
                            f"{s}[{nob.abs().min()},{nob.abs().max():A{nob.mean()}N{n_nan}nans}]"
                        )
                    return f"{s}[{obj.min()},{obj.max()}:A{obj.to(float).mean()}N{n_nan}nans]"
                if obj.dtype in {torch.complex64, torch.complex128}:
                    return f"{s}[{obj.abs().min()},{obj.abs().max()}:A{obj.abs().mean()}]"
                return f"{s}[{obj.min()},{obj.max()}:A{obj.to(float).mean()}]"
            i = torch_dtype_to_onnx_dtype(obj.dtype)
            prefix = ("G" if obj.get_device() >= 0 else "C") if with_device else ""
            if not with_shape:
                return f"{prefix}T{i}r{len(obj.shape)}"
            return f"{prefix}T{i}s{'x'.join(map(str, obj.shape))}"

    if is_dataclass(obj):
        # That includes torch.export.Dim.AUTO, torch.export.Dim.DYNAMIC so they need to be
        # handled before that.
        values = {f.name: getattr(obj, f.name, None) for f in fields(obj)}
        values = {k: v for k, v in values.items() if v is not None}
        s = string_type(
            values,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            ignore=ignore,
            limit=limit,
        )
        return f"{obj.__class__.__name__}{s[4:]}"

    if obj.__class__.__name__ == "OrtValue":
        if not obj.has_value():
            return "OV(<novalue>)"
        if not obj.is_tensor():
            return "OV(NOTENSOR)"
        if with_min_max:
            try:
                t = obj.numpy()
            except Exception:
                return "OV(NO-NUMPY:FIXIT)"
            dev = ("G" if obj.device_name() == "Cuda" else "C") if with_device else ""
            return f"{dev}OV({string_type(t, with_shape=with_shape, with_min_max=with_min_max)})"
        dt = obj.element_type()
        shape = obj.shape()
        dev = ("G" if obj.device_name() == "Cuda" else "C") if with_device else ""
        if with_shape:
            return f"{dev}OV{dt}s{'x'.join(map(str, shape))}"
        return f"{dev}OV{dt}r{len(shape)}"

    if obj.__class__.__name__ == "SymbolicTensor":
        return _string_tensor(obj, "ST", with_shape, with_device)

    if (
        obj.__class__.__name__ in {"DynamicCache"}
        and hasattr(obj, "layers")
        and any(lay.__class__.__name__ != "DynamicLayer" for lay in obj.layers)
    ):
        slay = []
        for lay in obj.layers:
            skeys = string_type(
                lay.keys,
                with_shape=with_shape,
                with_min_max=with_min_max,
                with_device=with_device,
                limit=limit,
            )
            svalues = string_type(
                lay.keys,
                with_shape=with_shape,
                with_min_max=with_min_max,
                with_device=with_device,
                limit=limit,
            )
            slay.append(f"{lay.__class__.__name__}({skeys}, {svalues})")
        return f"{obj.__class__.__name__}({', '.join(slay)})"

    if obj.__class__.__name__ in {
        "DynamicCache",
        "SlidingWindowCache",
        "StaticCache",
        "HybridCache",
    }:
        from ..torch.in_transformers.cache_helper import CacheKeyValue

        ca = CacheKeyValue(obj)
        kc = string_type(
            ca.key_cache,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        vc = string_type(
            ca.value_cache,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        return f"{obj.__class__.__name__}(key_cache={kc}, value_cache={vc})"

    if obj.__class__.__name__ == "StaticLayer":
        kc = string_type(
            list(obj.keys),
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        vc = string_type(
            list(obj.values),
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        return f"{obj.__class__.__name__}(keys={kc}, values={vc})"

    if obj.__class__.__name__ == "EncoderDecoderCache":
        att = string_type(
            obj.self_attention_cache,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        cross = string_type(
            obj.cross_attention_cache,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        return (
            f"{obj.__class__.__name__}(self_attention_cache={att}, "
            f"cross_attention_cache={cross})"
        )

    if has_torch:
        import torch.utils._pytree as pytree

        if obj.__class__ in pytree.SUPPORTED_NODES:
            from ..torch.in_transformers.cache_helper import flatten_unflatten_for_dynamic_shapes

            args = flatten_unflatten_for_dynamic_shapes(obj)
            att = string_type(
                args,
                with_shape=with_shape,
                with_min_max=with_min_max,
                with_device=with_device,
                limit=limit,
            )
            return f"{obj.__class__.__name__}[serialized]({att})"

    if type(obj).__name__ == "Node" and hasattr(obj, "meta"):
        # torch.fx.node.Node
        return f"%{obj.target}"
    if type(obj).__name__ == "ValueInfoProto":
        return f"OT{obj.type.tensor_type.elem_type}"

    if obj.__class__.__name__ == "BatchFeature":
        s = string_type(
            obj.data,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        return f"BatchFeature(data={s})"

    if obj.__class__.__name__ == "BatchEncoding":
        s = string_type(
            obj.data,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        return f"BatchEncoding(data={s})"

    if obj.__class__.__name__ == "VirtualTensor":

        def _torch_sym_int_to_str(value: "torch.SymInt") -> Union[int, str]:  #  noqa: F821
            if isinstance(value, str):
                return value
            if hasattr(value, "node") and isinstance(value.node, str):
                return f"{value.node}"

            from torch.fx.experimental.sym_node import SymNode

            if hasattr(value, "node") and isinstance(value.node, SymNode):
                # '_expr' is safer than expr
                return str(value.node._expr).replace(" ", "")

            try:
                val_int = int(value)
                return val_int
            except (
                TypeError,
                ValueError,
                AttributeError,
                torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode,
            ):
                pass

            raise AssertionError(f"Unable to convert {value!r} into string")

        return (
            f"{obj.__class__.__name__}(name={obj.name!r}, "
            f"dtype={obj.dtype}, shape={tuple(_torch_sym_int_to_str(_) for _ in obj.shape)})"
        )

    if obj.__class__.__name__ == "KeyValuesWrapper":
        import transformers

        assert isinstance(
            obj, transformers.cache_utils.KeyValuesWrapper
        ), f"Unexpected type {type(obj)}"
        s = string_type(
            list(obj),
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        return f"{obj.__class__.__name__}[{obj.cache_type}]{s}"

    if obj.__class__.__name__ == "DynamicLayer":
        import transformers

        assert isinstance(
            obj, transformers.cache_utils.DynamicLayer
        ), f"Unexpected type {type(obj)}"
        s1 = string_type(
            obj.keys,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        s2 = string_type(
            obj.values,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        return f"{obj.__class__.__name__}(keys={s1}, values={s2})"

    if has_torch:
        if isinstance(obj, torch.nn.Module):
            return f"{obj.__class__.__name__}(...)"

        if isinstance(obj, (torch.device, torch.dtype, torch.memory_format, torch.layout)):
            return f"{obj.__class__.__name__}({obj})"

        if isinstance(  # TreeSpec, MappingKey, SequenceKey
            obj, (pytree.TreeSpec, pytree.MappingKey, pytree.SequenceKey)
        ):
            return repr(obj).replace(" ", "").replace("\n", " ")

        if isinstance(obj, torch.fx.proxy.Proxy):
            return repr(obj)

    if ignore:
        return f"{obj.__class__.__name__}(...)"

    if obj.__class__.__name__.endswith("Config"):
        import transformers.configuration_utils as tcu

        if isinstance(obj, tcu.PretrainedConfig):
            s = str(obj.to_diff_dict()).replace("\n", "").replace(" ", "")
            return f"{obj.__class__.__name__}(**{s})"
    if obj.__class__.__name__ in {"TorchModelContainer", "InferenceSession"}:
        return f"{obj.__class__.__name__}(...)"
    if obj.__class__.__name__ == "Results":
        import ultralytics

        assert isinstance(obj, ultralytics.engine.results.Results), f"Unexpected type={type(obj)}"
        return f"ultralytics.{obj.__class__.__name__}(...)"
    if obj.__class__.__name__ == "FakeTensorMode":
        return f"{obj}"
    if obj.__class__.__name__ == "FakeTensorContext":
        return "FakeTensorContext(...)"
    if obj.__class__.__name__ == "Chat":
        import transformers.utils.chat_template_utils as ctu

        assert isinstance(obj, ctu.Chat), f"unexpected type {type(obj)}"
        msg = string_type(
            obj.messages,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        return f"Chat({msg})"

    if obj.__class__.__name__ == "Value":
        import onnx_ir

        if isinstance(obj, onnx_ir.Value):
            return f"ir.{obj.__class__.__name__}({obj})"

    if obj.__class__.__name__.endswith("Type"):
        return f"{obj.__class__.__name__}(...)"

    if obj.__class__.__name__ == "DataFrame":
        import pandas

        assert isinstance(obj, pandas.DataFrame), f"unexpected type {type(obj)}"
        s = string_type(
            dict(zip(obj.columns, [obj[c].values for c in obj.columns])),
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        return f"DataFrame({s})"

    raise TypeError(f"Unsupported type {type(obj).__name__!r} - {type(obj)} ({has_torch=})")


def string_signature(sig: Any) -> str:
    """Displays the signature of a functions."""

    def _k(p, kind):
        for name in dir(p):
            if getattr(p, name) == kind:
                return name
        return repr(kind)

    text = [" __call__ ("]
    for p in sig.parameters:
        pp = sig.parameters[p]
        kind = repr(pp.kind)
        t = f"{p}: {pp.annotation}" if pp.annotation is not inspect._empty else p
        if pp.default is not inspect._empty:
            t = f"{t} = {pp.default!r}"
        if kind == pp.VAR_POSITIONAL:
            t = f"*{t}"
        le = (30 - len(t)) * " "
        text.append(f"    {t}{le}|{_k(pp,kind)}")
    text.append(
        f") -> {sig.return_annotation}" if sig.return_annotation is not inspect._empty else ")"
    )
    return "\n".join(text)


def string_sig(f: Any, kwargs: Optional[Dict[str, Any]] = None) -> str:
    """
    Displays the signature of a function if the default
    if the given value is different from
    """
    if hasattr(f, "__init__") and kwargs is None:
        fct = f.__init__
        kwargs = f.__dict__
        name = f.__class__.__name__
    else:
        fct = f
        name = f.__name__

    if kwargs is None:
        kwargs = {}
    rows = []
    sig = inspect.signature(fct)
    for p in sig.parameters:
        pp = sig.parameters[p]
        d = pp.default
        if d is inspect._empty:
            if p in kwargs:
                v = kwargs[p]
                rows.append(f"{p}={v!r}" if not isinstance(v, enum.IntEnum) else f"{p}={v.name}")
            continue
        v = kwargs.get(p, d)
        if d != v:
            rows.append(f"{p}={v!r}" if not isinstance(v, enum.IntEnum) else f"{p}={v.name}")
            continue
    atts = ", ".join(rows)
    return f"{name}({atts})"


def get_sig_kwargs(obj: Any) -> Dict[str, Any]:
    """
    Returns the current values of the parameters of an object's ``__init__`` method.

    :param obj: an object with an ``__init__`` method
    :return: a dictionary mapping parameter names to their current values
    """
    sig = inspect.signature(obj.__init__)
    kwargs = {}
    for p in sig.parameters:
        if p == "self":
            continue
        pp = sig.parameters[p]
        if pp.kind in (pp.VAR_POSITIONAL, pp.VAR_KEYWORD):
            continue
        if hasattr(obj, p):
            kwargs[p] = getattr(obj, p)
        elif pp.default is not inspect.Parameter.empty:
            kwargs[p] = pp.default
    return kwargs


def make_hash(obj: Any) -> str:
    """
    Returns a simple hash of ``id(obj)`` in four letter.
    """
    aa = id(obj) % (26**3)
    return f"{chr(65 + aa // 26 ** 2)}{chr(65 + (aa // 26) % 26)}{chr(65 + aa % 26)}"


def flatten_object(x: Any, drop_keys: bool = False) -> Any:
    """
    Flattens the object.
    It accepts some common classes used in deep learning.

    :param x: any object
    :param drop_keys: drop the keys if a dictionary is flattened.
        Keeps the order defined by the dictionary if False, sort them if True.
    :return: flattened object
    """
    if x is None:
        return x
    if isinstance(x, (list, tuple)):
        res = []
        for i in x:
            if i is None or hasattr(i, "shape") or isinstance(i, (int, float, str)):
                res.append(i)
            else:
                res.extend(flatten_object(i, drop_keys=drop_keys))
        return tuple(res) if isinstance(x, tuple) else res
    if isinstance(x, dict):
        # We flatten the keys.
        if drop_keys:
            return flatten_object(list(x.values()), drop_keys=drop_keys)
        return flatten_object(list(x.items()), drop_keys=drop_keys)

    if x.__class__.__name__ in {"DynamicCache", "StaticCache", "HybridCache"}:
        from ..torch.in_transformers.cache_helper import CacheKeyValue

        return CacheKeyValue(x).aslist()

    if x.__class__.__name__ == "EncoderDecoderCache":
        res = [*flatten_object(x.self_attention_cache), *flatten_object(x.cross_attention_cache)]
        return tuple(res)
    if hasattr(x, "to_tuple"):
        return flatten_object(x.to_tuple(), drop_keys=drop_keys)
    if hasattr(x, "shape"):
        # A tensor. Nothing to do.
        return x
    raise TypeError(
        f"Unexpected type {type(x)} for x, drop_keys={drop_keys}, "
        f"content is {string_type(x, with_shape=True)}"
    )


def _make_debug_info(msg, level, debug_info) -> Optional[List[str]]:
    return [*(debug_info if debug_info else []), f"{' ' * level}{msg}"]


def max_diff(
    expected: Any,
    got: Any,
    level: int = 0,
    flatten: bool = False,
    debug_info: Optional[List[str]] = None,
    begin: int = 0,
    end: int = -1,
    _index: int = 0,
    allow_unique_tensor_with_list_of_one_element: bool = True,
    hist: Optional[Union[bool, List[float]]] = None,
    skip_none: bool = False,
) -> Dict[str, Union[float, int, Tuple[Any, ...]]]:
    """
    Returns the maximum discrepancy.

    :param expected: expected values
    :param got: values
    :param level: for embedded outputs, used for debug purpposes
    :param flatten: flatten outputs
    :param debug_info: debug information
    :param begin: first output to considered
    :param end: last output to considered (-1 for the last one)
    :param _index: used with begin and end
    :param allow_unique_tensor_with_list_of_one_element:
        allow a comparison between a single tensor and a list of one tensor
    :param hist: compute an histogram of the discrepancies
    :param skip_none: skips none value
    :return: dictionary with many values

    * abs: max absolute error
    * rel: max relative error
    * sum: sum of the errors
    * n: number of outputs values, if there is one
        output, this number will be the number of elements
        of this output
    * dnan: difference in the number of nan
    * dev: tensor on the same device, if applicable

    You may use :func:`string_diff` to display the discrepancies in one string.
    """
    if expected is None and got is None:
        return dict(abs=0, rel=0, sum=0, n=0, dnan=0)

    _dkws_ = dict(
        level=level + 1, begin=begin, end=end, _index=_index, hist=hist, skip_none=skip_none
    )
    _dkws = {**_dkws_, "flatten": flatten}
    _dkwsf = {**_dkws_, "flatten": False}

    _debug = lambda msg: _make_debug_info(msg, level, debug_info)  # noqa: E731

    if allow_unique_tensor_with_list_of_one_element:
        if hasattr(expected, "shape") and isinstance(got, (list, tuple)) and len(got) == 1:
            return max_diff(
                expected,
                got[0],
                level=level,
                flatten=False,
                debug_info=debug_info,
                allow_unique_tensor_with_list_of_one_element=False,
                hist=hist,
                skip_none=skip_none,
            )
        return max_diff(
            expected,
            got,
            level=level,
            flatten=flatten,
            debug_info=debug_info,
            begin=begin,
            end=end,
            _index=_index,
            allow_unique_tensor_with_list_of_one_element=False,
            hist=hist,
            skip_none=skip_none,
        )

    if expected.__class__.__name__ == "CausalLMOutputWithPast":
        if got.__class__.__name__ == "CausalLMOutputWithPast":
            return max_diff(
                [expected.logits, *flatten_object(expected.past_key_values)],
                [got.logits, *flatten_object(got.past_key_values)],
                debug_info=_debug(expected.__class__.__name__),
                **_dkws,
            )
        return max_diff(
            [expected.logits, *flatten_object(expected.past_key_values)],
            got,
            debug_info=_debug(expected.__class__.__name__),
            **_dkws,
        )

    if hasattr(expected, "to_tuple"):
        return max_diff(expected.to_tuple(), got, debug_info=_debug("to_tuple1"), **_dkws)

    if hasattr(got, "to_tuple"):
        return max_diff(expected, got.to_tuple(), debug_info=_debug("to_tuple2"), **_dkws)

    if isinstance(expected, (tuple, list)):
        if len(expected) == 1 and not isinstance(got, type(expected)):
            return max_diff(expected[0], got, debug_info=_debug("lt2"), **_dkws)
        if not isinstance(got, (tuple, list)):
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)

        if len(got) != len(expected):
            if flatten:
                # Let's flatten.
                flat_a = flatten_object(expected, drop_keys=True)
                flat_b = flatten_object(got, drop_keys=True)
                return max_diff(
                    flat_a,
                    flat_b,
                    debug_info=[
                        *(debug_info if debug_info else []),
                        (f"{' ' * level}flatten[{string_type(expected)},{string_type(got)}]"),
                    ],
                    **_dkwsf,
                )

            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)

        am, rm, sm, n, dn, drep, dd = 0, 0, 0.0, 0.0, 0, None, None
        for ip, (e, g) in enumerate(zip(expected, got)):
            d = max_diff(
                e,
                g,
                level=level + 1,
                debug_info=[
                    *(debug_info if debug_info else []),
                    f"{' ' * level}[{ip}] so far abs {am} - rel {rm}",
                ],
                begin=begin,
                end=end,
                _index=_index + ip,
                flatten=flatten,
                hist=hist,
                skip_none=skip_none,
            )
            am = max(am, d["abs"])
            dn = max(dn, d["dnan"])
            rm = max(rm, d["rel"])
            sm += d["sum"]  # type: ignore
            n += d["n"]  # type: ignore
            if "rep" in d:
                if drep is None:
                    drep = d["rep"].copy()
                else:
                    for k, v in d["rep"].items():
                        drep[k] += v
            if "dev" in d and d["dev"] is not None:
                if dd is None:
                    dd = d["dev"]
                else:
                    dd += d["dev"]  # type: ignore[operator]

        res = dict(abs=am, rel=rm, sum=sm, n=n, dnan=dn)
        if dd is not None:
            res["dev"] = dd
        if drep:
            res["rep"] = drep
        return res  # type: ignore

    if isinstance(expected, dict):
        assert begin == 0 and end == -1, (
            f"begin={begin}, end={end} not compatible with dictionaries, "
            f"keys={sorted(expected)}"
        )
        if isinstance(got, dict):
            if len(expected) != len(got):
                return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
            if set(expected) != set(got):
                return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
            keys = sorted(expected)
            return max_diff(
                [expected[k] for k in keys],
                [got[k] for k in keys],
                debug_info=_debug("dict1"),
                **_dkws,
            )

        if not isinstance(got, (tuple, list)):
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
        if len(expected) != len(got):
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
        return max_diff(list(expected.values()), got, debug_info=_debug("dict2"), **_dkws)

    if isinstance(expected, np.ndarray) or isinstance(got, np.ndarray):
        dev = None
        if not isinstance(expected, np.ndarray) and hasattr(expected, "numpy"):
            from ..torch.torch_helper import to_numpy

            dev = 0 if expected.device.type == "cpu" else 1
            expected = to_numpy(expected)

        if not isinstance(got, np.ndarray) and hasattr(got, "numpy"):
            from ..torch.torch_helper import to_numpy

            dev = 0 if got.device.type == "cpu" else 1
            got = to_numpy(got)
        if isinstance(got, (list, tuple)):
            got = np.array(got)
        if isinstance(expected, (list, tuple)):
            expected = np.array(expected)

        if _index < begin or (end != -1 and _index >= end):
            # out of boundary
            res = dict(abs=0.0, rel=0.0, sum=0.0, n=0.0, dnan=0)
            if dev is not None:
                res["dev"] = dev  # type: ignore[operator]
            return res  # type: ignore[return-value]
        if isinstance(expected, (int, float)):
            if isinstance(got, np.ndarray) and len(got.shape) == 0:
                got = float(got)
            if isinstance(got, (int, float)):
                if expected == got:
                    return dict(abs=0.0, rel=0.0, sum=0.0, n=0.0, dnan=0)
                res = dict(
                    abs=abs(expected - got),
                    rel=abs(expected - got) / (abs(expected) + 1e-5),
                    sum=abs(expected - got),
                    n=1,
                    dnan=0,
                )
                if dev is not None:
                    res["dev"] = dev
                return res  # type: ignore[return-value]
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
        if expected.dtype in (np.complex64, np.complex128):
            if got.dtype == expected.dtype:
                got = np.real(got)
            elif got.dtype not in (np.float32, np.float64):
                return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
            expected = np.real(expected)

        if expected.shape != got.shape:
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
        # nan are replace by 1e10, any discrepancies in that order of magnitude
        # is likely caused by nans
        exp_cpu = np.nan_to_num(expected.astype(np.float64), nan=1e10)
        got_cpu = np.nan_to_num(got.astype(np.float64), nan=1e10)
        diff = np.abs(got_cpu - exp_cpu)
        ndiff = np.abs(np.isnan(expected).astype(int) - np.isnan(got).astype(int))
        rdiff = diff / (np.abs(exp_cpu) + 1e-3)
        if diff.size == 0:
            abs_diff, rel_diff, sum_diff, n_diff, nan_diff = (
                (0, 0, 0, 0, 0)
                if exp_cpu.size == got_cpu.size
                else (np.inf, np.inf, np.inf, 0, np.inf)
            )
            argm = None
        else:
            abs_diff, rel_diff, sum_diff, n_diff, nan_diff = (
                float(diff.max()),
                float(rdiff.max()),
                float(diff.sum()),
                float(diff.size),
                float(ndiff.sum()),
            )
            argm = tuple(map(int, np.unravel_index(diff.argmax(), diff.shape)))

        res: Dict[str, float] = dict(  # type: ignore
            abs=abs_diff, rel=rel_diff, sum=sum_diff, n=n_diff, dnan=nan_diff, argm=argm
        )
        if dev is not None:
            res["dev"] = dev
        if hist:
            if isinstance(hist, bool):
                hist = np.array([0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100], dtype=diff.dtype)
            res["rep"] = {f">{h}": (diff > h).sum().item() for h in hist}
        return res  # type: ignore

    import torch

    if isinstance(expected, torch.Tensor) and isinstance(got, torch.Tensor):
        dev = 0 if expected.device == got.device else 1
        if _index < begin or (end != -1 and _index >= end):
            # out of boundary
            return dict(abs=0.0, rel=0.0, sum=0.0, n=0.0, dnan=0, dev=dev)
        if expected.dtype in (torch.complex64, torch.complex128):
            if got.dtype == expected.dtype:
                got = torch.view_as_real(got)
            elif got.dtype not in (torch.float32, torch.float64):
                return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
            expected = torch.view_as_real(expected)

        if expected.shape != got.shape:
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
        # nan are replace by 1e10, any discrepancies in that order of magnitude
        # is likely caused by nans
        exp_cpu = expected.to(torch.float64).nan_to_num(1e10)
        got_cpu = got.to(torch.float64).nan_to_num(1e10)
        if got_cpu.device != exp_cpu.device:
            if torch.device("cuda:0") in {got_cpu.device, exp_cpu.device}:
                got_cpu = got_cpu.to("cuda:0")
                exp_cpu = exp_cpu.to("cuda:0")
                expected = expected.to("cuda:0")
                got = got.to("cuda:0")
            else:
                got_cpu = got_cpu.detach().to("cpu")
                exp_cpu = exp_cpu.detach().to("cpu")
                expected = expected.to("cpu")
                got = got.to("cpu")
        diff = (got_cpu - exp_cpu).abs()
        ndiff = (expected.isnan().to(int) - got.isnan().to(int)).abs()
        rdiff = diff / (exp_cpu.abs() + 1e-3)
        if diff.numel() > 0:
            abs_diff, rel_diff, sum_diff, n_diff, nan_diff = (
                float(diff.max().detach()),
                float(rdiff.max().detach()),
                float(diff.sum().detach()),
                float(diff.numel()),
                float(ndiff.sum().detach()),
            )
            argm = tuple(map(int, torch.unravel_index(diff.argmax(), diff.shape)))
        elif got_cpu.numel() == exp_cpu.numel():
            abs_diff, rel_diff, sum_diff, n_diff, nan_diff = (0.0, 0.0, 0.0, 0.0, 0.0)
            argm = None
        else:
            abs_diff, rel_diff, sum_diff, n_diff, nan_diff = (
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
            )
            argm = None

        res: Dict[str, float] = dict(  # type: ignore
            abs=abs_diff, rel=rel_diff, sum=sum_diff, n=n_diff, dnan=nan_diff, argm=argm, dev=dev
        )
        if hist:
            if isinstance(hist, bool):
                hist = torch.tensor([0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100], dtype=diff.dtype)
            res["rep"] = {f">{h}": (diff > h).sum().item() for h in hist}
        return res  # type: ignore

    if isinstance(expected, int) and isinstance(got, torch.Tensor):
        # a size
        if got.shape != tuple():
            return dict(  # type: ignore
                abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf, argm=np.inf
            )
        return dict(  # type: ignore
            abs=abs(expected - got.item()),
            rel=abs((expected - got.item()) / max(1, expected)),
            sum=abs(expected - got.item()),
            n=1,
            dnan=0,
        )

    if "SquashedNormal" in expected.__class__.__name__:
        values = (expected.mean, expected.scale)
        return max_diff(values, got, debug_info=_debug("SquashedNormal"), **_dkws)

    import torch.utils._pytree as pytree

    if expected.__class__ in pytree.SUPPORTED_NODES:
        if got.__class__ not in pytree.SUPPORTED_NODES:
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
        expected_args, _spec = pytree.tree_flatten(expected)
        got_args, _spec = pytree.tree_flatten(got)
        return max_diff(
            expected_args, got_args, debug_info=_debug(expected.__class__.__name__), **_dkws
        )

    # backup function in case pytorch does not know how to serialize.
    if expected.__class__.__name__ == "DynamicCache":
        if got.__class__.__name__ == "DynamicCache":
            from ..torch.in_transformers.cache_helper import CacheKeyValue

            expected = CacheKeyValue(expected)
            got = CacheKeyValue(got)
            return max_diff(
                [expected.key_cache, expected.value_cache],
                [got.key_cache, got.value_cache],
                hist=hist,
            )
        if isinstance(got, tuple) and len(got) == 2:
            from ..torch.in_transformers.cache_helper import CacheKeyValue

            if not isinstance(expected, CacheKeyValue):
                expected = CacheKeyValue(expected)
            return max_diff(
                [expected.key_cache, expected.value_cache],
                [got[0], got[1]],
                debug_info=_debug(expected.__class__.__name__),
                **_dkws,
            )
        raise AssertionError(
            f"DynamicCache not fully implemented with classes "
            f"{expected.__class__.__name__!r} and {got.__class__.__name__!r}, "
            f"and expected={string_type(expected)}, got={string_type(got)},\n"
            f"level={level}"
        )

    if expected.__class__.__name__ == "StaticCache":
        if got.__class__.__name__ == "StaticCache":
            from ..torch.in_transformers.cache_helper import CacheKeyValue

            cae = CacheKeyValue(expected)
            cag = CacheKeyValue(got)
            return max_diff(
                [cae.key_cache, cae.value_cache], [cag.key_cache, cag.value_cache], hist=hist
            )
        if isinstance(got, tuple) and len(got) == 2:
            from ..torch.in_transformers.cache_helper import CacheKeyValue

            cae = CacheKeyValue(expected)
            return max_diff(
                [cae.key_cache, cae.value_cache],
                [got[0], got[1]],
                debug_info=_debug(expected.__class__.__name__),
                **_dkws,
            )
        raise AssertionError(
            f"StaticCache not fully implemented with classes "
            f"{expected.__class__.__name__!r} and {got.__class__.__name__!r}, "
            f"and expected={string_type(expected)}, got={string_type(got)},\n"
            f"level={level}"
        )

    if expected.__class__.__name__ == "CacheKeyValue":
        from ..torch.in_transformers.cache_helper import CacheKeyValue

        if got.__class__.__name__ == "CacheKeyValue":
            return max_diff(
                [expected.key_cache, expected.value_cache],
                [got.key_cache, got.value_cache],
                hist=hist,
            )
        if isinstance(got, tuple) and len(got) == 2:
            return max_diff(
                [expected.key_cache, expected.value_cache],
                [got[0], got[1]],
                debug_info=_debug(expected.__class__.__name__),
                **_dkws,
            )
        raise AssertionError(
            f"CacheKeyValue not fully implemented with classes "
            f"{expected.__class__.__name__!r} and {got.__class__.__name__!r}, "
            f"and expected={string_type(expected)}, got={string_type(got)},\n"
            f"level={level}"
        )

    if expected.__class__.__name__ == "EncoderDecoderCache":
        if got.__class__.__name__ == "EncoderDecoderCache":
            return max_diff(
                [expected.self_attention_cache, expected.cross_attention_cache],
                [got.self_attention_cache, got.cross_attention_cache],
                hist=hist,
            )
        if isinstance(got, tuple) and len(got) == 2:
            return max_diff(
                [expected.self_attention_cache, expected.cross_attention_cache],
                [got[0], got[1]],
                debug_info=_debug(expected.__class__.__name__),
                **_dkws,
            )
        raise AssertionError(
            f"EncoderDecoderCache not fully implemented with classes "
            f"{expected.__class__.__name__!r} and {got.__class__.__name__!r}, "
            f"and expected={string_type(expected)}, got={string_type(got)},\n"
            f"level={level}"
        )

    if expected.__class__.__name__ == "KeyValuesWrapper":
        if got.__class__.__name__ != expected.__class__.__name__:
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
        if got.cache_type != expected.cache_type:
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
        return max_diff(
            list(expected), list(got), debug_info=_debug(expected.__class__.__name__), **_dkws
        )

    if skip_none and (expected is None or got is None):
        return {"abs": 0, "rel": 0, "dnan": 0, "n": 0, "sum": 0}

    raise AssertionError(
        f"Not implemented with implemented with expected="
        f"{string_type(expected)} ({type(expected)}), got={string_type(got)},\n"
        f"level={level}"
    )


def size_type(dtype: Any) -> int:
    """Returns the element size for an element type."""
    if isinstance(dtype, int):
        from onnx import TensorProto

        # It is a TensorProto.DATATYPE
        if dtype in {
            TensorProto.DOUBLE,
            TensorProto.INT64,
            TensorProto.UINT64,
            TensorProto.COMPLEX64,
        }:
            return 8
        if dtype in {TensorProto.FLOAT, TensorProto.INT32, TensorProto.UINT32}:
            return 4
        if dtype in {
            TensorProto.FLOAT16,
            TensorProto.BFLOAT16,
            TensorProto.INT16,
            TensorProto.UINT16,
        }:
            return 2
        if dtype in {
            TensorProto.INT8,
            TensorProto.UINT8,
            TensorProto.BOOL,
            TensorProto.FLOAT8E4M3FN,
            TensorProto.FLOAT8E4M3FNUZ,
            TensorProto.FLOAT8E5M2,
            TensorProto.FLOAT8E5M2FNUZ,
            getattr(TensorProto, "FLOAT8E8M0", None),
        }:
            return 1
        if dtype in {TensorProto.COMPLEX128}:
            return 16
        from .onnx_helper import onnx_dtype_name

        raise AssertionError(
            f"Unable to return the element size for type {onnx_dtype_name(dtype)}"
        )

    if dtype == np.float64 or dtype == np.int64:
        return 8
    if dtype == np.float32:
        return 4
    if dtype == np.float16 or dtype == np.int16:
        return 2
    if dtype == np.int32:
        return 4
    if dtype == np.int8:
        return 1
    if hasattr(np, "uint64"):
        # it fails on mac
        if dtype == np.uint64:
            return 8
        if dtype == np.uint32:
            return 4
        if dtype == np.uint16:
            return 2
        if dtype == np.uint8:
            return 1

    import torch

    if dtype in {torch.float64, torch.int64}:
        return 8
    if dtype in {torch.float32, torch.int32}:
        return 4
    if dtype in {torch.float16, torch.int16, torch.bfloat16}:
        return 2
    if dtype in {torch.int8, torch.uint8, torch.bool}:
        return 1
    if hasattr(torch, "uint64"):
        # it fails on mac
        if dtype in {torch.uint64}:
            return 8
        if dtype in {torch.uint32}:
            return 4
        if dtype in {torch.uint16}:
            return 2
    import ml_dtypes

    if dtype == ml_dtypes.bfloat16:
        return 2
    raise AssertionError(f"Unexpected dtype={dtype}")


def string_diff(diff: Dict[str, Any], js: bool = False, ratio: bool = False, **kwargs) -> str:
    """
    Renders discrepancies returned by :func:`max_diff` into one string.

    :param diff: differences
    :param js: json format
    :param ratio: display mismatch ratio
    :param kwargs: addition values to add in the json format
    """
    if js:
        if "rep" in diff:
            rep = diff["rep"]
            diff = {**{k: v for k, v in diff.items() if k != "rep"}, **rep}
            if ratio:
                for k, v in rep.items():
                    diff[f"%{k}"] = v / diff["n"]
                diff["mean"] = diff["sum"] / diff["n"]
            diff.update(kwargs)
        return json.dumps(diff)

    # dict(abs=, rel=, sum=, n=n_diff, dnan=)
    if "dev" in diff:
        ddiff = {k: v for k, v in diff.items() if k != "dev"}
        return f"{string_diff(ddiff)}, dev={diff['dev']}"
    suffix = ""
    if "rep" in diff:
        rows = []
        for k, v in diff["rep"].items():
            if v > 0:
                rows.append(f"#{v}{k}")
        suffix = "-".join(rows)
        if suffix:
            suffix = f"/{suffix}"
    if "argm" in diff:
        sa = (
            ",".join(map(str, diff["argm"]))
            if isinstance(diff["argm"], tuple)
            else str(diff["argm"])
        )
        suffix += f",amax={sa}"
    if diff.get("dnan", None):
        if diff["abs"] == 0 or diff["rel"] == 0:
            return f"abs={diff['abs']}, rel={diff['rel']}, dnan={diff['dnan']}{suffix}"
        return f"abs={diff['abs']}, rel={diff['rel']}, n={diff['n']}, dnan={diff['dnan']}{suffix}"
    if diff["abs"] == 0 or diff["rel"] == 0:
        return f"abs={diff['abs']}, rel={diff['rel']}{suffix}"
    return f"abs={diff['abs']}, rel={diff['rel']}, n={diff['n']}{suffix}"

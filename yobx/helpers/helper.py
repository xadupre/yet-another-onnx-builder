import enum
import inspect
from dataclasses import is_dataclass, fields
from typing import Any, Callable, Dict, Optional
import numpy as np


def _string_tensor(obj, cls: str, with_shape: bool, with_device: bool, verbose: int) -> str:
    from .torch_helper import torch_dtype_to_onnx_dtype

    i = torch_dtype_to_onnx_dtype(obj.dtype)
    prefix = ("G" if obj.get_device() >= 0 else "C") if with_device else ""
    if not with_shape:
        if verbose:
            print(f"[string_type] {cls}1:{type(obj)}")
        return f"{prefix}{cls}{i}r{len(obj.shape)}"
    if verbose:
        print(f"[string_type] {cls}2:{type(obj)}")
    return f"{prefix}{cls}{i}s{'x'.join(map(str, obj.shape))}"


def string_type(
    obj: Any,
    with_shape: bool = False,
    with_min_max: bool = False,
    with_device: bool = False,
    ignore: bool = False,
    limit: int = 20,
    verbose: int = 0,
) -> str:
    """
    Displays the types of an object as a string.

    :param obj: any
    :param with_shape: displays shapes as well
    :param with_min_max: displays information about the values
    :param with_device: display the device
    :param ignore: if True, just prints the type for unknown types
    :param verbose: verbosity (to show the path it followed to get that print)
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
    by :func:`onnx_diagnostic.helpers.onnx_helper.onnx_dtype_name`.

    .. runpython::
        :showcode:

        from yobx.helpers import string_type

        print(string_type((1, ["r", 6.6])))

    With pytorch:

    .. runpython::
        :showcode:

        import torch
        from tobx.helpers import string_type

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
        if verbose:
            print(f"[string_type] A:{type(obj)}")
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
                verbose=verbose,
            )
            if verbose:
                print(f"[string_type] C:{type(obj)}")
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
                    verbose=verbose,
                )
                for o in obj
            )
            if verbose:
                print(f"[string_type] D:{type(obj)}")
            return f"({js})"
        tt = string_type(
            obj[0],
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            ignore=ignore,
            limit=limit,
            verbose=verbose,
        )
        if with_min_max and all(isinstance(_, (int, float, bool)) for _ in obj):
            mini, maxi, avg = min(obj), max(obj), sum(float(_) for _ in obj) / len(obj)
            if verbose:
                print(f"[string_type] E:{type(obj)}")
            return f"#{len(obj)}({tt},...)[{mini},{maxi}:A[{avg}]]"
        if verbose:
            print(f"[string_type] F:{type(obj)}")
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
                    verbose=verbose,
                )
                for o in obj
            )
            if verbose:
                print(f"[string_type] G:{type(obj)}")
            return f"#{len(obj)}[{js}]"
        tt = string_type(
            obj[0],
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            ignore=ignore,
            limit=limit,
            verbose=verbose,
        )
        if with_min_max and all(isinstance(_, (int, float, bool)) for _ in obj):
            mini, maxi, avg = min(obj), max(obj), sum(float(_) for _ in obj) / len(obj)
            if verbose:
                print(f"[string_type] H:{type(obj)}")
            return f"#{len(obj)}[{tt},...][{mini},{maxi}:{avg}]"
        if verbose:
            print(f"[string_type] I:{type(obj)}")
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
                    verbose=verbose,
                )
                for o in obj
            )
            if verbose:
                print(f"[string_type] J:{type(obj)}")
            return f"{{{js}}}"
        if with_min_max and all(isinstance(_, (int, float, bool)) for _ in obj):
            mini, maxi, avg = min(obj), max(obj), sum(float(_) for _ in obj) / len(obj)
            if verbose:
                print(f"[string_type] K:{type(obj)}")
            return f"{{...}}#{len(obj)}[{mini},{maxi}:A{avg}]"
        if verbose:
            print(f"[string_type] L:{type(obj)}")
        return f"{{...}}#{len(obj)}" if with_shape else "{...}"
    # dict
    if isinstance(obj, dict) and type(obj) is dict:
        if len(obj) == 0:
            if verbose:
                print(f"[string_type] M:{type(obj)}")
            return "{}"

        import torch

        if all(isinstance(k, int) for k in obj) and all(
            isinstance(
                v,
                (
                    str,
                    torch.export.dynamic_shapes._Dim,
                    torch.export.dynamic_shapes._DerivedDim,
                    torch.export.dynamic_shapes._DimHint,
                ),
            )
            for v in obj.values()
        ):
            # This is dynamic shapes
            rows = []
            for k, v in obj.items():
                if isinstance(v, str):
                    rows.append(f"{k}:DYN({v})")
                else:
                    rows.append(f"{k}:{string_type(v, verbose=verbose)}")
            if verbose:
                print(f"[string_type] DS0:{type(obj)}")
            return f"{{{','.join(rows)}}}"

        kws = dict(
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            ignore=ignore,
            limit=limit,
            verbose=verbose,
        )
        s = ",".join(f"{kv[0]}:{string_type(kv[1],**kws)}" for kv in obj.items())
        if all(isinstance(k, int) for k in obj):
            if verbose:
                print(f"[string_type] N:{type(obj)}")
            return f"{{{s}}}"
        if verbose:
            print(f"[string_type] O:{type(obj)}")
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
                    if verbose:
                        print(f"[string_type] A1:{type(obj)}")
                    return f"{s}[N{n_nan}nans]"
                if verbose:
                    print(f"[string_type] A2:{type(obj)}")
                return f"{s}[{nob.min()},{nob.max()}:A{nob.astype(float).mean()}N{n_nan}nans]"
            if verbose:
                print(f"[string_type] A3:{type(obj)}")
            return f"{s}[{obj.min()},{obj.max()}:A{obj.astype(float).mean()}]"
        i = np_dtype_to_tensor_dtype(obj.dtype)
        if not with_shape:
            if verbose:
                print(f"[string_type] A4:{type(obj)}")
            return f"A{i}r{len(obj.shape)}"
        if verbose:
            print(f"[string_type] A5:{type(obj)}")
        return f"A{i}s{'x'.join(map(str, obj.shape))}"

    import torch

    # Dim, SymInt
    if isinstance(obj, torch.export.dynamic_shapes._DerivedDim):
        if verbose:
            print(f"[string_type] Y1:{type(obj)}")
        return "DerivedDim"
    if isinstance(obj, torch.export.dynamic_shapes._Dim):
        if verbose:
            print(f"[string_type] Y2:{type(obj)}")
        return f"Dim({obj.__name__})"
    if isinstance(obj, torch.SymInt):
        if verbose:
            print(f"[string_type] Y3:{type(obj)}")
        return "SymInt"
    if isinstance(obj, torch.SymFloat):
        if verbose:
            print(f"[string_type] Y4:{type(obj)}")
        return "SymFloat"

    if isinstance(obj, torch.export.dynamic_shapes._DimHint):
        cl = (
            torch.export.dynamic_shapes._DimHintType
            if hasattr(torch.export.dynamic_shapes, "_DimHintType")
            else torch.export.Dim
        )
        if obj in (torch.export.Dim.DYNAMIC, cl.DYNAMIC):
            if verbose:
                print(f"[string_type] Y8:{type(obj)}")
            return "DYNAMIC"
        if obj in (torch.export.Dim.AUTO, cl.AUTO):
            if verbose:
                print(f"[string_type] Y9:{type(obj)}")
            return "AUTO"
        if verbose:
            print(f"[string_type] Y7:{type(obj)}")
        return str(obj).replace("DimHint(DYNAMIC)", "DYNAMIC").replace("DimHint(AUTO)", "AUTO")

    if isinstance(obj, bool):
        if with_min_max:
            if verbose:
                print(f"[string_type] W1:{type(obj)}")
            return f"bool={obj}"
        if verbose:
            print(f"[string_type] W2:{type(obj)}")
        return "bool"
    if isinstance(obj, int):
        if with_min_max:
            if verbose:
                print(f"[string_type] W3:{type(obj)}")
            return f"int={obj}"
        if verbose:
            print(f"[string_type] W4:{type(obj)}")
        return "int"
    if isinstance(obj, float):
        if with_min_max:
            if verbose:
                print(f"[string_type] W6:{type(obj)}")
            return f"float={obj}"
        if verbose:
            print(f"[string_type] W8:{type(obj)}")
        return "float"
    if isinstance(obj, str):
        if verbose:
            print(f"[string_type] W9:{type(obj)}")
        return "str"
    if isinstance(obj, slice):
        if verbose:
            print(f"[string_type] W10:{type(obj)}")
        return "slice"

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
            verbose=verbose,
        )
        if verbose:
            print(f"[string_type] B:{type(obj)}")
        return f"{obj.__class__.__name__}{s[4:]}"

    # Tensors
    if isinstance(obj, torch._subclasses.fake_tensor.FakeTensor):
        return _string_tensor(obj, "F", with_shape, with_device, verbose)

    if isinstance(obj, torch.Tensor):
        from .torch_helper import torch_dtype_to_onnx_dtype

        if with_min_max:
            s = string_type(obj, with_shape=with_shape, with_device=with_device)
            if len(obj.shape) == 0:
                if verbose:
                    print(f"[string_type] T1:{type(obj)}")
                return f"{s}={obj}"
            if obj.numel() == 0:
                if verbose:
                    print(f"[string_type] T2:{type(obj)}")
                return f"{s}[empty]"
            n_nan = obj.reshape((-1,)).isnan().to(int).sum()
            if n_nan > 0:
                nob = obj.reshape((-1,))
                nob = nob[~nob.isnan()]
                if obj.dtype in {torch.complex64, torch.complex128}:
                    if verbose:
                        print(f"[string_type] T3:{type(obj)}")
                    return f"{s}[{nob.abs().min()},{nob.abs().max():A{nob.mean()}N{n_nan}nans}]"
                if verbose:
                    print(f"[string_type] T5:{type(obj)}")
                return f"{s}[{obj.min()},{obj.max()}:A{obj.to(float).mean()}N{n_nan}nans]"
            if obj.dtype in {torch.complex64, torch.complex128}:
                if verbose:
                    print(f"[string_type] T6:{type(obj)}")
                return f"{s}[{obj.abs().min()},{obj.abs().max()}:A{obj.abs().mean()}]"
            if verbose:
                print(f"[string_type] T7:{type(obj)}")
            return f"{s}[{obj.min()},{obj.max()}:A{obj.to(float).mean()}]"
        i = torch_dtype_to_onnx_dtype(obj.dtype)
        prefix = ("G" if obj.get_device() >= 0 else "C") if with_device else ""
        if not with_shape:
            if verbose:
                print(f"[string_type] T8:{type(obj)}")
            return f"{prefix}T{i}r{len(obj.shape)}"
        if verbose:
            print(f"[string_type] T9:{type(obj)}")
        return f"{prefix}T{i}s{'x'.join(map(str, obj.shape))}"

    if obj.__class__.__name__ == "OrtValue":
        if not obj.has_value():
            if verbose:
                print(f"[string_type] V1:{type(obj)}")
            return "OV(<novalue>)"
        if not obj.is_tensor():
            if verbose:
                print(f"[string_type] V2:{type(obj)}")
            return "OV(NOTENSOR)"
        if with_min_max:
            from .torch_helper import to_numpy

            try:
                t = to_numpy(obj)
            except Exception:
                # pass unable to convert into numpy (bfloat16, ...)
                if verbose:
                    print(f"[string_type] V3:{type(obj)}")
                return "OV(NO-NUMPY:FIXIT)"
            if verbose:
                print(f"[string_type] V4:{type(obj)}")
            dev = ("G" if obj.device_name() == "Cuda" else "C") if with_device else ""
            return f"{dev}OV({string_type(t, with_shape=with_shape, with_min_max=with_min_max)})"
        dt = obj.element_type()
        shape = obj.shape()
        dev = ("G" if obj.device_name() == "Cuda" else "C") if with_device else ""
        if with_shape:
            if verbose:
                print(f"[string_type] V5:{type(obj)}")
            return f"{dev}OV{dt}s{'x'.join(map(str, shape))}"
        if verbose:
            print(f"[string_type] V6:{type(obj)}")
        return f"{dev}OV{dt}r{len(shape)}"

    if obj.__class__.__name__ == "SymbolicTensor":
        return _string_tensor(obj, "ST", with_shape, with_device, verbose)

    # others classes

    if obj.__class__.__name__ == "MambaCache":
        c = string_type(
            obj.conv_states,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
            verbose=verbose,
        )
        d = string_type(
            obj.ssm_states,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
            verbose=verbose,
        )
        if verbose:
            print(f"[string_type] CACHE1:{type(obj)}")
        return f"MambaCache(conv_states={c}, ssm_states={d})"

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
                verbose=verbose,
            )
            svalues = string_type(
                lay.keys,
                with_shape=with_shape,
                with_min_max=with_min_max,
                with_device=with_device,
                limit=limit,
                verbose=verbose,
            )
            slay.append(f"{lay.__class__.__name__}({skeys}, {svalues})")
        return f"{obj.__class__.__name__}({', '.join(slay)})"

    if obj.__class__.__name__ in {
        "DynamicCache",
        "SlidingWindowCache",
        "StaticCache",
        "HybridCache",
    }:
        from .cache_helper import CacheKeyValue

        ca = CacheKeyValue(obj)
        kc = string_type(
            ca.key_cache,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
            verbose=verbose,
        )
        vc = string_type(
            ca.value_cache,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
            verbose=verbose,
        )
        if verbose:
            print(f"[string_type] CACHE2:{type(obj)}")
        return f"{obj.__class__.__name__}(key_cache={kc}, value_cache={vc})"

    if obj.__class__.__name__ == "StaticLayer":
        kc = string_type(
            list(obj.keys),
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
            verbose=verbose,
        )
        vc = string_type(
            list(obj.values),
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
            verbose=verbose,
        )
        if verbose:
            print(f"[string_type] SL:{type(obj)}")
        return f"{obj.__class__.__name__}(keys={kc}, values={vc})"

    if obj.__class__.__name__ == "EncoderDecoderCache":
        att = string_type(
            obj.self_attention_cache,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
            verbose=verbose,
        )
        cross = string_type(
            obj.cross_attention_cache,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
            verbose=verbose,
        )
        if verbose:
            print(f"[string_type] CACHE3:{type(obj)}")
        return (
            f"{obj.__class__.__name__}(self_attention_cache={att}, "
            f"cross_attention_cache={cross})"
        )

    if obj.__class__ in torch.utils._pytree.SUPPORTED_NODES:
        from .cache_helper import flatten_unflatten_for_dynamic_shapes

        args = flatten_unflatten_for_dynamic_shapes(obj)
        att = string_type(
            args,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
            verbose=verbose,
        )
        if verbose:
            print(f"[string_type] DS:{type(obj)}")
        return f"{obj.__class__.__name__}[serialized]({att})"

    if type(obj).__name__ == "Node" and hasattr(obj, "meta"):
        # torch.fx.node.Node
        if verbose:
            print(f"[string_type] TT1:{type(obj)}")
        return f"%{obj.target}"
    if type(obj).__name__ == "ValueInfoProto":
        if verbose:
            print(f"[string_type] OO1:{type(obj)}")
        return f"OT{obj.type.tensor_type.elem_type}"

    if obj.__class__.__name__ == "BatchFeature":
        s = string_type(
            obj.data,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
            verbose=verbose,
        )
        if verbose:
            print(f"[string_type] TT2:{type(obj)}")
        return f"BatchFeature(data={s})"

    if obj.__class__.__name__ == "BatchEncoding":
        s = string_type(
            obj.data,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
            verbose=verbose,
        )
        if verbose:
            print(f"[string_type] TT3:{type(obj)}")
        return f"BatchEncoding(data={s})"

    if obj.__class__.__name__ == "VirtualTensor":
        if verbose:
            print(f"[string_type] TT4:{type(obj)}")

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
        if verbose:
            print(f"[string_type] KW0:{type(obj)}")
        s = string_type(
            list(obj),
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
            verbose=verbose,
        )
        return f"{obj.__class__.__name__}[{obj.cache_type}]{s}"

    if obj.__class__.__name__ == "DynamicLayer":
        import transformers

        assert isinstance(
            obj, transformers.cache_utils.DynamicLayer
        ), f"Unexpected type {type(obj)}"
        if verbose:
            print(f"[string_type] LY0:{type(obj)}")
        s1 = string_type(
            obj.keys,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
            verbose=verbose,
        )
        s2 = string_type(
            obj.values,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
            verbose=verbose,
        )
        return f"{obj.__class__.__name__}(keys={s1}, values={s2})"

    if isinstance(obj, torch.nn.Module):
        if verbose:
            print(f"[string_type] MM:{type(obj)}")
        return f"{obj.__class__.__name__}(...)"

    if isinstance(obj, (torch.device, torch.dtype, torch.memory_format, torch.layout)):
        if verbose:
            print(f"[string_type] TT7:{type(obj)}")
        return f"{obj.__class__.__name__}({obj})"

    if isinstance(  # TreeSpec, MappingKey, SequenceKey
        obj,
        (
            torch.utils._pytree.TreeSpec,
            torch.utils._pytree.MappingKey,
            torch.utils._pytree.SequenceKey,
        ),
    ):
        if verbose:
            print(f"[string_type] TT8:{type(obj)}")
        return repr(obj).replace(" ", "").replace("\n", " ")

    if isinstance(obj, torch.fx.proxy.Proxy):
        return repr(obj)

    if ignore:
        if verbose:
            print(f"[string_type] CACHE4:{type(obj)}")
        return f"{obj.__class__.__name__}(...)"

    if obj.__class__.__name__.endswith("Config"):
        import transformers.configuration_utils as tcu

        if isinstance(obj, tcu.PretrainedConfig):
            if verbose:
                print(f"[string_type] CONFIG:{type(obj)}")
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
            verbose=verbose,
        )
        return f"Chat({msg})"

    if verbose:
        print(f"[string_type] END:{type(obj)}")
    raise AssertionError(f"Unsupported type {type(obj).__name__!r} - {type(obj)}")


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


def string_sig(f: Callable, kwargs: Optional[Dict[str, Any]] = None) -> str:
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


def make_hash(obj: Any) -> str:
    """
    Returns a simple hash of ``id(obj)`` in four letter.
    """
    aa = id(obj) % (26**3)
    return f"{chr(65 + aa // 26 ** 2)}{chr(65 + (aa // 26) % 26)}{chr(65 + aa % 26)}"

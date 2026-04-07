import contextlib
from itertools import zip_longest
import os
from typing import Any, Dict, Generator, List, Tuple
import numpy as np
from onnx import NodeProto
from ..helpers import string_type
from ..helpers.onnx_helper import (
    dtype_to_tensor_dtype,
    tensor_dtype_to_np_dtype,
    str_tensor_proto_type,
)
from ..xexpressions import simplify_expression
from ..xshape._shape_helper import DYNAMIC_SHAPE, STATIC_SHAPE, all_int, all_int_or_str


@contextlib.contextmanager
def _unset_fake_temporarily() -> Generator:
    import torch

    old = torch._C._unset_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE)
    try:
        yield old
    finally:
        if old is not None:
            torch._C._set_dispatch_mode(old)


@contextlib.contextmanager
def _maybe_disable_fake_tensor_mode() -> Generator:
    try:
        yield
    finally:
        pass


class _ExtraPackages:
    """Lazy availability checks for optional heavy dependencies (torch, tensorflow).

    Calling ``self._has_torch`` or ``self._has_tensorflow`` performs a one-time
    import attempt for the respective package and caches the result so that
    subsequent calls are cheap.  If the environment variable ``NOTORCH=1`` (or
    ``NOTF=1`` for TensorFlow) is set the import is skipped unconditionally and
    the property returns ``False``, which is useful in test environments that
    must not load those frameworks.

    Once a package is confirmed available the corresponding module object is
    cached in ``self._torch`` / ``self._tensorflow`` and exposed through the
    ``torch`` / ``tensorflow`` properties.  For torch, ``torch._subclasses``
    is also cached and :func:`_maybe_disable_fake_tensor_mode` is installed as
    ``self.maybe_disable_fake_tensor_mode``.
    """

    def __init__(self):
        self.maybe_disable_fake_tensor_mode = contextlib.nullcontext

        if os.environ.get("NOTORCH", "0") in ("1", "true"):
            self._has_torch_ = False
            self._torch = None
            self._TracingInt = None
        else:
            self._has_torch_ = None
            self._torch = None
            self._TracingInt = None

        if os.environ.get("NOTF", "0") in ("1", "true"):
            self._has_tensorflow_ = False
            self._tensorflow = None
        else:
            self._has_tensorflow_ = None
            self._tensorflow = None

    @property
    def torch(self):
        assert self._has_torch, "torch is missing"
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    @property
    def TracingInt(self):
        assert self._has_torch, "torch is missing"
        if self._TracingInt is None:
            from ..torch.new_tracing.shape import TracingInt

            self._TracingInt = TracingInt
        return self._TracingInt

    @property
    def _has_torch(self) -> bool:
        """Return ``True`` if *torch* is importable.

        On the first call the property tries to ``import torch`` and caches
        the outcome in ``self._has_torch_``.  All subsequent calls return the
        cached value without touching the import system.  Setting the
        environment variable ``NOTORCH=1`` before instantiation forces the
        property to return ``False`` without attempting any import.
        """
        if self._has_torch_ is not None:
            return self._has_torch_

        try:
            import torch
            import torch._subclasses

            self._has_torch_ = hasattr(torch, "__version__")
            self._torch = torch
            self.torch_subclasses = torch._subclasses
            self.maybe_disable_fake_tensor_mode = _maybe_disable_fake_tensor_mode
        except (NameError, ImportError, AttributeError):
            self._has_torch_ = False
            self._torch = None

        return self._has_torch_

    @property
    def tensorflow(self):
        assert self._has_tensorflow, "tensorflow is missing"
        if self._tensorflow is None:
            import tensorflow

            self._tensorflow = tensorflow
        return self._tensorflow

    @property
    def _has_tensorflow(self) -> bool:
        """Return ``True`` if *tensorflow* is importable.

        On the first call the property tries to ``import tensorflow`` and
        caches the outcome in ``self._has_tensorflow_``.  All subsequent calls
        return the cached value without touching the import system.  Setting
        the environment variable ``NOTF=1`` before instantiation forces the
        property to return ``False`` without attempting any import.
        """
        if self._has_tensorflow_ is not None:
            return self._has_tensorflow_

        try:
            import tensorflow

            self._has_tensorflow_ = hasattr(tensorflow, "__version__")
            self._tensorflow = tensorflow
        except (NameError, ImportError, AttributeError):
            self._has_tensorflow_ = False
            self._tensorflow = None

        return self._has_tensorflow_


class _BuilderRuntime:
    """
    Computes the output of a couple of nodes knowing their inputs.
    It supports numpy and torch tensors. Most of the function are
    used while exporting a model, by :meth:`_InferenceRuntime.compute_constant
    <yobx.xshape._inference_runtime._InferenceRuntime.compute_constant>`.
    """

    def onnx_dtype_to_torch_dtype(self, itype: int) -> "torch.dtype":  # noqa: F821
        """See :func:`yobx.torch.torch_helper.onnx_dtype_to_torch_dtype`."""
        from ..torch.torch_helper import onnx_dtype_to_torch_dtype

        return onnx_dtype_to_torch_dtype(itype)

    def onnx_dtype_to_np_dtype(self, itype: int) -> np.dtype:
        """See :func:`yobx.helpers.onnx_helper.tensor_dtype_to_np_dtype`."""
        from ..helpers.onnx_helper import tensor_dtype_to_np_dtype

        return tensor_dtype_to_np_dtype(itype)

    def container_type(self, v) -> str:
        if isinstance(v, np.ndarray):
            return "numpy"
        if hasattr(v, "detach") and self._has_torch and isinstance(v, self.torch.Tensor):
            return "torch"
        if hasattr(v, "ref") and self._has_tensorflow and isinstance(v, self.tensorflow.Tensor):
            return "tensorflow"
        raise TypeError(f"Unexpected type {type(v)} for a value{self.get_debug_msg()}")

    def consistent_tensor_feeds(
        self, feeds: Dict[str, Any], node: NodeProto
    ) -> Tuple[str, Dict[str, Any]]:
        types = {self.container_type(v) for v in feeds.values() if v is not None}
        if len(types) == 1:
            return types.pop(), feeds
        if types == {"numpy", "torch"}:
            res = {}
            for k, v in feeds.items():
                if isinstance(v, np.ndarray):
                    itype = dtype_to_tensor_dtype(v.dtype)
                    ttype = self.onnx_dtype_to_torch_dtype(itype)
                    x = self.make_torch_tensor_from_np_array(v.copy()).to(ttype)
                    assert "FakeTensor" not in str(type(x)), (
                        f"FakeTensor {node.output[0]!r} cannot be a constant {type(x)}, "
                        f"node.op_type={node.op_type!r}, type={self.torch.Tensor}"
                        f"{self.get_debug_msg()}"
                    )
                    res[k] = x
                else:
                    res[k] = v
            return "torch", res
        if types == {"numpy", "tensorflow"}:
            return "tensorflow", {
                k: self.tensorflow.convert_to_tensor(v) if isinstance(v, np.ndarray) else v
                for k, v in feeds.items()
            }
        raise ValueError(f"Not implemented for {types=}.")

    def _apply_slice_to_shape(
        self, shape: STATIC_SHAPE, indices: List[slice], axes: List[int], expand_axes: List[int]
    ) -> STATIC_SHAPE:
        assert isinstance(shape, tuple), f"Unexpected type {type(shape)} for shape: {shape}"
        assert isinstance(indices, list), f"Unexpected type {type(indices)} for index: {indices}"
        assert isinstance(axes, list), f"Unexpected type {type(axes)} for index: {axes}"
        assert len(axes) in (1, len(indices)), f"Mismatch lengths {len(indices)} != {len(axes)}"

        if all(isinstance(i, slice) for i in indices):
            new_shape = []
            for index, axis_ in zip(indices, axes):
                axis = axis_ if axis_ >= 0 else (axis_ + len(shape)) % len(shape)
                while len(new_shape) < axis:
                    assert shape[len(new_shape)] >= 0, (
                        f"Negative value in shape {shape}, indices={indices}, "
                        f"axes={axes}, expand_axes={expand_axes}"
                    )
                    new_shape.append(shape[len(new_shape)])
                assert axis < len(shape), (
                    f"axis={axis} is out of order (shape={shape}, "
                    f"indices={indices}, axes={axes}){self.get_debug_msg()}"
                )
                n = shape[axis]
                start = index.start or 0
                end = index.stop or n
                diff = end - start
                dim = diff // index.step if index.step else diff
                dim = max(dim, 0)
                assert dim >= 0, (
                    f"Negative dim={dim}, axis={axis}, shape={shape}, indices={indices}, "
                    f"axes={axes}, expand_axes={expand_axes}"
                )
                new_shape.append(dim)
        elif all_int(indices):
            assert len(axes) == 1, (
                f"Unable to guess new shape from shape={shape}, "
                f"indices={indices}, axes={axes}, expand_axes={expand_axes}"
            )
            new_shape = [len(indices), *shape[1:]]
        else:
            raise RuntimeError(
                f"Unable to guess new shape from shape={shape}, "
                f"indices={indices}, axes={axes}, expand_axes={expand_axes}"
            )
        for a in shape[len(new_shape) :]:
            assert a >= 0, (
                f"Negative value in shape {shape}, indices={indices}, "
                f"axes={axes}, expand_axes={expand_axes}"
            )
            new_shape.append(a)
        for e in expand_axes:
            new_shape.insert(e, 1)
        return tuple(new_shape)

    def _apply_reshape_to_shape(
        self, input_shape: DYNAMIC_SHAPE, new_shape: STATIC_SHAPE
    ) -> DYNAMIC_SHAPE:
        """Returns the shape of the output of a node Reshape."""
        assert isinstance(
            input_shape, tuple
        ), f"unexpected type {type(input_shape)} for input_shape."
        assert isinstance(new_shape, tuple), f"unexpected type {type(new_shape)} for input_shape."
        assert all_int(new_shape), f"unexpected type for a dimension in {new_shape}"

        # handling zeros --> keeps the original dimension
        new_new_shape = []
        for i, sh in enumerate(new_shape):
            if sh == 0:
                assert i < len(
                    input_shape
                ), f"Unable to apply reshape {new_shape} to input shape {input_shape}"
                new_new_shape.append(input_shape[i])
                continue
            new_new_shape.append(sh)
        new_shape = tuple(new_new_shape)

        if -1 not in new_shape:
            return new_shape

        if all_int(input_shape):
            size = int(np.prod(input_shape))
            div = np.prod([i for i in new_shape if i != -1])
            if div == 0:
                return tuple((int(i) if i >= 0 else 0) for i in new_shape)
            return tuple((int(i) if i >= 0 else int(size // div)) for i in new_shape)
        if all_int_or_str(input_shape) and new_shape == (1, -1):
            # common case
            return (1, "*".join(map(str, input_shape)))

        mul, div = [], []
        muli, divi = 1, 1
        for s, n in zip_longest(input_shape, new_shape):
            if s is None:
                s = 1
            if n is None:
                n = 1
            if isinstance(s, str) and isinstance(n, str):
                if s != n:
                    mul.append(s)
                    div.append(n)
            elif isinstance(s, str):
                mul.append(s)
                if n != -1:
                    divi *= n
            else:
                muli *= s
                if n != -1:
                    divi *= n

        if not mul and not div:
            assert muli % divi == 0, (
                f"Inconsistency between input_shape={input_shape} "
                f"and new_shape={new_shape}{self.get_debug_msg()}"
            )
            rest = muli // divi
        else:
            if muli != 1:
                mul.append(str(muli))
            if divi != 1:
                div.append(str(divi))
            if not mul:
                mul = ["1"]
            if not div:
                rest = (
                    mul[0]
                    if len(mul) == 1
                    else simplify_expression(f"{'*'.join(f'({s})' for s in mul)}")
                )
            elif not mul:
                rest = simplify_expression(
                    f"1//({div[0]})"
                    if len(div) == 1
                    else f"1//({'*'.join(f'({s})' for s in div)})"
                )
            else:
                rest = simplify_expression(
                    f"(({'*'.join(f'({s})' for s in mul)})"
                    f"//({'*'.join(f'({s})' for s in div)}))"
                )
        return tuple(s if s != -1 else rest for s in new_shape)

    def _apply_expand_to_shape(
        self, input_shape: DYNAMIC_SHAPE, new_shape: STATIC_SHAPE
    ) -> DYNAMIC_SHAPE:
        """Returns the shape of the output of a node Reshape."""
        assert isinstance(
            input_shape, tuple
        ), f"unexpected type {type(input_shape)} for input_shape."
        assert isinstance(new_shape, tuple), f"unexpected type {type(new_shape)} for input_shape."

        if -1 not in new_shape and 1 not in new_shape:
            return new_shape

        assert len(new_shape) >= len(input_shape), (
            f"inconsistent behaviour, new_shape={new_shape}, "
            f"input_shape={input_shape}{self.get_debug_msg()}"
        )
        if len(input_shape) < len(new_shape):
            input_shape = (1,) * (len(new_shape) - len(input_shape)) + input_shape
        nsh = []
        for i, s in enumerate(new_shape):
            if s == 1:
                assert i < len(input_shape), (
                    f"Unexpected scenario new_shape={new_shape}, "
                    f"input_shape={input_shape}{self.get_debug_msg()}"
                )
                nsh.append(input_shape[i])
                continue
            if s == 0:
                nsh.append(0)
                continue
            if i < len(input_shape):
                if isinstance(s, str) and isinstance(input_shape[i], str):
                    if s != input_shape[
                        i
                    ] and not self.evaluate_dimension_equality_with_constraints(
                        s, input_shape[i]
                    ):
                        return None
                    nsh.append(s)
                    continue
                if isinstance(s, str) and isinstance(input_shape[i], int):
                    if input_shape[i] == 1:
                        nsh.append(s)
                        continue
                    # (1, 1, 1024) with (1, 1, 'input_dim_13')
                    # The output is 1024 if input_dim_13 is not zero, which we don't know.
                    return None
            assert isinstance(s, int) or (i < len(input_shape) and input_shape[i] == 1), (
                f"Unable to compute expanded shape at position {i} when trying "
                f"to expand shape {input_shape} with {new_shape}{self.get_debug_msg()}"
            )
            nsh.append(s)
        return tuple(nsh)

    def _apply_transpose(
        self, node: NodeProto, feeds: Dict[str, "torch.Tensor"]  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        perm = None
        for att in node.attribute:
            if att.name == "perm":
                perm = tuple(att.ints)
                break
        assert perm, f"perm not here in node {node}"
        x = feeds[node.input[0]]
        assert len(x.shape) == len(perm), (
            f"Shape mismatch between x.shape={x.shape} and perm={perm!r}, "
            f"node is {self.pretty_node(node)}{self.get_debug_msg()}"
        )
        if hasattr(x, "detach") and self._has_torch:
            if isinstance(x, np.ndarray):
                # Type conversion between numpy and torch is not robust.
                itype = dtype_to_tensor_dtype(x.dtype)
                ttype = self.onnx_dtype_to_torch_dtype(itype)
                x = self.torch.from_numpy(x.copy()).to(ttype)
            return [self.torch.permute(x, perm).to(x.dtype)]
        return [x.transpose(perm).astype(x.dtype)]

    def _apply_expand(
        self, node: NodeProto, feeds: Dict[str, "torch.Tensor"]  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        x = feeds[node.input[0]]
        new_shape = feeds[node.input[1]]
        if hasattr(x, "detach") and self._has_torch and isinstance(x, self.torch.Tensor):
            if len(x.shape) == 0:
                if len(new_shape) == 0:
                    return x
                import torch

                return [torch.full(tuple(new_shape), x)]
            shape_x = (
                x.shape
                if len(x.shape) == len(new_shape)
                else ((1,) * (len(new_shape) - len(x.shape)) + x.shape)
            )
            try:
                return [x.expand(tuple(max(s, int(i)) for s, i in zip(shape_x, new_shape)))]
            except RuntimeError as e:
                raise RuntimeError(
                    f"Unable to compute the constant, new_shape={new_shape}, "
                    f"x.shape={x.shape}, node={node}\n{self.pretty_text()}"
                ) from e
        ones = np.ones(tuple(int(i) for i in new_shape), dtype=x.dtype)
        return [(x * ones).astype(x.dtype)]

    def _apply_squeeze(
        self, node: NodeProto, feeds: Dict[str, "torch.Tensor"]  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        x = feeds[node.input[0]]
        if len(node.input) == 1:
            # No axis.
            return [x.squeeze()]
        axis = feeds[node.input[1]]
        if len(axis.shape) == 0:
            return [np.squeeze(x, (int(axis),))]
        return [x.squeeze(tuple(int(i) for i in axis))]

    def _apply_unsqueeze(
        self, node: NodeProto, feeds: Dict[str, "torch.Tensor"]  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        x = feeds[node.input[0]]
        axis = feeds[node.input[1]]
        if isinstance(x, np.ndarray):
            if len(axis.shape) == 0:
                return [np.expand_dims(x, (int(axis),))]
            return [np.expand_dims(x, tuple(int(i) for i in axis))]
        if isinstance(axis, np.ndarray):
            axis = [int(axis)] if axis.shape == tuple() else axis.tolist()
        if len(axis) == 1:
            if isinstance(x, (np.int64, np.int32)):
                return [np.array([x])]
            return (
                [np.expand_dims(x, int(axis[0]))]
                if isinstance(x, np.ndarray)
                else [x.unsqueeze(int(axis[0]))]
            )
        assert len(axis) > 0, f"axis={axis} is null"
        for a in axis:
            x = np.expand_dims(x, int(a)) if isinstance(x, np.ndarray) else x.unsqueeze(int(a))
        return [x]

    def _apply_cast(
        self, node: NodeProto, feeds: Dict[str, "torch.Tensor"]  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        x = feeds[node.input[0]]
        if not isinstance(x, np.ndarray) and (
            not hasattr(x, "detach")
            or not self._has_torch
            or not isinstance(x, self.torch.Tensor)
        ):
            # Maybe a float, then we process it as a float, tensor.to only works
            # on tensors.
            assert isinstance(
                x, (float, int, np.float32, np.float64, np.float16, np.int32, np.int64)
            ), f"Unexpected type {type(x)} for {node.input[0]!r} (node.name={node.name!r})"
            res = self._apply_cast(node, {node.input[0]: np.array(x)})
            return [res[0]]
        to, saturate = None, 1
        for att in node.attribute:
            if att.name == "to":
                to = att.i
            elif att.name == "saturate":
                saturate = att.i
        assert to is not None, f"to not here in node {node}"
        assert to != 8 and to < 17, f"Cast not implemented for to={to}, {str_tensor_proto_type()}"
        del saturate
        if not self._has_torch:
            ttype = tensor_dtype_to_np_dtype(to)
            return [x.astype(ttype)]
        if isinstance(x, np.ndarray):
            # Type conversion between numpy and torch is not robust.
            itype = dtype_to_tensor_dtype(x.dtype)
            ttype = self.onnx_dtype_to_torch_dtype(itype)
            x = self.make_torch_tensor_from_np_array(x).to(ttype)
            assert "FakeTensor" not in str(type(x)), (
                f"FakeTensor {node.output[0]!r} cannot be a constant {type(x)}, "
                f"node.op_type={node.op_type!r}, type={self.torch.Tensor}"
                f"{self.pretty_text()}"
            )
            assert isinstance(x, self.torch.Tensor), (
                f"Unexpected type {type(x)} for x for node type {node.op_type}, "
                f"name={node.name}, inputs={node.input}, outputs={node.output}"
            )
            ttype = self.onnx_dtype_to_torch_dtype(to)
            return [x.to(ttype)]
        assert (
            hasattr(x, "detach") and self._has_torch and isinstance(x, self.torch.Tensor)
        ), "unexpected configuration"
        ttype = self.onnx_dtype_to_torch_dtype(to)
        return [x.to(ttype)]

    def _apply_unary_function(
        self, node: NodeProto, feeds: Dict[str, "torch.Tensor"]  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        x = feeds[node.input[0]]
        itype = dtype_to_tensor_dtype(x.dtype)
        if isinstance(x, np.ndarray):
            ttype = tensor_dtype_to_np_dtype(itype)
            if node.op_type == "Sqrt":
                return [np.sqrt(x).astype(ttype)]
            if node.op_type == "Exp":
                return [np.exp(x).astype(ttype)]
            if node.op_type == "Reciprocal":
                return [(np.array([1], dtype=x.dtype) / x).astype(ttype)]
            raise AssertionError(
                f"Not implemented for op_type={node.op_type!r}, node={node}, feeds={feeds}"
            )

        ttype = self.onnx_dtype_to_torch_dtype(itype)
        if node.op_type == "Sqrt":
            return [self.torch.sqrt(x).to(ttype)]
        if node.op_type == "Exp":
            return [self.torch.exp(x).to(ttype)]
        if node.op_type == "Reciprocal":
            return [(self.torch.tensor([1], dtype=x.dtype) / x).to(ttype)]
        raise AssertionError(
            f"Not implemented for op_type={node.op_type!r}, node={node}, "
            f"feeds={string_type(feeds, with_shape=True)}"
        )

    def _apply_trilu(
        self, node: NodeProto, feeds: Dict[str, "torch.Tensor"]  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        upper = True
        for att in node.attribute:
            if att.name == "upper":
                upper = att.i
                break
        assert len(node.input) in (1, 2), (
            f"Unexpected number of inputs (inputs={node.input}) "
            f"for Trilu{self.get_debug_msg()}"
        )
        x = feeds[node.input[0]]
        k = feeds[node.input[1]] if len(node.input) > 1 else np.array(0, dtype=np.int64)
        assert len(x.shape) > 0, (
            f"x cannot be empty but shape is {x.shape}, execution of Trilu "
            f"failed{self.get_debug_msg()}"
        )
        if hasattr(x, "detach") and self._has_torch and isinstance(x, self.torch.Tensor):
            assert isinstance(k, self.torch.Tensor), (
                f"Expecting a tensor for {node.input[1]!r} but got "
                f"{type(k)}{self.get_debug_msg()}"
            )
            ak = k.detach().cpu()
            iak = int(ak) if len(ak.shape) == 0 else int(ak[0])
            assert iak <= 1, f"Unexpected value for k={k}{self.get_debug_msg()}"
            return [self.torch.triu(x, iak) if upper else self.torch.tril(x, iak)]

        assert isinstance(k, np.ndarray), (
            f"Expecting a tensor for {node.input[1]!r} but got "
            f"{type(k)}{self.get_debug_msg()}"
        )
        iak = int(k) if len(k.shape) == 0 else int(k[0])
        return [np.triu(x, iak) if upper else np.tril(x, iak)]

    def _apply_binary_op(
        self, node: NodeProto, feeds: Dict[str, "torch.Tensor"]  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        a, b = feeds[node.input[0]], feeds[node.input[1]]
        if a.dtype != b.dtype:
            a = self._to_torch_tensor(a)
            b = self._to_torch_tensor(b)
        try:
            if node.op_type == "Add":
                return [a + b]
            if node.op_type == "Mul":
                return [a * b]
            if node.op_type == "Sub":
                return [a - b]
            if node.op_type == "Div":
                return [a / b]
            if node.op_type == "Pow":
                return [a**b]
            raise AssertionError(f"{node.op_type!r} not implemented")
        except RuntimeError as e:
            raise AssertionError(
                f"Unable to multiply two objects of dtype {a.dtype}, {b.dtype} and "
                f"shapes {a.shape}, {b.shape}, node.op_type={node.op_type!r}, "
                f"node.name={node.name!r}, inputs={node.input}, outputs={node.output}"
            ) from e

    def make_torch_tensor_from_np_array(self, arr: np.ndarray) -> "torch.Tensor":  # noqa: F821
        """Converts a numpy array to a torch tensor."""
        return self.torch.from_numpy(arr)

    def _apply_where(
        self, node: NodeProto, feeds: Dict[str, "torch.Tensor"]  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        engine, new_feeds = self.consistent_tensor_feeds(feeds, node)
        if engine == "numpy":
            y = np.where(*[new_feeds[k] for k in node.input])
            return [y]
        if engine == "torch":
            y = self.torch.where(*[new_feeds[k] for k in node.input])
            return [y]
        raise RuntimeError(f"engine {engine!r} not implemented")

    def _apply_slice(
        self, node: NodeProto, feeds: Dict[str, "torch.Tensor"]  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        new_feeds = {}
        for k, v in feeds.items():
            if hasattr(v, "detach") and self._has_torch and isinstance(v, np.ndarray):
                # Type conversion between numpy and torch is not robust.
                itype = dtype_to_tensor_dtype(v.dtype)
                ttype = self.onnx_dtype_to_torch_dtype(itype)
                x = self.torch.from_numpy(v)
                assert x.dtype == ttype, (
                    f"Unexpected conversion from numpy {v.dtype} to "
                    f"{x.dtype} != {ttype}{self.get_debug_msg()}"
                )

                assert "FakeTensor" not in str(type(x)), (
                    f"FakeTensor {node.output[0]!r} cannot be a constant {type(x)}, "
                    f"node.op_type={node.op_type!r}, type={self.torch.Tensor}"
                    f"{self.get_debug_msg()}"
                )
                new_feeds[k] = x
            else:
                new_feeds[k] = v
        assert len(node.input) >= 3, (
            f"Node {node.op_type} (name={node.name!r}) has not enough "
            f"inputs {node.input}\n{self.pretty_text()}"
        )
        data, starts, ends = [new_feeds[k] for k in node.input[:3]]
        axes = new_feeds[node.input[3]] if len(node.input) > 3 and node.input[3] else None
        steps = new_feeds[node.input[4]] if len(node.input) > 4 and node.input[4] else None

        if axes is None:
            if steps is None:
                slices = [slice(s, e) for s, e in zip(starts, ends)]
            else:
                slices = [slice(s, e, d) for s, e, d in zip(starts, ends, steps)]
        else:
            if steps is None:
                slices = [slice(0, a) for a in data.shape]
                for s, e, a in zip(starts, ends, axes):
                    slices[a] = slice(s, e)
            else:
                slices = [slice(0, a) for a in data.shape]
                for s, e, a, d in zip(starts, ends, axes, steps):
                    slices[a] = slice(s, e, d)
        res = data[tuple(slices)]
        assert len(res.shape) == 0 or min(res.shape) > 0, (
            f"Empty shape found {res.shape} after Slice when x.shape={data.shape}, "
            f"starts={starts}, ends={ends}, axes={axes}, steps={steps}, "
            f"node.name={node.name!r}, input names={node.input}, "
            f"slices={slices}"
        )
        assert len(res.shape) == len(data.shape), (
            f"Shape mismatch input shape is {data.shape}, output shape is {res.shape}, "
            f"axes={axes}, starts={starts}, ends={ends}, steps={steps}, "
            f"node is {self.pretty_node(node)}{self.pretty_text()}"
        )
        return [res]

    def _apply_shape_on_shape(
        self, node: NodeProto, shape: Tuple[int, ...]
    ) -> "torch.Tensor":  # noqa: F821
        if node.attribute:
            start = 0
            end = None
            for att in node.attribute:
                if att.name == "start":
                    start = att.i
                elif att.name == "end":
                    end = att.i
            shape = shape[start:] if end is None else shape[start:end]
        if self._has_torch:
            return [self.torch.from_numpy(np.array(shape, dtype=np.int64))]
        return [np.array(shape, dtype=np.int64)]

    def _apply_shape(
        self, node: NodeProto, feeds: Dict[str, "torch.Tensor"]  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        shape = tuple(map(int, feeds[node.input[0]].shape))
        return self._apply_shape_on_shape(node, shape)

"""Lowers exported PyTorch programs through the native ONNX graph engines."""

import operator
import time

import numpy
import torch
from onnx_light import onnx
from onnx_light.onnx import helper

from ...builder.onnxlight import OnnxLightOptimizationOptions
from ...container import ExportArtifact, FunctionPieces
from ...helpers.mini_onnx_builder import proto_from_array
from ...xbuilder.function_options import FunctionOptions
from ..export_options import ConvertingLibrary, ExportOptions, TracingMode
from ..torch_helper import torch_dtype_to_onnx_dtype
from .graph_builder import TorchOnnxLightGraphBuilder


def _dynamic_shapes(value):
    """Normalizes named dimensions without replacing PyTorch export constraints."""
    if isinstance(value, str):
        return torch.export.Dim(value)
    if isinstance(value, dict):
        return {key: _dynamic_shapes(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_dynamic_shapes(item) for item in value)
    if isinstance(value, list):
        return [_dynamic_shapes(item) for item in value]
    return value


def _shape(value, specification=None):
    """Extracts input tensor annotations and requested symbolic dimension names."""
    shape = [int(dim) if isinstance(dim, int) else str(dim) for dim in value.shape]
    if specification is not None:
        axes = (
            specification.items() if isinstance(specification, dict) else enumerate(specification)
        )
        for axis, dimension in axes:
            if isinstance(dimension, str):
                shape[axis] = dimension
            elif hasattr(dimension, "__name__"):
                shape[axis] = dimension.__name__
    return tuple(shape)


class NativeTorchInterpreter:
    """Owns FX bookkeeping while native engines own graph and shape state.

    ATen lowering only emits native operators. It does not run the legacy FX
    interpreter, Python shape rules, Python graph builder or Python optimizer.
    Unsupported operators fail at their FX node instead of falling back.
    """

    unary = {
        "relu": "Relu",
        "abs": "Abs",
        "neg": "Neg",
        "exp": "Exp",
        "log": "Log",
        "sqrt": "Sqrt",
        "rsqrt": "Sqrt",
        "sigmoid": "Sigmoid",
        "tanh": "Tanh",
        "sin": "Sin",
        "cos": "Cos",
        "floor": "Floor",
        "ceil": "Ceil",
        "reciprocal": "Reciprocal",
        "erf": "Erf",
        "sign": "Sign",
        "logical_not": "Not",
    }
    binary = {
        "add": "Add",
        "sub": "Sub",
        "mul": "Mul",
        "div": "Div",
        "true_divide": "Div",
        "pow": "Pow",
        "maximum": "Max",
        "minimum": "Min",
        "eq": "Equal",
        "ne": "Equal",
        "lt": "Less",
        "le": "LessOrEqual",
        "gt": "Greater",
        "ge": "GreaterOrEqual",
        "logical_and": "And",
        "logical_or": "Or",
        "logical_xor": "Xor",
        "bitwise_and": "BitwiseAnd",
    }

    def __init__(self, builder, graph_module, dispatcher=None, raise_list=None):
        self.builder = builder
        self.module = graph_module
        self.dispatcher = dispatcher
        self.raise_list = set(raise_list or ())
        self.values = {}
        self.builder.torch = torch

    def resolve(self, value):
        """Resolves FX references without evaluating tensors in Python."""
        if isinstance(value, torch.fx.Node):
            return self.values[value]
        if isinstance(value, tuple):
            return tuple(self.resolve(item) for item in value)
        if isinstance(value, list):
            return [self.resolve(item) for item in value]
        if isinstance(value, dict):
            return {key: self.resolve(item) for key, item in value.items()}
        return value

    def initializer(self, name, value):
        """Copies parameter or constant storage into a native tensor initializer."""
        if isinstance(value, torch.Tensor):
            from torch._subclasses.fake_tensor import FakeTensor

            if isinstance(value, FakeTensor):
                raise NotImplementedError(
                    "Native Torch export requires concrete parameter storage."
                )
            value = proto_from_array(value.detach().cpu(), name=name)
        return self.builder.make_initializer(name, value)

    def tensor(self, value, dtype=None):
        """Normalizes scalar inputs and PyTorch-prescribed dtype promotion."""
        if isinstance(value, str):
            if dtype and self.builder.get_type(value) != dtype:
                return self.builder.op.Cast(value, to=dtype)
            return value
        if isinstance(value, torch.Tensor):
            value = self.initializer("", value)
            return self.tensor(value, dtype)
        if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            raise NotImplementedError(
                "Symbolic scalar values must be represented by exported FX operators."
            )
        if dtype is not None:
            value = numpy.asarray(value, dtype=self.builder.onnx_dtype_to_np_dtype(dtype))
        elif not isinstance(value, numpy.ndarray):
            value = numpy.asarray(value)
        return self.builder.make_initializer("", value)

    def shape_tensor(self, dimensions):
        """Builds a runtime shape tensor from scalar values and shape operators."""
        if any(not isinstance(value, (int, str)) for value in dimensions):
            raise NotImplementedError(
                "Shape dimensions must be integers or exported size values."
            )
        if all(isinstance(value, int) for value in dimensions):
            return numpy.array(dimensions, dtype=numpy.int64)
        parts = [
            (
                self.builder.op.Unsqueeze(value, numpy.array([0], dtype=numpy.int64))
                if isinstance(value, str)
                else numpy.array([value], dtype=numpy.int64)
            )
            for value in dimensions
        ]
        return self.builder.op.Concat(*parts, axis=0)

    def output_dtype(self, node):
        """Reads PyTorch dtype promotion metadata, not inferred ONNX shapes."""
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor):
            return torch_dtype_to_onnx_dtype(value.dtype)
        if isinstance(value, (torch.SymBool, bool)):
            return onnx.TensorProto.BOOL
        if isinstance(value, (torch.SymInt, int)):
            return onnx.TensorProto.INT64
        return None

    def assert_tensor_metadata(self, node):
        """Discharges only assertions proved by exported tensor metadata."""
        from torch.fx.experimental.symbolic_shapes import statically_known_true

        arguments = {
            argument.name: value
            for argument, value in zip(node.target._schema.arguments, node.args)
        }
        arguments.update(node.kwargs)
        source = arguments["a"]
        value = source.meta.get("val") if isinstance(source, torch.fx.Node) else source
        if not isinstance(value, torch.Tensor):
            raise NotImplementedError("Tensor metadata assertions require export metadata.")

        def metadata(item):
            return item.meta.get("val") if isinstance(item, torch.fx.Node) else item

        scalar_metadata = {"dtype": value.dtype, "device": value.device, "layout": value.layout}
        for key in ("size", "stride", *scalar_metadata):
            expected = arguments.get(key)
            if expected is None:
                continue
            if key in ("size", "stride"):
                dimensions = value.shape if key == "size" else value.stride()
                if len(expected) != len(dimensions):
                    raise ValueError(f"Tensor metadata assertion failed for {key}: rank.")
                pairs = zip(dimensions, expected)
            else:
                pairs = [(scalar_metadata[key], expected)]
            for left, right in pairs:
                right = metadata(right)
                if right is None:
                    raise NotImplementedError(f"Cannot prove tensor metadata assertion: {key}.")
                equal = left == right
                if isinstance(equal, bool):
                    if not equal:
                        raise ValueError(f"Tensor metadata assertion failed for {key}.")
                elif not statically_known_true(equal):
                    raise NotImplementedError(
                        f"Cannot prove dynamic tensor metadata assertion: {key}."
                    )
        return None

    def broadcast_tensors(self, tensors, name):
        """Expands tensors to a shared runtime shape without promoting their dtypes."""
        g = self.builder
        if not tensors:
            return []
        rank = max(g.get_rank(tensor) for tensor in tensors)
        shape = numpy.ones(rank, dtype=numpy.int64)
        for tensor in tensors:
            padding = rank - g.get_rank(tensor)
            other = g.op.Shape(tensor)
            if padding:
                other = g.op.Concat(numpy.ones(padding, dtype=numpy.int64), other, axis=0)
            # A zero dimension broadcasts with one to zero, not to their maximum.
            shape = g.op.Where(g.op.Equal(shape, numpy.array(1, dtype=numpy.int64)), other, shape)
        return [
            g.op.Expand(tensor, shape, outputs=[g.unique_name(f"{name}_{index}")])
            for index, tensor in enumerate(tensors)
        ]

    def remainder(self, node, args, outputs):
        """Computes floor remainder with exact integer and floating fmod semantics."""
        g = self.builder
        dtype = self.output_dtype(node)
        floating = dtype in (
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.FLOAT16,
            onnx.TensorProto.BFLOAT16,
        )
        compute_dtype = (
            onnx.TensorProto.FLOAT
            if dtype in (onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16)
            else dtype
        )
        left, right = (
            self.tensor(self.tensor(value, dtype), compute_dtype) for value in args[:2]
        )
        if not floating:
            return g.op.Mod(left, right, fmod=0, outputs=outputs)
        remainder = g.op.Mod(left, right, fmod=1)
        zero = self.tensor(0, compute_dtype)
        adjust = g.op.And(
            g.op.Not(g.op.Equal(remainder, zero)),
            g.op.Xor(g.op.Less(remainder, zero), g.op.Less(right, zero)),
        )
        result = g.op.Where(adjust, g.op.Add(remainder, right), remainder)
        return (
            g.op.Cast(result, to=dtype, outputs=outputs)
            if dtype != compute_dtype
            else g.op.Identity(result, outputs=outputs)
        )

    def run(self):
        """Lowers all nodes after placeholders have been registered."""
        for node in self.module.graph.nodes:
            if node.name in self.raise_list:
                raise RuntimeError(f"Native Torch export stopped at {node.name!r}.")
            if node.op == "placeholder":
                if node not in self.values:
                    raise ValueError(f"Unregistered Torch input {node.name!r}.")
            elif node.op == "get_attr":
                value = self.module
                for part in str(node.target).split("."):
                    value = getattr(value, part)
                self.values[node] = (
                    self.initializer(self.builder.unique_name(node.name), value)
                    if isinstance(value, torch.Tensor)
                    else value
                )
            elif node.op == "call_function":
                self.values[node] = self.call(node)
            elif node.op == "output":
                return self.resolve(node.args[0])
            else:
                raise NotImplementedError(
                    f"Native Torch export does not support FX {node.op!r} at {node.name!r}."
                )
        raise ValueError("The exported Torch graph has no output.")

    def branch(self, module, operands, prefix):
        """Builds an ONNX If branch with native inference and lexical captures."""
        child = self.builder.empty_copy(as_function=True)
        interpreter = NativeTorchInterpreter(child, module, self.dispatcher, self.raise_list)
        placeholders = [node for node in module.graph.nodes if node.op == "placeholder"]
        if len(placeholders) != len(operands):
            raise ValueError("Conditional branch input count does not match its operands.")
        for node, value in zip(placeholders, operands):
            if not isinstance(value, str):
                raise NotImplementedError("Conditional operands must be tensors.")
            if not child.has_name(value):
                child.make_tensor_input(
                    value, self.builder.get_type(value), self.builder.get_shape(value)
                )
            interpreter.values[node] = value
        with child.prefix_name_context(prefix):
            outputs = interpreter.run()
        outputs = (outputs,) if isinstance(outputs, str) else outputs
        for output in outputs:
            child.make_tensor_output(output)
        graph = child.to_native(optimize=False).graph
        graph.ClearField("input")
        return graph, len(outputs)

    def call(self, node):
        """Lowers one supported ATen or higher-order call into native operators."""
        g = self.builder
        args = self.resolve(node.args)
        kwargs = self.resolve(node.kwargs)
        name = g.unique_name(node.name)
        target = node.target
        if self.dispatcher is not None and self.dispatcher.find_function(target) is not None:
            return self.custom(node, args, kwargs, name)
        if target is operator.getitem:
            if isinstance(args[0], (list, tuple)):
                return args[0][args[1]]
            raise NotImplementedError("Tensor indexing must be exported as an ATen operator.")
        if str(target) in ("cond", "higher_order.cond"):
            predicate, true_module, false_module, operands = args
            true_graph, count = self.branch(true_module, operands, name + "_then")
            false_graph, other_count = self.branch(false_module, operands, name + "_else")
            if count != other_count:
                raise ValueError("Conditional branches must return the same number of tensors.")
            result = g.op.If(
                predicate,
                then_branch=true_graph,
                else_branch=false_graph,
                outputs=[g.unique_name(name + f"_{index}") for index in range(count)],
            )
            return (result,) if isinstance(result, str) else result
        python_ops = {
            operator.add: "Add",
            operator.sub: "Sub",
            operator.mul: "Mul",
            operator.gt: "Greater",
            operator.ge: "GreaterOrEqual",
            operator.lt: "Less",
            operator.le: "LessOrEqual",
            operator.eq: "Equal",
        }
        if target in python_ops:
            if all(isinstance(value, (int, float)) for value in args):
                return target(*args)
            return g.make_node(
                python_ops[target], [self.tensor(a, onnx.TensorProto.INT64) for a in args], name
            )
        schema = getattr(target, "_schema", None)
        if schema is None or not schema.name.startswith("aten::"):
            return self.custom(node, args, kwargs, name)
        op = schema.name.split("::", 1)[1]
        outputs = [name]
        if op == "_assert_tensor_metadata":
            return self.assert_tensor_metadata(node)
        if op == "broadcast_tensors":
            return self.broadcast_tensors(args[0], name)
        if op == "remainder":
            return self.remainder(node, args, outputs)
        if op in ("zeros_like", "ones_like"):
            if kwargs.get("layout") not in (None, torch.strided):
                raise NotImplementedError("Native tensor factories require strided layout.")
            return g.op.Expand(
                self.tensor(int(op == "ones_like"), self.output_dtype(node)),
                g.op.Shape(args[0]),
                outputs=outputs,
            )
        if op == "square":
            dtype = self.output_dtype(node)
            value = self.tensor(args[0], dtype)
            if dtype in (onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16):
                value = self.tensor(value, onnx.TensorProto.FLOAT)
                return g.op.Cast(g.op.Mul(value, value), to=dtype, outputs=outputs)
            return g.op.Mul(value, value, outputs=outputs)
        if op in self.unary:
            operand = (
                self.tensor(args[0], onnx.TensorProto.BOOL) if op == "logical_not" else args[0]
            )
            value = g.make_node(self.unary[op], [operand], outputs)
            return g.op.Reciprocal(value) if op == "rsqrt" else value
        if op in self.binary:
            dtype = self.output_dtype(node)
            if dtype == onnx.TensorProto.BOOL and op in ("eq", "ne", "lt", "le", "gt", "ge"):
                operands = [
                    value.meta["val"] if isinstance(value, torch.fx.Node) else value
                    for value in node.args[:2]
                ]
                dtype = torch_dtype_to_onnx_dtype(torch.result_type(*operands))
            left, right = (self.tensor(value, dtype) for value in args[:2])
            if dtype == onnx.TensorProto.BFLOAT16 and op in ("eq", "ne", "lt", "le", "gt", "ge"):
                left, right = (
                    self.tensor(value, onnx.TensorProto.FLOAT) for value in (left, right)
                )
            alpha = kwargs.get("alpha", args[2] if len(args) > 2 and op in ("add", "sub") else 1)
            if alpha != 1:
                right = g.op.Mul(right, self.tensor(alpha, dtype))
            if op == "div" and (kwargs.get("rounding_mode") is not None or len(args) > 2):
                raise NotImplementedError(
                    "Native Torch division currently requires rounding_mode=None."
                )
            if op in ("div", "true_divide") and dtype in (
                onnx.TensorProto.FLOAT16,
                onnx.TensorProto.BFLOAT16,
            ):
                left, right = (
                    self.tensor(value, onnx.TensorProto.FLOAT) for value in (left, right)
                )
                return g.op.Cast(g.op.Div(left, right), to=dtype, outputs=outputs)
            if op == "ne":
                return g.op.Not(g.op.Equal(left, right), outputs=outputs)
            if op == "bitwise_and":
                if dtype == onnx.TensorProto.BOOL:
                    return g.op.And(left, right, outputs=outputs)
                if g.main_opset < 18:
                    raise NotImplementedError("Native bitwise_and requires opset 18 or newer.")
            return g.make_node(self.binary[op], [left, right], outputs)
        if op == "linear":
            value = g.op.MatMul(args[0], g.op.Transpose(args[1], perm=[1, 0]))
            return (
                g.op.Add(value, args[2], outputs=outputs)
                if len(args) > 2 and args[2] is not None
                else g.op.Identity(value, outputs=outputs)
            )
        if op in ("matmul", "mm", "bmm"):
            return g.op.MatMul(*args, outputs=outputs)
        if op == "addmm":
            beta, alpha = kwargs.get("beta", 1), kwargs.get("alpha", 1)
            if beta == 0:
                product = g.op.MatMul(args[1], args[2])
                return (
                    g.op.Identity(product, outputs=outputs)
                    if alpha == 1
                    else g.op.Mul(
                        product, self.tensor(alpha, self.output_dtype(node)), outputs=outputs
                    )
                )
            return g.op.Gemm(
                args[1], args[2], args[0], alpha=float(alpha), beta=float(beta), outputs=outputs
            )
        if op in ("t", "transpose", "permute"):
            rank = g.get_rank(args[0])
            permutation = list(range(rank))
            if op == "t":
                if rank > 2:
                    raise ValueError("aten.t expects a tensor with at most two dimensions.")
                permutation.reverse()
            elif op == "transpose":
                first, second = args[1] % rank, args[2] % rank
                permutation[first], permutation[second] = permutation[second], permutation[first]
            else:
                permutation = list(args[1])
            return g.op.Transpose(args[0], perm=permutation, outputs=outputs)
        if op in ("view", "reshape", "_unsafe_view"):
            return g.op.Reshape(args[0], self.shape_tensor(args[1]), outputs=outputs)
        if op == "sym_size":
            return g.op.Gather(
                g.op.Shape(args[0]), numpy.array(args[1], dtype=numpy.int64), outputs=outputs
            )
        if op == "sym_numel":
            return g.op.Size(args[0], outputs=outputs)
        if op in ("sum", "mean", "amax", "amin", "prod"):
            reduction = {
                "sum": "ReduceSum",
                "mean": "ReduceMean",
                "amax": "ReduceMax",
                "amin": "ReduceMin",
                "prod": "ReduceProd",
            }[op]
            value = self.tensor(args[0], self.output_dtype(node))
            axes = kwargs.get("dim", args[1] if len(args) > 1 else None)
            keep = kwargs.get("keepdim", args[2] if len(args) > 2 else False)
            inputs = [value]
            if axes is not None:
                inputs.append(numpy.asarray(axes, dtype=numpy.int64).reshape(-1))
            return getattr(g.op, reduction + "AnyOpset")(
                *inputs, keepdims=int(keep), outputs=outputs
            )
        if op in ("cat", "stack"):
            values = args[0]
            axis = kwargs.get("dim", args[1] if len(args) > 1 else 0)
            if op == "stack":
                values = [
                    g.op.Unsqueeze(value, numpy.array([axis], dtype=numpy.int64))
                    for value in values
                ]
            return g.op.Concat(*values, axis=axis, outputs=outputs)
        if op == "unsqueeze":
            return g.op.UnsqueezeAnyOpset(
                args[0], numpy.array([args[1]], dtype=numpy.int64), outputs=outputs
            )
        if op == "squeeze":
            axes = args[1] if len(args) > 1 else None
            return (
                g.op.Squeeze(args[0], outputs=outputs)
                if axes is None
                else g.op.SqueezeAnyOpset(
                    args[0], numpy.asarray(axes, dtype=numpy.int64).reshape(-1), outputs=outputs
                )
            )
        if op == "select":
            return g.op.Gather(
                args[0], numpy.array(args[2], dtype=numpy.int64), axis=args[1], outputs=outputs
            )
        if op == "where":
            dtype = self.output_dtype(node)
            return g.op.Where(
                self.tensor(args[0], onnx.TensorProto.BOOL),
                self.tensor(args[1], dtype),
                self.tensor(args[2], dtype),
                outputs=outputs,
            )
        if op == "slice":
            axis, start, end = args[1:4]
            step = args[4] if len(args) > 4 else 1
            return g.op.Slice(
                args[0],
                numpy.array([start], dtype=numpy.int64),
                numpy.array([end], dtype=numpy.int64),
                numpy.array([axis], dtype=numpy.int64),
                numpy.array([step], dtype=numpy.int64),
                outputs=outputs,
            )
        if op in ("detach", "clone", "contiguous", "alias", "lift_fresh_copy"):
            return g.op.Identity(args[0], outputs=outputs)
        if op == "_to_copy":
            dtype = kwargs.get("dtype")
            return (
                g.op.Identity(args[0], outputs=outputs)
                if dtype is None
                else g.op.Cast(args[0], to=torch_dtype_to_onnx_dtype(dtype), outputs=outputs)
            )
        if op in ("softmax", "log_softmax", "_softmax", "_log_softmax"):
            value = self.tensor(args[0], self.output_dtype(node))
            return g.make_node(
                "LogSoftmax" if "log" in op else "Softmax", [value], outputs, axis=args[1]
            )
        if op == "gelu":
            if g.main_opset < 20:
                raise NotImplementedError(
                    "Native Torch Gelu currently requires opset 20 or newer."
                )
            return g.op.Gelu(
                args[0], approximate=kwargs.get("approximate", "none"), outputs=outputs
            )
        if op == "flatten":
            start, end = args[1] if len(args) > 1 else 0, args[2] if len(args) > 2 else -1
            if end not in (-1, g.get_rank(args[0]) - 1) or start not in (0, 1):
                raise NotImplementedError(
                    "Native flatten currently supports complete trailing dimensions."
                )
            return (
                g.op.Reshape(args[0], numpy.array([-1], dtype=numpy.int64), outputs=outputs)
                if start == 0
                else g.op.Flatten(args[0], axis=1, outputs=outputs)
            )
        if op in ("dropout", "feature_dropout"):
            if args[2] if len(args) > 2 else kwargs.get("train", True):
                raise NotImplementedError(
                    "Training-mode dropout is not supported by native Torch export."
                )
            return g.op.Identity(args[0], outputs=outputs)
        if op in ("conv1d", "conv2d", "conv3d", "convolution"):
            return self.convolution(op, args, kwargs, outputs)
        return self.custom(node, args, kwargs, name)

    def convolution(self, op, args, kwargs, outputs):
        """Lowers convolution attributes without computing output shapes."""
        rank = self.builder.get_rank(args[1]) - 2

        def attribute(position, key, default):
            value = kwargs.get(key, args[position] if len(args) > position else default)
            return [value] * rank if isinstance(value, int) else value

        stride = attribute(3, "stride", 1)
        padding = attribute(4, "padding", 0)
        dilation = attribute(5, "dilation", 1)
        transposed = op == "convolution" and args[6]
        groups = (
            args[8]
            if op == "convolution"
            else kwargs.get("groups", args[6] if len(args) > 6 else 1)
        )
        attributes = {"strides": stride, "dilations": dilation, "group": groups}
        if isinstance(padding, str):
            attributes["auto_pad"] = {"same": "SAME_UPPER", "valid": "VALID"}[padding]
        else:
            attributes["pads"] = list(padding) * 2
        if transposed:
            attributes["output_padding"] = args[7]
        return self.builder.make_node(
            "ConvTranspose" if transposed else "Conv", args[:3], outputs, **attributes
        )

    def custom(self, node, args, kwargs, name):
        """Calls an explicit custom lowering or rejects an unsupported FX target."""
        converter = (
            self.dispatcher.find_function(node.target) if self.dispatcher is not None else None
        )
        if converter is not None:
            value = node.meta.get("val")
            outputs = (
                [self.builder.unique_name(f"{name}_{index}") for index in range(len(value))]
                if isinstance(value, (tuple, list))
                else [name]
            )
            state: dict[str, bool | torch.dtype] = {"native": True}
            if isinstance(value, torch.Tensor):
                state["dtype"] = value.dtype
            return converter(self.builder, state, outputs, *args, **kwargs)
        raise NotImplementedError(
            f"Native Torch export does not support {node.target!s} at FX node {node.name!r}; "
            "no Python builder or optimizer fallback is available."
        )


def to_onnx(
    mod,
    args=None,
    kwargs=None,
    input_names=None,
    target_opset=None,
    as_function=False,
    options=None,
    verbose=0,
    return_builder=False,
    raise_list=None,
    dynamic_shapes=None,
    optimize=True,
    dispatcher=None,
    large_model=False,
    external_threshold=1024,
    export_options=None,
    return_optimize_report=False,
    filename=None,
    inline=True,
    export_modules_as_functions=False,
    function_options=None,
    output_names=None,
    output_dynamic_shapes=None,
    validate_onnx=False,
    return_ep=False,
):
    """Exports a supported Torch program using only native ONNX graph engines.

    Tensor arguments, captured parameters, dynamic dimensions, standard ATen
    operations and ``torch.cond`` are supported. Nested argument pytrees,
    mutations, alternative tracing frontends and module-boundary function
    preservation are rejected explicitly. Standalone function export promotes
    weights to function inputs or embeds native Constant nodes.
    """
    if export_modules_as_functions:
        raise NotImplementedError(
            "Native Torch export does not yet preserve submodule boundaries."
        )
    if output_dynamic_shapes is not None:
        raise NotImplementedError("Native output dimensions are inferred from input dimensions.")
    if options is not None and not isinstance(options, OnnxLightOptimizationOptions):
        raise TypeError("Native Torch export requires OnnxLightOptimizationOptions.")
    if dispatcher is not None and not callable(getattr(dispatcher, "find_function", None)):
        raise TypeError("dispatcher must expose find_function.")
    export_options = ExportOptions() if export_options is None else export_options
    if not isinstance(export_options, ExportOptions):
        raise TypeError("export_options must be ExportOptions.")
    if function_options is not None and not isinstance(function_options, FunctionOptions):
        raise TypeError("function_options must be FunctionOptions.")
    if (
        export_options.tracing != TracingMode.DEFAULT
        or export_options.jit
        or export_options.dynamo
        or export_options.fake
        or export_options.converting_library != ConvertingLibrary.DEFAULT
        or export_options.strategy == "transformers"
    ):
        raise NotImplementedError("Native Torch export currently uses torch.export.export only.")
    if export_options.save_ep or export_options.validate_ep:
        raise NotImplementedError("ExportOptions.save_ep and validate_ep are not yet supported.")
    if export_options.tracing_module_leaves or export_options.backed_size_oblivious not in (
        "auto",
        False,
    ):
        raise NotImplementedError(
            "Custom tracing leaves and backed-size-oblivious patches are unsupported."
        )
    from .onnx_export import get_default_aten_as_function

    if export_options.aten_as_function and (
        export_options.aten_as_function is True
        or set(export_options.aten_as_function) != set(get_default_aten_as_function())
    ):
        raise NotImplementedError(
            "Native Torch export does not yet preserve ATen function boundaries."
        )
    if function_options is not None and (
        function_options.move_initializer_to_constant
        or function_options.inline
        or function_options.merge_allowed
        or function_options.rename_allowed
        or function_options.external_threshold != 2**25
    ):
        raise NotImplementedError(
            "Native standalone functions support name, domain and return_initializer options."
        )
    function_export = as_function or (
        function_options is not None and function_options.export_as_function
    )
    if function_options is not None and not function_export:
        raise NotImplementedError("Native function options require standalone function export.")
    if function_export and (large_model or filename or validate_onnx):
        raise NotImplementedError(
            "Standalone functions do not support container export, saving "
            "or automatic validation."
        )
    if kwargs is None and isinstance(args, dict):
        kwargs, args = args, ()
    kwargs = dict(kwargs or {})
    args = (args,) if isinstance(args, torch.Tensor) else tuple(args or ())
    if isinstance(mod, torch.export.ExportedProgram):
        program = mod
        if not args and not kwargs and program.example_inputs is not None:
            args, kwargs = program.example_inputs
    else:
        if not isinstance(mod, torch.nn.Module):
            raise TypeError("Native Torch export expects a Module or ExportedProgram.")
        if any(isinstance(value, (list, tuple, dict)) for value in (*args, *kwargs.values())):
            raise NotImplementedError("Native Torch export currently requires flat arguments.")
        program = torch.export.export(
            mod,
            args,
            kwargs=kwargs,
            dynamic_shapes=_dynamic_shapes(dynamic_shapes),
            strict=export_options.strict,
            prefer_deferred_runtime_asserts_over_guards=(
                export_options.prefer_deferred_runtime_asserts_over_guards
            ),
        )
    if export_options.decomposition_table or export_options.remove_inplace:
        decompositions = (
            export_options.get_decomposition_table() if export_options.decomposition_table else {}
        )
        program = program.run_decompositions(decompositions)
    from torch.export.graph_signature import (
        ConstantArgument,
        InputKind,
        OutputKind,
        TensorArgument,
    )

    if any(spec.kind != OutputKind.USER_OUTPUT for spec in program.graph_signature.output_specs):
        raise NotImplementedError(
            "Native Torch export does not support mutations or training outputs."
        )
    if target_opset is None:
        from ... import DEFAULT_TARGET_OPSET

        target_opset = DEFAULT_TARGET_OPSET
    builder = TorchOnnxLightGraphBuilder(
        target_opset, optimization_options=options, verbose=verbose
    )
    interpreter = NativeTorchInterpreter(builder, program.graph_module, dispatcher, raise_list)
    placeholders = [node for node in program.graph.nodes if node.op == "placeholder"]
    tensor_specs = [
        spec
        for spec in program.graph_signature.input_specs
        if spec.kind == InputKind.USER_INPUT and isinstance(spec.arg, TensorArgument)
    ]
    if input_names is not None and len(input_names) != len(tensor_specs):
        raise ValueError("input_names must match the number of tensor inputs.")
    declared_names = (
        list(input_names)
        if input_names is not None
        else [str(spec.arg.name) for spec in tensor_specs]
    )
    if len(set(declared_names)) != len(declared_names):
        raise ValueError("Tensor input names must be unique.")
    for name in declared_names:
        builder.unique_name(name)
    started = time.perf_counter()
    tensor_index = 0
    user_index = 0
    for node, spec in zip(placeholders, program.graph_signature.input_specs):
        if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER, InputKind.CONSTANT_TENSOR):
            if spec.target is None:
                raise ValueError(f"Exported {spec.kind.name} input {node.name!r} has no target.")
            value = (
                program.state_dict[spec.target]
                if spec.target in program.state_dict
                else program.constants[spec.target]
            )
            interpreter.values[node] = interpreter.initializer(
                builder.unique_name(str(spec.target)), value
            )
        elif spec.kind == InputKind.USER_INPUT:
            if isinstance(spec.arg, TensorArgument):
                name = input_names[tensor_index] if input_names is not None else node.name
                annotation = None
                if isinstance(dynamic_shapes, dict):
                    annotation = dynamic_shapes.get(str(node.target))
                elif dynamic_shapes is not None:
                    annotation = dynamic_shapes[user_index]
                value = node.meta["val"]
                builder.make_tensor_input(
                    name, torch_dtype_to_onnx_dtype(value.dtype), _shape(value, annotation)
                )
                interpreter.values[node] = name
                tensor_index += 1
            elif isinstance(spec.arg, ConstantArgument):
                interpreter.values[node] = spec.arg.value
            else:
                raise NotImplementedError(
                    f"Native Torch export does not support user input {spec.arg!r}."
                )
            user_index += 1
        else:
            raise NotImplementedError(f"Unsupported exported input kind {spec.kind!r}.")
    outputs = interpreter.run()
    outputs = (outputs,) if isinstance(outputs, (str, torch.Tensor)) else outputs
    if output_names is not None and len(output_names) != len(outputs):
        raise ValueError("output_names must match the exported tensor output count.")
    if output_names is not None and len(set(output_names)) != len(output_names):
        raise ValueError("Tensor output names must be unique.")
    for index, value in enumerate(outputs):
        name = value if isinstance(value, str) else interpreter.tensor(value)
        if output_names is not None and output_names[index] != name:
            name = builder.op.Identity(name, outputs=[output_names[index]])
        elif name in builder.output_names:
            name = builder.op.Identity(name, outputs=[builder.unique_name(name + "_output")])
        builder.make_tensor_output(name)
    artifact = builder.to_onnx(
        optimize=optimize,
        inline=inline,
        large_model=large_model,
        external_threshold=external_threshold,
        return_optimize_report=True,
    )
    report = artifact.report
    if report is None:
        raise RuntimeError("Native Torch export requires the requested optimization report.")
    report.update(
        {"torch_backend": "native", "time_torch_lowering": time.perf_counter() - started}
    )
    if verbose:
        model = artifact.get_proto(include_weights=False)
        if not isinstance(model, onnx.ModelProto):
            raise TypeError(f"Expected a native ModelProto, got {type(model)!r}.")
        print(
            f"[nativeTorch] {len(model.graph.node)} ONNX nodes, "
            f"{report.extra['rewrites']} native rewrites"
        )
    if function_export:
        artifact = _function_artifact(artifact, function_options)
    if return_ep:
        artifact.ep = program
    if filename:
        artifact.save(filename)
    if validate_onnx:
        from onnxruntime import InferenceSession

        session = InferenceSession(
            artifact.get_proto().SerializeToString(), providers=["CPUExecutionProvider"]
        )
        flat_inputs, _ = torch.utils._pytree.tree_flatten((args, kwargs))
        values = [value for value in flat_inputs if isinstance(value, torch.Tensor)]
        if len(values) != len(builder.input_names):
            raise ValueError("Validation tensor arguments do not match the exported inputs.")
        named_inputs = {
            name: value.detach().cpu().numpy() for name, value in zip(builder.input_names, values)
        }
        feeds = {name: named_inputs[name] for name in artifact.input_names}
        actual = session.run(None, feeds)
        with torch.no_grad():
            expected = program.module()(*args, **kwargs)
        expected, _ = torch.utils._pytree.tree_flatten(expected)
        if len(expected) != len(actual):
            raise ValueError("Validation result count does not match the exported outputs.")
        tolerance = float(validate_onnx) if isinstance(validate_onnx, float) else 1e-5
        for got, want in zip(actual, expected):
            if not isinstance(got, numpy.ndarray):
                raise TypeError(
                    f"Native Torch validation expects tensor outputs, got {type(got)!r}."
                )
            value = want.detach().cpu().numpy() if isinstance(want, torch.Tensor) else want
            numpy.testing.assert_allclose(got, value, atol=tolerance, rtol=tolerance)
    return artifact


def _function_artifact(artifact, options):
    """Packages a natively optimized graph as a native standalone function."""
    model = artifact.proto
    names = [str(value.name) for value in model.graph.input]
    constants = []
    weights = {}
    for tensor in model.graph.initializer:
        name = str(tensor.name)
        if options is not None and options.return_initializer:
            names.append(name)
            weights[name] = tensor
        else:
            constants.append(helper.make_node("Constant", [], [name], value=tensor))
    function = helper.make_function(
        options.domain if options is not None and options.domain else "yobx.torch",
        options.name if options is not None and options.name else "forward",
        names,
        [str(value.name) for value in model.graph.output],
        [*constants, *model.graph.node],
        list(model.opset_import),
    )
    return ExportArtifact(
        proto=function,
        report=artifact.report,
        builder=artifact.builder,
        function=FunctionPieces(
            initializers_name=list(weights) or None,
            initializers_dict=weights or None,
            initializers_renaming={name: name for name in weights} or None,
            nested_functions=list(model.functions) or None,
        ),
    )

from typing import Any, Callable, Dict, List, Optional
from onnx_light.onnx import (
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
    helper,
    numpy_helper,
)
from onnx_light.onnx.reference import ReferenceEvaluator
from .ops._native_op import NativeOpKernel
from .ops.op__extended_add_add_mul_mul import (
    AddAdd,
    AddMul,
    AddSharedInput,
    MulAdd,
    MulMul,
    MulSharedInput,
    MulSub,
    SubMul,
)
from .ops.op__extended_mul_sigmoid import MulSigmoid
from .ops.op__extended_negxplus1 import NegXplus1
from .ops.op__extended_replace_zero import ReplaceZero
from .ops.op__extended_rotary import Rotary
from .ops.op__extended_scatternd_of_shape import MaskedScatterNDOfShape, ScatterNDOfShape
from .ops.op__extended_transpose_cast import Transpose2DCastFP16, Transpose2DCastFP32
from .ops.op__extended_tri_matrix import TriMatrix
from .ops.op__overwrite_argminmax import ArgMax, ArgMin
from .ops.op__overwrite_comparison import Greater, Less
from .ops.op__overwrite_compress import Compress
from .ops.op__overwrite_reduce import ReduceMax, ReduceMean, ReduceMin
from .ops.op_attention import Attention
from .ops.op_bias_softmax import BiasSoftmax
from .ops.op_complex import (
    ComplexModule,
    ComplexMul,
    ComplexMulConj,
    FftC2r,
    FftIrfft2,
    Istft,
    ToComplex,
)
from .ops.op_fast_gelu import FastGelu
from .ops.op_fused_matmul import FusedMatMul
from .ops.op_fused_matmul_activation import FusedMatMulActivation
from .ops.op_gemm_fast_gelu import GemmFastGelu
from .ops.op_memcpy_host import MemcpyFromHost, MemcpyToHost
from .ops.op_qlinear_average_pool import QLinearAveragePool
from .ops.op_qlinear_conv import QLinearConv
from .ops.op_quick_gelu import QuickGelu
from .ops.op_skip_layer_normalization import SkipLayerNormalization
from .ops.op_simplified_layer_normalization import SimplifiedLayerNormalization


class ExtendedReferenceEvaluator(ReferenceEvaluator):
    """Executes models with the onnx-light native runtime.

    Operators execute through the native runtime. Project NumPy kernels, including
    full-precision comparison and index reductions, are explicitly registered
    through its custom-kernel API. Missing kernels never trigger another evaluator
    or runtime.

    ``run(None, feeds)`` returns outputs in graph declaration order. ``run(feeds_list)``
    maps positional values to graph inputs. Models, serialized models, filenames,
    graphs, individual nodes, functions and export artifacts are accepted.

    ``new_ops`` accepts :class:`NativeOpKernel` subclasses. Alternatively,
    ``register_custom_kernel(domain, op_type, fn)`` registers a callable with
    signature ``fn(node, *inputs)`` returning an array or a tuple of arrays.
    """

    default_ops: List[type[NativeOpKernel]] = [
        ArgMax,
        ArgMin,
        Greater,
        Less,
        Compress,
        ReduceMax,
        ReduceMean,
        ReduceMin,
        Attention,
        BiasSoftmax,
        ComplexModule,
        ComplexMul,
        ComplexMulConj,
        FftC2r,
        FftIrfft2,
        Istft,
        FastGelu,
        FusedMatMul,
        FusedMatMulActivation,
        GemmFastGelu,
        MemcpyFromHost,
        MemcpyToHost,
        QLinearConv,
        QLinearAveragePool,
        QuickGelu,
        SimplifiedLayerNormalization,
        SkipLayerNormalization,
        ToComplex,
        AddAdd,
        AddMul,
        AddSharedInput,
        MaskedScatterNDOfShape,
        MulAdd,
        MulMul,
        MulSharedInput,
        MulSigmoid,
        MulSub,
        NegXplus1,
        ReplaceZero,
        Rotary,
        ScatterNDOfShape,
        SubMul,
        Transpose2DCastFP16,
        Transpose2DCastFP32,
        TriMatrix,
    ]

    @staticmethod
    def filter_ops(proto, new_ops, opsets):
        """Selects the highest compatible version of each custom kernel."""
        if opsets is None and isinstance(proto, (ModelProto, FunctionProto)):
            opsets = {d.domain: d.version for d in proto.opset_import}
        best = {}
        renamed = set()
        for kernel in new_ops:
            name, separator, version = kernel.__name__.rpartition("_")
            if not separator or not version.isdecimal():
                continue
            version = int(version)
            if opsets is not None and version > opsets.get(kernel.op_domain, 1):
                continue
            renamed.add(kernel.__name__)
            key = kernel.op_domain, name
            if key not in best or best[key][0] < version:
                best[key] = (version, kernel)
        result = [kernel for kernel in new_ops if kernel.__name__ not in renamed]
        for (domain, name), (_, kernel) in best.items():
            result.append(type(name, (kernel,), {"op_domain": domain}))
        return result

    def __init__(
        self,
        proto: Any,
        opsets: Optional[Dict[str, int]] = None,
        functions: Optional[List] = None,
        verbose: int = 0,
        new_ops: Optional[List[type[NativeOpKernel]]] = None,
        **kwargs,
    ):
        from ..container.export_artifact import ExportArtifact

        if isinstance(proto, ExportArtifact):
            proto = proto.get_proto(include_weights=True)
        node_proto = proto if isinstance(proto, NodeProto) else None
        if node_proto is not None:
            from ..helpers.onnx_helper import get_hidden_inputs

            proto = helper.make_graph(
                [node_proto],
                str(node_proto.name or node_proto.op_type),
                [],
                [
                    helper.make_tensor_value_info(str(name), 0, None)
                    for name in node_proto.output
                    if name
                ],
            )
            inputs = list(dict.fromkeys(str(name) for name in node_proto.input if name))
            inputs.extend(sorted(get_hidden_inputs(proto) - set(inputs)))
            proto.input.extend(
                helper.make_tensor_value_info(str(name), 0, None) for name in inputs
            )
            if opsets is None:
                opsets = {"": 18}
                if node_proto.domain:
                    opsets[str(node_proto.domain)] = 1
        proto = self._load_proto(proto)
        self.proto_ = node_proto if node_proto is not None else proto
        functions = [
            function.proto_ if isinstance(function, ExtendedReferenceEvaluator) else function
            for function in functions or []
        ]
        if any(not isinstance(function, FunctionProto) for function in functions):
            raise TypeError("functions must contain native FunctionProto objects.")
        if isinstance(proto, GraphProto):
            proto = helper.make_model(
                proto,
                opset_imports=[
                    helper.make_opsetid(domain, version)
                    for domain, version in (opsets or {"": 18}).items()
                ],
                functions=functions,
            )
        elif isinstance(proto, ModelProto) and (functions or opsets):
            model = ModelProto()
            model.CopyFrom(proto)
            model.functions.extend(functions)
            if opsets:
                model.ClearField("opset_import")
                model.opset_import.extend(
                    helper.make_opsetid(domain, version) for domain, version in opsets.items()
                )
            proto = model
        self._extra_functions = functions
        kernels = [*(new_ops or []), *self.default_ops]
        for kernel in kernels:
            if not isinstance(kernel, type) or not issubclass(kernel, NativeOpKernel):
                raise TypeError(
                    "new_ops requires NativeOpKernel subclasses; use "
                    "register_custom_kernel(domain, op_type, fn) for native callbacks."
                )
        self._kernel_classes = self.filter_ops(proto, kernels, opsets)
        self._native_options = {"verbose": verbose, **kwargs}
        self._registered_callbacks: Dict[tuple[str, str], Callable] = {}
        self._execution_graph = proto.graph if isinstance(proto, ModelProto) else proto
        # The wheel's initializer cache loses STRING payloads. The native feed
        # path preserves them and supports overriding initializer values.
        self._string_initializers = (
            {
                str(value.name): numpy_helper.to_array(value)
                for value in self._execution_graph.initializer
                if value.data_type == TensorProto.STRING
            }
            if isinstance(self._execution_graph, GraphProto)
            else {}
        )
        super().__init__(proto, verbose=verbose, **kwargs)
        registered = set()
        for kernel in self._kernel_classes:
            key = kernel.op_domain, kernel.__name__
            if key not in registered:
                self.register_custom_kernel(*key, kernel(None, {}))
                registered.add(key)

    def register_custom_kernel(self, domain, op_type, fn):
        """Registers an explicit callback, including for attributed function calls."""
        super().register_custom_kernel(domain, op_type, fn)
        self._registered_callbacks[domain, op_type] = fn

    def unregister_custom_kernel(self, domain, op_type):
        """Unregisters an explicit callback."""
        super().unregister_custom_kernel(domain, op_type)
        self._registered_callbacks.pop((domain, op_type), None)

    @property
    def input_types(self):
        """Returns the declared graph input types."""
        graph = self._execution_graph
        if isinstance(graph, FunctionProto):
            return None
        return [value.type for value in graph.input if value.name in self.input_names]

    @property
    def output_types(self):
        """Returns the declared graph output types."""
        graph = self._execution_graph
        if isinstance(graph, FunctionProto):
            return None
        return [value.type for value in graph.output]

    @property
    def rt_inits_(self):
        """Returns graph initializers as arrays."""
        graph = self._execution_graph
        if isinstance(graph, FunctionProto):
            return {}
        return {value.name: numpy_helper.to_array(value) for value in graph.initializer}

    def run(self, output_names, feed_inputs=None, attributes=None, intermediate=False):
        """Returns requested outputs, or a dictionary including intermediate values."""
        if feed_inputs is None and isinstance(output_names, list):
            if len(output_names) != len(self.input_names):
                raise ValueError(
                    f"Expected {len(self.input_names)} inputs, got {len(output_names)}."
                )
            feed_inputs = dict(zip(self.input_names, output_names))
            output_names = None
        if self._string_initializers:
            feed_inputs = {**self._string_initializers, **(feed_inputs or {})}
        if attributes or (isinstance(self.proto_, FunctionProto) and self._extra_functions):
            if not isinstance(self.proto_, FunctionProto):
                raise NotImplementedError("Run attributes are only supported for functions.")
            if intermediate:
                raise NotImplementedError(
                    "Intermediate outputs of wrapped function calls are not supported."
                )
            function = self.proto_
            model = helper.make_model(
                helper.make_graph(
                    [
                        helper.make_node(
                            function.name,
                            list(function.input),
                            list(function.output),
                            domain=function.domain,
                            **(attributes or {}),
                        )
                    ],
                    function.name,
                    [helper.make_tensor_value_info(name, 0, None) for name in function.input],
                    [helper.make_tensor_value_info(name, 0, None) for name in function.output],
                ),
                functions=[*self._extra_functions, function],
                opset_imports=[*function.opset_import, helper.make_opsetid(function.domain, 1)],
            )
            evaluator = ReferenceEvaluator(model, **self._native_options)
            for (domain, op_type), fn in self._registered_callbacks.items():
                evaluator.register_custom_kernel(domain, op_type, fn)
            return evaluator.run(output_names, feed_inputs)
        if intermediate:
            graph = self._execution_graph
            names = list(
                dict.fromkeys(name for node in graph.node for name in node.output if name)
            )
            values = super().run(names, feed_inputs)
            return {"": None, **self.rt_inits_, **feed_inputs, **dict(zip(names, values))}
        return super().run(output_names, feed_inputs)

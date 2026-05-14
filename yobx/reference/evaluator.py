from typing import Any, Dict, List, Optional, Union
from onnx import FunctionProto, ModelProto, NodeProto, TypeProto
from onnx.defs import get_schema
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun
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
from .ops.op__overwrite_gather_elements import GatherElements
from .ops.op__overwrite_gather import Gather
from .ops.op__overwrite_scatter_elements import ScatterElements
from .ops.op_attention import Attention
from .ops.op_bias_softmax import BiasSoftmax
from .ops.op_complex import ComplexModule, ComplexMul, ComplexMulConj, ToComplex
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
    """
    Extends :class:`onnx.reference.ReferenceEvaluator` with additional operator
    kernels for non-standard domains such as ``com.microsoft``.

    The evaluator allows testing scenarios outside what a standard ONNX backend
    can handle, such as optimization patterns that rely on ONNX Runtime contrib
    operators (e.g. :class:`FusedMatMul <yobx.reference.ops.op_fused_matmul.FusedMatMul>`,
    :class:`QuickGelu <yobx.reference.ops.op_quick_gelu.QuickGelu>`).

    **Basic usage** — run an ONNX model with standard operators:

    .. runpython::
        :showcode:

        import numpy as np
        import onnx.helper as oh
        import onnx
        from yobx.reference import ExtendedReferenceEvaluator

        TFLOAT = onnx.TensorProto.FLOAT
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Add", ["X", "Y"], ["Z"])],
                "add_graph",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [None, None]),
                    oh.make_tensor_value_info("Y", TFLOAT, [None, None]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        ref = ExtendedReferenceEvaluator(model)
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        (result,) = ref.run(None, {"X": x, "Y": x})
        print(result)

    **Using contrib operators** — run a ``com.microsoft`` operator:

    .. runpython::
        :showcode:

        import numpy as np
        import onnx.helper as oh
        import onnx
        from yobx.reference import ExtendedReferenceEvaluator

        TFLOAT = onnx.TensorProto.FLOAT
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("FusedMatMul", ["X", "Y"], ["Z"], domain="com.microsoft")],
                "fused_mm",
                [
                    oh.make_tensor_value_info("X", TFLOAT, None),
                    oh.make_tensor_value_info("Y", TFLOAT, None),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, None)],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
        )
        ref = ExtendedReferenceEvaluator(model)
        a = np.arange(4, dtype=np.float32).reshape(2, 2)
        (result,) = ref.run(None, {"X": a, "Y": a})
        print(result)

    **Adding custom operators** — pass extra :class:`OpRun
    <onnx.reference.op_run.OpRun>` subclasses via ``new_ops``:

    .. runpython::
        :showcode:

        import numpy as np
        import onnx.helper as oh
        import onnx
        from onnx.reference.op_run import OpRun
        from yobx.reference import ExtendedReferenceEvaluator

        TFLOAT = onnx.TensorProto.FLOAT

        class MyCustomOp(OpRun):
            op_domain = "my.domain"

            def _run(self, X):
                return (X * 2,)

        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("MyCustomOp", ["X"], ["Z"], domain="my.domain")],
                "custom_graph",
                [oh.make_tensor_value_info("X", TFLOAT, [None])],
                [oh.make_tensor_value_info("Z", TFLOAT, [None])],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("my.domain", 1)],
            ir_version=10,
        )
        ref = ExtendedReferenceEvaluator(model, new_ops=[MyCustomOp])
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (result,) = ref.run(None, {"X": x})
        print(result)

    The ``new_ops`` list is *merged* with :attr:`default_ops`; you do not need
    to re-list the built-in contrib operators.

    The class overloads or adds the following operators by default:

    .. runpython::
        :showcode:

        import pprint
        from yobx.reference import ExtendedReferenceEvaluator

        pprint.pprint(ExtendedReferenceEvaluator.default_ops)
    """

    default_ops: List[type[OpRun]] = [
        # overwritten
        Gather,
        GatherElements,
        ScatterElements,
        # com.microsoft
        Attention,
        BiasSoftmax,
        ComplexModule,
        ComplexMul,
        ComplexMulConj,
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
        # yaourt.ortops.fused_kernel.cuda
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
        if opsets is None and isinstance(proto, (ModelProto, FunctionProto)):
            opsets = {d.domain: d.version for d in proto.opset_import}
        best = {}
        renamed = {}
        for cl in new_ops:
            if "_" not in cl.__name__:
                continue
            vers = cl.__name__.split("_")
            try:
                v = int(vers[-1])
            except ValueError:
                # not a version
                continue
            if opsets is not None and v > opsets.get(cl.op_domain, 1):
                continue
            renamed[cl.__name__] = cl
            key = cl.op_domain, "_".join(vers[:-1])
            if key not in best or best[key][0] < v:
                best[key] = (v, cl)

        modified = []
        for cl in new_ops:
            if cl.__name__ not in renamed:
                modified.append(cl)
        for k, v in best.items():
            atts = {"domain": k[0]}
            bases = (v[1],)
            if not hasattr(v[1], "op_schema"):
                atts["op_schema"] = get_schema(k[1], v[0], domain=v[1].op_domain)
            new_cl = type(k[1], bases, atts)
            modified.append(new_cl)

        new_ops = modified
        return new_ops

    def __init__(
        self,
        proto: Any,
        opsets: Optional[Dict[str, int]] = None,
        functions: Optional[List[Union[ReferenceEvaluator, FunctionProto]]] = None,
        verbose: int = 0,
        new_ops: Optional[List[type[OpRun]]] = None,
        **kwargs,
    ):
        from ..container.export_artifact import ExportArtifact

        if isinstance(proto, ExportArtifact):
            # Unwrap: prefer the container (for large models) then the proto.
            proto = proto.container if proto.container is not None else proto.proto

        if new_ops is None:
            new_ops = ExtendedReferenceEvaluator.default_ops
        else:
            new_ops = new_ops.copy()
            new_ops.extend(ExtendedReferenceEvaluator.default_ops)
        new_ops = ExtendedReferenceEvaluator.filter_ops(proto, new_ops, opsets)

        ReferenceEvaluator.__init__(
            self,
            proto,
            opsets=opsets,
            functions=functions,
            verbose=verbose,
            new_ops=new_ops,
            **kwargs,
        )

    def run(self, *args, **kwargs):
        """See :meth:`onnx.reference.ReferenceEvaluator.run`."""
        if len(args) == 1 and isinstance(args[0], list):
            feeds = dict(zip(self.input_names, args[0]))
            return self.run(None, feeds, **kwargs)
        if isinstance(self.proto_, FunctionProto):
            return self._run_function(*args, **kwargs)
        return ReferenceEvaluator.run(self, *args, **kwargs)

    def _load_impl(self, node: NodeProto, input_types: TypeProto | None = None) -> Any:
        res = super()._load_impl(node, input_types)
        assert (
            not hasattr(res, "op_domain") or res.op_domain == node.domain
        ), f"Domain mismatch {res.op_domain!r} != {node.domain} for node={node}"
        return res

    def _run_function(
        self,
        output_names,
        feed_inputs: Dict[str, Any],
        attributes: Optional[Dict[str, Any]] = None,
        intermediate: bool = False,
    ) -> Union[Dict[str, Any], List[Any]]:  # type: ignore
        if output_names is None:
            output_names = self.output_names

        # step 1: inputs and initializers
        results = {"": None}  # optional input
        results.update(self.rt_inits_)  # type: ignore[arg-type]
        results.update(feed_inputs)
        for k, v in self.rt_inits_.items():
            self._log(2, " +C %s: %s", k, v)  # type: ignore[arg-type]
        for k, v in feed_inputs.items():
            self._log(2, " +I %s: %s", k, v)  # type: ignore[arg-type]

        # step 2: execute nodes
        for node in self.rt_nodes_:
            self._log(1, "%s(%s) -> %s", node.op_type, node.input, node.output)
            for i in node.input:
                if i not in results:
                    raise RuntimeError(
                        f"Unable to find input {i!r} in known results {sorted(results)}, "
                        f"self.rt_inits_ has {sorted(self.rt_inits_)}, "
                        f"feed_inputs has {sorted(feed_inputs)}."
                    )
            inputs = [results[i] for i in node.input]
            linked_attributes = {}
            if node.has_linked_attribute and attributes:
                linked_attributes["linked_attributes"] = attributes
            if node.need_context():
                outputs = node.run(*inputs, context=results, **linked_attributes)
            else:
                outputs = node.run(*inputs, **linked_attributes)
            for name, value in zip(node.output, outputs):
                self._log(2, " + %s: %s", name, value)  # type: ignore[arg-type]
                results[name] = value

        # return the results
        if intermediate:
            return results

        for name in output_names:
            if name not in results:
                raise RuntimeError(
                    f"Unable to find output name {name!r} "
                    f"in {sorted(results)}, proto is\n{self.proto_}"
                )
        return [results[name] for name in output_names]

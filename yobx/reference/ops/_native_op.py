"""Adapts project-specific NumPy kernels to the native runtime callback API."""

from onnx_light.onnx import helper
from onnx_light.onnx.reference import ReferenceEvaluator


class NativeOpKernel:
    """Adapts an existing project kernel without depending on a Python operator runtime."""

    op_domain = ""

    def __init__(self, node, run_params=None):
        self.node = node

    def __call__(self, node, *inputs):
        attributes = {
            attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute
        }
        attributes = {
            name: value.decode("utf-8") if isinstance(value, bytes) else value
            for name, value in attributes.items()
        }
        outputs = self._run(*inputs, **attributes)
        return tuple(outputs[: len(node.output)])


def evaluate_native_operator(op_type, *inputs, output_count=1, **attributes):
    """Executes a standard operator through native kernels, without a fallback."""
    names = [f"input_{i}" if value is not None else "" for i, value in enumerate(inputs)]
    outputs = [f"output_{i}" for i in range(output_count)]
    graph = helper.make_graph(
        [
            helper.make_node(
                op_type,
                names,
                outputs,
                **{name: value for name, value in attributes.items() if value is not None},
            )
        ],
        op_type,
        [
            helper.make_tensor_value_info(
                name, helper.np_dtype_to_tensor_dtype(value.dtype), value.shape
            )
            for name, value in zip(names, inputs)
            if name
        ],
        [helper.make_tensor_value_info(name, 0, None) for name in outputs],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    return ReferenceEvaluator(model).run(
        None, {name: value for name, value in zip(names, inputs) if name}
    )

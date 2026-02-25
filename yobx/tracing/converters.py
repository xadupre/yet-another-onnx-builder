import onnx.helper as oh
import onnx.numpy_helper as onh
import torch
from .helpers import to_list


def convert_linear(node, module, inp, out, inits):
    prefix = node.target
    w_name = f"{prefix}.weight"
    inits.append(onh.from_array(module.weight.detach().cpu().numpy(), name=w_name))
    if module.bias is not None:
        b_name = f"{prefix}.bias"
        inits.append(onh.from_array(module.bias.detach().cpu().numpy(), name=b_name))
        return [
            oh.make_node(
                "Gemm", inputs=[inp[0], w_name, b_name], outputs=[out], transB=1
            )
        ]
    return [oh.make_node("Gemm", inputs=[inp[0], w_name], outputs=[out], transB=1)]


def make_activation_converter(op_name: str):
    def _convert(node, module, inp, out, inits):
        return [oh.make_node(op_name, inputs=[inp[0]], outputs=[out])]

    return _convert


def convert_dropout(node, module, inp, out, inits):
    return [oh.make_node("Identity", inputs=[inp[0]], outputs=[out])]


def convert_batchnorm(node, module, inp, out, inits):
    prefix = node.target
    names = [
        (f"{prefix}.weight", module.weight),
        (f"{prefix}.bias", module.bias),
        (f"{prefix}.running_mean", module.running_mean),
        (f"{prefix}.running_var", module.running_var),
    ]
    for name, tensor in names:
        inits.append(onh.from_array(tensor.detach().cpu().numpy(), name=name))
    return [
        oh.make_node(
            "BatchNormalization",
            inputs=[inp[0]] + [n for n, _ in names],
            outputs=[out],
            epsilon=module.eps,
            momentum=1.0 - module.momentum,
        )
    ]


def convert_layernorm(node, module, inp, out, inits):
    prefix = node.target
    inputs = [inp[0]]
    if module.weight is not None:
        w_name = f"{prefix}.weight"
        inits.append(onh.from_array(module.weight.detach().cpu().numpy(), name=w_name))
        inputs.append(w_name)
        if module.bias is not None:
            b_name = f"{prefix}.bias"
            inits.append(
                onh.from_array(module.bias.detach().cpu().numpy(), name=b_name)
            )
            inputs.append(b_name)
    return [
        oh.make_node(
            "LayerNormalization",
            inputs=inputs,
            outputs=[out],
            epsilon=module.eps,
            axis=-len(module.normalized_shape),
        )
    ]


def convert_conv(node, module, inp, out, inits):
    prefix = node.target
    ndim = module.weight.ndim - 2  # spatial dimensions
    w_name = f"{prefix}.weight"
    inits.append(onh.from_array(module.weight.detach().cpu().numpy(), name=w_name))
    inputs = [inp[0], w_name]
    if module.bias is not None:
        b_name = f"{prefix}.bias"
        inits.append(onh.from_array(module.bias.detach().cpu().numpy(), name=b_name))
        inputs.append(b_name)
    padding = to_list(module.padding, ndim)
    return [
        oh.make_node(
            "Conv",
            inputs=inputs,
            outputs=[out],
            kernel_shape=to_list(module.kernel_size, ndim),
            strides=to_list(module.stride, ndim),
            pads=padding + padding,
            group=module.groups,
            dilations=to_list(module.dilation, ndim),
        )
    ]


def make_pool_converter(op_name: str, ndim: int):
    def _convert(node, module, inp, out, inits):
        stride = module.stride
        if stride is None:
            stride = module.kernel_size
        padding = to_list(module.padding, ndim)
        kwargs = dict(
            kernel_shape=to_list(module.kernel_size, ndim),
            strides=to_list(stride, ndim),
            pads=padding + padding,
        )
        if op_name == "AveragePool":
            kwargs["count_include_pad"] = (
                1 if getattr(module, "count_include_pad", True) else 0
            )
        return [oh.make_node(op_name, inputs=[inp[0]], outputs=[out], **kwargs)]

    return _convert


def convert_flatten(node, module, inp, out, inits):
    return [
        oh.make_node("Flatten", inputs=[inp[0]], outputs=[out], axis=module.start_dim)
    ]


def build_module_converter_registry() -> dict:
    return {
        torch.nn.Linear: convert_linear,
        torch.nn.ReLU: make_activation_converter("Relu"),
        torch.nn.Sigmoid: make_activation_converter("Sigmoid"),
        torch.nn.Tanh: make_activation_converter("Tanh"),
        torch.nn.Dropout: convert_dropout,
        torch.nn.Dropout1d: convert_dropout,
        torch.nn.Dropout2d: convert_dropout,
        torch.nn.BatchNorm1d: convert_batchnorm,
        torch.nn.BatchNorm2d: convert_batchnorm,
        torch.nn.BatchNorm3d: convert_batchnorm,
        torch.nn.LayerNorm: convert_layernorm,
        torch.nn.Conv1d: convert_conv,
        torch.nn.Conv2d: convert_conv,
        torch.nn.Conv3d: convert_conv,
        torch.nn.MaxPool1d: make_pool_converter("MaxPool", 1),
        torch.nn.MaxPool2d: make_pool_converter("MaxPool", 2),
        torch.nn.AvgPool1d: make_pool_converter("AveragePool", 1),
        torch.nn.AvgPool2d: make_pool_converter("AveragePool", 2),
        torch.nn.Flatten: convert_flatten,
    }


def find_converter(submodule: "torch.nn.Module", registry: dict):
    """Look up a converter for *submodule* by walking its MRO."""
    for cls in type(submodule).__mro__:
        if cls in registry:
            return registry[cls]
    return None

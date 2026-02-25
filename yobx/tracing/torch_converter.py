"""
Converter module for transforming PyTorch models to ONNX format.

The model is treated as a pipeline: the root ``nn.Module`` is the
pipeline entry-point and its leaf sub-modules (those with no children)
are the individual pipeline steps.  Conversion uses ``torch.fx``
symbolic tracing to decompose the pipeline into its constituent
operations and maps each to the corresponding ONNX operator.
"""

import onnx
import onnx.helper as oh
import torch
from torch.fx.passes.shape_prop import ShapeProp
from .helpers import torch_dtype_to_onnx, make_value_info
from .converters import build_module_converter_registry, find_converter


def fx_graph_to_onnx(
    graph_module: "torch.fx.GraphModule",
    args: tuple,
    *,
    input_names: list[str] | None,
    output_names: list[str] | None,
    dynamic_axes: dict | None,
    opset_version: int,
) -> onnx.ModelProto:
    """Translate an FX ``GraphModule`` into an ``onnx.ModelProto``."""
    ShapeProp(graph_module).propagate(*args)

    registry = build_module_converter_registry()
    onnx_nodes: list = []
    initializers: list = []
    graph_inputs: list = []
    graph_outputs: list = []
    # Maps FX node name → ONNX tensor name used in the graph.
    name_map: dict[str, str] = {}
    placeholder_idx = 0

    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            tensor_name = (
                input_names[placeholder_idx]
                if input_names and placeholder_idx < len(input_names)
                else node.name
            )
            name_map[node.name] = tensor_name
            meta = node.meta.get("tensor_meta")
            onnx_dtype = (
                torch_dtype_to_onnx(meta.dtype) if meta else onnx.TensorProto.FLOAT
            )
            shape = list(meta.shape) if meta else None
            dynamic_dims = (dynamic_axes or {}).get(tensor_name)
            graph_inputs.append(
                make_value_info(tensor_name, onnx_dtype, shape, dynamic_dims)
            )
            placeholder_idx += 1

        elif node.op == "call_module":
            submodule = graph_module.get_submodule(node.target)
            inp = [name_map[a.name] for a in node.args if hasattr(a, "name")]
            out = node.name
            converter = find_converter(submodule, registry)
            if converter is None:
                raise NotImplementedError(
                    f"No ONNX converter registered for module type "
                    f"{type(submodule).__name__!r}. "
                    "Extend the registry to add support."
                )
            onnx_nodes.extend(converter(node, submodule, inp, out, initializers))
            name_map[node.name] = out

        elif node.op == "output":
            out_args = node.args[0]
            out_nodes = (
                list(out_args) if isinstance(out_args, (list, tuple)) else [out_args]
            )
            for idx, out_node in enumerate(out_nodes):
                src_name = name_map[out_node.name]
                final_name = (
                    output_names[idx]
                    if output_names and idx < len(output_names)
                    else src_name
                )
                if final_name != src_name:
                    onnx_nodes.append(
                        oh.make_node(
                            "Identity", inputs=[src_name], outputs=[final_name]
                        )
                    )
                meta = out_node.meta.get("tensor_meta")
                onnx_dtype = (
                    torch_dtype_to_onnx(meta.dtype) if meta else onnx.TensorProto.FLOAT
                )
                shape = list(meta.shape) if meta else None
                graph_outputs.append(make_value_info(final_name, onnx_dtype, shape))

    graph = oh.make_graph(
        onnx_nodes,
        "pipeline",
        graph_inputs,
        graph_outputs,
        initializer=initializers,
    )
    return oh.make_model(
        graph,
        opset_imports=[oh.make_opsetid("", opset_version)],
    )


def get_leaf_modules(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    """Returns the leaf sub-modules of *model* (those with no children).

    Leaf modules are the final steps of the pipeline.  A module with
    no registered sub-modules is considered a leaf.

    :param model: Root ``torch.nn.Module`` to inspect.
    :returns: Ordered dict mapping qualified name to leaf module.

    Example::

        import torch.nn as nn
        from onnx_pipe import get_leaf_modules

        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        leaves = get_leaf_modules(model)
        # {'0': Linear(...), '1': ReLU(), '2': Linear(...)}
    """
    leaves: dict[str, torch.nn.Module] = {}
    for name, module in model.named_modules():
        if name and not list(module.children()):
            leaves[name] = module
    # If the root has no children at all, it is itself the only leaf.
    if not leaves:
        leaves[""] = model
    return leaves


def convert_torch_to_onnx(
    model: torch.nn.Module,
    args: tuple,
    *,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict | None = None,
    opset_version: int = 17,
    output_path: str | None = None,
) -> onnx.ModelProto:
    """
    Converts a PyTorch ``nn.Module`` to ONNX format.

    The model is treated as a **pipeline**: the root module is the
    overall pipeline and its leaf sub-modules are the individual
    processing steps.  Conversion is performed by symbolically tracing
    the model with ``torch.fx`` and translating each operation in the
    resulting graph to the corresponding ONNX operator.

    :param model: PyTorch model (``torch.nn.Module``) to convert.
        The model is temporarily set to evaluation mode during tracing
        and restored to its original training mode afterwards.
    :param args: Tuple of example inputs used for shape inference.
        The shapes and dtypes must match those expected by the model's
        ``forward`` method.
    :param input_names: Optional list of names to assign to the model's
        input nodes in the ONNX graph.
    :param output_names: Optional list of names to assign to the
        model's output nodes in the ONNX graph.
    :param dynamic_axes: Optional dict specifying dynamic (variable-
        length) axes, e.g. ``{"input": {0: "batch_size"}}``.
    :param opset_version: ONNX opset version to target (default 17).
    :param output_path: Optional file path to save the ONNX model.
        If *None*, the model is only returned in memory.
    :returns: The exported ``onnx.ModelProto``.
    :raises TypeError: If *model* is not a ``torch.nn.Module``.
    :raises ValueError: If *args* is empty.

    Example::

        import torch
        import torch.nn as nn
        from onnx_pipe import convert_torch_to_onnx

        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        dummy_input = torch.zeros(1, 4)
        onnx_model = convert_torch_to_onnx(model, (dummy_input,))
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"model must be a torch.nn.Module, got {type(model).__name__!r}"
        )
    if not args:
        raise ValueError("args must be a non-empty tuple of example inputs")

    training = model.training
    model.eval()
    try:
        graph_module = torch.fx.symbolic_trace(model)
        onnx_model = fx_graph_to_onnx(
            graph_module,
            args,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
        )
    finally:
        model.train(training)

    onnx.checker.check_model(onnx_model)

    if output_path is not None:
        onnx.save(onnx_model, output_path)

    return onnx_model

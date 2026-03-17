import re
from typing import Dict
import numpy as np
import onnx
import onnx.numpy_helper as onh
from ..reference import ExtendedReferenceEvaluator as Inference
from ..helpers.onnx_helper import onnx_dtype_name, pretty_onnx, get_hidden_inputs


def _mermaid_id(prefix: str, index: int) -> str:
    return f"{prefix}{index}"


def _mermaid_safe(text: str) -> str:
    """Escape characters that break Mermaid node labels (double quotes → single quotes)."""
    return text.replace('"', "'")


def _make_node_label_mermaid(node: onnx.NodeProto, tiny_inits: Dict[str, str]) -> str:
    els = [f"{node.domain}.{node.op_type}" if node.domain else node.op_type, "("]
    ee = [tiny_inits.get(i, ".") if i else "" for i in node.input]
    for att in node.attribute:
        if att.name == "to":
            ee.append(f"{att.name}={onnx_dtype_name(att.i)}")
        elif att.name in {"axis", "value_int", "stash_type", "start", "end"}:
            ee.append(f"{att.name}={att.i}")
        elif att.name in {"value_float"}:
            ee.append(f"{att.name}={att.f}")
        elif att.name in {"value_floats"}:
            ee.append(f"{att.name}={att.floats}")
        elif att.name in {"value_ints", "perm"}:
            ee.append(f"{att.name}={att.ints}")
    els.append(", ".join(ee))
    els.append(")")
    if node.op_type == "Constant":
        els.extend([" -> ", node.output[0]])
    return _mermaid_safe("".join(els))


def _make_edge_label_mermaid(value_info: onnx.ValueInfoProto) -> str:
    itype = value_info.type.tensor_type.elem_type
    if itype == onnx.TensorProto.UNDEFINED:
        return ""
    shape = tuple(
        d.dim_param if d.dim_param else d.dim_value for d in value_info.type.tensor_type.shape.dim
    )
    res = [
        str(a)
        for a in [("?" if isinstance(s, str) and s.startswith("unk") else s) for s in shape]
    ]
    return f"{onnx_dtype_name(itype)}({','.join(res)})"


_MAX_INLINE_CONSTANT_SIZE = 10


def to_mermaid(model: onnx.ModelProto) -> str:
    """
    Converts an ONNX model into a `Mermaid <https://mermaid.js.org/>`_ flowchart string.

    The returned string can be embedded directly in a Markdown fenced code block
    with the language tag ``mermaid``, or rendered with the Mermaid CLI / live editor.

    Here is a small example::

        import numpy as np
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh
        from yobx.translate.mermaid_helper import to_mermaid

        TFLOAT = onnx.TensorProto.FLOAT
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "Y"], ["added"]),
                    oh.make_node("MatMul", ["added", "W"], ["mm"]),
                    oh.make_node("Relu", ["mm"], ["Z"]),
                ],
                "add_matmul_relu",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq", 4]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["batch", "seq", 4]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, ["batch", "seq", 2])],
                [
                    onh.from_array(
                        np.zeros((4, 2), dtype=np.float32),
                        name="W",
                    )
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        print(to_mermaid(model))

    Node colour legend:

    * **green** (``classDef input``) – graph inputs
    * **yellow** (``classDef init``) – initializers / constant weights
    * **light-grey** (``classDef op``) – ONNX operators
    * **light-blue** (``classDef output``) – graph outputs
    """
    _unique: Dict[int, int] = {}

    def _mkn(obj: object) -> int:
        id_obj = id(obj)
        if id_obj in _unique:
            return _unique[id_obj]
        i = len(_unique)
        _unique[id_obj] = i
        return i

    builder = None
    try:
        from ..xshape import BasicShapeBuilder

        builder = BasicShapeBuilder()
        builder.run_model(model)
    except Exception:
        builder = None

    edge_label: Dict[str, str] = {}
    if builder is not None:
        for node in model.graph.node:
            for name in node.output:
                if name and builder.has_type(name) and builder.has_shape(name):
                    itype = builder.get_type(name)
                    if itype == onnx.TensorProto.UNDEFINED:
                        continue
                    shape = builder.get_shape(name)
                    res = [
                        str(a)
                        for a in [
                            ("?" if isinstance(s, str) and s.startswith("unk") else s)
                            for s in shape
                        ]
                    ]
                    edge_label[name] = f"{onnx_dtype_name(itype)}({','.join(res)})"

    rows = ["flowchart TD"]

    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    nodes = list(model.graph.node)
    inits = list(model.graph.initializer)
    tiny_inits: Dict[str, str] = {}
    name_to_ids: Dict[str, str] = {}

    for inp in inputs:
        if not inp.name:
            continue
        lab = _make_edge_label_mermaid(inp)
        node_id = _mermaid_id("I_", _mkn(inp))
        display = _mermaid_safe(f"{inp.name}\\n{lab}" if lab else inp.name)
        rows.append(f'    {node_id}["{display}"]:::input')
        name_to_ids[inp.name] = node_id
        edge_label[inp.name] = lab

    # Small constant --> initializer
    output_names = {n.name for n in outputs}
    for node in nodes:
        if node.op_type != "Constant" or node.output[0] in output_names:
            continue
        skip = False
        for att in node.attribute:
            if att.name == "value" and (
                len(att.t.dims) > 1 or np.prod(tuple(att.t.dims)) > _MAX_INLINE_CONSTANT_SIZE
            ):
                skip = True
                break
        if skip:
            continue

        sess = Inference(node)
        value = sess.run(None, {})[0]  # type: ignore
        inits.append(onh.from_array(value, name=node.output[0]))

    for init in inits:
        if init.name in name_to_ids:
            # hide optional inputs already listed as graph inputs
            continue
        shape = tuple(init.dims)
        if len(shape) == 0 or (len(shape) == 1 and shape[0] < _MAX_INLINE_CONSTANT_SIZE):  # type: ignore[operator]
            a = onh.to_array(init)
            tiny_inits[init.name] = (
                str(a) if len(shape) == 0 else f"[{', '.join([str(i) for i in a])}]"
            )
        else:
            ls = f"{onnx_dtype_name(init.data_type)}({', '.join(map(str, shape))})"
            node_id = _mermaid_id("i_", _mkn(init))
            display = _mermaid_safe(f"{init.name}\\n{ls}")
            rows.append(f'    {node_id}["{display}"]:::init')
            name_to_ids[init.name] = node_id
            edge_label[init.name] = ls

    for node in nodes:
        if node.op_type == "Constant" and node.output[0] in tiny_inits:
            continue
        node_id = f"{re.sub(r'[^A-Za-z0-9_]', '_', node.op_type)}_{_mkn(node)}"
        label = _make_node_label_mermaid(node, tiny_inits)
        rows.append(f'    {node_id}["{label}"]:::op')
        name_to_ids.update({o: node_id for o in node.output if o})

    # edges
    done = set()
    for node in nodes:
        node_id = f"{re.sub(r'[^A-Za-z0-9_]', '_', node.op_type)}_{_mkn(node)}"
        for i in node.input:
            if not i or i in tiny_inits:
                continue
            if i not in name_to_ids:
                raise ValueError(f"Unable to find {i!r}\n{pretty_onnx(model)}")
            src = name_to_ids[i]
            edge = (src, node_id)
            if edge in done:
                continue
            done.add(edge)
            lab = edge_label.get(i, "")
            if lab:
                rows.append(f'    {src} -->|"{_mermaid_safe(lab)}"| {node_id}')
            else:
                rows.append(f"    {src} --> {node_id}")
        if node.op_type in {"Scan", "Loop", "If"}:
            unique = set()
            for att in node.attribute:
                if att.type == onnx.AttributeProto.GRAPH:
                    unique |= get_hidden_inputs(att.g)
            for i in unique:
                if i in tiny_inits:
                    continue
                src = name_to_ids[i]
                edge = (src, node_id)
                if edge in done:
                    continue
                done.add(edge)
                rows.append(f"    {src} -.-> {node_id}")

    # outputs
    for out in outputs:
        if not out.name:
            continue
        lab = _make_edge_label_mermaid(out)
        node_id = _mermaid_id("O_", _mkn(out))
        display = _mermaid_safe(f"{out.name}\\n{lab}" if lab else out.name)
        rows.append(f'    {node_id}["{display}"]:::output')
        src = name_to_ids[out.name]
        rows.append(f"    {src} --> {node_id}")

    rows.extend(
        [
            "    classDef input fill:#aaeeaa",
            "    classDef init fill:#cccc00",
            "    classDef op fill:#cccccc",
            "    classDef output fill:#aaaaee",
        ]
    )

    return "\n".join(rows)

import re
import subprocess
from typing import Dict
import numpy as np
import onnx
import onnx.numpy_helper as onh
from ..reference import ExtendedReferenceEvaluator as Inference
from .onnx_helper import onnx_dtype_name, pretty_onnx, get_hidden_inputs


def _make_node_label(node: onnx.NodeProto, tiny_inits: Dict[str, str]) -> str:
    els = [f"{node.domain}.\\n{node.op_type}" if node.domain else node.op_type, "\\n("]
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
    res = "".join(els)
    if len(res) < 40:
        return res.replace("\\n(", "(")
    return res


def _make_edge_label(value_info: onnx.ValueInfoProto, multi_line: bool = False) -> str:
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
    sshape = ",".join(res)
    if multi_line and len(sshape) > 30:
        sshape = ",\\n".join(res)
    return f"{onnx_dtype_name(itype)}({sshape})"


def to_dot(model: onnx.ModelProto) -> str:
    """
    Converts a model into a dot graph.
    Here is an example:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(batch, seq, 4)"])
            I_Y(["Y FLOAT(batch, seq, 4)"])
            i_W["W FLOAT(4, 2)"]

            Add_0[["Add(., .)"]]
            MatMul_1[["MatMul(., .)"]]
            Relu_2[["Relu(.)"]]

            I_X -->|"FLOAT(batch, seq, 4)"| Add_0
            I_Y -->|"FLOAT(batch, seq, 4)"| Add_0
            Add_0 -->|"FLOAT(batch, seq, 4)"| MatMul_1
            i_W -->|"FLOAT(4, 2)"| MatMul_1
            MatMul_1 -->|"FLOAT(batch, seq, 2)"| Relu_2

            O_Z(["Z FLOAT(batch, seq, 2)"])
            Relu_2 --> O_Z

            class I_X,I_Y,O_Z ioNode
            class i_W initNode
            class Add_0,MatMul_1,Relu_2 opNode
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

    op_type_colors = {
        "Shape": "#d2a81f",
        "MatMul": "#ee9999",
        "Transpose": "#ee99ee",
        "Reshape": "#eeeeee",
        "Squeeze": "#eeeeee",
        "Unsqueeze": "#eeeeee",
    }

    edge_label = {}
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
                    sshape = ",".join(res)
                    if len(sshape) > 30:
                        sshape = ",\\n".join(res)
                    edge_label[name] = f"{onnx_dtype_name(itype)}({sshape})"

    rows = [
        "digraph {",
        (
            "  graph [rankdir=TB, splines=true, overlap=false, nodesep=0.2, "
            "ranksep=0.2, fontsize=8];"
        ),
        '  node [style="rounded,filled", color="#888888", fontcolor="#222222", shape=box];',
        "  edge [arrowhead=vee, fontsize=7, labeldistance=-5, labelangle=0];",
    ]
    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    nodes = list(model.graph.node)
    inits = list(model.graph.initializer)
    tiny_inits = {}
    name_to_ids = {}

    for inp in inputs:
        if not inp.name:
            continue
        lab = _make_edge_label(inp)
        rows.append(f'  I_{_mkn(inp)} [label="{inp.name}\\n{lab}", fillcolor="#aaeeaa"];')
        name_to_ids[inp.name] = f"I_{_mkn(inp)}"
        edge_label[inp.name] = _make_edge_label(inp, multi_line=True)

    # Small constant --> initializer
    output_names = {n.name for n in outputs}
    for node in nodes:
        if node.op_type != "Constant" or node.output[0] in output_names:
            continue
        skip = False
        for att in node.attribute:
            if att.name == "value" and (len(att.t.dims) > 1 or np.prod(tuple(att.t.dims)) > 10):
                skip = True
                break
        if skip:
            continue

        sess = Inference(node)
        value = sess.run(None, {})[0]  # type: ignore
        inits.append(onh.from_array(value, name=node.output[0]))

    for init in inits:
        if init.name in name_to_ids:
            # hide optional inputs
            continue
        shape = tuple(init.dims)
        if len(shape) == 0 or (len(shape) == 1 and shape[0] < 10):  # type: ignore[operator]
            a = onh.to_array(init)
            tiny_inits[init.name] = (
                str(a) if len(shape) == 0 else f"[{', '.join([str(i) for i in a])}]"
            )
        else:
            ls = f"{onnx_dtype_name(init.data_type)}({', '.join(map(str,shape))})"
            rows.append(f'  i_{_mkn(init)} [label="{init.name}\\n{ls}", fillcolor="#cccc00"];')
            name_to_ids[init.name] = f"i_{_mkn(init)}"
            edge_label[init.name] = ls

    for node in nodes:
        if node.op_type == "Constant" and node.output[0] in tiny_inits:
            continue
        color = op_type_colors.get(node.op_type, "#cccccc")
        label = _make_node_label(node, tiny_inits)
        rows.append(f'  {node.op_type}_{_mkn(node)} [label="{label}", fillcolor="{color}"];')
        name_to_ids.update({o: f"{node.op_type}_{_mkn(node)}" for o in node.output if o})

    # nodes
    done = set()
    for node in nodes:
        names = list(node.input)
        for i in names:
            if not i or i in tiny_inits:
                continue
            if i not in name_to_ids:
                raise ValueError(f"Unable to find {i!r}\n{pretty_onnx(model)}")
            edge = name_to_ids[i], f"{node.op_type}_{_mkn(node)}"
            if edge in done:
                continue
            done.add(edge)
            lab = edge_label.get(i, "")
            if lab:
                ls = ",".join([f'label="{lab}"'])
                lab = f" [{ls}]"
            rows.append(f"  {edge[0]} -> {edge[1]}{lab};")
        if node.op_type in {"Scan", "Loop", "If"}:
            unique = set()
            for att in node.attribute:
                if att.type == onnx.AttributeProto.GRAPH:
                    unique |= get_hidden_inputs(att.g)
            for i in unique:
                if i in tiny_inits:
                    continue
                edge = name_to_ids[i], f"{node.op_type}_{_mkn(node)}"
                if edge in done:
                    continue
                done.add(edge)
                rows.append(f"  {edge[0]} -> {edge[1]} [style=dotted];")

    # outputs
    for out in outputs:
        if not out.name:
            continue
        lab = _make_edge_label(out)
        rows.append(f'  O_{_mkn(out)} [label="{out.name}\\n{lab}", fillcolor="#aaaaee"];')
        edge = name_to_ids[out.name], f"O_{_mkn(out)}"
        rows.append(f"  {edge[0]} -> {edge[1]};")

    rows.append("}")
    return "\n".join(rows)


def _mermaid_id(name: str, prefix: str = "") -> str:
    """Returns a valid mermaid node identifier derived from *name*."""
    safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not safe or safe[0].isdigit():
        safe = "_" + safe
    return f"{prefix}{safe}" if prefix else safe


def _make_mermaid_node_label(node: onnx.NodeProto, tiny_inits: Dict[str, str]) -> str:
    """Builds the human-readable label shown inside a mermaid operation node."""
    parts = [f"{node.domain}.{node.op_type}" if node.domain else node.op_type, "("]
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
            ee.append(f"{att.name}={list(att.ints)}")
    parts.append(", ".join(ee))
    parts.append(")")
    if node.op_type == "Constant":
        parts.extend([" -> ", node.output[0]])
    res = "".join(parts)
    # Escape characters that break mermaid label parsing
    res = res.replace('"', "#quot;").replace("<", "#lt;").replace(">", "#gt;")
    return res


def _make_mermaid_edge_label(value_info: onnx.ValueInfoProto) -> str:
    """Builds the edge label string from a *ValueInfoProto* for mermaid."""
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
    sshape = ", ".join(res)
    return f"{onnx_dtype_name(itype)}({sshape})"


def to_mermaid(model: onnx.ModelProto) -> str:
    """
    Converts a model into a `Mermaid <https://mermaid.js.org/>`_ graph string.

    The output can be embedded in a Sphinx ``.. mermaid::`` directive or any
    Markdown renderer that supports Mermaid.

    :param model: ONNX model to visualise
    :return: Mermaid ``graph TD`` source string

    Color convention (matching :func:`to_dot`):

    * **green** (``#dfd``) – graph inputs and outputs
    * **yellow** (``#cccc00``) – large initializers / weight tensors
    * **pink** (``#f9f``) – small inline constants (shown as rectangle nodes)
    * **blue** (``#bbf``) – all other operator nodes
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
                    edge_label[name] = f"{onnx_dtype_name(itype)}({', '.join(res)})"

    rows = [
        "graph TD",
        "",
        "    classDef ioNode fill:#dfd,stroke:#333,color:#333",
        "    classDef initNode fill:#cccc00,stroke:#333,color:#333",
        "    classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333",
        "    classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333",
        "",
    ]

    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    nodes = list(model.graph.node)
    inits = list(model.graph.initializer)
    tiny_inits: Dict[str, str] = {}
    name_to_ids: Dict[str, str] = {}

    # Input nodes
    for inp in inputs:
        if not inp.name:
            continue
        lab = _make_mermaid_edge_label(inp)
        node_id = _mermaid_id(inp.name, "I_")
        label_text = f"{inp.name}"
        if lab:
            label_text += f" {lab}"
        rows.append(f'    {node_id}(["{label_text}"])')
        name_to_ids[inp.name] = node_id
        if lab:
            edge_label[inp.name] = lab

    # Inline small constants (Constant nodes whose output is small)
    output_names = {n.name for n in outputs}
    for node in nodes:
        if node.op_type != "Constant" or node.output[0] in output_names:
            continue
        skip = False
        for att in node.attribute:
            if att.name == "value" and (len(att.t.dims) > 1 or np.prod(tuple(att.t.dims)) > 10):
                skip = True
                break
        if skip:
            continue

        sess = Inference(node)
        value = sess.run(None, {})[0]  # type: ignore
        inits.append(onh.from_array(value, name=node.output[0]))

    # Initializer nodes
    for init in inits:
        if init.name in name_to_ids:
            continue
        shape = tuple(init.dims)
        if len(shape) == 0 or (len(shape) == 1 and shape[0] < 10):  # type: ignore[operator]
            a = onh.to_array(init)
            tiny_inits[init.name] = (
                str(a) if len(shape) == 0 else f"[{', '.join([str(i) for i in a])}]"
            )
        else:
            ls = f"{onnx_dtype_name(init.data_type)}({', '.join(map(str, shape))})"
            node_id = _mermaid_id(init.name, "i_")
            rows.append(f'    {node_id}["{init.name} {ls}"]')
            name_to_ids[init.name] = node_id
            edge_label[init.name] = ls

    rows.append("")

    # Operator nodes
    for node in nodes:
        if node.op_type == "Constant" and node.output[0] in tiny_inits:
            continue
        node_id = f"{_mermaid_id(node.op_type)}_{_mkn(node)}"
        label = _make_mermaid_node_label(node, tiny_inits)
        rows.append(f'    {node_id}[["{label}"]]')
        name_to_ids.update({o: node_id for o in node.output if o})

    rows.append("")

    # Edges: inputs → operator
    done = set()
    for node in nodes:
        if node.op_type == "Constant" and node.output[0] in tiny_inits:
            continue
        node_id = f"{_mermaid_id(node.op_type)}_{_mkn(node)}"
        for i in node.input:
            if not i or i in tiny_inits:
                continue
            if i not in name_to_ids:
                raise ValueError(f"Unable to find {i!r}\n{pretty_onnx(model)}")
            src = name_to_ids[i]
            edge = (src, node_id, i)
            if edge in done:
                continue
            done.add(edge)
            lab = edge_label.get(i, "")
            if lab:
                rows.append(f"    {src} -->|\"{lab}\"| {node_id}")
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
                edge = (src, node_id, i)
                if edge in done:
                    continue
                done.add(edge)
                rows.append(f"    {src} -.-> {node_id}")

    rows.append("")

    # Output nodes
    for out in outputs:
        if not out.name:
            continue
        lab = _make_mermaid_edge_label(out)
        node_id = _mermaid_id(out.name, "O_")
        label_text = f"{out.name}"
        if lab:
            label_text += f" {lab}"
        rows.append(f'    {node_id}(["{label_text}"])')
        src = name_to_ids[out.name]
        rows.append(f"    {src} --> {node_id}")

    rows.append("")

    # Class assignments
    io_ids = [_mermaid_id(inp.name, "I_") for inp in inputs if inp.name]
    io_ids += [_mermaid_id(out.name, "O_") for out in outputs if out.name]
    if io_ids:
        rows.append(f"    class {','.join(io_ids)} ioNode")

    init_ids = []
    for init in model.graph.initializer:
        if init.name not in tiny_inits and init.name not in {inp.name for inp in inputs}:
            init_ids.append(_mermaid_id(init.name, "i_"))
    if init_ids:
        rows.append(f"    class {','.join(init_ids)} initNode")

    const_ids = []
    for node in nodes:
        if node.op_type == "Constant" and node.output[0] not in tiny_inits:
            const_ids.append(f"{_mermaid_id(node.op_type)}_{_mkn(node)}")
    if const_ids:
        rows.append(f"    class {','.join(const_ids)} constNode")

    op_ids = []
    for node in nodes:
        if node.op_type == "Constant" and node.output[0] in tiny_inits:
            continue
        node_id = f"{_mermaid_id(node.op_type)}_{_mkn(node)}"
        if node_id not in const_ids:
            op_ids.append(node_id)
    if op_ids:
        rows.append(f"    class {','.join(op_ids)} opNode")

    return "\n".join(rows)


def to_svg(dot: str) -> str:
    """
    Converts a DOT string into an SVG string by calling the *dot* command-line tool.

    :param dot: DOT graph source, e.g. as returned by :func:`to_dot`
    :return: SVG content as a UTF-8 string
    :raises FileNotFoundError: if the *dot* executable is not found on ``PATH``
    :raises RuntimeError: if *dot* exits with a non-zero return code
    """
    try:
        proc = subprocess.run(["dot", "-Tsvg"], input=dot.encode(), capture_output=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "The dot executable was not found. "
            "Please install Graphviz and ensure it is on your PATH."
        ) from e
    if proc.returncode != 0:
        raise RuntimeError(
            f"dot exited with return code {proc.returncode}:\n{proc.stderr.decode()}"
        )
    return proc.stdout.decode()

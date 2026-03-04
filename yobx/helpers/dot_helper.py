from __future__ import annotations
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

    .. gdot::
        :script: DOT-SECTION

        import numpy as np
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh
        from yobx.helpers.dot_helper import to_dot

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
        dot = to_dot(model)
        print("DOT-SECTION", dot)
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
        value = sess.run(None, {})[0]
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


def to_svg(dot: str) -> str:
    """
    Converts a DOT string into an SVG string by calling the *dot* command-line tool.

    :param dot: DOT graph source, e.g. as returned by :func:`to_dot`
    :return: SVG content as a UTF-8 string
    :raises FileNotFoundError: if the *dot* executable is not found on ``PATH``
    :raises RuntimeError: if *dot* exits with a non-zero return code
    """
    try:
        proc = subprocess.run(
            ["dot", "-Tsvg"],
            input=dot.encode(),
            capture_output=True,
        )
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

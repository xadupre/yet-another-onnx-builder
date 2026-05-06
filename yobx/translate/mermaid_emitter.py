import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import onnx

from ..helpers.onnx_helper import get_hidden_inputs, onnx_dtype_name
from .base_emitter import BaseEmitter

_MAX_INLINE_CONSTANT_SIZE = 10


def _mermaid_safe(text: str) -> str:
    """Escape characters that break Mermaid node labels (double quotes → single quotes)."""
    return text.replace('"', "'")


def _shape_label(elem_type: int, shape: Tuple) -> str:
    """Returns a ``'DTYPE(dim,dim,...)'`` label string, or an empty string when undefined."""
    if elem_type == onnx.TensorProto.UNDEFINED:
        return ""
    res = [str("?" if isinstance(s, str) and s.startswith("unk") else s) for s in shape]
    return f"{onnx_dtype_name(elem_type)}({','.join(res)})"


class MermaidEmitter(BaseEmitter):
    """
    Emitter that converts an ONNX graph into a
    `Mermaid <https://mermaid.js.org/>`_ ``flowchart TD`` string.

    Intended to be used with :class:`~yobx.translate.translator.Translator`::

        from yobx.translate.mermaid_helper import MermaidEmitter
        from yobx.translate.translator import Translator

        emitter = MermaidEmitter()
        tr = Translator(model, emitter=emitter)
        print(tr.export(as_str=True))

    Node colour legend:

    * **green** (``classDef input``) – graph inputs
    * **yellow** (``classDef init``) – initializers / constant weights
    * **light-grey** (``classDef op``) – ONNX operators
    * **light-blue** (``classDef output``) – graph outputs
    """

    def __init__(self, edge_labels: Optional[Dict[str, str]] = None):
        """
        :param edge_labels: optional pre-computed edge labels mapping
            tensor name to ``"DTYPE(shape)"`` string (e.g. from
            ``BasicShapeBuilder``).  When *None*, labels are derived
            from input/output type annotations only.
        """
        self._counter: int = 0
        # Accumulated state from events (main graph only)
        self._inputs: List[Tuple[str, int, Tuple]] = []
        self._initializers: List[Tuple[str, onnx.TensorProto, np.ndarray]] = []
        self._nodes: List[Tuple[str, List, List, str, Dict]] = []
        self._outputs: List[Tuple[str, int, Tuple]] = []
        # Pre-supplied shape/type labels (tensor name → "DTYPE(shape)")
        self._edge_label: Dict[str, str] = dict(edge_labels) if edge_labels else {}
        # Guard against accumulating function-body events
        self._in_function: bool = False

    # ------------------------------------------------------------------
    # BaseEmitter interface
    # ------------------------------------------------------------------

    def join(self, rows: List[str], single_line: bool = False) -> str:
        """Join Mermaid lines into a single string."""
        return "\n".join(rows)

    def _emit_start(self, **kwargs: Dict[str, Any]) -> List[str]:
        return []

    def _emit_begin_graph(self, **kwargs: Dict[str, Any]) -> List[str]:
        return []

    def _emit_end_graph(self, **kwargs: Dict[str, Any]) -> List[str]:
        return []

    def _emit_initializer(self, **kwargs: Dict[str, Any]) -> List[str]:
        if self._in_function:
            return []
        self._initializers.append((kwargs["name"], kwargs["init"], kwargs["value"]))
        return []

    def _emit_input(self, **kwargs: Dict[str, Any]) -> List[str]:
        if self._in_function:
            return []
        name = kwargs["name"]
        elem_type = kwargs.get("elem_type", onnx.TensorProto.UNDEFINED)
        shape = kwargs.get("shape", ())
        self._inputs.append((name, elem_type, shape))
        lab = _shape_label(elem_type, shape)
        if lab:
            self._edge_label.setdefault(name, lab)
        return []

    def _emit_output(self, **kwargs: Dict[str, Any]) -> List[str]:
        if self._in_function:
            return []
        name = kwargs["name"]
        elem_type = kwargs.get("elem_type", onnx.TensorProto.UNDEFINED)
        shape = kwargs.get("shape", ())
        self._outputs.append((name, elem_type, shape))
        return []

    def _emit_node(self, **kwargs: Dict[str, Any]) -> List[str]:
        if self._in_function:
            return []
        self._nodes.append(
            (
                kwargs["op_type"],
                list(kwargs["inputs"]),
                list(kwargs["outputs"]),
                kwargs.get("domain", ""),
                kwargs.get("atts", {}),
            )
        )
        return []

    def _emit_sparse_initializer(self, **kwargs: Dict[str, Any]) -> List[str]:
        return []

    def _emit_to_onnx_model(self, **kwargs: Dict[str, Any]) -> List[str]:
        return self._build_mermaid()

    def _emit_to_onnx_function(self, **kwargs: Dict[str, Any]) -> List[str]:
        return []

    def _emit_begin_function(self, **kwargs: Dict[str, Any]) -> List[str]:
        self._in_function = True
        return []

    def _emit_end_function(self, **kwargs: Dict[str, Any]) -> List[str]:
        self._in_function = False
        return []

    def _emit_function_input(self, **kwargs: Dict[str, Any]) -> List[str]:
        return []

    def _emit_function_output(self, **kwargs: Dict[str, Any]) -> List[str]:
        return []

    def _emit_function_attributes(self, **kwargs: Dict[str, Any]) -> List[str]:
        return []

    # ------------------------------------------------------------------
    # Core Mermaid assembly
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        i = self._counter
        self._counter += 1
        return i

    def _build_mermaid(self) -> List[str]:
        """Assembles all accumulated events into Mermaid flowchart lines."""
        rows: List[str] = ["flowchart TD"]
        name_to_id: Dict[str, str] = {}
        tiny_inits: Dict[str, str] = {}
        output_names: Set[str] = {name for name, _, _ in self._outputs}

        # ---- inputs ----
        for name, elem_type, shape in self._inputs:
            if not name:
                continue
            lab = self._edge_label.get(name) or _shape_label(elem_type, shape)
            node_id = f"I_{self._next_id()}"
            display = _mermaid_safe(f"{name}\\n{lab}" if lab else name)
            rows.append(f'    {node_id}["{display}"]:::input')
            name_to_id[name] = node_id

        # ---- fold small Constant nodes into tiny_inits ----
        for op_type, _inputs, outputs, _domain, atts in self._nodes:
            if op_type != "Constant":
                continue
            if outputs and outputs[0] in output_names:
                continue
            val = self._extract_constant_value(atts)
            if val is None or val.ndim > 1 or val.size > _MAX_INLINE_CONSTANT_SIZE:
                continue
            tiny_str = (
                str(val.flat[0]) if val.ndim == 0 else f"[{', '.join(str(x) for x in val.flat)}]"
            )
            if outputs:
                tiny_inits[outputs[0]] = tiny_str

        # ---- initializer nodes ----
        for name, init_proto, value in self._initializers:
            if name in name_to_id:
                # already listed as a graph input
                continue
            shape = tuple(init_proto.dims)
            if len(shape) == 0 or (len(shape) == 1 and shape[0] < _MAX_INLINE_CONSTANT_SIZE):
                tiny_inits[name] = (
                    str(value.flat[0])
                    if len(shape) == 0
                    else f"[{', '.join(str(x) for x in value.flat)}]"
                )
            else:
                ls = f"{onnx_dtype_name(init_proto.data_type)}({', '.join(map(str, shape))})"
                node_id = f"i_{self._next_id()}"
                display = _mermaid_safe(f"{name}\\n{ls}")
                rows.append(f'    {node_id}["{display}"]:::init')
                name_to_id[name] = node_id
                self._edge_label.setdefault(name, ls)

        # ---- operator nodes ----
        node_ids: Dict[int, str] = {}
        for idx, (op_type, _inputs, outputs, domain, atts) in enumerate(self._nodes):
            if op_type == "Constant" and outputs and outputs[0] in tiny_inits:
                continue
            safe_op = re.sub(r"[^A-Za-z0-9_]", "_", op_type)
            node_id = f"{safe_op}_{self._next_id()}"
            node_ids[idx] = node_id
            label = self._make_op_label(op_type, domain, _inputs, outputs, atts, tiny_inits)
            rows.append(f'    {node_id}["{label}"]:::op')
            for o in outputs:
                if o:
                    name_to_id[o] = node_id

        # ---- edges ----
        done: Set[Tuple[str, str]] = set()
        for idx, (op_type, inputs, _outputs, _domain, atts) in enumerate(self._nodes):
            if idx not in node_ids:
                continue
            node_id = node_ids[idx]
            for inp in inputs:
                if not inp or inp in tiny_inits:
                    continue
                if inp not in name_to_id:
                    continue
                src = name_to_id[inp]
                edge = (src, node_id)
                if edge in done:
                    continue
                done.add(edge)
                lab = self._edge_label.get(inp, "")
                if lab:
                    rows.append(f'    {src} -->|"{_mermaid_safe(lab)}"| {node_id}')
                else:
                    rows.append(f"    {src} --> {node_id}")
            # dotted arrows for hidden inputs of control-flow ops
            if op_type in {"Scan", "Loop", "If"}:
                hidden = self._hidden_inputs(atts)
                for inp in hidden:
                    if inp in tiny_inits or inp not in name_to_id:
                        continue
                    src = name_to_id[inp]
                    edge = (src, node_id)
                    if edge in done:
                        continue
                    done.add(edge)
                    rows.append(f"    {src} -.-> {node_id}")

        # ---- output nodes ----
        for name, elem_type, shape in self._outputs:
            if not name:
                continue
            lab = self._edge_label.get(name) or _shape_label(elem_type, shape)
            node_id = f"O_{self._next_id()}"
            display = _mermaid_safe(f"{name}\\n{lab}" if lab else name)
            rows.append(f'    {node_id}["{display}"]:::output')
            if name in name_to_id:
                rows.append(f"    {name_to_id[name]} --> {node_id}")

        # ---- class definitions ----
        rows += [
            "    classDef input fill:#aaeeaa,stroke:#00aa00,color:#000",
            "    classDef init fill:#cccc00,stroke:#888800,color:#000",
            "    classDef op fill:#cccccc,stroke:#666666,color:#000",
            "    classDef output fill:#aaaaee,stroke:#0000aa,color:#000",
        ]
        return rows

    @staticmethod
    def _make_op_label(
        op_type: str,
        domain: str,
        inputs: List[str],
        outputs: List[str],
        atts: Dict,
        tiny_inits: Dict[str, str],
    ) -> str:
        """Builds the display label for an operator node."""
        full_op = f"{domain}.{op_type}" if domain else op_type
        args = [tiny_inits.get(i, ".") if i else "" for i in inputs]
        for att_name, (att_proto, att_val) in atts.items():
            if att_proto.type == onnx.AttributeProto.GRAPH:
                continue
            if att_name == "to":
                args.append(f"{att_name}={onnx_dtype_name(int(att_val))}")
            elif att_name in {"axis", "value_int", "stash_type", "start", "end"}:
                args.append(f"{att_name}={att_val}")
            elif att_name == "value_float":
                args.append(f"{att_name}={att_val}")
            elif att_name in {"value_floats", "value_ints", "perm"}:
                v = att_val.tolist() if isinstance(att_val, np.ndarray) else list(att_val)
                args.append(f"{att_name}={v}")
        label = f"{full_op}({', '.join(args)})"
        if op_type == "Constant" and outputs:
            label += f" -> {outputs[0]}"
        return _mermaid_safe(label)

    @staticmethod
    def _extract_constant_value(atts: Dict) -> Optional[np.ndarray]:
        """Extracts the constant value from a ``Constant`` node's ``atts`` dict."""
        if "value" in atts:
            _, val = atts["value"]
            if isinstance(val, np.ndarray):
                return val
        if "value_float" in atts:
            _, val = atts["value_float"]
            return np.array(val, dtype=np.float32)
        if "value_int" in atts:
            _, val = atts["value_int"]
            return np.array(val, dtype=np.int64)
        if "value_floats" in atts:
            _, val = atts["value_floats"]
            return np.asarray(val, dtype=np.float32)
        if "value_ints" in atts:
            _, val = atts["value_ints"]
            return np.asarray(val, dtype=np.int64)
        return None

    @staticmethod
    def _hidden_inputs(atts: Dict) -> Set[str]:
        """Returns tensor names used inside subgraph attributes (for ``Scan``/``Loop``/``If``)."""
        hidden: Set[str] = set()
        for _att_name, (att_proto, _att_val) in atts.items():
            if att_proto.type == onnx.AttributeProto.GRAPH:
                hidden |= get_hidden_inputs(att_proto.g)
        return hidden

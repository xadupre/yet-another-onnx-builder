from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from onnx import AttributeProto, TensorProto
from .base_emitter import BaseEmitter
from .translator import Translator

# Mapping from ONNX element type integer to name string
_ELEMENT_TYPE_NAME = {
    getattr(TensorProto, k): k
    for k in dir(TensorProto)
    if isinstance(getattr(TensorProto, k), int) and "_" not in k
}


class InnerEmitter(BaseEmitter):
    """Converts event into proper onnx.helper code."""

    def __init__(self):
        self.ir_version = None

    def render_attribute_value(self, value: Any) -> Tuple[List[str], str]:
        """
        Renders an attribute value into a string.

        :param value: value to converter
        :return: rows to append before, actual value
        """
        if value[0].type == AttributeProto.GRAPH:
            tr = Translator(value[0].g, emitter=self)
            rows = tr.export(as_str=False, single_line=False)
            new_rows = [f"def _make_local_graph_{value[0].name}():"]
            for line in rows:
                if "oh.make_model" in line:
                    break
                new_rows.append("    " + line)
            new_rows.append("    return graph")
            new_rows.append(f"{value[0].name} = _make_local_graph_{value[0].name}()")
            return new_rows, value[0].name

        return super().render_attribute_value(value)

    def _make_attribute(
        self, name: str, attr_type: int, ref_attr_name: Optional[str] = None
    ) -> str:
        assert (
            ref_attr_name is not None
        ), f"Cannot create attribute with name={name!r}, attr_type={attr_type}."
        return (
            f"make_ref_attribute(key={name!r}, attr_type={attr_type}, "
            f"ref_attr_name={ref_attr_name!r})"
        )

    def join(self, rows: List[str], single_line: bool = False) -> str:
        "Returns the separators. `single_line` is unused."
        return "\n".join(rows)

    def _emit_start(self, **kwargs: Dict[str, Any]) -> List[str]:
        self.ir_version = kwargs.get("ir_version", None)
        lines = ["opset_imports = ["]
        opsets = kwargs.get("opsets", {})
        for k, v in opsets.items():
            lines.append(f"    oh.make_opsetid({k!r}, {v!r}),")
        lines.append("]")
        return lines

    def _emit_to_onnx_model(self, **kwargs: Dict[str, Any]) -> List[str]:
        lines = [
            "model = oh.make_model(",
            "    graph,",
            "    functions=functions,",
            "    opset_imports=opset_imports,",
        ]
        if self.ir_version:
            lines.append(f"    ir_version={self.ir_version},")
        lines.append(")")
        return lines

    def _emit_begin_graph(self, **kwargs: Dict[str, Any]) -> List[str]:
        lines = [
            "inputs = []",
            "outputs = []",
            "nodes = []",
            "initializers = []",
            "sparse_initializers = []",
            "functions = []",
        ]
        return lines

    def _emit_end_graph(self, **kwargs: Dict[str, Any]) -> List[str]:
        name = kwargs.get("name", "noname")
        lines = [
            "graph = oh.make_graph(",
            "    nodes,",
            f"    {name!r},",
            "    inputs,",
            "    outputs,",
            "    initializers,",
            "    sparse_initializer=sparse_initializers,",
            ")",
        ]
        return lines

    def _emit_initializer(self, **kwargs: Dict[str, Any]) -> List[str]:
        name = kwargs["name"]
        value = kwargs["value"]
        repl = {"bool": "bool_", "object": "object_", "str": "str_"}
        fra = "onh.from_array"
        sdtype = repl.get(str(value.dtype), str(value.dtype))
        sdtype = f"np.{sdtype}" if hasattr(np, sdtype) else f"ml_dtypes.{sdtype}"

        return [
            "initializers.append(",
            f"    {fra}(",
            f"        np.array({value.tolist()}, dtype={sdtype}),",
            f"        name={name!r}",
            "    )",
            ")",
        ]

    def _emit_io(self, container: str, **kwargs: Dict[str, Any]) -> List[str]:
        name = kwargs["name"]
        elem_type = kwargs.get("elem_type", None)
        shape = kwargs.get("shape", None)
        if elem_type and shape:
            return [
                f"{container}.append(oh.make_tensor_value_info({name!r}, "
                f"onnx.TensorProto.{_ELEMENT_TYPE_NAME[elem_type]}, shape={shape!r}))"
            ]
        if elem_type:
            return [
                f"{container}.append(oh.make_tensor_value_info({name!r}, "
                f"onnx.TensorProto.{_ELEMENT_TYPE_NAME[elem_type]}, shape=[]))"
            ]
        return [
            f"{container}.append(oh.make_tensor_value_info({name!r}, "
            f"onnx.TensorProto.UNDEFINED, []))"
        ]

    def _emit_input(self, **kwargs: Dict[str, Any]) -> List[str]:
        return self._emit_io("inputs", **kwargs)

    def _emit_output(self, **kwargs: Dict[str, Any]) -> List[str]:
        return self._emit_io("outputs", **kwargs)

    def _emit_node(self, **kwargs: Dict[str, Any]) -> List[str]:
        op_type = kwargs["op_type"]
        inputs = list(kwargs["inputs"])
        outputs = list(kwargs["outputs"])
        before_lines = []
        domain = kwargs.get("domain", "")
        atts = kwargs.get("atts", {})

        # Separate regular attrs from ref attrs (ref_attr_name attributes, value is None)
        regular_atts = []
        ref_attr_appends = []
        for k, v in atts.items():
            before, value = self.render_attribute_value(v)
            before_lines.extend(before)
            if v[1] is None and getattr(v[0], "ref_attr_name", ""):
                # ref attribute: must be appended to node.attribute separately
                ref_attr_appends.append(f"node.attribute.append({value})")
            else:
                regular_atts.append((k, value))

        # Build argument list for oh.make_node
        args = [
            f"        {op_type!r},",
            f"        {inputs},",
            f"        {outputs},",
        ]
        if domain:
            args.append(f"        domain={domain!r},")
        for k, value in regular_atts:
            args.append(f"        {k}={value},")
        # Remove trailing comma from last arg
        if args:
            args[-1] = args[-1].rstrip(",")

        call_lines = ["    oh.make_node(", *args, "    )"]

        if ref_attr_appends:
            # Use a temporary node variable to allow appending ref attributes
            lines = [*before_lines, "node = (", *call_lines, ")"]
            lines.extend(ref_attr_appends)
            lines.append("nodes.append(node)")
        else:
            lines = [*before_lines, "nodes.append(", *call_lines, ")"]
        return lines

    def _emit_begin_function(self, **kwargs: Dict[str, Any]) -> List[str]:
        opsets = kwargs.get("opsets", {})
        lines = [
            "",
            "opset_imports_f = [",
            *[f"    oh.make_opsetid({k!r}, {v!r})," for k, v in opsets.items()],
            "]",
            f"name_f = {kwargs['name']!r}",
            f"domain_f = {kwargs['domain']!r}",
            "nodes = []",
            "inputs = []",
            "outputs = []",
            "atts = []",
        ]
        return lines

    def _emit_to_onnx_function(self, **kwargs: Dict[str, Any]) -> List[str]:
        return []

    def _emit_function_input(self, **kwargs: Dict[str, Any]) -> List[str]:
        return [f"inputs.append({kwargs['name']!r})"]

    def _emit_function_output(self, **kwargs: Dict[str, Any]) -> List[str]:
        return [f"outputs.append({kwargs['name']!r})"]

    def _emit_function_attributes(self, **kwargs: Dict[str, Any]) -> List[str]:
        atts = kwargs["attributes"]
        if isinstance(atts, list) and all(isinstance(t, str) for t in atts):
            return [f"atts.extend({atts!r})"]
        raise NotImplementedError(f"Unable to process function attributes {atts!r}.")

    def _emit_end_function(self, **kwargs: Dict[str, Any]) -> List[str]:
        lines = [
            "function = oh.make_function(",
            "    domain_f,",
            "    name_f,",
            "    inputs,",
            "    outputs,",
            "    nodes,",
            "    attributes=atts,",
            "    opset_imports=opset_imports_f,",
            ")",
            "try:",
            "    functions.append(function)",
            "except NameError:",
            "    pass",
        ]
        return lines


class InnerEmitterCompact(InnerEmitter):
    """
    Converts events into compact single-expression ONNX code.

    Instead of building lists of nodes/inputs/outputs and then assembling them,
    this emitter produces a single nested expression::

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node('Reshape', ['X', 'shape'], ['reshaped']),
                    oh.make_node('Transpose', ['reshaped'], ['Y'], perm=[1, 0]),
                ],
                'simple',
                [oh.make_tensor_value_info('X', onnx.TensorProto.FLOAT, (None, None))],
                [oh.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, (None, None))],
                [onh.from_array(np.array([-1, 1], dtype=np.int64), name='shape')],
            ),
            functions=[],
            opset_imports=[oh.make_opsetid('', 17)],
            ir_version=8,
        )
    """

    def __init__(self):
        super().__init__()
        self._opsets: Dict[str, int] = {}
        # Main graph state
        self._c_nodes: List[Tuple[List[str], str]] = []
        self._c_inputs: List[str] = []
        self._c_outputs: List[str] = []
        self._c_initializers: List[str] = []
        self._c_graph_name: str = "graph"
        # Function state
        self._c_in_function: bool = False
        self._c_func_nodes: List[Tuple[List[str], str]] = []
        self._c_func_inputs: List[str] = []
        self._c_func_outputs: List[str] = []
        self._c_func_attributes: List[str] = []
        self._c_func_opsets: Dict[str, int] = {}
        self._c_func_name: str = ""
        self._c_func_domain: str = ""
        self._c_function_vars: List[str] = []
        self._c_function_count: int = 0
        self._c_node_counter: int = 0

    def _make_value_info_expr(
        self, name: str, elem_type: Optional[int], shape: Optional[tuple]
    ) -> str:
        if elem_type and shape is not None:
            return (
                f"oh.make_tensor_value_info({name!r}, "
                f"onnx.TensorProto.{_ELEMENT_TYPE_NAME[elem_type]}, {shape!r})"
            )
        if elem_type:
            return (
                f"oh.make_tensor_value_info({name!r}, "
                f"onnx.TensorProto.{_ELEMENT_TYPE_NAME[elem_type]}, [])"
            )
        return f"oh.make_tensor_value_info({name!r}, onnx.TensorProto.UNDEFINED, [])"

    def render_attribute_value(self, value: Any) -> Tuple[List[str], str]:
        """Override to use a plain InnerEmitter for subgraph attributes."""
        if value[0].type == AttributeProto.GRAPH:
            tr = Translator(value[0].g, emitter=InnerEmitter())
            rows = tr.export(as_str=False, single_line=False)
            new_rows = [f"def _make_local_graph_{value[0].name}():"]
            for line in rows:
                if "oh.make_model" in line:
                    break
                new_rows.append("    " + line)
            new_rows.append("    return graph")
            new_rows.append(f"{value[0].name} = _make_local_graph_{value[0].name}()")
            return new_rows, value[0].name
        return super().render_attribute_value(value)

    def _emit_start(self, **kwargs: Dict[str, Any]) -> List[str]:
        self.ir_version = kwargs.get("ir_version", None)
        self._opsets = kwargs.get("opsets", {})
        self._c_function_vars = []
        self._c_function_count = 0
        self._c_node_counter = 0
        self._c_in_function = False
        return []

    def _emit_begin_graph(self, **kwargs: Dict[str, Any]) -> List[str]:
        self._c_nodes = []
        self._c_inputs = []
        self._c_outputs = []
        self._c_initializers = []
        self._c_graph_name = kwargs.get("name", "graph")
        self._c_in_function = False
        return []

    def _emit_end_graph(self, **kwargs: Dict[str, Any]) -> List[str]:
        self._c_graph_name = kwargs.get("name", self._c_graph_name)
        return []

    def _emit_initializer(self, **kwargs: Dict[str, Any]) -> List[str]:
        name = kwargs["name"]
        value = kwargs["value"]
        repl = {"bool": "bool_", "object": "object_", "str": "str_"}
        fra = "onh.from_array"
        sdtype = repl.get(str(value.dtype), str(value.dtype))
        sdtype = f"np.{sdtype}" if hasattr(np, sdtype) else f"ml_dtypes.{sdtype}"
        expr = f"{fra}(np.array({value.tolist()}, dtype={sdtype}), name={name!r})"
        self._c_initializers.append(expr)
        return []

    def _emit_input(self, **kwargs: Dict[str, Any]) -> List[str]:
        name = kwargs["name"]
        elem_type = kwargs.get("elem_type", None)
        shape = kwargs.get("shape", None)
        self._c_inputs.append(self._make_value_info_expr(name, elem_type, shape))
        return []

    def _emit_output(self, **kwargs: Dict[str, Any]) -> List[str]:
        name = kwargs["name"]
        elem_type = kwargs.get("elem_type", None)
        shape = kwargs.get("shape", None)
        self._c_outputs.append(self._make_value_info_expr(name, elem_type, shape))
        return []

    def _emit_node(self, **kwargs: Dict[str, Any]) -> List[str]:
        op_type = kwargs["op_type"]
        inputs = list(kwargs["inputs"])
        outputs = list(kwargs["outputs"])
        domain = kwargs.get("domain", "")
        atts = kwargs.get("atts", {})

        before_lines: List[str] = []
        regular_atts: List[str] = []
        ref_attr_appends: List[str] = []
        for k, v in atts.items():
            before, value = self.render_attribute_value(v)
            before_lines.extend(before)
            if v[1] is None and getattr(v[0], "ref_attr_name", ""):
                ref_attr_appends.append(f"_cnode.attribute.append({value})")
            else:
                regular_atts.append(f"{k}={value}")

        node_args = [f"{op_type!r}", str(inputs), str(outputs)]
        if domain:
            node_args.append(f"domain={domain!r}")
        node_args.extend(regular_atts)
        expr = f"oh.make_node({', '.join(node_args)})"

        if ref_attr_appends:
            node_var = f"_cnode_{self._c_node_counter}"
            self._c_node_counter += 1
            setup = [f"{node_var} = {expr}", *ref_attr_appends]
            entry: Tuple[List[str], str] = (before_lines + setup, node_var)
        else:
            entry = (before_lines, expr)

        if self._c_in_function:
            self._c_func_nodes.append(entry)
        else:
            self._c_nodes.append(entry)
        return []

    def _emit_begin_function(self, **kwargs: Dict[str, Any]) -> List[str]:
        self._c_in_function = True
        self._c_func_nodes = []
        self._c_func_inputs = []
        self._c_func_outputs = []
        self._c_func_attributes = []
        self._c_func_name = kwargs["name"]
        self._c_func_domain = kwargs["domain"]
        self._c_func_opsets = kwargs.get("opsets", {})
        return []

    def _emit_function_input(self, **kwargs: Dict[str, Any]) -> List[str]:
        self._c_func_inputs.append(kwargs["name"])
        return []

    def _emit_function_output(self, **kwargs: Dict[str, Any]) -> List[str]:
        self._c_func_outputs.append(kwargs["name"])
        return []

    def _emit_function_attributes(self, **kwargs: Dict[str, Any]) -> List[str]:
        atts = kwargs["attributes"]
        if isinstance(atts, list):
            self._c_func_attributes.extend(str(a) for a in atts)
        return []

    def _emit_to_onnx_function(self, **kwargs: Dict[str, Any]) -> List[str]:
        return []

    def _emit_end_function(self, **kwargs: Dict[str, Any]) -> List[str]:
        before_lines: List[str] = []
        node_exprs: List[str] = []
        for bl, expr in self._c_func_nodes:
            before_lines.extend(bl)
            node_exprs.append(expr)

        opset_exprs = [
            f"oh.make_opsetid({k!r}, {v!r})" for k, v in self._c_func_opsets.items()
        ]
        opsets_str = "[" + ", ".join(opset_exprs) + "]"

        var_name = f"_function_{self._c_function_count}"
        self._c_function_count += 1

        lines: List[str] = [*before_lines]
        lines.append(f"{var_name} = oh.make_function(")
        lines.append(f"    {self._c_func_domain!r},")
        lines.append(f"    {self._c_func_name!r},")
        lines.append(f"    {self._c_func_inputs!r},")
        lines.append(f"    {self._c_func_outputs!r},")
        lines.append("    [")
        for ne in node_exprs:
            lines.append(f"        {ne},")
        lines.append("    ],")
        if self._c_func_attributes:
            lines.append(f"    attributes={self._c_func_attributes!r},")
        lines.append(f"    opset_imports={opsets_str},")
        lines.append(")")

        self._c_function_vars.append(var_name)
        self._c_in_function = False
        return lines

    def _emit_to_onnx_model(self, **kwargs: Dict[str, Any]) -> List[str]:
        before_lines: List[str] = []
        node_exprs: List[str] = []
        for bl, expr in self._c_nodes:
            before_lines.extend(bl)
            node_exprs.append(expr)

        opset_exprs = [f"oh.make_opsetid({k!r}, {v!r})" for k, v in self._opsets.items()]
        opsets_str = "[" + ", ".join(opset_exprs) + "]"
        fns_str = "[" + ", ".join(self._c_function_vars) + "]"

        lines: List[str] = [*before_lines]
        lines.append("model = oh.make_model(")
        lines.append("    oh.make_graph(")
        lines.append("        [")
        for ne in node_exprs:
            lines.append(f"            {ne},")
        lines.append("        ],")
        lines.append(f"        {self._c_graph_name!r},")
        lines.append("        [")
        for ie in self._c_inputs:
            lines.append(f"            {ie},")
        lines.append("        ],")
        lines.append("        [")
        for oe in self._c_outputs:
            lines.append(f"            {oe},")
        lines.append("        ],")
        if self._c_initializers:
            lines.append("        [")
            for init_e in self._c_initializers:
                lines.append(f"            {init_e},")
            lines.append("        ],")
        lines.append("    ),")
        lines.append(f"    functions={fns_str},")
        lines.append(f"    opset_imports={opsets_str},")
        if self.ir_version:
            lines.append(f"    ir_version={self.ir_version},")
        lines.append(")")
        return lines


class InnerEmitterShortInitializer(InnerEmitter):
    """
    Converts event into proper code.
    Initializers are replaced by random values if too big.
    """

    def _emit_initializer(self, **kwargs: Dict[str, Any]) -> List[str]:
        name = kwargs["name"]
        value = kwargs["value"]
        repl = {"bool": "bool_", "object": "object_", "str": "str_"}
        fra = "onh.from_array"
        sdtype = repl.get(str(value.dtype), str(value.dtype))
        sdtype = f"np.{sdtype}" if hasattr(np, sdtype) else f"ml_dtypes.{sdtype}"
        if value.size <= 16:
            return [
                "initializers.append(",
                f"    {fra}(",
                f"        np.array({value.tolist()}, dtype={sdtype}),",
                f"        name={name!r}",
                "    )",
                ")",
            ]
        if "int" in sdtype:
            return [
                f"value = np.random.randint(0, 10, size={value.shape}).astype({sdtype})",
                "initializers.append(",
                f"    {fra}(",
                f"        np.array(value, dtype={sdtype}),",
                f"        name={name!r}",
                "    )",
                ")",
            ]
        return [
            f"value = np.random.randn({', '.join(map(str, value.shape))}).astype({sdtype})",
            "initializers.append(",
            f"    {fra}(",
            f"        np.array(value, dtype={sdtype}),",
            f"        name={name!r}",
            "    )",
            ")",
        ]

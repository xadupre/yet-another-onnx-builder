import functools
import re
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh


class NodeCoordinates:
    """
    A way to localize a node,
    path is a tuple of three information, node index, node type, node name.
    """

    __slots__ = ("node", "path")

    def __init__(
        self,
        node: Union[
            onnx.TensorProto, onnx.NodeProto, onnx.SparseTensorProto, onnx.ValueInfoProto, str
        ],
        path: Tuple[Tuple[int, str, str], ...],
    ):
        assert isinstance(path, tuple), f"Unexpected type {type(path)} for path"
        assert all(isinstance(t, tuple) for t in path), f"Unexpected type in path={path}"
        self.node = node
        self.path = path

    def __str__(self) -> str:
        "usual"
        if isinstance(self.node, str):
            return f"{self.path_to_str()} :: {self.node!r}"
        return f"{self.path_to_str()} :: {pretty_onnx(self.node)}"

    def path_to_str(self) -> str:
        "Strings representing coordinates."
        return "x".join(f"({':'.join(map(str, t))})" for t in self.path)


class ResultFound:
    """Class returned by :func:`enumerate_results`."""

    __slots__ = ("consumer", "name", "producer")

    def __init__(
        self,
        name: str,
        producer: Optional[NodeCoordinates],
        consumer: Optional[NodeCoordinates],
    ):
        assert isinstance(name, str), f"unexpected type {type(name)} for name"
        self.name = name
        self.producer = producer
        self.consumer = consumer

    def __str__(self) -> str:
        "usuals"
        return (
            f"<< {self.name} - {self.consumer}"
            if self.producer is None
            else f">> {self.name} - {self.producer}"
        )


@functools.cache
def onnx_dtype_name(itype: int, exc: bool = True) -> str:
    """
    Returns the ONNX name for a specific element type.

    .. runpython::
        :showcode:

        import onnx
        from yobx.helpers.onnx_helper import onnx_dtype_name

        itype = onnx.onnx.TensorProto.BFLOAT16
        print(onnx_dtype_name(itype))
        print(onnx_dtype_name(7))
    """
    for k in dir(onnx.onnx.TensorProto):
        if k.upper() == k and k not in {"DESCRIPTOR", "EXTERNAL", "DEFAULT"}:
            v = getattr(onnx.onnx.TensorProto, k)
            if v == itype:
                return k
    if exc:
        raise ValueError(f"Unexpected value itype: {itype}")
    if itype == 0:
        return "UNDEFINED"
    return "UNEXPECTED"


def np_dtype_to_tensor_dtype(dtype: np.dtype) -> int:
    """Converts a numpy dtype to an onnx element type."""
    return oh.np_dtype_to_tensor_dtype(dtype)


def dtype_to_tensor_dtype(dt: Union[np.dtype, "torch.dtype"]) -> int:  # type: ignore[arg-type,name-defined] # noqa: F821
    """
    Converts a torch dtype or numpy dtype into a onnx element type.

    :param to: dtype
    :return: onnx type
    """
    try:
        return np_dtype_to_tensor_dtype(dt)
    except (KeyError, TypeError, ValueError):
        pass
    from .torch_helper import torch_dtype_to_onnx_dtype

    return torch_dtype_to_onnx_dtype(dt)  # type: ignore[arg-type]


def tensor_dtype_to_np_dtype(tensor_dtype: int) -> np.dtype:
    """
    Converts a onnx.TensorProto's data_type to corresponding numpy dtype.
    It can be used while making tensor.

    :param tensor_dtype: onnx.TensorProto's data_type
    :return: numpy's data_type
    """
    return oh.tensor_dtype_to_np_dtype(tensor_dtype)


def pretty_onnx(
    onx: Union[
        onnx.AttributeProto,
        onnx.FunctionProto,
        onnx.GraphProto,
        onnx.ModelProto,
        onnx.NodeProto,
        onnx.SparseTensorProto,
        onnx.TensorProto,
        onnx.ValueInfoProto,
        str,
    ],
    with_attributes: bool = False,
    highlight: Optional[Set[str]] = None,
    shape_inference: bool = False,
) -> str:
    """
    Displays an onnx proto in a better way.

    :param with_attributes: displays attributes as well, if only a node is printed
    :param highlight: to highlight some names
    :param shape_inference: run shape inference before printing the model
    :return: text
    """
    assert onx is not None, "onx cannot be None"
    if isinstance(onx, str):
        onx = onnx.load(onx, load_external_data=False)
    assert onx is not None, "onx cannot be None"

    if shape_inference:
        assert isinstance(
            onx, onnx.ModelProto
        ), f"shape inference only works for ModelProto, not {type(onx)}"
        onx = onnx.shape_inference.infer_shapes(onx)

    if isinstance(onx, onnx.ValueInfoProto):
        name = onx.name
        itype = onx.type.tensor_type.elem_type
        shape = tuple((d.dim_param or d.dim_value) for d in onx.type.tensor_type.shape.dim)
        shape_str = ",".join(map(str, shape))
        return f"{onnx_dtype_name(itype, exc=False)}[{shape_str}] {name}"

    if isinstance(onx, onnx.TypeProto):
        itype = onx.tensor_type.elem_type
        shape = tuple((d.dim_param or d.dim_value) for d in onx.tensor_type.shape.dim)
        shape_str = ",".join(map(str, shape))
        return f"{onnx_dtype_name(itype, exc=False)}[{shape_str}]"

    if isinstance(onx, onnx.AttributeProto):
        att = onx
        if att.type == onnx.AttributeProto.INT:
            return f"{att.name}={att.i}"
        if att.type == onnx.AttributeProto.INTS:
            return f"{att.name}={att.ints}"
        if att.type == onnx.AttributeProto.FLOAT:
            return f"{att.name}={att.f}"
        if att.type == onnx.AttributeProto.FLOATS:
            return f"{att.name}={att.floats}"
        if att.type == onnx.AttributeProto.STRING:
            return f"{att.name}={att.s!r}"
        if att.type == onnx.AttributeProto.TENSOR:
            v = onh.to_array(att.t)
            assert hasattr(v, "reshape"), f"not a tensor {type(v)}"
            assert hasattr(v, "shape"), f"not a tensor {type(v)}"
            vf = v.reshape((-1,))
            if vf.size < 10:
                tt = f"[{', '.join(map(str, vf))}]"
            else:
                tt = f"[{', '.join(map(str, vf[:10]))}, ...]"
            if len(v.shape) != 1:
                return f"{att.name}=tensor({tt}, dtype={v.dtype}).reshape({v.shape})"
            return f"{att.name}=tensor({tt}, dtype={v.dtype})"
        raise NotImplementedError(f"pretty_onnx not implemented yet for AttributeProto={att!r}")

    if isinstance(onx, onnx.NodeProto):

        def _high(n):
            if highlight and n in highlight:
                return f"**{n}**"
            return n

        text = (
            f"{onx.op_type}({', '.join(map(_high, onx.input))})"
            f" -> {', '.join(map(_high, onx.output))}"
        )
        if onx.domain:
            text = f"{onx.domain}.{text}"
        if not with_attributes or not onx.attribute:
            return text
        rows = []
        for att in onx.attribute:
            rows.append(pretty_onnx(att))
        if len(rows) > 1:
            suffix = "\n".join(f"    {s}" for s in rows)
            return f"{text}\n{suffix}"
        return f"{text}  ---  {rows[0]}"

    if isinstance(onx, onnx.TensorProto):
        shape = "x".join(str(d) for d in onx.dims)  # type: ignore[assignment]
        return f"onnx.TensorProto:{onx.data_type}:{shape}:{onx.name}"

    assert not isinstance(
        onx, onnx.SparseTensorProto
    ), "Sparseonnx.TensorProto is not handled yet."

    from ._onnx_simple_text_plot import onnx_simple_text_plot

    if isinstance(onx, onnx.FunctionProto):
        return (
            f"function: {onx.name}[{onx.domain}]\n"
            f"{onnx_simple_text_plot(onx, recursive=True)}"  # pyrefly: ignore[bad-argument-type]
        )
    return onnx_simple_text_plot(onx, recursive=True)  # pyrefly: ignore[bad-argument-type]


def enumerate_results(
    proto: Union[onnx.FunctionProto, onnx.GraphProto, onnx.ModelProto, Sequence[onnx.NodeProto]],
    name: Union[Set[str], str],
    verbose: int = 0,
    coordinates: Optional[List[Tuple[int, str, str]]] = None,
) -> Iterator[ResultFound]:
    """
    Iterates on all nodes, attributes to find where a name is used.

    :param proto: a proto
    :param name: name or names to find
    :param verbose: verbosity
    :param coordinates: coordinates of a node
    :return: iterator on :class:`ResultFound`
    """
    if not isinstance(name, set):
        name = {name}
    coordinates = coordinates or []
    assert all(
        isinstance(c, tuple) for c in coordinates
    ), f"Unexpected type in coordinates={coordinates}"
    indent = "  " * len(coordinates)
    if isinstance(proto, onnx.ModelProto):
        if verbose:
            print(f"[enumerate_results] {indent}searching for {name!r} into ModelProto...")
        yield from enumerate_results(proto.graph, name, verbose=verbose, coordinates=coordinates)
    elif isinstance(proto, onnx.FunctionProto):
        if verbose:
            print(f"[enumerate_results] {indent}searching for {name!r} into FunctionProto...")
        for i in proto.input:
            if i in name:
                r = ResultFound(
                    i,
                    NodeCoordinates(i, tuple([*coordinates, (-1, "INPUT", "")])),  # noqa: C409
                    None,
                )
                if verbose > 1:
                    print(f"[enumerate_results] {indent}-- {r}")
                yield r
        yield from enumerate_results(proto.node, name, verbose=verbose, coordinates=coordinates)
        for i in proto.output:
            if i in name:
                r = ResultFound(
                    i,
                    None,
                    NodeCoordinates(
                        i, tuple([*coordinates, (len(proto.node), "OUTPUT", "")])  # noqa: C409
                    ),
                )
                if verbose > 1:
                    print(f"[enumerate_results] {indent}-- {r}")
                yield r
    elif isinstance(proto, onnx.GraphProto):
        if verbose:
            print(f"[enumerate_results] {indent}searching for {name!r} into GraphProto...")
        for i in proto.initializer:
            if i.name in name:
                r = ResultFound(
                    i.name,
                    NodeCoordinates(i, tuple([*coordinates, (-1, "INIT", "")])),  # noqa: C409
                    None,
                )
                if verbose > 1:
                    print(f"[enumerate_results] {indent}-- {r}")
                yield r
        for i in proto.sparse_initializer:
            if i.values.name in name:
                r = ResultFound(
                    i.values.name,
                    NodeCoordinates(i, tuple([*coordinates, (-1, "INIT", "")])),  # noqa: C409
                    None,
                )
                if verbose > 1:
                    print(f"[enumerate_results] {indent}-- {r}")
                yield r
        for i in proto.input:
            if i.name in name:
                r = ResultFound(
                    i.name,
                    NodeCoordinates(i, tuple([*coordinates, (-1, "INPUT", "")])),  # noqa: C409
                    None,
                )
                if verbose > 1:
                    print(f"[enumerate_results] {indent}-- {r}")
                yield r
        yield from enumerate_results(proto.node, name, verbose=verbose, coordinates=coordinates)
        for i in proto.output:
            if i.name in name:
                r = ResultFound(
                    i.name,
                    None,
                    NodeCoordinates(
                        i, tuple([*coordinates, (len(proto.node), "OUTPUT", "")])  # noqa: C409
                    ),
                )
                if verbose > 1:
                    print(f"[enumerate_results] {indent}-- {r}")
                yield r
    else:
        if verbose:
            print(f"[enumerate_results] {indent}searching for {name!r} into List[NodeProto]...")
        for node_i, node in enumerate(proto):
            if set(node.input) & name:
                for n in node.input:
                    if n in name:
                        r = ResultFound(
                            n,
                            None,
                            NodeCoordinates(
                                node,
                                tuple(  # noqa: C409
                                    [*coordinates, (node_i, node.op_type, node.name)]
                                ),
                            ),
                        )
                        if verbose > 1:
                            print(f"[enumerate_results] {indent}-- {r}")
                        yield r
            if node.op_type in {"If", "Scan", "Loop", "SequenceMap"}:
                for att in node.attribute:
                    if att.type == onnx.AttributeProto.GRAPH:
                        yield from enumerate_results(
                            att.g,
                            name,
                            verbose=verbose,
                            coordinates=[*coordinates, (node_i, node.op_type, node.name)],
                        )
            if set(node.output) & name:
                for n in node.output:
                    if n in name:
                        r = ResultFound(
                            n,
                            NodeCoordinates(
                                node,
                                tuple(  # noqa: C409
                                    [*coordinates, (node_i, node.op_type, node.name)]
                                ),
                            ),
                            None,
                        )
                        if verbose > 1:
                            print(f"[enumerate_results] {indent}-- {r}")
                        yield r
    if verbose:
        print(f"[enumerate_results] {indent}done")


def shadowing_names(
    proto: Union[onnx.FunctionProto, onnx.GraphProto, onnx.ModelProto, Sequence[onnx.NodeProto]],
    verbose: int = 0,
    existing: Optional[Set[str]] = None,
    shadow_context: Optional[Set[str]] = None,
    post_shadow_context: Optional[Set[str]] = None,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Returns the shadowing names, the names created in the main graph
    after they were created in a subgraphs and the names created by the nodes.
    """
    if isinstance(proto, onnx.ModelProto):
        return shadowing_names(proto.graph)
    if isinstance(proto, onnx.GraphProto):
        assert (
            existing is None and shadow_context is None
        ), "existing must be None if nodes is None"
        return shadowing_names(
            proto.node,
            verbose=verbose,
            existing={i.name for i in proto.initializer}
            | {i.values.name for i in proto.sparse_initializer}
            | {i.name for i in proto.input if i.name},
            shadow_context=set(),
            post_shadow_context=set(),
        )
    if isinstance(proto, onnx.FunctionProto):
        assert (
            existing is None and shadow_context is None
        ), "existing must be None if nodes is None"
        return shadowing_names(
            proto.node,
            verbose=verbose,
            existing=set(i for i in proto.input if i),
            shadow_context=set(),
            post_shadow_context=set(),
        )

    assert (
        existing is not None and shadow_context is not None
    ), "existing must not be None if nodes is not None"
    shadow = set()
    shadow_context = shadow_context.copy()
    existing = existing.copy()
    created = set()
    post_shadow = set()
    for node in proto:
        not_empty = set(n for n in node.input if n)
        intersection = not_empty & existing
        assert len(intersection) == len(
            not_empty
        ), f"One input in {not_empty}, node={pretty_onnx(node)} was not found in {existing}"
        for att in node.attribute:
            if att.type == onnx.AttributeProto.GRAPH:
                g = att.g
                shadow |= {i.name for i in g.input} & shadow_context
                shadow |= {i.name for i in g.initializer} & shadow_context
                shadow |= {i.values.name for i in g.sparse_initializer} & shadow_context
                s, _ps, c = shadowing_names(
                    g.node, verbose=verbose, existing=existing, shadow_context=existing
                )
                shadow |= s
                created |= c

        not_empty = set(n for n in node.output if n)
        post_shadow |= not_empty & created
        shadow |= not_empty & shadow_context
        existing |= not_empty
        created |= not_empty
    return shadow, post_shadow, created


def get_hidden_inputs(graph: onnx.GraphProto) -> Set[str]:
    """
    Returns the hidden inputs (inputs coming from an upper context)
    used by a subgraph. It excludes empty names.
    """
    hidden = set()
    memo = (
        {i.name for i in graph.initializer}
        | {i.values.name for i in graph.sparse_initializer}
        | {i.name for i in graph.input}
    )
    for node in graph.node:
        for i in node.input:
            if i and i not in memo:
                hidden.add(i)
        for att in node.attribute:
            if att.type == onnx.AttributeProto.GRAPH and att.g:
                hid = get_hidden_inputs(att.g)
                less = set(h for h in hid if h not in memo)
                hidden |= less
        memo |= set(node.output)
    return hidden


def _validate_graph(
    g: onnx.GraphProto,
    existing: Set[str],
    verbose: int = 0,
    watch: Optional[Set[str]] = None,
    path: Optional[Sequence[str]] = None,
) -> List[Union[onnx.NodeProto, onnx.TensorProto, onnx.ValueInfoProto]]:
    found: List[Union[onnx.NodeProto, onnx.TensorProto, onnx.ValueInfoProto]] = []
    path = path or ["root"]
    set_init = {i.name for i in g.initializer}
    set_input = {i.name for i in g.input}
    existing |= set_init | set_input
    if watch and set_init & watch:
        if verbose:
            print(f"-- found init {set_init & watch} in {path}")
        found.extend([i for i in g.initializer if i.name in set_init & watch])
    if watch and set_input & watch:
        if verbose:
            print(f"-- found input {set_input & watch} in {path}")
        found.extend([i for i in g.input if i.name in set_input & watch])
    try:
        import tqdm

        loop = tqdm.tqdm(g.node) if verbose else g.node
    except ImportError:
        loop = g.node

    for node in loop:
        ins = set(node.input) & existing
        if ins != set(node.input):
            raise AssertionError(
                f"One input is missing from node.input={node.input}, "
                f"existing={ins}, path={'/'.join(path)}, "
                f"node: {node.op_type}[{node.name}]"
            )
        if watch and ins & watch:
            if verbose:
                print(
                    f"-- found input {ins & watch} in "
                    f"{'/'.join(path)}/{node.op_type}[{node.name}]"
                )
            found.append(node)
        for att in node.attribute:
            if att.type == onnx.AttributeProto.GRAPH:
                found.extend(
                    _validate_graph(
                        att.g,
                        existing.copy(),
                        watch=watch,
                        path=[*path, f"{node.op_type}[{node.name}]"],
                        verbose=verbose,
                    )
                )
        existing |= set(node.output)
        if watch and set(node.output) & watch:
            if verbose:
                print(
                    f"-- found output {set(node.output) & watch} "
                    f"in {'/'.join(path)}/{node.op_type}[{node.name}]"
                )
            found.append(node)
    out = {o.name for o in g.output}
    ins = out & existing
    assert ins == out, f"One output is missing, out={out}, existing={ins}, path={path}"
    return found


def _validate_function(g: onnx.FunctionProto, verbose: int = 0, watch: Optional[Set[str]] = None):
    existing: Set[str] = set(g.input)
    found: List[Union[onnx.NodeProto, onnx.TensorProto, onnx.ValueInfoProto]] = []
    for node in g.node:
        ins = set(node.input) & existing
        if ins != set(node.input):
            raise AssertionError(
                f"One input is missing from node.input={node.input}, existing={ins}"
            )
        if watch and ins & watch:
            if verbose:
                print(f"-- found input {ins & watch} in {node.op_type}[{node.name}]")
            found.append(node)
        for att in node.attribute:
            if att.type == onnx.AttributeProto.GRAPH:
                found.extend(
                    _validate_graph(att.g, existing.copy(), path=[g.name], verbose=verbose)
                )
        existing |= set(node.output)
        if watch and set(node.output) & watch:
            if verbose:
                print(
                    f"-- found output {set(node.output) & watch} "
                    f"in {node.op_type}[{node.name}]"
                )
    out = set(g.output)
    ins = out & existing
    if ins != out:
        raise AssertionError(f"One output is missing, out={out}, existing={ins}, path={g.name}")
    return found


def onnx_find(
    onx: Union[str, onnx.ModelProto], verbose: int = 0, watch: Optional[Set[str]] = None
) -> List[Union[onnx.NodeProto, onnx.TensorProto]]:
    """
    Looks for node producing or consuming some results.

    :param onx: model
    :param verbose: verbosity
    :param watch: names to search for
    :return: list of nodes
    """

    if isinstance(onx, str):
        onx = onnx.load(onx, load_external_data=False)
    found = []
    found.extend(_validate_graph(onx.graph, set(), verbose=verbose, watch=watch))
    for f in onx.functions:
        found.extend(_validate_function(f, watch=watch, verbose=verbose))
    if verbose and found:
        print(f"-- found {len(found)} nodes")
    return found


def check_for_non_recursivity(
    node_indices: List[int],
    node_list: List[Optional[onnx.NodeProto]],
    inputs: Union[Set[str], Sequence[str]],
    outputs: Union[Set[str], Sequence[str]],
    exc: bool = True,
) -> List[int]:
    """
    We need to check that any of this output is not required
    by one input from the function itself, that would mean one node
    needs an output of the function and is also required by the function:
    it is probably missing from the initial set.

    :param node_indices: node_indices part of the subset
    :param node_list: list of nodes
    :param inputs: input names to consider
    :param outputs: output names which cannot be involved in input names
    :param exc: raise an exception as soon as possible it becomes impossible
    :return: list of nodes to add to make the list of node consistence
        with the list of inputs and outputs (they should be recomputed)
    """
    original_set_inputs = inputs if isinstance(inputs, set) else set(inputs)
    set_inputs = original_set_inputs.copy()
    original_set_outputs = outputs if isinstance(outputs, set) else set(outputs)
    subset = set(node_indices)
    before_inputs = set()
    indexed_node = list(enumerate(node_list))
    additional_nodes: List[int] = []
    for ind, node in indexed_node[::-1]:
        if not node:
            continue
        s_outputs = set(o for o in node.output if o)
        if ind in subset:
            # The node is part of the subset.
            if s_outputs & set_inputs:
                set_inputs |= set(i for i in node.input if i)
                if node.op_type in {"Scan", "If", "Loop"}:
                    # there are hidden inputs
                    for att in node.attribute:
                        if att.type == onnx.AttributeProto.GRAPH:
                            set_inputs |= get_hidden_inputs(att.g)
            if original_set_outputs & set_inputs:
                raise ValueError(
                    f"Results {original_set_outputs & set_inputs} "
                    f"are needed for inputs {inputs} "
                    f"but also requires {outputs} which is not allowed."
                )
        else:
            # Not part of the function. Let's check
            if s_outputs & original_set_inputs:
                before_inputs |= set(i for i in node.input if i)
                if node.op_type in {"Scan", "If", "Loop"}:
                    # there are hidden inputs
                    for att in node.attribute:
                        if att.type == onnx.AttributeProto.GRAPH:
                            before_inputs |= get_hidden_inputs(att.g)
            if original_set_outputs & before_inputs:
                if exc:
                    raise ValueError(
                        f"Results {original_set_outputs & before_inputs} "
                        f"are needed for inputs {inputs} "
                        f"but also requires {outputs} which is not allowed."
                    )
                additional_nodes.append(ind)
    return additional_nodes


def _select_nodes_from_metadata_with_regex(
    model: onnx.ModelProto, prefix: Union[str, Tuple[str, ...]], regex: str
) -> Tuple[Dict[str, List[int]], Set[str]]:
    reg = re.compile(regex)
    unique_values = set()
    unique: Dict[str, List[int]] = {}
    for i, node in enumerate(model.graph.node):
        selected = False
        for data in node.metadata_props:
            if data.key.startswith(prefix):
                values = re.split("[,:]", data.value)
                for v in values:
                    if not v:
                        continue
                    if reg.match(v):
                        if v not in unique:
                            unique[v] = []
                        unique[v].append(i)
                        selected = True
                        break
                    unique_values.add(v)
                if selected:
                    break
    return unique, unique_values


def unknown_names_within_nodes(nodes: List[onnx.NodeProto]) -> Set[str]:
    """Returns the list of unknown results from a list of nodes."""
    not_known: Set[str] = set()
    for node in nodes[::-1]:
        not_known -= {o for o in node.output if o}
        not_known |= {i for i in node.input if i}
        if node.op_type in {"Scan", "If", "Loop"}:
            # there are hidden inputs
            for att in node.attribute:
                if att.type == onnx.AttributeProto.GRAPH:
                    not_known |= get_hidden_inputs(att.g)
    return not_known


def make_subfunction(
    name: str,
    nodes: List[onnx.NodeProto],
    opset_imports: Sequence[onnx.OperatorSetIdProto],
    output_names: List[str],
    domain: str = "local_function",
) -> onnx.FunctionProto:
    """
    Creates a function with the given list of nodes.
    It computes the minimum list of inputs needed for this model.
    The function assumes the nodes are sorted.

    :param name: function name
    :param nodes: list of nodes
    :param opset_imports: opset import
    :param output_names: desired outputs
    :param domain: function domain
    :return: model proto
    """
    return oh.make_function(
        domain,
        name,
        nodes=nodes,
        inputs=sorted(unknown_names_within_nodes(nodes)),
        outputs=output_names,
        opset_imports=opset_imports,
    )


def _find_used_names(node_list, node_indices):
    # find all the outputs the subset of nodes produces
    possible_outputs = set()
    for i_node in node_indices:
        if not node_list[i_node]:
            continue
        possible_outputs |= {o for o in node_list[i_node].output if o}
    # find all requires input from the other nodes
    set_indices = set(node_indices)
    not_known = set()
    ranges = list(range(len(node_list)))
    for i_node in ranges[::-1]:
        if i_node in set_indices:
            continue
        node = node_list[i_node]
        if not node:
            continue
        not_known -= {o for o in node.output if o}
        not_known |= {i for i in node.input if i}
        if node.op_type in {"Scan", "If", "Loop"}:
            # there are hidden inputs
            for att in node.attribute:
                if att.type == onnx.AttributeProto.GRAPH:
                    not_known |= get_hidden_inputs(att.g)
    # output
    selection = possible_outputs & not_known
    assert selection, (
        f"No output is needed, possible_outputs={sorted(possible_outputs)}, "
        f"not_known={sorted(not_known)}"
    )
    return sorted(selection)


def make_model_with_local_functions(
    model: onnx.ModelProto,
    regex: str = ".*[.]layers[.][0-9]+[.]forward$",
    domain: str = "local_function",
    metadata_key_prefix: Union[str, Tuple[str, ...]] = ("namespace", "source["),
    allow_extensions: bool = True,
    verbose: int = 0,
) -> onnx.ModelProto:
    """
    Selects nodes based on a regular expression, using metadata
    ``'namespace'``. It is going to look into every value
    matching the regular expression and partition the nodes based
    on the unique values the regular expression finds.
    Every set of nodes it replaced by a call to a local function.

    :param model: model proto
    :param regex: regular expression
    :param domain: function domain
    :param metadata_keys: list of metadata keys to consider,
        every value is split into multiple ones.
    :param allow_extensions: allows the function to take nodes outside
        a partition if there are not already inside another partition
    :param verbose: verbosity
    :return: model proto

    Example:

    .. runpython::
        :showcode:

        import numpy as np
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh
        from yobx.helpers.onnx_helper import (
            make_model_with_local_functions,
            pretty_onnx,
        )

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [320, 1280])],
                [oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [3, 5, 320, 640])],
                [
                    onh.from_array(
                        np.random.rand(3, 5, 1280, 640).astype(np.float32), name="Y"
                    ),
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 320, 1280], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 1280, 640], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 320, 640], dtype=np.int64), name="shape3"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        for i_node in [0, 1, 2, 3]:
            node = model.graph.node[i_node]
            meta = node.metadata_props.add()
            meta.key = f"source[{i_node}]"
            meta.value = f"LLL{i_node//3}"

        print("-- model before --")
        print(pretty_onnx(model))
        print()
        print("-- metadata --")
        for node in model.graph.node:
            text = (
                f" -- [{node.metadata_props[0].key}: {node.metadata_props[0].value}]"
                if node.metadata_props
                else ""
            )
            print(
                f"-- {node.op_type}({', '.join(node.input)}) -> "
                f"{', '.join(node.output)}{text}"
            )
        print()

        new_model = make_model_with_local_functions(
            model, "^LLL[01]$", metadata_key_prefix="source[", verbose=1
        )

        print()
        print("-- model after --")
        print(pretty_onnx(new_model))
    """
    prefix = (
        metadata_key_prefix if isinstance(metadata_key_prefix, tuple) else (metadata_key_prefix,)
    )
    unique, unique_values = _select_nodes_from_metadata_with_regex(model, prefix, regex)

    # sets of nodes.
    if not unique:
        if verbose:
            print(f"[make_model_with_local_functions] no match in {sorted(unique_values)}")
        return model

    if verbose:
        print(f"[make_model_with_local_functions] matched {len(unique)} partitions")
        for un, nid in unique.items():
            print(f"[make_model_with_local_functions] {un!r}: {len(nid)} nodes")
            for ind in nid[:5]:
                print(f"     {pretty_onnx(model.graph.node[ind])}")
            if len(nid) > 5:
                print("     ...")
    functions = []
    new_nodes: List[Optional[onnx.NodeProto]] = list(model.graph.node)
    processed: Dict[str, onnx.FunctionProto] = {}
    unique_as_set = {k: set(v) for k, v in unique.items()}
    while len(processed) < len(unique):
        for key, node_indices in unique.items():
            if key in processed:
                # already processed
                continue
            function_name = key.strip().replace(".", "_")
            if verbose:
                print(
                    f"[make_model_with_local_functions] move {len(node_indices)} "
                    f"nodes in partition {key!r} (function={function_name!r})"
                )
            outputs = _find_used_names(new_nodes, node_indices)
            # pyrefly: ignore[bad-assignment]
            function_nodes: List[onnx.NodeProto] = [
                new_nodes[i] for i in node_indices if new_nodes[i]
            ]

            function_inputs = unknown_names_within_nodes(function_nodes)
            additional_nodes = check_for_non_recursivity(
                node_indices, new_nodes, function_inputs, outputs, exc=not allow_extensions
            )
            if additional_nodes:
                if not allow_extensions:
                    raise ValueError(
                        f"Function for key={key!r} cannot be added because "
                        f"it must steal a node outside the partition, node ids "
                        f"{additional_nodes} are needed for inputs {function_inputs} "
                        f"but also requires {outputs} which is not allowed."
                    )
                # Additional nodes are needed to make the function consistence.
                # We check they are not in conflict with other partitions not
                # yet processed.
                set_add = set(additional_nodes)
                for k, v in unique_as_set.items():
                    if v & set_add:
                        raise ValueError(
                            f"Function for key={key!r} cannot be added because "
                            f"it is conflict with other key {k!r} with node ids "
                            f"{set_add & v} are needed for inputs {function_inputs} "
                            f"but also requires {outputs} which is not allowed."
                        )
                # If no exception, everything is fine, let's add the nodes.
                node_indices.extend(additional_nodes)
                node_indices[:] = sorted(node_indices)
                # Inputs and outputs needed to be recomputed. Let's do that in another
                # iteration.
                if verbose:
                    print(
                        f"[make_model_with_local_functions] add {len(additional_nodes)} "
                        f"nodes in partition {key!r}"
                    )
                continue

            lf = make_subfunction(
                function_name,
                function_nodes,
                model.opset_import,
                outputs,
                domain=domain,
            )

            check_for_non_recursivity(node_indices, new_nodes, lf.input, lf.output)

            if verbose:
                print(
                    f"[make_model_with_local_functions] add function {function_name}"
                    f"({', '.join(lf.input)}) -> {', '.join(lf.output)}"
                )
            functions.append(lf)
            maxi = max(node_indices)
            for i in node_indices:
                new_nodes[i] = None
            new_nodes[maxi] = oh.make_node(lf.name, lf.input, lf.output, domain=lf.domain)
            processed[key] = lf

    return oh.make_model(
        oh.make_graph(
            [n for n in new_nodes if n],
            model.graph.name,
            model.graph.input,
            model.graph.output,
            model.graph.initializer,
            doc_string=model.graph.doc_string,
            value_info=model.graph.value_info,
            sparse_initializer=model.graph.sparse_initializer,
        ),
        ir_version=model.ir_version,
        opset_imports=(
            model.opset_import
            if domain in {d.domain for d in model.opset_import}
            else [*model.opset_import, oh.make_opsetid(domain, 1)]
        ),
        functions=[*model.functions, *functions],
    )

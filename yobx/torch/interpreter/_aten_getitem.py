"""
Standalone implementations of the ``getitem`` operator and its slice/index
helpers.  These functions mirror the methods that used to live on
:class:`~yobx.torch.interpreter.interpreter.FxGraphInterpreter` but accept
a :class:`~yobx.xbuilder.GraphBuilder` as their first argument so they can
be used independently of the interpreter.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from onnx import TensorProto
from ...typing import GraphBuilderTorchProtocol
from ...container.model_container import _get_type
from ...helpers import string_type
from ...xshape._shape_helper import all_int
from ...xexpressions.rename_expressions import parse_expression_tokens
from ._aten_functions import _aten_tensor_int1


def _getitem_verify_new_shape(
    g: GraphBuilderTorchProtocol, sts: Optional[Dict[str, Any]], outputs: List[str], shape
) -> None:
    """Registers previously-unseen dynamic dimension tokens found in *shape*.

    Walks over every dimension of *shape* and, for each symbolic integer that
    consists of a single token not yet present in ``g.dynamic_objects``, adds
    that token as a new dynamic object and records the source axis / tensor
    name in ``g.dynamic_dimensions_source``.

    :param g: the graph builder
    :param sts: known shapes and types (unused; present for signature consistency
        with other aten functions)
    :param outputs: list of output tensor names; ``outputs[0]`` is used for
        source bookkeeping
    :param shape: the shape tuple (may contain :class:`torch.SymInt` or
        :class:`~yobx.torch.new_tracing.shape.TracingInt` entries)
    """
    assert hasattr(g, "torch"), "torch module is added as an attribute to avoid import"
    assert hasattr(g, "TracingInt"), "TracingInt class is added as an attribute to avoid import"
    assert hasattr(g, "_torch_sym_int_to_str"), "expecting _torch_sym_int_to_str specific method"
    assert hasattr(g, "dynamic_objects"), "expecting dynamic_objects specific method"
    assert hasattr(
        g, "dynamic_dimensions_source"
    ), "expecting dynamic_dimensions_source specific attribute"
    output_name = outputs[0]
    for axis, dim in enumerate(shape):
        if isinstance(dim, (g.torch.SymInt, g.TracingInt)):
            if isinstance(dim, g.TracingInt):
                sdim = dim.value
            else:
                sdim = g._torch_sym_int_to_str(dim)
            tokens = parse_expression_tokens(sdim)
            if len(tokens) == 1:
                # Only one token, possibly new.
                t = tokens.pop()
                if t not in g.dynamic_objects:
                    g.add_dynamic_object(t, t)
                    assert isinstance(output_name, str), (
                        f"Unexpected type for dim={dim!r}, axis={axis}, shape={shape}, "
                        f"outputs[0]={output_name!r}, t={t!r}"
                    )
                    source = dict(axis=axis, input_name=output_name)
                    if t in g.dynamic_dimensions_source:
                        g.dynamic_dimensions_source[t].append(source)
                    else:
                        g.dynamic_dimensions_source[t] = [source]


def _getitem_slice(
    g: GraphBuilderTorchProtocol,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    input_name: str,
    index_slice: slice,
    axes: List[int],
    expand_axes: List[int],
    name: str = "_getitem_slice",
):
    """Emits ONNX ``Slice`` (and optional ``Unsqueeze``/``Squeeze``) nodes for a
    subscript expression of the form ``tensor[a:b, c:d, ...]``.

    :param g: the graph builder
    :param sts: known shapes and types; when ``None`` the function sets type and
        shape on the output itself
    :param outputs: list of output tensor names; ``outputs[0]`` is the result
    :param input_name: name of the ONNX tensor to slice
    :param index_slice: list of :class:`slice` / :class:`int` /
        :class:`torch.fx.Node` objects, one per axis in *axes*
    :param axes: axes to slice (one entry per element of *index_slice*)
    :param expand_axes: axes to unsqueeze after slicing
    :param name: base name for generated ONNX nodes
    :returns: the name of the result tensor
    """
    assert hasattr(g, "torch"), "torch module is added as an attribute to avoid import"
    assert hasattr(g, "_apply_slice_to_shape"), "expecting method _apply_slice_to_shape"
    output_name = outputs[0]
    assert isinstance(axes, list), f"Unexpected type {type(axes)} for axes"
    assert all_int(axes), f"Expected only integer axis but got {axes}"
    assert len(axes) == len(
        index_slice  # type: ignore
    ), f"Length mismatch {len(axes)} != {len(index_slice)}"  # type: ignore

    # axes
    aaxes = np.array(axes, dtype=np.int64)
    axes_name = g.unique_name(f"{output_name}_axis")
    g.make_initializer(axes_name, aaxes, source="_getitem_slice.axis.1")

    shape_value = None
    if g.has_shape(input_name):
        shape_value = g.get_shape(input_name)

    starts = []
    ends = []
    steps = []
    shape_name = None
    end_name = None
    concat = False
    squeeze_axes: List[int] = []
    for axis_, aslice in zip(axes, index_slice):  # type: ignore
        axis = axis_
        if isinstance(aslice, int):
            # integer
            starts.append(aslice)
            ends.append(aslice + 1)
            steps.append(1)
            continue

        assert isinstance(
            aslice, (slice, int, g.torch.fx.Node)
        ), f"Unexpected type {type(aslice)} ({aslice}) in {index_slice}"
        if isinstance(aslice, g.torch.fx.Node):
            # Dynamic integer index: treat as i:i+1 slice and squeeze the axis afterwards.
            starts.append(aslice)
            slice_end_name = g.unique_name(f"{output_name}_slice_end_{axis_}")
            g.make_node(
                "Add",
                [
                    aslice.name,
                    g.make_initializer(
                        "", np.array(1, dtype=np.int64), source="_getitem_slice.int_end"
                    ),
                ],
                [slice_end_name],
                name=f"{name}_int_end",
                sts=None,
            )
            ends.append(slice_end_name)
            steps.append(1)
            concat = True
            squeeze_axes.append(axis_)
            continue

        starts.append(aslice.start or 0)

        if aslice.stop is None:
            if shape_value is None or not isinstance(shape_value[axis], int):
                if shape_name is None:
                    shape_name = g.unique_name(f"{output_name}_shape")
                    g.make_node("Shape", [input_name], [shape_name], name=f"{name}A")

                aaxis = np.array([axis], dtype=np.int64)
                axis_name = g.unique_name(f"{output_name}_axis_{axis}")
                g.make_initializer(axis_name, aaxis, source="_getitem_slice.axis.2")

                end_name = g.unique_name(f"{output_name}_end")
                g.make_node(
                    "GatherElements",
                    [shape_name, axis_name],
                    [end_name],
                    name=f"{name}B",
                    sts=None,
                )
                ends.append(end_name)
                concat = True
            else:
                ends.append(shape_value[axis])
        else:
            vstop = aslice.stop.name if hasattr(aslice.stop, "name") else aslice.stop
            concat |= isinstance(vstop, str)
            ends.append(vstop)

        steps.append(aslice.step if aslice.step else 1)

    # if concat: one end is coming from a shape
    if concat:
        iends = []
        for i in ends:
            if isinstance(i, str):
                if g.get_rank(i) == 0:
                    iends.append(
                        g.op.UnsqueezeAnyOpset(i, np.array([0], dtype=np.int64), name=f"{name}C")
                    )
                else:
                    assert (
                        g.get_rank(i) == 1
                    ), f"Unexpected rank={g.get_rank(i)} for {i!r}{g.get_debug_msg()}"
                    iends.append(i)
            else:
                assert isinstance(i, int), f"Unexpected value for end={i!r}{g.get_debug_msg()}"
                iends.append(np.array([i], dtype=np.int64))
        if len(iends) > 1:
            conc_ends = g.op.Concat(*iends, axis=0, name=f"{name}D")
        else:
            conc_ends = g.op.Identity(iends[0], name=f"{name}E")
    else:
        assert all_int(
            ends
        ), f"Unexpected value for ends={ends}: {[type(_) for _ in ends]}{g.get_debug_msg()}"
        conc_ends = g.make_initializer(
            "", np.array(ends, dtype=np.int64), source="_getitem_slice.1"
        )

    assert all_int(steps), (
        f"Not implemented for steps={steps} (types are "
        f"{[type(c) for c in steps]}){g.get_debug_msg()}"
    )
    if all_int(starts):
        conc_starts = g.make_initializer(
            g.unique_name(f"{output_name}_start"),
            np.array(starts, dtype=np.int64),
            source="_getitem_slice.2",
        )
    else:
        istarts = []
        for i in starts:
            si = i.name if hasattr(i, "name") else i
            if isinstance(si, str):
                if g.get_rank(si) == 0:
                    istarts.append(
                        g.op.UnsqueezeAnyOpset(si, np.array([0], dtype=np.int64), name=f"{name}C")
                    )
                else:
                    assert (
                        g.get_rank(si) == 1
                    ), f"Unexpected rank={g.get_rank(i)} for {si!r}{g.get_debug_msg()}"
                    istarts.append(si)
            else:
                assert isinstance(si, int), f"Unexpected value for end={si!r}{g.get_debug_msg()}"
                istarts.append(np.array([si], dtype=np.int64))
        if len(istarts) > 1:
            conc_starts = g.op.Concat(*istarts, axis=0, name=f"{name}SD")
        else:
            conc_starts = g.op.Identity(istarts[0], name=f"{name}SE")

    inputs: List[str] = [  # type: ignore
        input_name,
        conc_starts,
        conc_ends,
        axes_name,
        g.make_initializer(
            g.unique_name(f"{output_name}_step"),
            np.array(steps, dtype=np.int64),
            source="_getitem_slice.3",
        ),
    ]

    if expand_axes and squeeze_axes:
        sliced = g.make_node("Slice", inputs, name=f"{name}F")
        unsqueezed = g.op.UnsqueezeAnyOpset(
            sliced, np.array(expand_axes, dtype=np.int64), name=f"{name}F2"
        )
        res = g.op.SqueezeAnyOpset(
            unsqueezed,
            np.array(squeeze_axes, dtype=np.int64),
            outputs=[output_name],
            name=f"{name}H",
        )
    elif expand_axes:
        sliced = g.make_node("Slice", inputs, name=f"{name}F")
        res = g.op.UnsqueezeAnyOpset(
            sliced, np.array(expand_axes, dtype=np.int64), outputs=[output_name], name=f"{name}F"
        )
    elif squeeze_axes:
        slice_name = g.unique_name(f"{output_name}_sliced")
        g.make_node("Slice", inputs, [slice_name], name=f"{name}G_sq")
        res = g.op.SqueezeAnyOpset(
            slice_name,
            np.array(squeeze_axes, dtype=np.int64),
            outputs=[output_name],
            name=f"{name}H_sq",
        )
    else:
        res = g.make_node("Slice", inputs, [output_name], name=f"{name}G")
    if not sts:
        dtype = g.get_type(inputs[0])
        g.set_type(output_name, dtype)
        if not concat and g.has_shape(inputs[0]):
            shape = g.get_shape(inputs[0])
            new_shape = g._apply_slice_to_shape(
                shape, index_slice, axes=axes, expand_axes=expand_axes
            )
            assert not g.has_shape(output_name) or new_shape == g.get_shape(output_name), (
                f"Shape for node {output_name!r} is already set to "
                f"{g.get_shape(output_name)} with type "
                f"{g.get_type(output_name)} (expecting {dtype}) "
                f"new_shape={new_shape}, shape={shape}, index_slice={index_slice}, "
                f"axes={axes}, expand_axes={expand_axes}"
                f"{g.get_debug_msg()}"
            )
            g.set_shape(output_name, new_shape)
        elif squeeze_axes and g.has_rank(inputs[0]):  # type: ignore
            # expand_axes is empty here (handled by the first branch above)
            g.set_rank(output_name, g.get_rank(inputs[0]) - len(squeeze_axes))
        elif expand_axes:
            g.set_rank(output_name, g.get_rank(inputs[0]) + len(expand_axes))
    return res


def getitem(  # noqa: F821
    g: GraphBuilderTorchProtocol,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    node: "torch.fx.Node",  # type: ignore # noqa: F821
):
    """Converts a ``getitem`` (``something[...]``) node to ONNX.

    The index may be another variable (a :class:`torch.fx.Node`), an integer,
    a :class:`slice`, a :class:`tuple`, or a list.

    :param g: the graph builder
    :param sts: known shapes and types; when ``None`` the function sets type and
        shape on the output itself
    :param outputs: list of output tensor names; ``outputs[0]`` is the result
    :param node: the :class:`torch.fx.Node` representing the subscript operation
    :returns: name of the result tensor (or ONNX node)
    """
    assert hasattr(g, "torch"), "torch module is added as an attribute to avoid import"
    args = node.args
    assert len(args) == 2
    node_output, index = args
    result_name = node_output.name
    val = node.meta.get("val", None)
    if val is not None:
        if isinstance(val, g.torch.Tensor):
            shape = val.shape
            dtype = _get_type(val.dtype)
            # the graph could be new if a function produces results
            # depending on the result values
            t_shape = tuple(shape)
            _getitem_verify_new_shape(g, sts, outputs, shape)
            # Let's set the shape if not null shape
            if len(t_shape) > 1 and not any(i == 0 for i in t_shape if isinstance(i, int)):
                g.set_shape(outputs[0], t_shape, allow_zero=all_int(t_shape) and t_shape == (0,))
            else:
                g.set_rank(outputs[0], len(t_shape))
            # When accessing a named tuple element, the ONNX converter may
            # have chosen a different dtype for that element (e.g., float32 for
            # the rstd output of _fused_rms_norm for numerical stability). Use the
            # already-established ONNX type to stay consistent with the Identity
            # node that will propagate the source type to this output.
            if isinstance(index, int):
                _name_index = f"{result_name}#{index}"
                if g.has_type(_name_index):
                    dtype = g.get_type(_name_index)
            g.set_type(outputs[0], dtype)
            g.set_device(outputs[0], val.get_device())
        elif isinstance(val, g.torch.SymInt):
            g.set_shape(outputs[0], tuple())
            g.set_type(outputs[0], TensorProto.INT64)
            g.set_device(outputs[0], -1)
        else:
            raise TypeError(
                f"Unexpected type {type(val)} in node {node!r}"
                f"\n{g.pretty_text(add_fx_graph=True)}"
            )

    if hasattr(index, "name"):
        # A dynamic index (torch.fx.Node)
        res = g.make_node("Gather", [result_name, index.name], [outputs[0]], name="getitemA")
        if not sts:
            g.set_type(outputs[0], g.get_type(result_name))
            g.set_rank(outputs[0], g.get_rank(result_name) + g.get_rank(index.name) - 1)
        return res

    if isinstance(index, int):
        name_index = f"{result_name}#{index}"
        if g.has_name(name_index):
            # The user wants to get a tensor from a tuple of tensors.
            return g.make_node("Identity", [name_index], [outputs[0]], name="getitemB_tuple")
        # The user means to access the first element of a tensor or a sequence.
        if g.is_sequence(result_name):
            # A sequence
            tpos = g.make_initializer("", np.array(index, dtype=np.int64), source="getitem.1")
            res = g.make_node(
                "SequenceAt", [result_name, tpos], [outputs[0]], name="getitemB_tuple"
            )
            if not sts:
                info = g.get_sequence(result_name)
                dtype = info["dtype"]
                if isinstance(dtype, tuple):
                    dtype = dtype[index]
                g.set_type(res, dtype)  # type: ignore
                if info["shapes"] is not None:
                    g.set_shape(res, info["shapes"][min(index, len(info["shapes"]) - 1)])  # type: ignore
                elif info["ranks"] is not None:
                    if isinstance(info["ranks"], int):
                        g.set_rank(res, info["ranks"])  # type: ignore
                    else:
                        g.set_rank(res, info["ranks"][min(index, len(info["ranks"]) - 1)])  # type: ignore
            return res
        else:
            # A tensor.
            res = g.op.SqueezeAnyOpset(
                g.op.Gather(
                    result_name, np.array([index], dtype=np.int64), name="getitemB_index"
                ),
                np.array([0], dtype=np.int64),
                name="getitemB_index",
                outputs=[outputs[0]],
            )
            if not sts:
                if g.has_type(result_name):
                    g.set_type(outputs[0], g.get_type(result_name))
                if g.has_device(result_name):
                    g.set_device(outputs[0], g.get_device(result_name))
                if g.has_shape(result_name):
                    g.set_shape(outputs[0], g.get_shape(result_name)[1:])
                elif g.has_rank(result_name):
                    g.set_rank(outputs[0], g.get_rank(result_name) - 1)
            return res

    if isinstance(index, slice):
        return _getitem_slice(
            g,
            sts,
            outputs,
            node_output.name,
            [index],  # type: ignore
            axes=[0],
            expand_axes=[],
            name="_getitem_slice1",
        )

    assert hasattr(g, "torch"), "torch module is added as an attribute to avoid import"
    if isinstance(index, g.torch.fx.immutable_collections.immutable_list):
        # something like x[[0, 2]]
        if all_int(index):
            # something like x[[0, 1]]
            axes = [0]
            return _aten_tensor_int1(
                g,  # type: ignore
                sts,
                outputs,
                node_output.name,
                index,
                axes=axes,
                expand_axes=[],
                name="_getitem_int1a",
            )

    if isinstance(index, tuple):
        assert hasattr(g, "torch"), "torch module is added as an attribute to avoid import"
        if all(isinstance(x, (slice, g.torch.fx.Node)) for x in index):
            return _getitem_slice(
                g,
                sts,
                outputs,
                node_output.name,
                list(index),  # type: ignore
                axes=list(range(len(index))),
                expand_axes=[],
                name="_getitem_slicen",
            )

        if all(x is Ellipsis or x is None or isinstance(x, slice) for x in index):
            # something like x[3:4]
            axes = []
            slices = []
            expand_axes = []
            ellipsis = False
            true_slice = False
            for i, ind in enumerate(index):
                if ind is Ellipsis:
                    assert not ellipsis, f"Second (...) found in index={index}"
                    ellipsis = True
                    continue
                if ind is None:
                    assert (
                        not ellipsis
                    ), f"An axis cannot be inserted after (...) in index={index}"
                    expand_axes.append(i)
                    continue
                axes.append(((i - len(index)) if ellipsis else i) - len(expand_axes))
                if (
                    not isinstance(ind, slice)
                    or ind.start is not None
                    or ind.stop is not None
                    or ind.step is not None
                ):
                    true_slice = True
                slices.append(ind)
            if true_slice:
                return _getitem_slice(
                    g,
                    sts,
                    outputs,
                    node_output.name,
                    slices,  # type: ignore
                    axes=axes,
                    expand_axes=expand_axes,
                    name="_getitem_slice2",
                )
            # It is just a node unsqueeze.
            res = g.op.UnsqueezeAnyOpset(
                str(node.args[0]),
                np.array(expand_axes, dtype=np.int64),
                name="getitem_unsqueeze",
                outputs=[outputs[0]],
            )
            return res

        raise RuntimeError(
            f"getitem: unexpected tuple {tuple(type(x) for x in index)} "
            f"for index={index}, node={node}, args={args}, val={val}, "
            f"types={string_type(args)}{g.get_debug_msg()}"
        )

    raise RuntimeError(
        f"getitem: unexpected type {type(index)} for index={index}, "
        f"node={node}, args={args}, val={val}, "
        f"types={string_type(args)}{g.get_debug_msg()}"
    )

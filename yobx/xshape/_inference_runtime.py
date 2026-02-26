import time
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import onnx
from onnx_diagnostic.reference import ExtendedReferenceEvaluator
from ..helpers import string_type
from ..helpers.onnx_helper import tensor_dtype_to_np_dtype
from ..helpers.torch_helper import onnx_dtype_to_torch_dtype
from ._shape_helper import (
    all_int,
    _reshape_shape,
    is_static_shape,
    reshape_implementation_with_zero,
)
from .shape_type_compute import set_shape_type_op_any, set_shape_type_custom


class _InferenceRuntime:
    """Sets shape and type."""

    def make_dimension_name_if_necessary(
        self, a: Union[int, str], b: Union[int, str], op: str
    ) -> str:
        """Creates a new dimension."""
        if op == "^":
            # very simple trick for the time being
            if a == b:
                # pyrefly: ignore [bad-return]
                return a  # type: ignore[return-value]
            if isinstance(a, str) and a.endswith(f"^{b}"):
                return a
            if isinstance(b, str) and b.startswith(f"{a}^"):
                return b
        if isinstance(a, str) and set(a) & set("+/*-^"):
            a = f"({a})"
        if isinstance(b, str) and set(b) & set("+/*-^"):
            b = f"({b})"
        return f"{a}{op}{b}"

    def _make_node_set_type_shape(self, node: onnx.NodeProto, exc: bool = False):
        """Updates shapes for a node."""
        update = self._make_node_set_type_shape_constant(node, {})
        if update is None:
            if node.domain == "":
                node.doc_string += "#Io1"
                # pyrefly: ignore [bad-argument-type]
                update = set_shape_type_op_any(self, node, exc=exc)  # type: ignore[arg-type]
            else:
                # Missing type means it is probably coming from an inlined function.
                node.doc_string += (
                    # pyrefly: ignore [missing-attribute]
                    "#Io3" if node.input and not self.has_type(node.input[0]) else "#Io2"  # type: ignore[attr-defined]
                )
                # pyrefly: ignore [bad-argument-type]
                update = set_shape_type_custom(self, node, exc=exc)  # type: ignore[arg-type]
        if update:
            # pyrefly: ignore [missing-attribute]
            self._calls.append(  # type: ignore[attr-defined]
                (node.name, node.domain, node.op_type, node.input, node.output, update)
            )
        # pyrefly: ignore [missing-attribute]
        assert update is not None or not self._debug_shape_missing, (  # type: ignore[attr-defined]
            f"Shape missing for node type {node.op_type!r}, inputs={node.input}, "
            # pyrefly: ignore [missing-attribute]
            f"outputs={node.output}\n----\n{node}\n{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        return update

    def update_node_constant(self, name: str, node: onnx.NodeProto) -> bool:
        """Updates a constant NodeProto."""
        assert isinstance(name, str), f"Unexpected type {type(name)} for name"
        assert node is None or isinstance(
            node, onnx.NodeProto
        ), f"Unexpected type {type(node)} for name={name!r}"
        if node is not None and node.op_type.startswith("Random"):
            return False
        # pyrefly: ignore [missing-attribute]
        if self.verbose > 2:  # type: ignore[attr-defined]
            print(
                # pyrefly: ignore [missing-attribute]
                f"[GraphBuilder-{self._hash()}.update_node_constant] new constant "  # type: ignore[attr-defined]
                f"{name!r}, node={None if node is None else node.op_type}"
            )
        assert (
            node is None
            or node.op_type == "Shape"
            # pyrefly: ignore [missing-attribute]
            or all(self.is_constant(i) for i in node.input if i not in {"", None, "None"})  # type: ignore[attr-defined]
        ), (
            # pyrefly: ignore [missing-attribute]
            f"Output {name!r} is constant (node={self.pretty_node(node)}) "  # type: ignore[attr-defined]
            f"only if every input from {node.input} is constant "
            # pyrefly: ignore [missing-attribute]
            f"but constants={[self.is_constant(i) for i in node.input]}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        # pyrefly: ignore [missing-attribute]
        self.constants_[name] = node  # type: ignore[attr-defined]
        return True

    def _make_node_set_type_shape_constant(
        self, node: onnx.NodeProto, sts: Optional[Dict[str, Any]]
    ):
        if node.domain != "":
            return

        # pyrefly: ignore [missing-attribute]
        if all(self.is_constant(i) for i in node.input):  # type: ignore[attr-defined]
            for o in node.output:
                self.update_node_constant(o, node)

        if node.op_type == "Constant":
            assert (
                len(node.attribute) == 0
                or node.attribute[0].name != "value"
                or node.attribute[0].type != onnx.AttributeProto.GRAPH
            ), f"{node}"
            if len(node.attribute) == 1 and node.attribute[0].name == "value":
                size = np.prod(node.attribute[0].t.dims)
            else:
                size = len(node.SerializeToString())
            # pyrefly: ignore [missing-attribute]
            assert size < self.optimization_options.constant_size, (  # type: ignore[attr-defined]
                f"A node Constant holds a tensor bigger than "
                # pyrefly: ignore [missing-attribute]
                f"the constant: {size} >= {self.optimization_options.constant_size}."  # type: ignore[attr-defined]
            )
            k = node.output[0]
            self.update_node_constant(k, node)
            node.doc_string += ":constant-3:"
            # pyrefly: ignore [missing-attribute]
            shape = self._get_tensor_shape(node)  # type: ignore[attr-defined]
            # pyrefly: ignore [missing-attribute]
            dtype = self._get_tensor_type(node)  # type: ignore[attr-defined]
            # pyrefly: ignore [missing-attribute]
            self.set_shape(k, shape)  # type: ignore[attr-defined]
            # pyrefly: ignore [missing-attribute]
            self.set_type(k, dtype)  # type: ignore[attr-defined]
            # pyrefly: ignore [missing-attribute]
            if self.verbose > 2 or np.prod(shape) > 100:  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                print(f"[GraphBuilder-{self._hash()}.5.make_node] {k}[{dtype}: {shape}]")  # type: ignore[attr-defined]
            return shape
        elif node.op_type == "ConstantOfShape":
            if len(node.attribute) == 1 and node.attribute[0].name == "value":
                itype = node.attribute[0].t.data_type
            else:
                itype = onnx.TensorProto.FLOAT
            # pyrefly: ignore [missing-attribute]
            self.set_type(node.output[0], itype)  # type: ignore[attr-defined]
            # pyrefly: ignore [missing-attribute]
            if self.is_constant(node.input[0]):  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                value = self.get_constant(  # type: ignore[attr-defined]
                    node.input[0], computed_value=True, as_shape=True, exc=False
                )
                if value is not None:
                    # This is needed when concatenating caches.
                    # pyrefly: ignore [missing-attribute]
                    self.set_shape(node.output[0], value, allow_zero=True)  # type: ignore[attr-defined]
                    node.doc_string += ":constant-9:"
                    return value
            # pyrefly: ignore [missing-attribute]
            vs = self.value_as_shape(node.input[0])  # type: ignore[attr-defined]
            if vs is not None:
                # pyrefly: ignore [missing-attribute]
                self.set_shape(node.output[0], vs, allow_zero=True)  # type: ignore[attr-defined]
                return vs
            # pyrefly: ignore [missing-attribute]
            if self.has_shape(node.input[0]):  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                shape = self.get_shape(node.input[0])  # type: ignore[attr-defined]
                if is_static_shape(shape):
                    # pyrefly: ignore [missing-attribute]
                    self.set_rank(node.output[0], shape[0])  # type: ignore[attr-defined]
                    return True
        elif node.op_type == "Identity":
            shape = None
            # pyrefly: ignore [missing-attribute]
            if self.has_shape(node.input[0]):  # type: ignore[attr-defined]
                # allow_zero is True but if it fails here, it means it did not fail
                # before when it should be.
                # pyrefly: ignore [missing-attribute]
                shape = self.get_shape(node.input[0])  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                self.set_shape(node.output[0], shape, allow_zero=True)  # type: ignore[attr-defined]
            # pyrefly: ignore [missing-attribute]
            elif self.has_rank(node.input[0]):  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                self.set_rank(node.output[0], self.get_rank(node.input[0]))  # type: ignore[attr-defined]
            # pyrefly: ignore [missing-attribute]
            if self.has_type(node.input[0]):  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                self.set_type(node.output[0], self.get_type(node.input[0]))  # type: ignore[attr-defined]
            # pyrefly: ignore [missing-attribute]
            if self.has_device(node.input[0]) and not self.has_device(node.output[0]):  # type: ignore[attr-defined]
                # Identity node are tricky. The onnx conversion usually ignores this.
                # .to(device) becomes an identity node, therefore, a device could already be
                # defined for the output (during optimization).
                # pyrefly: ignore [missing-attribute]
                self.set_device(node.output[0], self.get_device(node.input[0]))  # type: ignore[attr-defined]
            # pyrefly: ignore [missing-attribute]
            if self.is_constant(node.input[0]):  # type: ignore[attr-defined]
                self.update_node_constant(node.output[0], node)
                node.doc_string += ":constant-4:"
            return shape
        elif node.op_type == "Expand":
            # pyrefly: ignore [missing-attribute]
            if self.has_type(node.input[0]):  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                self.set_type(node.output[0], self.get_type(node.input[0]))  # type: ignore[attr-defined]
            # pyrefly: ignore [missing-attribute]
            if self.has_device(node.input[0]):  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                self.set_device(node.output[0], self.get_device(node.input[0]))  # type: ignore[attr-defined]
            if (
                # pyrefly: ignore [missing-attribute]
                self.has_shape(node.input[0])  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                and is_static_shape(self.get_shape(node.input[0]))  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                and self.is_constant(node.input[1])  # type: ignore[attr-defined]
            ):
                cst, _ = self.compute_constant(node.input[1], exc=False, only_array=True)
                if cst is not None:
                    # pyrefly: ignore [missing-attribute]
                    assert not isinstance(cst, self.torch._subclasses.fake_tensor.FakeTensor), (  # type: ignore[attr-defined]
                        f"self.compute_constant returns a FakeTensor for {node.input[1]!r}"
                        # pyrefly: ignore [missing-attribute]
                        f"\n{self.pretty_text()}"  # type: ignore[attr-defined]
                    )
                    assert (
                        # pyrefly: ignore [missing-attribute]
                        not self.has_rank(node.input[1]) or self.get_rank(node.input[1]) == 1  # type: ignore[attr-defined]
                    ), (
                        # pyrefly: ignore [missing-attribute]
                        f"Unexpected rank {self.get_rank(node.input[1])} for {node.input[1]!r}"  # type: ignore[attr-defined]
                        # pyrefly: ignore [missing-attribute]
                        f"{self.get_debug_msg()}"  # type: ignore[attr-defined]
                    )
                    # pyrefly: ignore [missing-attribute]
                    with self.maybe_disable_fake_tensor_mode():  # type: ignore[attr-defined]
                        assert len(cst.shape) == 1 and cst[-1] > 0, (
                            f"Unexpected shape {cst.shape} "
                            f"for computed constant {node.input[1]!r}, "
                            # pyrefly: ignore [missing-attribute]
                            f"input={node.input}, cst={cst}{self.get_debug_msg()}"  # type: ignore[attr-defined]
                        )
                        # pyrefly: ignore [missing-attribute]
                        shape = self.get_shape(node.input[0])  # type: ignore[attr-defined]
                        new_shape = tuple(int(i) for i in cst)
                    if len(shape) < len(new_shape):
                        shape = (1,) * (len(new_shape) - len(shape)) + shape
                    new_shape = tuple(max(i, j) for i, j in zip(shape, new_shape))
                    # pyrefly: ignore [missing-attribute]
                    self.set_shape(node.output[0], new_shape, allow_zero=0 in shape)  # type: ignore[attr-defined]
                    return new_shape
        elif node.op_type == "Reshape":
            # pyrefly: ignore [missing-attribute]
            if self.has_type(node.input[0]):  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                self.set_type(node.output[0], self.get_type(node.input[0]))  # type: ignore[attr-defined]
            # pyrefly: ignore [missing-attribute]
            if self.has_device(node.input[0]):  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                self.set_device(node.output[0], self.get_device(node.input[0]))  # type: ignore[attr-defined]
            # pyrefly: ignore [missing-attribute]
            if self.is_constant(node.input[1]):  # type: ignore[attr-defined]
                cst, _ = self.compute_constant(
                    node.input[1], exc=False, only_array=True, allow_empty=True
                )
                if cst is not None:
                    shape_cst = tuple(int(i) for i in cst)
                    if 0 in shape_cst:
                        # pyrefly: ignore [missing-attribute]
                        if self.has_shape(node.input[0]):  # type: ignore[attr-defined]
                            # pyrefly: ignore [missing-attribute]
                            sh = self.get_shape(node.input[0])  # type: ignore[attr-defined]
                            shape_cst_last_zero = shape_cst[
                                : len(shape_cst) - 1 - shape_cst[::-1].index(0) + 1
                            ]
                            assert len(sh) >= len(shape_cst_last_zero), (
                                f"Shape discrepancies for name={node.input[0]!r} "
                                f"node.name={node.name!r} "
                                f"between sh={sh} and shape_cst={shape_cst}, "
                                f"shape_cst_last_zero={shape_cst_last_zero}"
                                # pyrefly: ignore [missing-attribute]
                                f"\ncst={cst}{self.get_debug_msg()}"  # type: ignore[attr-defined]
                            )
                            shape_cst = tuple(
                                [
                                    shape_cst[i] if shape_cst[i] != 0 else sh[i]
                                    for i in range(len(shape_cst))
                                ]
                            )
                        else:
                            shape_cst = None  # type: ignore[assignment]
                    if shape_cst is not None:
                        if -1 in shape_cst:
                            # pyrefly: ignore [missing-attribute]
                            if self.has_shape(node.input[0]):  # type: ignore[attr-defined]
                                # pyrefly: ignore [missing-attribute]
                                sh = self.get_shape(node.input[0])  # type: ignore[attr-defined]
                                if is_static_shape(sh):
                                    new_shape = _reshape_shape(sh, shape_cst)
                                    # pyrefly: ignore [missing-attribute]
                                    self.set_shape(node.output[0], new_shape, allow_zero=0 in sh)  # type: ignore[attr-defined]
                                    node.doc_string += ":constant-7a:"
                                    return new_shape
                        else:
                            # pyrefly: ignore [missing-attribute]
                            self.set_shape(node.output[0], shape_cst)  # type: ignore[attr-defined]
                            node.doc_string += ":constant-7b:"
                            return shape_cst
        elif node.op_type == "Shape":
            ret_shape = None
            # pyrefly: ignore [missing-attribute]
            self.set_type(node.output[0], onnx.TensorProto.INT64)  # type: ignore[attr-defined]
            # pyrefly: ignore [missing-attribute]
            self.set_device(node.output[0], -1)  # type: ignore[attr-defined]
            # pyrefly: ignore [missing-attribute]
            if self.has_rank(node.input[0]):  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                rk = self.get_rank(node.input[0])  # type: ignore[attr-defined]
                if len(node.attribute) == 0:
                    # pyrefly: ignore [missing-attribute]
                    self.set_shape(node.output[0], (rk,))  # type: ignore[attr-defined]
                else:
                    # pyrefly: ignore [missing-attribute]
                    start = self.get_attribute_with_default(node, "start", 0)  # type: ignore[attr-defined]
                    if start < 0:
                        start += rk
                    # pyrefly: ignore [missing-attribute]
                    end = self.get_attribute_with_default(node, "end", rk)  # type: ignore[attr-defined]
                    if end < 0:
                        end += rk
                    # pyrefly: ignore [missing-attribute]
                    self.set_shape(node.output[0], (end - start,))  # type: ignore[attr-defined]
                    ret_shape = (end - start,)
            elif node.attribute:
                # pyrefly: ignore [missing-attribute]
                start = self.get_attribute_with_default(node, "start", 0)  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                end = self.get_attribute_with_default(node, "end", None)  # type: ignore[attr-defined]
                if end is not None and end - start > 0:
                    # pyrefly: ignore [missing-attribute]
                    self.set_shape(node.output[0], (end - start,))  # type: ignore[attr-defined]
                else:
                    # pyrefly: ignore [missing-attribute]
                    self.set_rank(node.output[0], 1)  # type: ignore[attr-defined]
                    # pyrefly: ignore [missing-attribute]
                    assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
                        f"Unable to compute the shape of this shape: "
                        # pyrefly: ignore [missing-attribute]
                        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
                    )
            else:
                # pyrefly: ignore [missing-attribute]
                self.set_rank(node.output[0], 1)  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                assert not self._debug_shape_missing, (  # type: ignore[attr-defined]
                    f"Unable to compute the shape of this shape: "
                    # pyrefly: ignore [missing-attribute]
                    f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
                )
            # pyrefly: ignore [missing-attribute]
            if self.is_constant(node.input[0]) or (  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                self.has_shape(node.input[0]) and all_int(self.get_shape(node.input[0]))  # type: ignore[attr-defined]
            ):
                self.update_node_constant(node.output[0], node)
                node.doc_string += ":constant-2:"
            return ret_shape
        elif node.op_type == "Size":
            # pyrefly: ignore [missing-attribute]
            self.set_type(node.output[0], onnx.TensorProto.INT64)  # type: ignore[attr-defined]
            # pyrefly: ignore [missing-attribute]
            self.set_device(node.output[0], -1)  # type: ignore[attr-defined]
            # pyrefly: ignore [missing-attribute]
            self.set_shape(node.output[0], tuple())  # type: ignore[attr-defined]
            # pyrefly: ignore [missing-attribute]
            if self.is_constant(node.input[0]):  # type: ignore[attr-defined]
                self.update_node_constant(node.output[0], node)
                node.doc_string += ":constant-2s:"
            return tuple()
        elif not sts:
            if node.op_type == "GatherElements":
                # pyrefly: ignore [missing-attribute]
                if self.has_type(node.input[0]):  # type: ignore[attr-defined]
                    # pyrefly: ignore [missing-attribute]
                    self.set_type(node.output[0], self.get_type(node.input[0]))  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                if self.has_device(node.input[0]):  # type: ignore[attr-defined]
                    # pyrefly: ignore [missing-attribute]
                    self.set_device(node.output[0], self.get_device(node.input[0]))  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                if self.has_shape(node.input[1]):  # type: ignore[attr-defined]
                    # pyrefly: ignore [missing-attribute]
                    self.set_shape(node.output[0], self.get_shape(node.input[1]))  # type: ignore[attr-defined]
                    # pyrefly: ignore [missing-attribute]
                    return self.get_shape(node.input[1])  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                elif self.has_rank(node.input[1]):  # type: ignore[attr-defined]
                    # pyrefly: ignore [missing-attribute]
                    self.set_rank(node.output[0], self.get_rank(node.input[1]))  # type: ignore[attr-defined]

    def compute_constant(
        self, name: str, exc: bool = True, only_array: bool = False, allow_empty: bool = False
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """
        Computes a constant.

        :param name: constant name
        :param exc: raises an exception if any failure
        :param only_array: do not return TensorProto
        :param allow_empty: allow empty result
        :return: constant

        If returns None if the constant is a FakeTensor.
        """
        # pyrefly: ignore [missing-attribute]
        assert self.is_constant(name), f"Name {name!r} is not a constant"  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        v = self.constants_[name]  # type: ignore[attr-defined]
        # It should not be None but a node as it is not an initializer.
        if isinstance(v, onnx.TensorProto):
            # pyrefly: ignore [missing-attribute]
            return self.get_constant(name, computed_value=True, exc=exc), None  # type: ignore[attr-defined]

        assert isinstance(
            v, onnx.NodeProto
        ), f"Unexpected type {type(v)} for constant name={name!r}"
        # pyrefly: ignore [missing-attribute]
        if self._debug_get_constant:  # type: ignore[attr-defined]
            # pyrefly: ignore [missing-attribute]
            print(f"[GraphBuilder-{self._hash()}.compute_constant] {self.pretty_node(v)}")  # type: ignore[attr-defined]

        if v.op_type == "Shape":
            # pyrefly: ignore [missing-attribute]
            if not self.has_shape(v.input[0]):  # type: ignore[attr-defined]
                # We stop.
                # pyrefly: ignore [missing-attribute]
                assert not self._debug_constant_folding, (  # type: ignore[attr-defined]
                    f"Unable to compute constant because {v.input[0]!r} has no shape"
                    # pyrefly: ignore [missing-attribute]
                    f"in node {self.pretty_node(v)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
                )
                return None, None
            # pyrefly: ignore [missing-attribute]
            shape = self.get_shape(v.input[0])  # type: ignore[attr-defined]
            if is_static_shape(shape):
                if v.attribute:
                    start = 0
                    end = None
                    for att in v.attribute:
                        if att.name == "start":
                            start = att.i
                        elif att.name == "end":
                            end = att.i
                    shape = shape[start:] if end is None else shape[start:end]
                    if self._debug_get_constant:  # type: ignore[attr-defined]
                        print(
                            # pyrefly: ignore [missing-attribute]
                            f"[GraphBuilder-{self._hash()}.compute_constant]     - SHAPE "  # type: ignore[attr-defined]
                            f"{name}: {shape}? start={start}, end={end}"
                        )
                elif self._debug_get_constant:  # type: ignore[attr-defined]
                    print(
                        # pyrefly: ignore [missing-attribute]
                        f"[GraphBuilder-{self._hash()}.compute_constant] "  # type: ignore[attr-defined]
                        f"    - SHAPE {name}: {shape}?"
                    )
                return np.array(shape, dtype=np.int64), {
                    # pyrefly: ignore [missing-attribute]
                    v.input[0]: self.ShapeConstant(v.input[0], shape, v)  # type: ignore[attr-defined]
                }

            # pyrefly: ignore [missing-attribute]
            if not self.is_constant(v.input[0]):  # type: ignore[attr-defined]
                # One exception here as the input maybe not
                # be constant but the shape may be known.
                assert all_int(shape), (
                    f"Shape must be static ({shape}) if shape is constant in {v} in "
                    # pyrefly: ignore [missing-attribute]
                    f"{self.pretty_node(v)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
                )
                # pyrefly: ignore [missing-attribute]
                with self.maybe_disable_fake_tensor_mode():  # type: ignore[attr-defined]
                    # pyrefly: ignore [missing-attribute]
                    output = self._apply_shape_on_shape(v, shape)  # type: ignore[attr-defined]
                    # pyrefly: ignore [missing-attribute]
                    if isinstance(output[0], self.torch.Tensor):  # type: ignore[attr-defined]
                        # We convert the tensor into numpy array,
                        # it is a small shape anyway so the FakeMode
                        # does not come up as an issue.
                        output = [output[0].detach().cpu().numpy()]
                    if self._debug_get_constant:  # type: ignore[attr-defined]
                        print(
                            # pyrefly: ignore [missing-attribute]
                            f"[GraphBuilder-{self._hash()}.compute_constant]     - A "  # type: ignore[attr-defined]
                            # pyrefly: ignore [missing-attribute]
                            f"{name}: {self.pretty_tensor(output[0])}"  # type: ignore[attr-defined]
                        )
                    # pyrefly: ignore [missing-attribute]
                    return output[0], {v.input[0]: self.ShapeConstant(v.input[0], shape, v)}  # type: ignore[attr-defined]
            # pyrefly: ignore [missing-attribute]
            assert not self._debug_constant_folding, (  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                f"Unable to compute constant for node {self.pretty_node(v)}"  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                f"{self.get_debug_msg()}"  # type: ignore[attr-defined]
            )
            return None, None

        # pyrefly: ignore [missing-attribute]
        feeds = {i: self.get_constant(i, exc=exc, computed_value=True) for i in v.input}  # type: ignore[attr-defined]
        for kval, val in feeds.items():
            if not exc and "FakeTensor" in str(type(val)):
                # pyrefly: ignore [missing-attribute]
                assert not self._debug_constant_folding, (  # type: ignore[attr-defined]
                    # pyrefly: ignore [missing-attribute]
                    f"Unable to compute constant for node {self.pretty_node(v)}"  # type: ignore[attr-defined]
                    # pyrefly: ignore [missing-attribute]
                    f"because a FakeTensor appeared{self.get_debug_msg()}"  # type: ignore[attr-defined]
                )
                return None, None
            assert "FakeTensor" not in str(type(val)), (
                f"FakeTensor {kval!r} cannot be an initializer {type(val)}, "
                f"v.op_type={v.op_type!r}"
                # pyrefly: ignore [missing-attribute]
                f"{self.get_debug_msg()}"  # type: ignore[attr-defined]
            )
            if val is None:
                # pyrefly: ignore [missing-attribute]
                assert not self._debug_constant_folding, (  # type: ignore[attr-defined]
                    # pyrefly: ignore [missing-attribute]
                    f"Unable to compute constant for node {self.pretty_node(v)}"  # type: ignore[attr-defined]
                    # pyrefly: ignore [missing-attribute]
                    f"because val=None{self.get_debug_msg()}"  # type: ignore[attr-defined]
                )
                return None, None

        # pyrefly: ignore [missing-attribute]
        with self.maybe_disable_fake_tensor_mode():  # type: ignore[attr-defined]
            if v.op_type == "Identity":
                # much faster this way
                output = [feeds[v.input[0]]]
            elif v.op_type == "Reshape":
                # much faster this way
                output = [
                    reshape_implementation_with_zero(feeds[v.input[0]], tuple(feeds[v.input[1]]))
                ]
            elif v.op_type in {
                "Add",
                "Div",
                "Mul",
                "Sub",
            }:
                # bypassing onnx.numpy_helper.from_array, too slow
                # pyrefly: ignore [missing-attribute]
                output = self._apply_binary_op(v, feeds)  # type: ignore[attr-defined]
            elif (
                v.op_type == "Pow"
                # pyrefly: ignore [missing-attribute]
                and self.has_type(v.input[0])  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                and self.has_type(v.input[1])  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                and self.get_type(v.input[0]) == self.get_type(v.input[1])  # type: ignore[attr-defined]
            ):
                # pyrefly: ignore [missing-attribute]
                output = self._apply_binary_op(v, feeds)  # type: ignore[attr-defined]
            elif v.op_type in {"Exp", "Log", "Reciprocal", "Sqrt"}:
                # bypassing onnx.numpy_helper.from_array, too slow
                # pyrefly: ignore [missing-attribute]
                output = self._apply_unary_function(v, feeds)  # type: ignore[attr-defined]
            elif hasattr(self, f"_apply_{v.op_type.lower()}"):
                output = getattr(self, f"_apply_{v.op_type.lower()}")(v, feeds)
            elif all(isinstance(v, np.ndarray) for v in feeds.values()):
                # pyrefly: ignore [missing-attribute]
                if v.op_type not in {"Constant", "ConstantOfShape"} and self.main_opset < 18:  # type: ignore[attr-defined]
                    # This functionality is not enabled before that opset.
                    if self._debug_get_constant:  # type: ignore[attr-defined]
                        print(
                            # pyrefly: ignore [missing-attribute]
                            f"[GraphBuilder-{self._hash()}.compute_constant] fails "  # type: ignore[attr-defined]
                            # pyrefly: ignore [missing-attribute]
                            f"because opset={self.main_opset} for name={name!r}, "  # type: ignore[attr-defined]
                            # pyrefly: ignore [missing-attribute]
                            f"node={self.pretty_node(v)}"  # type: ignore[attr-defined]
                        )
                    # pyrefly: ignore [missing-attribute]
                    assert not self._debug_constant_folding, (  # type: ignore[attr-defined]
                        # pyrefly: ignore [missing-attribute]
                        f"Unable to compute constant opset={self.main_opset}<18"  # type: ignore[attr-defined]
                        # pyrefly: ignore [missing-attribute]
                        f"for name={name!r}{self.get_debug_msg()}"  # type: ignore[attr-defined]
                    )
                    return None, None

                # Let's avoid big computation on CPU.
                max_dim = 0
                for _v in feeds.values():
                    max_dim = max(max_dim, np.prod(_v.shape))
                if max_dim >= 2**22:
                    # pyrefly: ignore [missing-attribute]
                    if self.verbose > 1:  # type: ignore[attr-defined]
                        print(
                            # pyrefly: ignore [missing-attribute]
                            f"[GraphBuilder-{self._hash()}.compute_constant] stop computing a "  # type: ignore[attr-defined]
                            f"constant as it may be too big, shapes are "
                            f"{[_.shape for _ in feeds.values()]}"
                        )
                    # pyrefly: ignore [missing-attribute]
                    assert not self._debug_constant_folding, (  # type: ignore[attr-defined]
                        # pyrefly: ignore [missing-attribute]
                        f"Unable to compute constant for node {self.pretty_node(v)}"  # type: ignore[attr-defined]
                        # pyrefly: ignore [missing-attribute]
                        f"because max_dim={max_dim} (shape={_v.shape}){self.get_debug_msg()}"  # type: ignore[attr-defined]
                    )
                    return None, None

                begin = time.perf_counter()
                ref = ExtendedReferenceEvaluator(v)
                try:
                    output = ref.run(None, feeds)
                except (ValueError, TypeError) as e:
                    sf = ", ".join(f"{k}:{v.dtype}:{v.shape}" for k, v in feeds.items())
                    # pyrefly: ignore [missing-attribute]
                    if "warnings" not in self._debug_msg:  # type: ignore[attr-defined]
                        # pyrefly: ignore [missing-attribute]
                        self._debug_msg["warnings"] = []  # type: ignore[attr-defined]
                    sv = str(v).replace("\n", " ")
                    # pyrefly: ignore [missing-attribute]
                    self._debug_msg["warnings"].append(f"Issue with v={sv}, feeds={sf}, e={e}")  # type: ignore[attr-defined]
                    # pyrefly: ignore [missing-attribute]
                    self.time_evaluation_constants_ += time.perf_counter() - begin  # type: ignore[attr-defined]
                    # pyrefly: ignore [missing-attribute]
                    assert not self._debug_constant_folding, (  # type: ignore[attr-defined]
                        # pyrefly: ignore [missing-attribute]
                        f"Unable to compute constant for node {self.pretty_node(v)}"  # type: ignore[attr-defined]
                        # pyrefly: ignore [missing-attribute]
                        f"due to {e}{self.get_debug_msg()}"  # type: ignore[attr-defined]
                    )
                    return None, None

                # pyrefly: ignore [missing-attribute]
                self.time_evaluation_constants_ += time.perf_counter() - begin  # type: ignore[attr-defined]
            else:
                # pyrefly: ignore [missing-attribute]
                assert not self._debug_constant_folding, (  # type: ignore[attr-defined]
                    # pyrefly: ignore [missing-attribute]
                    f"Unable to compute constant for node {self.pretty_node(v)}, "  # type: ignore[attr-defined]
                    f"feeds={string_type(feeds, with_shape=True, with_min_max=True, limit=20)}"
                    # pyrefly: ignore [missing-attribute]
                    f"{self.get_debug_msg()}"  # type: ignore[attr-defined]
                )
                return None, None

            cst = None
            for n, val in zip(v.output, output):
                assert not isinstance(val, tuple), f"Unexpected type {type(val)} for n={n!r}"
                assert "FakeTensor" not in str(type(val)), (
                    f"FakeTensor detected {type(val)} in constant {name!r}, "
                    # pyrefly: ignore [missing-attribute]
                    f"v.op_type={v.op_type!r}{self.get_debug_msg()}"  # type: ignore[attr-defined]
                )
                # pyrefly: ignore [missing-attribute]
                if self.has_type(n):  # type: ignore[attr-defined]
                    # numpy changes the expected type sometimes
                    # (like transpose(x: float36) --> float32)
                    # pyrefly: ignore [missing-attribute]
                    itype = self.get_type(n)  # type: ignore[attr-defined]
                    if hasattr(val, "detach"):
                        val = val.to(onnx_dtype_to_torch_dtype(itype))
                    else:
                        val = val.astype(tensor_dtype_to_np_dtype(itype))
                # pyrefly: ignore [missing-attribute]
                self.constants_computed_[n] = val  # type: ignore[attr-defined]
                if name == n:
                    cst = val

        assert (
            # pyrefly: ignore [missing-attribute]
            len(cst.shape) == 0  # type: ignore[union-attr]
            # pyrefly: ignore [no-matching-overload]
            or min(cst.shape) > 0  # type: ignore[union-attr]
            or (v.op_type in {"ConstantOfShape", "Cast", "Identity", "Constant"})
        ), (
            f"Output has empty shape {cst.shape}, name={name!r} "  # type: ignore[union-attr]
            # pyrefly: ignore [missing-attribute]
            f"v.op_type={v.op_type!r}, v.name={v.name!r}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        )
        assert cst is not None, f"Constant {name!r} was not found in {v.output}"
        if hasattr(self, "torch") and isinstance(
            cst, self.torch._subclasses.fake_tensor.FakeTensor
        ):
            # pyrefly: ignore [missing-attribute]
            assert not self._debug_constant_folding, (  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                f"Unable to compute constant for node {self.pretty_node(v)}"  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                f"because a FakeTensor appeared{self.get_debug_msg()}"  # type: ignore[attr-defined]
            )
            return None, None
        if self._debug_get_constant:  # type: ignore[attr-defined]
            print(
                # pyrefly: ignore [missing-attribute]
                f"[GraphBuilder-{self._hash()}.compute_constant] "  # type: ignore[attr-defined]
                # pyrefly: ignore [missing-attribute]
                f"    - A {name}: {self.pretty_tensor(cst)}"  # type: ignore[attr-defined]
            )
        assert (
            # pyrefly: ignore [missing-attribute]
            not self._debug_constant_folding or cst is not None  # type: ignore[attr-defined]
        # pyrefly: ignore [missing-attribute]
        ), f"Unable to compute constant for node {self.pretty_node(v)}{self.get_debug_msg()}"  # type: ignore[attr-defined]
        return cst, feeds

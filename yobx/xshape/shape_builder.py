from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import onnx
import onnx.helper as oh
from ..helpers import string_type
from ..helpers.onnx_helper import dtype_to_tensor_dtype
from ._shape_helper import DYNAMIC_SHAPE
from ..xexpressions import evaluate_expression, simplify_expression
from ..xexpressions.rename_expressions import rename_dynamic_dimensions, rename_dynamic_expression
from ..helpers.onnx_helper import (
    element_wise_binary_op_types,
    element_wise_op_cmp_types,
    unary_like_op_types,
)


def make_hash(obj: Any) -> str:
    """Returns a simple hash of ``id(obj)`` in four letter."""
    aa = id(obj) % (26**3)
    return f"{chr(65 + aa // 26 ** 2)}{chr(65 + (aa // 26) % 26)}{chr(65 + aa % 26)}"


class ShapeBuilder:
    """
    API for a class computing shapes in an ONNX model.

    The main implementation is :class:`BasicShapeBuilder
    <yobx.xshape.shape_builder_impl.BasicShapeBuilder>`.
    It walks through all the nodes of an ONNX model and infers output shapes
    and types based on the input shapes, using symbolic expressions when the
    exact integer values are not known.

    **Symbolic expressions** — When a dimension cannot be determined as a plain
    integer (e.g. because it depends on a dynamic input dimension), it is stored
    as a Python-arithmetic string expression built from the input dimension names.
    For instance, concatenating tensors of shapes ``("batch", "seq1")`` and
    ``("batch", "seq2")`` along axis 1 yields output shape
    ``("batch", "seq1+seq2")``.  The supported operators inside a symbolic
    expression are ``+``, ``-``, ``*``, ``//``, ``%`` and ``^``
    (where ``^`` means ``max``).  Expressions are automatically simplified
    by :func:`simplify_expression
    <yobx.xshape.simplify_expressions.simplify_expression>` before being stored,
    so ``d + f - f`` becomes ``d`` and ``2*seq//2`` becomes ``seq``.
    Once concrete values are available they can be resolved with
    :meth:`evaluate_shape` or :func:`evaluate_expression
    <yobx.xshape.evaluate_expressions.evaluate_expression>`.

    .. runpython::
        :showcode:

        import onnx
        import onnx.helper as oh
        from yobx.xshape import BasicShapeBuilder

        TFLOAT = onnx.TensorProto.FLOAT

        # Build a small model: Z = Concat(X, Y, axis=1)
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Concat", ["X", "Y"], ["Z"], axis=1)],
                "graph",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq1"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["batch", "seq2"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )

        builder = BasicShapeBuilder()
        builder.run_model(model)

        print("input names :", builder.input_names)
        print("output names:", builder.output_names)
        print("shape of Z  :", builder.get_shape("Z"))
        print("type of Z   :", builder.get_type("Z"))

    **Constraint mechanism** — When a broadcasting operation aligns a symbolic
    dimension (e.g. ``"d_model"``) with a concrete integer (e.g. ``64``), the
    concrete value is used immediately as the output dimension and the equality
    ``"d_model" = 64`` is stored as a *constraint*.  This avoids the need to
    backtrack through earlier nodes when the concrete value is later discovered.
    Constraints are inspected with :meth:`get_registered_constraints` and are
    used internally for dimension renaming and equality checks.
    See :ref:`l-design-shape` for details.
    """

    _op_type_element_wise_types = element_wise_binary_op_types()
    _op_type_element_wise_cmp_types = element_wise_op_cmp_types()
    _op_type_unary_like = unary_like_op_types()

    _debug_shape_missing = False

    @property
    def input_names(self) -> List[str]:
        """Returns the list of input names of the model."""
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    @property
    def output_names(self) -> List[str]:
        """Returns the list of output names of the model."""
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def get_debug_msg(self, limit: int = 1000) -> str:
        """
        Returns a string providing as much information as possible
        to help the developer understand why a conversion failed.

        :param limit: limit the string if the model is big
        :return: many pieces of information about the on going conversion
        """
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def has_opset(self, name: str) -> bool:
        """Tells if opset `name` is defined."""
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def get_opset(self, name: str) -> int:
        """
        Returns the opset version for domain `name`.

        :param name: domain name
        :return: domain version or 0 if not defined
        """
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def set_opset(self, name: str, itype: int):
        """
        Sets the opset version for domain `name`.

        :param name: domain name
        :param version: domain version
        """
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def has_shape(self, name: str) -> bool:
        """Tells if `name` has a shape."""
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def get_shape(self, name: str) -> DYNAMIC_SHAPE:
        """
        Returns the shape of result *name* as a tuple.
        Each dimension is either an integer or a string (symbolic dimension).

        :param name: result name
        :return: shape as a tuple of integers and/or strings
        """
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def set_shape(self, name: str, shape: DYNAMIC_SHAPE):
        """
        Sets the shape for result *name*.

        :param name: result name
        :param shape: tuple of integers and/or strings (symbolic dimensions)
        """
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def has_type(self, name: str) -> bool:
        """Tells if `name` has a type."""
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def get_type(self, name: str) -> int:
        """
        Returns the element type of result *name* as an ONNX integer
        (e.g. ``onnx.TensorProto.FLOAT == 1``).

        :param name: result name
        :return: element type as an integer
        """
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def set_type(self, name: str, itype: int):
        """
        Sets the element type for result *name*.

        :param name: result name
        :param itype: element type as an ONNX integer
                      (e.g. ``onnx.TensorProto.FLOAT == 1``)
        """
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def has_rank(self, name: str) -> bool:
        """Tells if `name` has a rank."""
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def get_rank(self, name: str) -> int:
        """
        Returns the rank (number of dimensions) of result *name*.

        :param name: result name
        :return: rank as an integer
        """
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def set_rank(self, name: str, rank: int):
        """
        Sets the rank (number of dimensions) for result *name*.

        :param name: result name
        :param rank: rank as an integer
        """
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def has_device(self, name: str) -> bool:
        """Tells if `name` has a device."""
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def get_device(self, name: str) -> int:
        """
        Returns the device of result *name*.

        :param name: result name
        :return: rank as an integer
        """
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def set_device(self, name: str, rank: int):
        """
        Sets the rank (number of dimensions) for result *name*.

        :param name: result name
        :param rank: rank as an integer
        """
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def _ensure_constraints_initialized(self):
        """Initializes ``constraints_`` if it has not been set yet by a subclass constructor."""
        if not hasattr(self, "constraints_"):
            self.constraints_: Dict[str, Set[Union[str, int]]] = {}

    def add_to_constraints(self, dim_name: str, value: Union[str, int, Set[Union[str, int]]]):
        """
        Adds a constraint associating a symbolic dimension name with a value or set of values.

        :param dim_name: symbolic dimension name (e.g. ``"batch"``)
        :param value: the value, name, or set of values/names to associate with that dimension
        """
        self._ensure_constraints_initialized()
        if dim_name not in self.constraints_:
            self.constraints_[dim_name] = set()
        if isinstance(value, set):
            self.constraints_[dim_name] |= value
        else:
            self.constraints_[dim_name].add(value)

    def register_constraint_dimension(self, dim_name: str, value: Any):
        """
        Registers a constraint associating a symbolic dimension name with a value.
        This allows to deal backward constraints after a single pass if the model.

        :param dim_name: symbolic dimension name (e.g. ``"batch"``)
        :param value: the value or set of values to associate with that dimension
        """
        self.add_to_constraints(dim_name, value)

    def get_registered_constraints(self) -> Dict[str, Set[Union[str, int]]]:
        """
        Returns the constraints registered so far.

        :return: mapping from dimension name to the set of values/names
                 it is constrained to be equal to
        """
        self._ensure_constraints_initialized()
        return self.constraints_

    def get_shape_renamed(self, name: str) -> DYNAMIC_SHAPE:
        """
        Returns the shape of result *name* using user-visible dimension names.

        After :meth:`_improves_dynamic_dimension_naming` has been called, symbolic
        dimension names that were given by the user (e.g. ``"batch"``, ``"seq_length"``)
        are substituted for the internal names (e.g. ``"s0"``, ``"s1"``).  When no
        renaming has been computed yet this falls back to :meth:`get_shape`.

        :param name: result name
        :return: shape tuple with user dimension names where available
        """
        if hasattr(self, "replacements_dimensions_") and name in self.replacements_dimensions_:
            return self.replacements_dimensions_[name]
        return self.get_shape(name)

    def _improves_dynamic_dimension_naming(
        self, original: Set[str], apply_replacements: bool = False
    ) -> Dict[str, str]:
        """
        Renames internal dynamic-dimension tokens (e.g. ``"s0"``, ``"DYN0"``) to
        user-visible names wherever the registered constraints allow it.

        After this method has been called :meth:`get_shape_renamed` will return
        the shapes with the preferred names.

        .. note::
            This method accesses ``self._known_shapes`` which must be a
            ``dict`` mapping result names to shape tuples.  This attribute
            is provided by concrete subclasses such as
            :class:`~yobx.xshape.shape_builder_impl.BasicShapeBuilder`.

        :param original: set of preferred (user-visible) dimension names to
                         try to substitute into the shapes
        :return: replacement dictionary mapping old token names to new names
        """
        constraints = self.get_registered_constraints()

        # Expand constraints: make every equality fully symmetric and transitive.
        expanded_constraints: Dict[str, Set[Union[str, int]]] = {}
        for k, v in constraints.items():
            expanded_constraints[k] = v.copy()
            for i in v:
                if i not in expanded_constraints:
                    expanded_constraints[i] = set()
                expanded_constraints[i] |= v
                expanded_constraints[i] |= {k}

        replacements = rename_dynamic_dimensions(expanded_constraints, original)

        if replacements:
            if not hasattr(self, "replacements_dimensions_"):
                self.replacements_dimensions_: Dict[str, DYNAMIC_SHAPE] = {}
                self.replacements_for_replacements_dimensions_ = replacements
            for k, v in self._known_shapes.items():  # type: ignore[attr-defined]
                if v is None:
                    continue
                self.replacements_dimensions_[k] = tuple(
                    (
                        simplify_expression(rename_dynamic_expression(_, replacements))
                        if isinstance(_, str)
                        else _
                    )
                    for _ in v
                )
        if apply_replacements and replacements:
            self._apply_shape_replacements(replacements)
        return replacements

    def _apply_shape_replacements(self, replacements: Dict[str, str]) -> Dict[str, str]:
        if not replacements:
            return

        # known_shapes
        updates = {}
        dim_updates = {}
        for name, shape in self._known_shapes.items():
            if not shape:
                continue
            new_shape = []
            update = False
            for s in shape:
                if isinstance(s, int):
                    new_shape.append(s)
                    continue
                ns = simplify_expression(rename_dynamic_expression(s, replacements))
                new_shape.append(ns)
                if ns != s:
                    dim_updates[s] = ns
                    update = True
            if update:
                updates[name] = tuple(new_shape)
        if updates:
            self._known_shapes.update(updates)

        # known_values_shapes
        v_updates = {}
        for name, shape in self._known_value_shape.items():
            if isinstance(shape, str):
                ns = simplify_expression(rename_dynamic_expression(shape, replacements))
                if ns != shape:
                    dim_updates[shape] = ns
                    update = True
                    v_updates[name] = ns
                continue
            if not isinstance(shape, tuple):
                continue
            if not shape:
                continue
            new_shape = []
            update = False
            for s in shape:
                if isinstance(s, int):
                    new_shape.append(s)
                    continue
                ns = simplify_expression(rename_dynamic_expression(s, replacements))
                new_shape.append(ns)
                if ns != s:
                    dim_updates[s] = ns
                    update = True
            if update:
                v_updates[name] = tuple(new_shape)
        if v_updates:
            self._known_value_shape.update(v_updates)
        if dim_updates:
            self._dynamic_alias.update(dim_updates)
        return dim_updates

    def _hash(self) -> str:
        return make_hash(self)

    def update_shapes(self, model: onnx.ModelProto):
        """Updates model shapes with the value stored inside this graph."""
        self._update_shapes_graph(model.graph)

    def _update_shapes_graph(self, graph: onnx.GraphProto):
        exclude = (
            set(i.name for i in graph.input)
            | set(i.name for i in graph.output)
            | set(i.name for i in graph.initializer)
            | set(i.name for i in graph.sparse_initializer)
        )
        include = set()
        for node in graph.node:
            include |= set(node.output)
        include -= exclude
        include -= set(i.name for i in graph.value_info)
        ordered_include = []
        for node in graph.node:
            for o in node.output:
                if o in include:
                    ordered_include.append(o)
        infos = []
        for k in ordered_include:
            if not self.has_shape(k):
                continue
            infos.append(oh.make_tensor_value_info(k, self.get_type(k), list(self.get_shape(k))))
        graph.value_info.extend(infos)

    def get_attribute(
        self, node: onnx.NodeProto, att_name: str, exc: bool = True
    ) -> Optional[onnx.AttributeProto]:
        """Returns an attribute for a node."""
        for att in node.attribute:
            if att.name == att_name:
                return att
        assert not exc, (
            f"Unable to find attribute {att_name!r} for node "
            f"type {node.op_type!r} in node {node}"
        )
        return None

    def get_attribute_with_default(
        self, node: onnx.NodeProto, name: str, default_value: Any
    ) -> Any:
        """
        Returns an attribute or its default value if missing.

        :param node: node
        :param name: attribute name
        :param default_value: default value
        :return: value
        """
        for att in node.attribute:
            if att.name == name:
                if att.type == onnx.AttributeProto.INT:
                    return att.i
                if att.type == onnx.AttributeProto.INTS:
                    return list(att.ints)
                if att.type == onnx.AttributeProto.FLOAT:
                    return att.f
                if att.type == onnx.AttributeProto.FLOATS:
                    return list(att.floats)
                if att.type == onnx.AttributeProto.STRING:
                    return att.s
                if att.type == onnx.AttributeProto.STRINGS:
                    return list(att.strings)
                raise TypeError(
                    f"Not implemented for attribute name {att.name!r}, attribute={att}"
                )
        return default_value

    def get_attributes_with_default(
        self, node: onnx.NodeProto, **default_values
    ) -> Dict[str, Any]:
        """
        Returns int or float attributes. If missing, the default value is returned
        if it is not None.

        :param node: node
        :param default_values: default values
        """
        res = {}
        for att in node.attribute:
            if att.name in default_values:
                if att.type == onnx.AttributeProto.INT:
                    res[att.name] = att.i
                elif att.type == onnx.AttributeProto.INTS:
                    res[att.name] = list(att.ints)
                elif att.type == onnx.AttributeProto.FLOAT:
                    res[att.name] = att.f
                elif att.type == onnx.AttributeProto.FLOATS:
                    res[att.name] = list(att.floats)
                elif att.type == onnx.AttributeProto.STRING:
                    res[att.name] = att.s
                elif att.type == onnx.AttributeProto.STRINGS:
                    res[att.name] = list(att.strings)
                else:
                    raise TypeError(
                        f"Not implemented for attribute name {att.name!r}, attribute={att}"
                    )
        for k, v in default_values.items():
            if k not in res and v is not None:
                res[k] = v
        res = {k: v for k, v in res.items() if v is not None}
        return res

    def pretty_node(
        self,
        node: Optional[onnx.NodeProto],
        limit: int = 80,
        short: bool = True,
        shape: bool = False,
    ) -> str:
        """
        Pretty rendering for a node.

        :param node: node to render
        :param limit: to show type and shapes after the limit
        :param short: do not display shape information on the left
        :param shape: show shape information below
        :return: string
        """
        if node is None:
            return "None"
        if shape:
            st = []
            for i in node.input:
                dt = self.get_type(i) if self.has_type(i) else "-"
                sh = (
                    "x".join(str(_).replace(" ", "") for _ in self.get_shape(i))
                    if self.has_shape(i)
                    else (f"rk={self.get_rank(i)}" if self.has_rank(i) else "?")
                )
                st.append(f"{i}:{dt}|{sh}")
            st.append("->")
            for i in node.output:
                dt = self.get_type(i) if self.has_type(i) else "-"
                sh = (
                    "x".join(str(_).replace(" ", "") for _ in self.get_shape(i))
                    if self.has_shape(i)
                    else (f"rk={self.get_rank(i)}" if self.has_rank(i) else "?")
                )
                st.append(f"{i}:{dt}|{sh}")
            shape_info = " ".join(st)
        else:
            shape_info = ""
        text = (
            (
                f"{node.op_type}[{node.domain}]: "
                f"{', '.join(node.input)} -> {', '.join(node.output)}"
            )
            if node.domain
            else f"{node.op_type}: {', '.join(node.input)} -> {', '.join(node.output)}"
        )
        if shape_info:
            text = f"{text} ## {shape_info}"
        if short:
            return text
        add = " " * abs(80 - len(text))
        text += add
        info = []
        for o in node.output:
            t = f"T{self.get_type(o)}" if self.has_type(o) else ""
            s = " x ".join(map(str, self.get_shape(o))) if self.has_shape(o) else ""
            info.append(": ".join([t, s]))
        if node.name:
            s = f"{text}|{' '.join(info)}"
            return f"{s}{' ' * (110 - len(s))}- {node.name}"
        return f"{text}|{' '.join(info)}"

    def map_value_info_dimension_with_true_values(
        self, name: str, tensor: np.ndarray, do_type: bool = True, do_shape: bool = True
    ):
        if do_type:
            assert self.has_type(name), f"Missing type for {name!r}{self.get_debug_msg()}"
            dtype = dtype_to_tensor_dtype(tensor.dtype)
            assert dtype == self.get_type(name), (
                f"Type mismatch for {name!r}, expecting "
                f"{self.get_type(name)}, got {dtype} in "
                f"{string_type(tensor, with_shape=True)}"
            )
            if not do_shape:
                return {}
        if do_shape:
            assert self.has_shape(name), f"Missing shape for {name!r}{self.get_debug_msg()}"
            res = {}
            shape = self.get_shape(name)
            for i, (value, dim) in enumerate(zip(tensor.shape, shape)):
                if isinstance(dim, str):
                    if dim in res:
                        assert res[dim] == value, (
                            f"Shape mismatch for {name!r} for dimension {i}, "
                            f"known dimensions are {shape}, got "
                            f"{string_type(tensor, with_shape=True)}"
                        )
                    res[dim] = value
                else:
                    assert dim == value, (
                        f"Shape mismatch for {name!r} for dimension {i}, "
                        f"expecting {dim}, got {string_type(tensor, with_shape=True)}"
                    )
            return res
        raise AssertionError(f"unexpected values {do_type=}, {do_shape=}")

    def evaluate_shape(self, name: str, context: Dict[str, int]) -> Tuple[int, ...]:
        shape = self.get_shape(name)
        return tuple(evaluate_expression(s, context) for s in shape)

    def evaluate_dimension_equality_with_constraints(self, d1: str, *args) -> bool:
        """Tells if two dimensions are equal."""
        assert len(args) == 1, f"Not implemented with d1={d1!r} and args={args}"
        d2 = args[0]
        return d1 == d2

    def compare_computed_shape_with_tensor(
        self,
        name: str,
        tensor: np.ndarray,
        context: Dict[str, int],
        do_shape: bool = True,
        do_type: bool = True,
    ) -> Tuple[Tuple[str, int, int], ...]:
        if do_type:
            assert self.has_type(name), f"Missing type for {name!r}."
            dtype = dtype_to_tensor_dtype(tensor.dtype)
            assert dtype == self.get_type(name), (
                f"Type mismatch for {name!r}, expecting "
                f"{self.get_type(name)}, got {dtype} in "
                f"{string_type(tensor, with_shape=True)}"
            )
            if not do_shape:
                return ((self.get_type(name), dtype, self.get_type(name)),)
        if do_shape:
            assert self.has_shape(name), f"Missing shape for {name!r}."
            computed = self.evaluate_shape(name, context=context)
            return tuple(zip(self.get_shape(name), tensor.shape, computed))

    def compare_with_true_inputs(
        self,
        inputs: Union[Dict[str, np.ndarray], List[np.ndarray]],
        outputs: Union[Dict[str, np.ndarray], List[np.ndarray]],
        exc: bool = True,
        do_shape: bool = True,
        do_type: bool = True,
    ) -> Dict[str, Tuple[Tuple[str, int, int], ...]]:
        """
        Compares the shape of the outputs with what the output shapes would return.

        :param inputs: inputs
        :param outputs: outputs
        :param exc: raises an exception when a discrepancy is met
        :param do_type: compares types
        :param do_shape: compares shapes
        :return: list of expression, expected value, computed value
        """
        if isinstance(inputs, list):
            inputs = dict(zip(self.input_names, inputs))
        if isinstance(outputs, list):
            outputs = dict(zip(self.output_names, outputs))

        context = {}
        for name in self.input_names:
            res = self.map_value_info_dimension_with_true_values(
                name, inputs[name], do_type=do_type, do_shape=do_shape
            )
            for k, v in res.items():
                if k not in context:
                    context[k] = v
                    continue
                if context[k] != res[k]:
                    assert not exc, (
                        f"Dimension mismatch for dimension {k!r}, previous value is "
                        f"{context[k]}, new value is {res[k]} for name={name!r}"
                    )

        final = {}
        for name, tensor in outputs.items():
            res = self.compare_computed_shape_with_tensor(
                name, tensor, context, do_type=do_type, do_shape=do_shape
            )
            for dim, expected, got in res:
                assert not exc or expected == got, (
                    f"Output dimension mismatch for {dim!r} for results {name!r}, "
                    f"expected is {expected!r}, got {got!r}."
                )
            final[name] = res
        return final

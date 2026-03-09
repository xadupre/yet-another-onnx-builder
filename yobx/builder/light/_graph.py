from enum import IntEnum
from typing import Any, Dict, List, Optional, Union
import numpy as np
from onnx import NodeProto, SparseTensorProto, TensorProto, TensorShapeProto, ValueInfoProto
from onnx.checker import check_model
from onnx.defs import onnx_opset_version
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array


class ProtoType(IntEnum):
    """
    The same code can be used to output a :class:`onnx.GraphProto`
    or a :class:`onnx.ModelProto`.
    """

    GRAPH = 2
    MODEL = 3


class OnnxGraph:
    """
    Accumulates nodes, inputs, outputs and initializers needed to build
    an ONNX graph using a fluent API.

    :param opset: main opset version (defaults to ``onnx_opset_version() - 1``)
    :param opsets: additional opsets as ``{domain: version}``
    :param ir_version: ONNX IR version; only applied to ModelProto outputs
    :param proto_type: :class:`ProtoType` - ``MODEL`` (default) or ``GRAPH``

    Simple example::

        from yobx.builder.light import start

        onx = start().vin("X").Neg().rename("Y").vout().to_onnx()
    """

    def __init__(
        self,
        opset: Optional[int] = None,
        opsets: Optional[Dict[str, int]] = None,
        ir_version: Optional[int] = None,
        proto_type: ProtoType = ProtoType.MODEL,
    ):
        if opsets is not None and "" in opsets:
            if opset is None:
                opset = opsets[""]
            elif opset != opsets[""]:
                raise ValueError("The main opset was specified twice with different values.")
        self.proto_type = proto_type
        self.opsets = opsets
        self.opset = opset
        self.ir_version = ir_version
        self.nodes: List[Union[NodeProto, TensorProto]] = []
        self.inputs: List[ValueInfoProto] = []
        self.outputs: List[ValueInfoProto] = []
        self.initializers: List[TensorProto] = []
        self.unique_names_: Dict[str, Any] = {}
        self.renames_: Dict[str, str] = {}

    def __repr__(self) -> str:
        parts = [f"{self.__class__.__name__}("]
        els = [
            repr(getattr(self, o)) for o in ["opset", "opsets"] if getattr(self, o) is not None
        ]
        parts.append(", ".join(els))
        parts.append(")")
        return "".join(parts)

    @property
    def input_names(self) -> List[str]:
        "Returns the input names."
        return [v.name for v in self.inputs]

    @property
    def output_names(self) -> List[str]:
        "Returns the output names."
        return [v.name for v in self.outputs]

    @property
    def main_opset(self) -> int:
        """Returns the opset version for the main ONNX domain (``""``)."""
        return self.opset if self.opset is not None else onnx_opset_version() - 1

    def has_opset(self, domain: str) -> int:
        """
        Returns the opset version for *domain*, or ``0`` if not registered.

        :param domain: domain name
        :return: opset version, or ``0`` if the domain is unknown
        """
        if domain == "":
            return self.main_opset
        if self.opsets is None:
            return 0
        return self.opsets.get(domain, 0)

    def get_opset(self, domain: str, exc: bool = True) -> int:
        """
        Returns the opset version for *domain*.

        :param domain: domain name
        :param exc: raise an ``AssertionError`` if the domain is not registered
        :return: version or ``0`` when *exc* is ``False`` and the domain is unknown
        """
        version = self.has_opset(domain)
        assert not exc or version, f"Domain {domain!r} is not registered."
        return version

    def set_opset(self, domain: str, version: int = 1) -> None:
        """
        Registers *domain* with the given opset *version*.

        If the domain is already registered with the same version this call is a
        no-op.  A version mismatch raises an ``AssertionError``.

        :param domain: domain name
        :param version: opset version (default: ``1``)
        """
        existing = self.has_opset(domain)
        if existing:
            assert existing == version, (
                f"Version mismatch for domain {domain!r}: existing={existing}, new={version}"
            )
            return
        if domain == "":
            self.opset = version
        else:
            if self.opsets is None:
                self.opsets = {}
            self.opsets[domain] = version

    def has_name(self, name: str) -> bool:
        "Returns ``True`` if *name* is already registered."
        return name in self.unique_names_

    def unique_name(self, prefix: str = "r", value: Optional[Any] = None) -> str:
        """
        Returns a new unique name.

        :param prefix: name prefix
        :param value: optional object to associate with the name
        :return: unique name string
        """
        name = prefix
        i = len(self.unique_names_)
        while name in self.unique_names_:
            name = f"{prefix}{i}"
            i += 1
        self.unique_names_[name] = value
        return name

    def make_input(
        self,
        name: str,
        elem_type: int = TensorProto.FLOAT,
        shape: Optional[Any] = None,
    ) -> ValueInfoProto:
        """
        Registers an input tensor and returns its :class:`onnx.ValueInfoProto`.

        :param name: input name (must be unique)
        :param elem_type: ONNX element type integer
        :param shape: optional shape (list of ints / strings / None for dynamic dims)
        :return: :class:`onnx.ValueInfoProto`
        """
        if self.has_name(name):
            raise ValueError(f"Name {name!r} is already taken.")
        var = make_tensor_value_info(name, elem_type, shape)
        self.inputs.append(var)
        self.unique_names_[name] = var
        return var

    def vin(
        self,
        name: str,
        elem_type: int = TensorProto.FLOAT,
        shape: Optional[Any] = None,
    ) -> "Var":  # noqa: F821
        """
        Declares a new graph input and returns a :class:`Var`.

        :param name: input name
        :param elem_type: ONNX element type integer (default: ``TensorProto.FLOAT``)
        :param shape: optional shape
        :return: :class:`Var`
        """
        from ._var import Var

        proto = self.make_input(name, elem_type=elem_type, shape=shape)
        return Var(
            self,
            proto.name,
            elem_type=proto.type.tensor_type.elem_type,
            shape=shape,
        )

    def make_output(
        self,
        name: str,
        elem_type: int = TensorProto.FLOAT,
        shape: Optional[Any] = None,
    ) -> ValueInfoProto:
        """
        Registers a graph output and returns its :class:`onnx.ValueInfoProto`.

        :param name: output name (must already exist as a node output or input)
        :param elem_type: ONNX element type integer
        :param shape: optional shape
        :return: :class:`onnx.ValueInfoProto`
        """
        if not self.has_name(name):
            raise ValueError(f"Name {name!r} does not exist.")
        var = make_tensor_value_info(name, elem_type, shape)
        self.outputs.append(var)
        self.unique_names_[name] = var
        return var

    def make_constant(self, value: np.ndarray, name: Optional[str] = None) -> TensorProto:
        """
        Adds a constant initializer.

        :param value: numpy array to store as initializer
        :param name: optional name (auto-generated when ``None``)
        :return: :class:`onnx.TensorProto`
        """
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(value)}.")
        if name is None:
            name = self.unique_name(prefix="cst")
        elif self.has_name(name):
            raise ValueError(f"Name {name!r} already exists.")
        tensor = from_array(value, name=name)
        self.unique_names_[name] = tensor
        self.initializers.append(tensor)
        return tensor

    def make_node(
        self,
        op_type: str,
        *inputs: Any,
        domain: str = "",
        n_outputs: int = 1,
        output_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> NodeProto:
        """
        Creates and registers an ONNX node.

        :param op_type: operator type name
        :param inputs: input :class:`Var` instances or numpy arrays
        :param domain: operator domain (default ``""`` for standard ops)
        :param n_outputs: number of output tensors
        :param output_names: explicit output names (auto-generated when ``None``)
        :param kwargs: operator attributes
        :return: :class:`onnx.NodeProto`
        """
        if output_names is None:
            output_names = [
                self.unique_name(prefix=f"r{len(self.nodes)}_{i}") for i in range(n_outputs)
            ]
        elif n_outputs != len(output_names):
            raise ValueError(f"Expected {n_outputs} output names but got {len(output_names)}.")
        input_names = []
        for inp in inputs:
            if hasattr(inp, "name"):
                input_names.append(inp.name)
            elif isinstance(inp, np.ndarray):
                cst = self.make_constant(inp)
                input_names.append(cst.name)
            elif inp is None:
                input_names.append("")
            else:
                raise TypeError(f"Unexpected input type {type(inp)}; expected Var or np.ndarray.")
        node = make_node(op_type, input_names, output_names, domain=domain, **kwargs)
        self.nodes.append(node)
        for out in output_names:
            self.unique_names_[out] = None
        if domain and (not self.opsets or domain not in self.opsets):
            raise RuntimeError(f"No opset version specified for domain {domain!r}.")
        return node

    def cst(self, value: np.ndarray, name: Optional[str] = None) -> "Var":  # noqa: F821
        """
        Adds a constant initializer and returns it as a :class:`Var`.

        :param value: constant numpy array
        :param name: optional name
        :return: :class:`Var`
        """
        from ._var import Var

        tensor = self.make_constant(value, name=name)
        return Var(self, tensor.name, elem_type=tensor.data_type, shape=tuple(tensor.dims))

    def true_name(self, name: str) -> str:
        """
        Resolves a (possibly renamed) name to its current name.

        :param name: original name
        :return: current name after applying all renames
        """
        if not isinstance(name, str):
            raise TypeError(f"Expected str, got {type(name)}.")
        while name in self.renames_:
            name = self.renames_[name]
        return name

    def get_var(self, name: str) -> "Var":  # noqa: F821
        """
        Returns the :class:`Var` corresponding to *name*.

        :param name: variable name
        :return: :class:`Var`
        """
        from ._var import Var

        tr = self.true_name(name)
        proto = self.unique_names_.get(tr)
        if proto is None:
            return Var(self, tr)
        if isinstance(proto, ValueInfoProto):
            return Var(
                self,
                proto.name,
                elem_type=proto.type.tensor_type.elem_type,
            )
        if isinstance(proto, TensorProto):
            return Var(self, proto.name, elem_type=proto.data_type, shape=tuple(proto.dims))
        return Var(self, tr)

    def rename(self, old_name: str, new_name: str) -> None:
        """
        Records a rename from *old_name* to *new_name*.  The rename is applied
        lazily when building the final proto.

        :param old_name: existing name
        :param new_name: desired new name
        """
        if not self.has_name(old_name):
            raise RuntimeError(f"Name {old_name!r} does not exist.")
        if self.has_name(new_name):
            raise RuntimeError(f"Name {new_name!r} already exists.")
        value = self.unique_names_[old_name]
        self.unique_names_[new_name] = value
        self.renames_[old_name] = new_name

    def _fix_name_tensor(
        self, obj: Union[TensorProto, SparseTensorProto, ValueInfoProto]
    ) -> Union[TensorProto, SparseTensorProto, ValueInfoProto]:
        true = self.true_name(obj.name)
        if true != obj.name:
            obj.name = true
        return obj

    def _ensure_shape(self, obj: ValueInfoProto) -> ValueInfoProto:
        """Ensures the shape field is present in a ValueInfoProto (required by onnx >= 1.20)."""
        if obj.type.HasField("tensor_type") and not obj.type.tensor_type.HasField("shape"):
            obj.type.tensor_type.shape.CopyFrom(TensorShapeProto())
        return obj

    def _fix_name_node(self, obj: NodeProto) -> NodeProto:
        new_inputs = [self.true_name(i) if i else i for i in obj.input]
        if list(new_inputs) != list(obj.input):
            del obj.input[:]
            obj.input.extend(new_inputs)
        new_outputs = [self.true_name(o) for o in obj.output]
        if list(new_outputs) != list(obj.output):
            del obj.output[:]
            obj.output.extend(new_outputs)
        return obj

    def to_onnx(self):
        """
        Builds and returns the ONNX proto.

        :return: :class:`onnx.GraphProto` when ``proto_type`` is ``GRAPH``,
                 :class:`onnx.ModelProto` otherwise
        """
        dense = [
            self._fix_name_tensor(i) for i in self.initializers if isinstance(i, TensorProto)
        ]
        graph = make_graph(
            [self._fix_name_node(n) for n in self.nodes],
            "light_api",
            [self._ensure_shape(self._fix_name_tensor(i)) for i in self.inputs],
            [self._ensure_shape(self._fix_name_tensor(o)) for o in self.outputs],
            dense,
        )
        if self.proto_type == ProtoType.GRAPH:
            return graph
        opsets = [make_opsetid("", self.opset or onnx_opset_version() - 1)]
        if self.opsets:
            for k, v in self.opsets.items():
                if k != "":
                    opsets.append(make_opsetid(k, v))
        model = make_model(graph, opset_imports=opsets)
        if self.ir_version:
            model.ir_version = self.ir_version
        check_model(model)
        return model

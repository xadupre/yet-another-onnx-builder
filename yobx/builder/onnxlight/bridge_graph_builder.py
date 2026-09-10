"""Adapts the published onnx-light native builder to the converter protocol."""

import contextlib
import importlib
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy
from onnx_light import onnx
from onnx_light.onnx import helper, numpy_helper
from onnx_light.onnx_core.graph_builder import GraphBuilder
from onnx_light.onnx_core.optimization import GraphGraph, standard_pattern_names
from onnx_light.onnx_core.shape_inference import ShapesContext, SymShape, SymTensor

from ...typing import DefaultConvertOptions

if TYPE_CHECKING:
    from ...xshape._shape_helper import ONNX_SHAPE

# Loading the native extension registers constant-folding kernels without an evaluator.
importlib.import_module("onnx_light.onnx_py._onnxpykernels")


@dataclass
class OnnxLightOptimizationOptions:
    """Configures native GraphGraph rewriting without Python optimizer semantics.

    ``patterns=None`` or ``"default"`` selects the wheel's standard patterns.
    An empty sequence disables pattern rewrites. Other strings must be exact
    native pattern names; Python pattern groups and Python patterns are rejected.
    GraphGraph owns its cleanup, recursive rewriting and constant-folding policy.
    Weight folding depends on the runtime kernels installed with onnx-light.
    """

    patterns: Optional[Union[str, Sequence[str]]] = None
    max_iter: int = -1

    def pattern_names(self):
        """Returns validated native pattern names."""
        available = set(standard_pattern_names())
        if self.patterns is None or self.patterns == "default":
            return sorted(available)
        names = [self.patterns] if isinstance(self.patterns, str) else list(self.patterns)
        if any(not isinstance(name, str) or name not in available for name in names):
            raise ValueError(
                f"Unsupported native patterns {names!r}; expected exact onnx-light "
                "standard pattern names, not Python pattern groups or instances."
            )
        return names

    def __post_init__(self):
        if not isinstance(self.max_iter, int) or self.max_iter < -1:
            raise ValueError("max_iter must be an integer greater than or equal to -1.")
        self.pattern_names()


def _native_proto(value, proto_type):
    """Copies a native protobuf without accepting foreign implementations."""
    if not isinstance(value, proto_type):
        raise TypeError(f"Expected native {proto_type.__name__}, not {type(value)!r}.")
    result = proto_type()
    result.ParseFromString(value.SerializeToString())
    return result


def _native_attribute_value(value):
    """Copies native attributes and subgraphs and converts array attributes."""
    if isinstance(value, numpy.ndarray):
        return numpy_helper.from_array(value)
    proto_type = getattr(onnx, type(value).__name__, None)
    if proto_type is not None and hasattr(value, "SerializeToString"):
        return _native_proto(value, proto_type)
    if isinstance(value, (tuple, list)):
        return [_native_attribute_value(item) for item in value]
    return value


def _native_shape(shape):
    """Normalizes unknown dimensions for the wheel's SymTensor constructor."""
    return ["" if dimension is None else dimension for dimension in shape]


def _compatible_ir_version(opsets):
    """Returns the ONNX format revision required by the declared schema versions."""
    standard = (
        (1, 3),
        (9, 4),
        (10, 5),
        (11, 6),
        (12, 7),
        (15, 8),
        (19, 9),
        (21, 10),
        (23, 11),
        (24, 12),
        (25, 13),
        (26, 14),
    )
    versions = [3]
    for opset in opsets:
        domain = str(opset.domain)
        if domain in ("", "ai.onnx"):
            versions.append(max(ir for version, ir in standard if opset.version >= version))
        elif domain == "ai.onnx.ml":
            versions.append({1: 3, 2: 6, 3: 8, 4: 9, 5: 10}.get(opset.version, standard[-1][1]))
    return max(versions)


class OnnxLightGraphBuilderOpset:
    """Normalizes operator convenience calls and historical axes signatures."""

    _multiple_outputs = {"TopK": 2, "Dropout": 2, "MaxPool": 2, "Unique": 4}
    _axes_versions = {
        "Squeeze": 13,
        "Unsqueeze": 13,
        "ReduceSum": 13,
        "ReduceMax": 18,
        "ReduceMin": 18,
        "ReduceMean": 18,
        "ReduceProd": 18,
        "ReduceLogSum": 18,
        "ReduceLogSumExp": 18,
        "ReduceL1": 18,
        "ReduceL2": 18,
        "ReduceSumSquare": 18,
    }

    def __init__(self, builder):
        self.builder = builder

    def __getattr__(self, op_type):
        if op_type == "DFTAnyOpset":
            return self._dft_any_opset
        if op_type.endswith("AnyOpset"):
            base = op_type[:-8]
            if base not in self._axes_versions:
                raise NotImplementedError(f"Native opset adapter does not support {op_type}.")
            return partial(self._any_opset, base)
        return partial(self.make_node, op_type)

    def _dft_any_opset(self, data, dft_length="", axis=None, **kwargs):
        if self.builder.main_opset < 17:
            raise ValueError("DFT requires opset 17 or newer.")
        if self.builder.main_opset < 20:
            if axis is not None:
                if isinstance(axis, str):
                    axis = self.builder.get_constant(axis)
                kwargs["axis"] = int(numpy.asarray(axis).item())
            return self.make_node("DFT", data, dft_length, **kwargs)
        if axis is None:
            return self.make_node("DFT", data, dft_length, **kwargs)
        if not isinstance(axis, str):
            axis = numpy.asarray(axis, dtype=numpy.int64).reshape(())
        return self.make_node("DFT", data, dft_length, axis, **kwargs)

    def _any_opset(self, op_type, *inputs, **kwargs):
        if len(inputs) not in (1, 2):
            raise ValueError(f"{op_type} expects one tensor and optional axes.")
        if len(inputs) == 2 and self.builder.main_opset < self._axes_versions[op_type]:
            axes = inputs[1]
            if isinstance(axes, str):
                axes = self.builder.get_constant(axes)
            kwargs["axes"] = numpy.asarray(axes, dtype=numpy.int64).reshape(-1).tolist()
            inputs = inputs[:1]
        return self.make_node(op_type, *inputs, **kwargs)

    def make_node(self, op_type, *inputs, outputs=None, **kwargs):
        """Creates an operator with optional inputs and array initializers."""
        is_split = op_type == "Split" and self.builder._domain(kwargs.get("domain", "")) == ""
        if outputs is None:
            if is_split:
                if "num_outputs" not in kwargs:
                    raise ValueError("Split requires outputs or a num_outputs attribute.")
                outputs = kwargs["num_outputs"]
            else:
                outputs = self._multiple_outputs.get(op_type, 1)
        if is_split and "num_outputs" in kwargs:
            count = (
                outputs
                if isinstance(outputs, int)
                else 1 if isinstance(outputs, str) else len(outputs)
            )
            if count != kwargs["num_outputs"]:
                raise ValueError("Split outputs count must match its num_outputs attribute.")
        normalized_inputs = [
            (
                value.name
                if isinstance(getattr(value, "name", None), str)
                and self.builder.has_name(value.name)
                else value
            )
            for value in inputs
        ]
        return self.builder.make_node(op_type, normalized_inputs, outputs=outputs, **kwargs)


class OnnxLightGraphBuilder:
    """Implements converter graph construction using native onnx-light engines.

    The persistent :attr:`shapes_context` owns symbolic inference and explicit
    annotations. The wheel does not expose mutable builder shape information, so
    changed annotations are synchronized through native model serialization
    before subsequent construction or optimization. No Python GraphBuilder,
    pattern optimizer, reference shape inference or reference runtime is used.

    Import, graph construction, optimization, inference and both export methods
    use only native protos. :meth:`to_onnx` wraps a native model in
    ``ExportArtifact`` and optionally stores large tensors in a native container.
    Sequence metadata, Torch exporter state and Python
    optimization/inference options are deliberately unsupported. Tensor ranks
    must be declared before inference because the wheel cannot distinguish an
    unknown rank from a scalar in ``SymTensor``.
    """

    supports_optimization_report = True

    def __init__(
        self,
        target_opset_or_existing_proto=18,
        ir_version=None,
        *,
        optimization_options=None,
        convert_options=None,
        verbose=0,
        as_function=False,
    ):
        if optimization_options is not None and not isinstance(
            optimization_options, OnnxLightOptimizationOptions
        ):
            raise TypeError(
                "The native backend requires OnnxLightOptimizationOptions, not "
                "Python OptimizationOptions."
            )
        self.optimization_options = optimization_options or OnnxLightOptimizationOptions()
        self.convert_options = convert_options or DefaultConvertOptions()
        self.verbose = verbose
        self.as_function = as_function
        self._devices = {}
        self._dimension_names = set()
        self._prefix_stack = []
        self._reserved_names = set()
        self._node_names = set()
        self._shape_names = set()
        self._annotations_dirty = False
        self._original_model = None
        self.shapes_context = ShapesContext()
        self.op = OnnxLightGraphBuilderOpset(self)
        self.anyop = self.op
        if isinstance(target_opset_or_existing_proto, (int, dict)):
            opsets = (
                {"": target_opset_or_existing_proto}
                if isinstance(target_opset_or_existing_proto, int)
                else dict(target_opset_or_existing_proto)
            )
            self._inner = GraphBuilder("graph")
            self.opsets = {}
            for domain, version in opsets.items():
                self.set_opset(domain, version)
            if "" not in self.opsets:
                raise ValueError("A standard ONNX opset must be specified.")
            self.ir_version = ir_version or 0
        elif isinstance(target_opset_or_existing_proto, onnx.ModelProto):
            model = _native_proto(target_opset_or_existing_proto, onnx.ModelProto)
            for value in model.graph.input:
                if value.type.HasField("tensor_type") and not value.type.tensor_type.HasField(
                    "shape"
                ):
                    raise NotImplementedError(
                        "Native inference requires a declared rank for input "
                        f"{str(value.name)!r}."
                    )
            self._original_model = model.SerializeToString()
            self.ir_version = ir_version or model.ir_version
            self._inner = GraphBuilder(model)
            self._node_names.update(str(node.name) for node in model.graph.node if node.name)
            self.opsets = {}
            for opset in model.opset_import:
                self.set_opset(str(opset.domain), opset.version)
            self.shapes_context.compute_shape_model(model, True)
            self._shape_names.update(self.shapes_context.names())
        else:
            raise TypeError(
                "The native builder expects an opset integer, dictionary or ModelProto."
            )

    @staticmethod
    def _domain(domain):
        return "" if domain == "ai.onnx" else domain

    @property
    def inner_builder(self):
        """Returns the underlying native GraphBuilder."""
        self._synchronize_annotations()
        return self._inner

    @property
    def main_opset(self):
        """Returns the standard ONNX opset."""
        return self.opsets[""]

    @property
    def inputs(self):
        """Returns native input value infos."""
        return list(self._inner.to_graph().input)

    @property
    def outputs(self):
        """Returns native output value infos."""
        return list(self._inner.to_graph().output)

    @property
    def input_names(self):
        """Returns declared input names."""
        return [str(value.name) for value in self.inputs]

    @property
    def output_names(self):
        """Returns declared output names."""
        return [str(value.name) for value in self.outputs]

    @property
    def nodes(self):
        """Returns a snapshot of native nodes, not a mutable builder registry."""
        return list(self._inner.to_graph().node)

    @property
    def initializers_dict(self):
        """Returns a snapshot of native initializer protos."""
        return {str(value.name): value for value in self._inner.to_graph().initializer}

    @property
    def functions(self):
        """Returns exported native local function definitions."""
        return {
            (str(value.domain), str(value.name)): value
            for value in self._inner.to_onnx().functions
        }

    def empty_copy(self, as_function=False):
        """Creates an independent native builder with the same converter options."""
        return type(self)(
            self.opsets,
            ir_version=self.ir_version,
            optimization_options=self.optimization_options,
            convert_options=self.convert_options,
            verbose=self.verbose,
            as_function=as_function,
        )

    def make_subset_builder(self, input_names, name="", domain="", add_local_functions=False):
        """Creates an independent native function builder from selected input descriptors."""
        builder = self.empty_copy(as_function=True)
        for input_name in input_names:
            builder.make_tensor_input(
                input_name,
                self.get_type(input_name) if self.has_type(input_name) else None,
                self.get_shape(input_name) if self.has_shape(input_name) else None,
                device=self.get_device(input_name) if self.has_device(input_name) else None,
            )
        if add_local_functions:
            model = builder._native_model()
            model.functions.extend(self.functions.values())
            builder._inner = GraphBuilder(model)
        return builder

    def make_tensor_value_info_from_name(self, name):
        """Returns a native value-info using the context's explicit metadata."""
        if not self.has_type(name) and not self.has_rank(name):
            value = onnx.ValueInfoProto()
            value.name = name
            return value
        return helper.make_tensor_value_info(
            name,
            self.get_type(name) if self.has_type(name) else 0,
            self.get_shape(name) if self.has_shape(name) else None,
        )

    @property
    def last_added_node(self):
        """Returns the most recently added native node."""
        nodes = self.nodes
        return nodes[-1] if nodes else None

    def get_opset(self, domain, exc=True):
        """Returns the registered domain version."""
        domain = self._domain(domain)
        if exc and domain not in self.opsets:
            raise KeyError(f"Unknown domain {domain!r}.")
        return self.opsets.get(domain, 0)

    def has_opset(self, domain):
        """Returns the domain version or zero."""
        return self.get_opset(domain, exc=False)

    def set_opset(self, domain, version=1):
        """Registers an opset with both native engines."""
        domain = self._domain(domain)
        if domain in self.opsets and self.opsets[domain] != version:
            raise ValueError(f"Conflicting opset for {domain!r}: {version}.")
        self.opsets[domain] = version
        self._inner.set_opset_version(domain, version)
        self.shapes_context.set_opset_version(domain, version)

    add_domain = set_opset

    def unique_name(self, prefix="value"):
        """Returns a collision-free name in the current prefix scope."""
        prefix = "__".join([*self._prefix_stack, prefix or "value"])
        name = prefix
        index = 2
        while name in self._reserved_names or name in self._node_names or self.has_name(name):
            name = f"{prefix}_{index}"
            index += 1
        self._reserved_names.add(name)
        return name

    def unique_dimension_name(self, prefix="dim"):
        """Returns a symbolic dimension name absent from native descriptors."""
        occupied = {
            dimension
            for name in self._shape_names
            for dimension in self.get_shape(name)
            if isinstance(dimension, str)
        } | self._dimension_names
        name = prefix or "dim"
        index = 2
        while name in occupied:
            name = f"{prefix or 'dim'}_{index}"
            index += 1
        self._dimension_names.add(name)
        return name

    @contextlib.contextmanager
    def prefix_name_context(self, prefix):
        """Scopes generated names to a nested converter prefix."""
        self._prefix_stack.append(prefix)
        try:
            yield
        finally:
            self._prefix_stack.pop()

    def has_name(self, name):
        """Returns whether the native builder knows a value."""
        return self._inner.has_name(name)

    def has_type(self, name):
        """Returns whether the native shape context knows a tensor dtype."""
        return self.shapes_context.has(name) and bool(self.shapes_context.get(name).dtype)

    def get_type(self, name):
        """Returns a tensor dtype from the native context."""
        if not self.has_type(name):
            raise KeyError(f"No tensor type is known for {name!r}.")
        return self.shapes_context.get(name).dtype

    def has_shape(self, name):
        """Returns whether the tensor shape, including its rank, is known."""
        return name in self._shape_names and self.shapes_context.has(name)

    def get_shape(self, name: str) -> "ONNX_SHAPE":
        """Returns native symbolic dimensions as a tuple."""
        if not self.has_shape(name):
            raise KeyError(f"No tensor shape is known for {name!r}.")
        return tuple(
            None if dimension == "" else dimension
            for dimension in self.shapes_context.get(name).shape.dims()
        )

    def _set_tensor(self, name, dtype, shape):
        previous = self.shapes_context.get(name) if self.shapes_context.has(name) else None
        tensor = SymTensor(dtype, _native_shape(shape))
        if previous is not None and previous.has_value_as_shape():
            tensor.set_value_as_shape(previous.value_as_shape())
        self.shapes_context.set(name, tensor)
        self._annotations_dirty = True

    def set_type(self, name, itype):
        """Sets an explicit dtype in the native context."""
        if self.has_type(name) and self.get_type(name) == itype:
            return
        shape = self.get_shape(name) if self.has_shape(name) else []
        self._set_tensor(name, int(itype), shape)

    def set_shape(self, name, shape, allow_zero=False):
        """Sets an explicit shape in the native context."""
        if shape is None or (not allow_zero and 0 in shape):
            raise ValueError(f"Invalid shape for {name!r}: {shape!r}.")
        if self.has_shape(name) and self.get_shape(name) == tuple(shape):
            return
        self._set_tensor(name, self.get_type(name) if self.has_type(name) else 0, list(shape))
        self._shape_names.add(name)

    def has_rank(self, name):
        """Returns whether the tensor rank is known."""
        return self.has_shape(name)

    def get_rank(self, name):
        """Returns the tensor rank."""
        return len(self.get_shape(name))

    def set_rank(self, name, value):
        """Records a rank using unknown native symbolic dimensions."""
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Invalid rank {value!r}.")
        if self.has_rank(name):
            if self.get_rank(name) != value:
                raise ValueError(f"Conflicting rank for {name!r}.")
            return
        self.set_shape(name, (None,) * value)

    def set_type_shape_unary_op(self, name, input_name, itype=None):
        """Copies a tensor annotation entirely within the native context."""
        if itype is not None or self.has_type(input_name):
            self.set_type(name, itype if itype is not None else self.get_type(input_name))
        if self.has_shape(input_name):
            self.set_shape(name, self.get_shape(input_name), allow_zero=True)
            return True
        return False

    def set_type_shape_or_rank(self, name, like):
        """Copies native tensor descriptors and explicit device metadata."""
        self.set_type_shape_unary_op(name, like)
        if self.has_device(like):
            self.set_device(name, self.get_device(like))

    def make_tensor_input(self, name, elem_type=None, shape=None, device=None):
        """Declares a tensor input with a native value-info proto."""
        if device is not None and not isinstance(device, int):
            raise TypeError(f"Expected an integer device index, not {type(device)!r}.")
        if self.has_name(name):
            raise ValueError(f"Input name {name!r} already exists.")
        value = helper.make_tensor_value_info(name, elem_type or 0, shape)
        self._inner.make_input(value)
        self.shapes_context.set(
            name, SymTensor(elem_type or 0, [] if shape is None else _native_shape(shape))
        )
        if shape is not None:
            self._shape_names.add(name)
        if device is not None:
            self.set_device(name, device)
        return name

    def make_tensor_output(
        self,
        name: Union[str, list[str], tuple[str, ...]],
        elem_type=None,
        shape=None,
        indexed=False,
        allow_untyped_output=False,
    ) -> Union[str, list[str]]:
        """Declares one or more native tensor outputs."""
        if indexed:
            raise NotImplementedError("Indexed output renaming is not supported.")
        if isinstance(name, (tuple, list)):
            for output in name:
                self.make_tensor_output(output, elem_type, shape, False, allow_untyped_output)
            return list(name)
        if not self.has_name(name):
            raise KeyError(f"Unknown output {name!r}.")
        if elem_type is not None:
            self.set_type(name, elem_type)
        if shape is not None:
            self.set_shape(name, shape, allow_zero=True)
        if not allow_untyped_output and not self.has_type(name):
            raise ValueError(f"Output {name!r} has no tensor dtype.")
        self._synchronize_annotations()
        self._inner.make_output(
            helper.make_tensor_value_info(
                name,
                self.get_type(name) if self.has_type(name) else 0,
                self.get_shape(name) if self.has_shape(name) else None,
            )
        )
        return name

    def make_initializer(
        self,
        name,
        value,
        give_unique_name=True,
        source=None,
        *,
        allow_empty=False,
        msg=None,
        parameter_name=None,
    ):
        """Adds an initializer after converting it to a native tensor."""
        if parameter_name:
            name = parameter_name
        if not name or (self.has_name(name) and give_unique_name):
            name = self.unique_name(name or "init")
        elif self.has_name(name):
            raise ValueError(f"Initializer name {name!r} already exists.")
        if type(value).__module__.startswith("torch") and hasattr(value, "detach"):
            from ...helpers.mini_onnx_builder import proto_from_array

            fake_tensor_type = importlib.import_module("torch._subclasses.fake_tensor").FakeTensor
            if isinstance(value, fake_tensor_type) or value.is_meta:
                raise NotImplementedError("Native initializers require concrete tensor storage.")
            value = proto_from_array(value.detach().cpu(), name=name)
        if isinstance(value, onnx.TensorProto):
            tensor = _native_proto(value, onnx.TensorProto)
            tensor.name = name
        else:
            if isinstance(value, numpy.generic):
                value = numpy.asarray(value)
            elif isinstance(value, int):
                value = numpy.array(value, dtype=numpy.int64)
            elif isinstance(value, float):
                value = numpy.array(value, dtype=numpy.float32)
            if not isinstance(value, numpy.ndarray):
                raise TypeError(f"Unsupported initializer value {type(value)!r}.")
            tensor = numpy_helper.from_array(value, name=name)
        self._inner.make_initializer(tensor)
        self.shapes_context.compute_shape_graph(
            helper.make_graph([], "initializer", [], [], [tensor])
        )
        self._shape_names.add(name)
        return name

    def make_node(
        self, op_type, inputs, outputs=1, domain="", attributes=None, name=None, **kwargs
    ):
        """Creates a native node and computes its shape with ShapesContext."""
        domain = self._domain(domain)
        if not self.has_opset(domain):
            raise ValueError(f"Register the opset for domain {domain!r} before creating nodes.")
        if isinstance(inputs, str):
            inputs = [inputs]
        normalized_inputs = [
            (
                value
                if isinstance(value, str)
                else "" if value is None else self.make_initializer("", value)
            )
            for value in inputs
        ]
        for value in normalized_inputs:
            if value and self.has_name(value) and not self.has_shape(value):
                raise NotImplementedError(
                    f"Native inference requires a declared rank for {value!r}; "
                    "use set_shape or set_rank before creating nodes."
                )
        if outputs is None:
            outputs = 1
        if isinstance(outputs, int):
            if outputs < 0:
                raise ValueError("The number of outputs cannot be negative.")
            outputs = [self.unique_name(op_type.lower()) for _ in range(outputs)]
        elif isinstance(outputs, str):
            outputs = [outputs]
        else:
            outputs = list(outputs)
        for output in outputs:
            if output and self.has_name(output):
                raise ValueError(f"Output name {output!r} already exists.")
        native_attributes = []
        if isinstance(attributes, dict):
            kwargs = {**attributes, **kwargs}
        elif attributes:
            native_attributes.extend(_native_proto(a, onnx.AttributeProto) for a in attributes)
        native_attributes.extend(
            helper.make_attribute(key, _native_attribute_value(value))
            for key, value in kwargs.items()
            if value is not None
        )
        self._synchronize_annotations()
        base_name = str(name) if name else self.unique_name(op_type)
        node_name = base_name
        index = 2
        while node_name in self._node_names:
            node_name = f"{base_name}_{index}"
            index += 1
        node = helper.make_node(
            op_type, normalized_inputs, outputs, domain=domain, name=node_name
        )
        node.attribute.extend(native_attributes)
        local_function = (
            self._inner.has_local_function(op_type) and (domain, op_type) in self.functions
        )
        native_inference_domain = domain in {
            "",
            "ai.onnx.ml",
            "ai.onnx.preview.training",
            "ai.onnx.training",
        }
        custom_inference = self.shapes_context.has_custom_shape_inference_function(
            domain, op_type
        )
        if not local_function and (native_inference_domain or custom_inference):
            self.shapes_context.compute_shape_node(node)
        self._inner.make_node(
            op_type, normalized_inputs, outputs, domain, node_name, native_attributes
        )
        if local_function:
            for output in outputs:
                if output and self._inner.has_shape(output):
                    self.shapes_context.set(output, self._inner.get_shape(output))
        self._node_names.add(node_name)
        self._shape_names.update(output for output in outputs if self.shapes_context.has(output))
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def _native_model(self):
        model = self._inner.to_onnx(ir_version=self.ir_version)
        self.shapes_context.apply_inferred_shapes_to_model(model)
        # The native apply method fills missing annotations, but does not replace
        # existing input annotations. Explicit converter overrides must win.
        for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]:
            name = str(value.name)
            if self.shapes_context.has(name) and value.type.HasField("tensor_type"):
                info = helper.make_tensor_value_info(
                    name,
                    self.shapes_context.get(name).dtype,
                    self.get_shape(name) if self.has_shape(name) else None,
                )
                value.ClearField("type")
                value.type.CopyFrom(info.type)
        return model

    def _synchronize_annotations(self):
        if self._annotations_dirty:
            self._inner = GraphBuilder(self._native_model())
            self._annotations_dirty = False

    def _function_artifact(self, options, optimize, inline):
        """Exports a native function with constants or explicit parameter inputs."""
        from ...container import ExportArtifact, FunctionPieces

        model, _, _ = self._export_native(optimize, inline)
        graph = model.graph
        graph.name = options.name
        promoted = {}
        constants = []
        for tensor in graph.initializer:
            value = numpy_helper.to_array(tensor)
            if options.return_initializer and (
                not options.move_initializer_to_constant
                or value.nbytes >= options.external_threshold
            ):
                promoted[str(tensor.name)] = value
                graph.input.append(
                    helper.make_tensor_value_info(
                        str(tensor.name), tensor.data_type, list(tensor.dims)
                    )
                )
            else:
                constants.append(
                    helper.make_node("Constant", [], [str(tensor.name)], value=tensor)
                )
        nodes = constants + list(graph.node)
        graph.ClearField("initializer")
        graph.ClearField("node")
        graph.node.extend(nodes)
        # Function serialization is owned by the native builder. Nested definitions
        # are returned alongside the proto because FunctionProto cannot contain them.
        native = GraphBuilder(model)
        native.set_opset_version(options.domain, self.opsets.get(options.domain, 1))
        function = native.to_function(options.domain)
        function.name = options.name
        return ExportArtifact(
            proto=function,
            builder=self,
            function=FunctionPieces(
                initializers_name=list(promoted),
                initializers_dict=promoted,
                initializers_renaming={name: name for name in promoted},
                nested_functions=list(self.functions.values()) if not inline else [],
            ),
        )

    def make_local_function(self, builder, function_options, optimize=False):
        """Registers converter functions through native model import."""
        options = function_options
        if not options.name or not options.domain:
            raise ValueError("A local function requires a name and a nonempty domain.")
        artifact = builder._function_artifact(options, optimize, options.inline)
        function = artifact.proto
        key = (str(function.domain), str(function.name))
        functions = self.functions
        nested_functions = {
            (str(nested.domain), str(nested.name)): nested
            for nested in artifact.function.nested_functions
        }
        for nested_key, nested in nested_functions.items():
            if nested_key in functions and self._function_signature(
                functions[nested_key]
            ) != self._function_signature(nested):
                raise ValueError(f"Conflicting nested local function {nested_key!r}.")
        if key in functions:
            existing = functions[key]
            if options.merge_allowed and self._function_signature(
                existing
            ) == self._function_signature(function):
                return self._add_function_initializers(artifact), key
            if not options.rename_allowed:
                raise ValueError(f"Local function {key!r} already exists.")
            index = 2
            while (key[0], f"{key[1]}_{index}") in functions:
                index += 1
            key = (key[0], f"{key[1]}_{index}")
            function.name = key[1]
        if not self.has_opset(key[0]):
            self.set_opset(key[0], 1)
        self._synchronize_annotations()
        model = self._native_model()
        for nested_key, nested in nested_functions.items():
            if nested_key not in functions:
                model.functions.append(nested)
        model.functions.append(function)
        self._inner = GraphBuilder(model)
        return self._add_function_initializers(artifact), key

    def _add_function_initializers(self, artifact):
        return [
            name if self.constant_is_equal_to(name, value) else self.make_initializer(name, value)
            for name, value in artifact.function.initializers_dict.items()
        ]

    @staticmethod
    def _function_signature(function):
        normalized = _native_proto(function, onnx.FunctionProto)
        opsets = sorted((str(opset.domain), opset.version) for opset in normalized.opset_import)
        normalized.ClearField("opset_import")
        normalized.opset_import.extend(
            helper.make_opsetid(domain, version) for domain, version in opsets
        )
        return normalized.SerializeToString()

    def inline_functions(self, verbose=0):
        """Inlines local functions using the native graph operation."""
        count = self.inner_builder.inline_local_functions()
        return count

    def remove_unused(self):
        """Removes unused nodes with the native cleanup pass."""
        return self.inner_builder.remove_unused_nodes()

    def remove_identity_nodes(self):
        """Removes internal identities while preserving native graph output names."""
        return self.inner_builder.remove_identity_nodes()

    def move_initializers_to_constant(self, full_parameter_name=False):
        """Lowers graph initializers to native Constant nodes for function export."""
        model = self._native_model()
        constants = [
            helper.make_node("Constant", [], [str(tensor.name)], value=tensor)
            for tensor in model.graph.initializer
        ]
        nodes = constants + list(model.graph.node)
        model.graph.ClearField("initializer")
        model.graph.ClearField("node")
        model.graph.node.extend(nodes)
        self._inner = GraphBuilder(model)

    def is_constant(self, name):
        """Queries constant ownership through native GraphGraph."""
        return GraphGraph(self.inner_builder, [], use_global_patterns=False).is_constant(name)

    def get_constant(
        self, name, exc=True, computed_value=False, as_shape=False, multiple_outputs=False
    ):
        """Returns a constant using only the native constant runtime."""
        if multiple_outputs:
            raise NotImplementedError("Multiple-output constant evaluation is not supported.")
        graph = GraphGraph(self.inner_builder, [], use_global_patterns=False)
        if not graph.is_constant(name):
            if exc:
                raise ValueError(f"{name!r} is not a native constant.")
            return None
        tensor = graph.get_computed_constant(name)
        if tensor is None:
            if exc:
                raise ValueError(f"No native runtime value is available for {name!r}.")
            return None
        value = numpy_helper.to_array(tensor)
        return tuple(value.reshape(-1).tolist()) if as_shape else value

    def constant_is_equal_to(self, name, value):
        """Compares a native computed constant without size-dependent shortcuts."""
        actual = self.get_constant(name, exc=False)
        if actual is None:
            return False
        assert isinstance(actual, numpy.ndarray), "Native tensor constants must be ndarrays."
        if isinstance(value, onnx.TensorProto):
            value = numpy_helper.to_array(value)
        value = numpy.asarray(value)
        return (
            actual.dtype == value.dtype
            and actual.shape == value.shape
            and numpy.array_equal(actual, value)
        )

    def get_dynamic_dimension(self, dimension, keep_const=False):
        """Builds a rank-one dimension tensor with native ONNX operators."""
        if isinstance(dimension, int):
            value = numpy.array([dimension], dtype=numpy.int64)
            return value if keep_const else self.make_initializer("", value)
        name = self.get_dimension_as_result(dimension)
        return (
            self.op.UnsqueezeAnyOpset(name, numpy.array([0], dtype=numpy.int64))
            if self.get_rank(name) == 0
            else name
        )

    def get_dimension_as_result(self, dimension):
        """Materializes a named dimension from native input shape descriptors."""
        if self.has_name(dimension):
            return dimension
        for name in self.input_names:
            if self.has_shape(name) and dimension in self.get_shape(name):
                return self.op.Gather(
                    self.op.Shape(name),
                    numpy.array(self.get_shape(name).index(dimension), dtype=numpy.int64),
                    outputs=[dimension],
                )
        raise ValueError(f"No input shape defines dimension {dimension!r}.")

    def value_as_shape(self, name):
        """Returns a shape value recorded by the native inference engine."""
        if not self.shapes_context.has(name):
            return None
        tensor = self.shapes_context.get(name)
        return tuple(tensor.value_as_shape().dims()) if tensor.has_value_as_shape() else None

    def set_value_shape(self, name, value):
        """Records a symbolic shape value in the native inference engine."""
        tensor = self.shapes_context.get(name)
        tensor.set_value_as_shape(SymShape(value))
        self.shapes_context.set(name, tensor)

    def is_sequence(self, name):
        """Returns whether the native context identifies a sequence."""
        return self.shapes_context.has_sequence(name)

    def get_sequence(self, name):
        """Rejects unsupported sequence metadata conversion."""
        raise NotImplementedError(
            "The native converter bridge does not expose sequence metadata."
        )

    def set_sequence(self, name, dtype, shapes=None, ranks=None, unknown=False):
        """Rejects unsupported sequence metadata conversion."""
        raise NotImplementedError(
            "The native converter bridge does not expose sequence metadata."
        )

    def has_device(self, name):
        """Returns whether explicit converter device metadata is available."""
        return name in self._devices

    def get_device(self, name):
        """Returns explicit converter device metadata."""
        return self._devices[name]

    def set_device(self, name, device):
        """Records device metadata without changing native inference or execution."""
        if not isinstance(device, int):
            raise TypeError(f"Expected an integer device index, not {type(device)!r}.")
        self._devices[name] = device

    def onnx_dtype_to_np_dtype(self, itype):
        """Returns the NumPy dtype through the native ONNX helper."""
        return helper.tensor_dtype_to_np_dtype(itype)

    def get_debug_msg(self):
        """Returns native builder diagnostics."""
        return f"\nOnnxLightGraphBuilder(opsets={self.opsets}, inputs={self.input_names})"

    def pretty_text(self):
        """Returns the native builder's graph diagnostics."""
        return self._inner.to_string()

    def _export_native(self, optimize, inline):
        """Exports a native model and optional native rewrite statistics."""
        builder = GraphBuilder(self._native_model())
        if inline:
            builder.inline_local_functions()
        rewrites = []
        native_report = None
        if optimize:
            rewrites, native_report = GraphGraph(
                builder, self.optimization_options.pattern_names(), use_global_patterns=False
            ).optimize(self.optimization_options.max_iter, report=True)
        model = builder.to_onnx(ir_version=self.ir_version)
        if self._original_model is not None:
            original = onnx.ModelProto()
            original.ParseFromString(self._original_model)
            model.graph.name = original.graph.name
            model.graph.doc_string = original.graph.doc_string
            model.graph.ClearField("metadata_props")
            model.graph.metadata_props.extend(original.graph.metadata_props)
            original.ClearField("graph")
            original.graph.CopyFrom(model.graph)
            original.ClearField("opset_import")
            original.opset_import.extend(model.opset_import)
            original.ClearField("functions")
            original.functions.extend(model.functions)
            original.ir_version = model.ir_version
            model = original
        self._canonical_domains(model)
        # Native rewrites may reuse a source name for several inserted nodes.
        self._normalize_node_names(model)
        return model, rewrites, native_report

    def to_native(self, optimize=True, inline=True):
        """Returns a native model without importing reference ONNX or its runtime.

        Construction, shape inference, optional optimization and serialization
        remain in onnx-light. The exported model preserves imported metadata
        and the explicitly requested or imported IR version. Otherwise it uses
        the native wheel's default IR version, unlike :meth:`to_onnx`, which
        selects a compatible IR version for the declared opsets.

        Returns:
            An ``onnx_light.onnx.ModelProto``.
        """
        return self._export_native(optimize, inline)[0]

    def to_onnx(
        self,
        optimize=True,
        large_model=False,
        external_threshold=1024,
        return_optimize_report=False,
        inline=True,
        function_options=None,
        mask_outputs=None,
        as_graph_proto=False,
    ):
        """Exports an artifact after optional native GraphGraph optimization."""
        if large_model and as_graph_proto:
            raise NotImplementedError("Large-model containers require a ModelProto export.")
        if function_options is not None and function_options.export_as_function:
            if (
                large_model
                or as_graph_proto
                or return_optimize_report
                or mask_outputs is not None
            ):
                raise ValueError("Function export cannot be combined with model export options.")
            return self._function_artifact(function_options, optimize, inline)
        if mask_outputs is not None:
            raise NotImplementedError("Native export does not support output masks.")
        from ...container import ExportArtifact, ExportReport, ExtendedModelContainer

        native_model, rewrites, native_report = self._export_native(optimize, inline)
        report = None
        if native_report is not None:
            if return_optimize_report:
                report = ExportReport(
                    stats=[
                        {
                            "pattern": str(rewrite.pattern_name),
                            "added": len(rewrite.added_nodes),
                            "removed": len(rewrite.matched_nodes),
                            "time_in": (rewrite.match_time_ns + rewrite.apply_time_ns) / 1e9,
                        }
                        for rewrite in rewrites
                    ],
                    extra={
                        "backend": "onnx-light",
                        **{
                            key: getattr(native_report, key)
                            for key in (
                                "iterations",
                                "rewrites",
                                "total_time_ns",
                                "matching_time_ns",
                                "rewriting_time_ns",
                                "cleanup_time_ns",
                                "constant_folding_time_ns",
                                "subgraph_optimization_time_ns",
                            )
                        },
                    },
                )
        elif return_optimize_report:
            report = ExportReport(extra={"backend": "onnx-light", "rewrites": 0})
        model = native_model
        if not self.ir_version:
            model.ir_version = _compatible_ir_version(model.opset_import)
        container = None
        if large_model:
            container = ExtendedModelContainer(model)
            container.externalize_initializers(external_threshold)
        return ExportArtifact(
            proto=model.graph if as_graph_proto else model,
            report=report,
            builder=self,
            container=container,
        )

    @classmethod
    def _canonical_domains(cls, model):
        """Normalizes the standard ONNX domain in native protobufs."""

        def graph_domains(graph):
            for node in graph.node:
                node.domain = cls._domain(node.domain)
                for attribute in node.attribute:
                    if attribute.HasField("g"):
                        graph_domains(attribute.g)
                    for subgraph in attribute.graphs:
                        graph_domains(subgraph)

        for opset in model.opset_import:
            opset.domain = cls._domain(opset.domain)
        graph_domains(model.graph)
        for function in model.functions:
            for opset in function.opset_import:
                opset.domain = cls._domain(opset.domain)
            graph_domains(function)

    @staticmethod
    def _normalize_node_names(model):
        """Deduplicates node names per scope without changing existing unique names."""

        def normalize(nodes):
            occupied = {str(node.name) for node in nodes if node.name}
            seen = set()
            for node in nodes:
                if node.name:
                    base = str(node.name)
                    name = base
                    if name in seen:
                        index = 2
                        name = f"{base}_{index}"
                        while name in occupied:
                            index += 1
                            name = f"{base}_{index}"
                        node.name = name
                        occupied.add(name)
                    seen.add(name)
                for attribute in node.attribute:
                    if attribute.HasField("g"):
                        normalize(attribute.g.node)
                    for graph in attribute.graphs:
                        normalize(graph.node)

        normalize(model.graph.node)
        for function in model.functions:
            normalize(function.node)

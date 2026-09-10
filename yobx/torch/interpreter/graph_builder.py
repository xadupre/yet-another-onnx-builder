"""Implements the onnx-light graph builder adapter used by Torch exporters."""

import ast
import os

import numpy
from onnx_light import onnx
from onnx_light.onnx import helper, numpy_helper
from onnx_light.onnx_core.graph_builder import GraphBuilder as NativeGraphBuilder

from ...builder.onnxlight import OnnxLightGraphBuilder
from ...container.model_container import _get_type
from ...xbuilder._wrap_dim import WrapDim
from ...xbuilder._wrap_sym import WrapSym
from ..new_tracing.shape import TracingInt
from ..torch_helper import torch_dtype_to_onnx_dtype


class TorchOnnxLightGraphBuilder(OnnxLightGraphBuilder):
    """Adapts the native graph builder to the PyTorch converter contract."""

    TEMPLATE_TYPE = 999
    WrapDim = WrapDim
    WrapSym = WrapSym
    TracingInt = TracingInt

    MINUS_TWO = numpy.array([-2], dtype=numpy.int64)
    MINUS_ONE = numpy.array([-1], dtype=numpy.int64)
    ONE = numpy.array([1], dtype=numpy.int64)
    ONE_NO_DIM = numpy.array(1, dtype=numpy.int64)
    ZERO = numpy.array([0], dtype=numpy.int64)
    ZERO_NO_DIM = numpy.array(0, dtype=numpy.int64)
    END = numpy.array([numpy.iinfo(numpy.int64).max], dtype=numpy.int64)

    def __init__(
        self,
        target_opset_or_existing_proto=18,
        input_names=None,
        as_function=False,
        optimization_options=None,
        args=None,
        kwargs=None,
        ir_version=None,
        verbose=0,
        infer_shapes_options=None,
        raise_list=None,
        dynamic_shapes=None,
        local_domain="local_function",
        signature=None,
        check_empty_source=False,
        graph_module=None,
        exe_path="",
        output_names=None,
        output_dynamic_shapes=None,
        convert_options=None,
        _parent=None,
        **legacy_kwargs,
    ):
        """Initializes native graph state and Torch-specific conversion metadata."""
        if legacy_kwargs:
            raise TypeError(
                f"Unsupported Torch graph builder options: {sorted(legacy_kwargs)!r}."
            )
        if infer_shapes_options not in (None, False, 0):
            raise ValueError("Native onnx-light shape inference is always enabled.")
        super().__init__(
            target_opset_or_existing_proto,
            ir_version=ir_version,
            optimization_options=optimization_options,
            convert_options=convert_options,
            verbose=verbose,
            as_function=as_function,
        )
        import torch

        self.torch = torch
        self._has_torch = True
        self.input_args = args
        self.input_kwargs = kwargs
        self.dynamic_shapes = dynamic_shapes
        self.output_dynamic_shapes = output_dynamic_shapes
        self.dynamic_objects = {}
        self.dynamic_objects_rev = {}
        self.dynamic_dimensions_source = {}
        self.dynamic_dimensions_source_flat = None
        self.output_dynamic_dimensions_source_flat = None
        self._dynamic_alias = {}
        self._dimension_equivalences = {}
        self._known_torch_value = {}
        self._registered_users = {}
        self.statistics_ = {}
        self._torch_unique_node_names = set()
        self.functions_builder = {}
        self.raise_list = raise_list
        self._raise_list = set(raise_list or ())
        self.local_domain = local_domain
        self.signature = signature
        self.check_empty_source = check_empty_source
        self.graph_module = graph_module
        self.user_defined_output_names = list(output_names or ())
        self.was_inputs_renamed = bool(input_names)
        self.update_dynamic_shape_when_input_name_is_defined = False
        self._requested_input_names = list(input_names or ())
        self.current_input = 0
        self._parent = _parent
        self._cache_shape = {}
        self._sequence_metadata = {}
        self._debug_msg = {"EXEPATH": exe_path}
        self._debug_print_node = set(os.environ.get("PRINTNAME", "").split(",")) - {""}
        self._register_dynamic_shape_specification(dynamic_shapes)
        self._register_dynamic_shape_specification(output_dynamic_shapes)

    def _register_dynamic_shape_specification(self, specification, input_name=None):
        if specification is None:
            return
        if isinstance(specification, dict):
            for key, value in specification.items():
                if isinstance(key, int):
                    self._register_dynamic_shape_specification(value, input_name)
                else:
                    self._register_dynamic_shape_specification(value, str(key))
            return
        if isinstance(specification, (list, tuple)):
            for value in specification:
                self._register_dynamic_shape_specification(value, input_name)
            return
        name = self._dimension_name(specification)
        if name is not None:
            self.add_dynamic_object(name, name, check_tokens=False)

    @staticmethod
    def _dimension_name(value):
        if isinstance(value, WrapDim):
            return value.name_as_string
        if isinstance(value, str):
            return value
        if hasattr(value, "__name__"):
            return value.__name__
        return None

    def empty_copy(self, as_function=False):
        """Creates an empty Torch builder preserving native and converter options."""
        builder = type(self)(
            self.opsets,
            ir_version=self.ir_version,
            optimization_options=self.optimization_options,
            convert_options=self.convert_options,
            verbose=self.verbose,
            as_function=as_function,
            local_domain=self.local_domain,
            raise_list=self.raise_list,
            _parent=self,
        )
        builder.dynamic_objects = self.dynamic_objects.copy()
        builder.dynamic_objects_rev = {
            key: list(values) for key, values in self.dynamic_objects_rev.items()
        }
        builder.dynamic_dimensions_source = {
            key: list(values) for key, values in self.dynamic_dimensions_source.items()
        }
        builder._dynamic_alias = self._dynamic_alias.copy()
        builder._dimension_equivalences = {
            key: set(values) for key, values in self._dimension_equivalences.items()
        }
        return builder

    def make_subset_builder(self, input_names, name="", domain="", add_local_functions=False):
        """Creates a Torch child builder containing selected input descriptors."""
        builder = super().make_subset_builder(
            input_names, name=name, domain=domain, add_local_functions=add_local_functions
        )
        builder.functions_builder = self.functions_builder.copy()
        return builder

    def make_tensor_input(
        self,
        name,
        elem_type=None,
        shape=None,
        device=None,
        default_initializer=None,
        marker="",
        users=None,
    ):
        """Declares a Torch tensor input and applies legacy input renaming."""
        del marker, users
        elem_type = _get_type(elem_type) if elem_type is not None else None
        input_name = (
            self._requested_input_names[self.current_input]
            if self.current_input < len(self._requested_input_names)
            else name
        )
        shape = self.verify_dynamic_shape(shape, name=input_name)
        self.current_input += 1
        normalized_device = self._device_index(device)
        super().make_tensor_input(input_name, elem_type, shape, device=normalized_device)
        if default_initializer is not None:
            tensor = numpy_helper.from_array(numpy.asarray(default_initializer), name=input_name)
            self._inner.make_initializer(tensor)
        if input_name != name:
            self.make_node("Identity", [input_name], [name], name="make_tensor_input_id")
        return name

    def make_tensor_output(
        self,
        name,
        elem_type=None,
        shape=None,
        indexed=False,
        allow_untyped_output=False,
        doc_string="",
    ):
        """Declares Torch tensor outputs while accepting converter documentation."""
        result = super().make_tensor_output(
            name,
            _get_type(elem_type) if elem_type is not None else None,
            self.verify_dynamic_shape(shape, name=name) if shape is not None else None,
            indexed=indexed,
            allow_untyped_output=allow_untyped_output,
        )
        if doc_string:
            model = self._native_model()
            names = [name] if isinstance(name, str) else list(name)
            for value in model.graph.output:
                if str(value.name) in names:
                    value.doc_string = doc_string
            self._inner = NativeGraphBuilder(model)
        return result

    def make_node(
        self, op_type, inputs, outputs=1, domain="", attributes=None, name=None, **kwargs
    ):
        """Creates a native node after removing Torch converter-only options."""
        kwargs.pop("check", None)
        metadata_props = kwargs.pop("metadata_props", None)
        if (
            op_type == "SequenceAt"
            and self._domain(domain) == ""
            and inputs
            and self.is_sequence(inputs[0])
        ):
            if attributes or kwargs:
                raise ValueError("SequenceAt does not accept attributes.")
            result = self._make_sequence_at(inputs, outputs, name)
            return result
        result = super().make_node(
            op_type,
            inputs,
            outputs=outputs,
            domain=domain,
            attributes=attributes,
            name=name,
            **kwargs,
        )
        if metadata_props:
            model = self._native_model()
            node = model.graph.node[-1]
            node.metadata_props.extend(
                onnx.StringStringEntryProto(key=str(key), value=str(value))
                for key, value in metadata_props.items()
            )
            self._inner = NativeGraphBuilder(model)
        return result

    @staticmethod
    def _device_index(device):
        if device is None or isinstance(device, int):
            return device
        if getattr(device, "type", None) == "cpu":
            return -1
        if getattr(device, "index", None) is not None:
            return int(device.index)
        raise TypeError(f"Unable to convert device {device!r} into an index.")

    def set_device(self, name, device, exc=True, keep_this_device=False):
        """Records a Torch device with optional consistency validation."""
        device = self._device_index(device)
        if self.has_device(name) and self.get_device(name) != device and not keep_this_device:
            if exc:
                raise ValueError(
                    f"Conflicting devices for {name!r}: {self.get_device(name)} and {device}."
                )
            return
        super().set_device(name, device)

    def _check_constants(self, prefix="before-inline", add=None):
        """Checks structural invariants for native Constant nodes."""

        def check(nodes, where):
            for node in nodes:
                if node.op_type != "Constant":
                    continue
                if node.input or len(node.output) != 1 or len(node.attribute) != 1:
                    raise AssertionError(f"Malformed Constant node at {where!r}: {node!r}.")

        model = self._native_model()
        check(model.graph.node, prefix)
        for function in model.functions:
            check(function.node, f"{prefix}-[{function.domain}.{function.name}]")
        if add is not None:
            if not isinstance(add, (onnx.FunctionProto, onnx.GraphProto)):
                raise TypeError(f"Unexpected constant container {type(add)!r}.")
            check(add.node, f"{prefix}-[add]")

    def _check_function_order(self):
        """Checks that local functions are defined before they are referenced."""
        model = self._native_model()
        known = set()
        standard = {"", "ai.onnx.ml", "com.microsoft", "ai.onnx.training"}
        for function in model.functions:
            key = (str(function.domain), str(function.name))
            for node in function.node:
                domain = str(node.domain)
                if domain in standard or domain.startswith("ai.") or ".onnx" in domain:
                    continue
                if (domain, str(node.op_type)) not in known:
                    raise AssertionError(
                        f"Function {key!r} references an unavailable local function "
                        f"{(domain, str(node.op_type))!r}."
                    )
            known.add(key)
        for node in model.graph.node:
            domain = str(node.domain)
            if (
                domain not in standard
                and not domain.startswith("ai.")
                and (domain, str(node.op_type)) not in known
            ):
                raise AssertionError(
                    f"Node {(domain, str(node.op_type))!r} has no local function."
                )

    def _check_two_shapes_are_compatible(self, old_shape, shape, register_int=True, name=None):
        """Checks ranks and static dimensions and records symbolic equivalences."""
        del register_int
        if len(old_shape) != len(shape):
            raise AssertionError(
                f"Rank mismatch for {name!r}: {tuple(old_shape)!r} != {tuple(shape)!r}."
            )
        for left, right in zip(old_shape, shape):
            left = self._normalize_dimension(left, add=False)
            right = self._normalize_dimension(right, add=False)
            if isinstance(left, int) and isinstance(right, int) and left != right:
                raise AssertionError(
                    f"Incompatible shapes for {name!r}: "
                    f"{tuple(old_shape)!r} != {tuple(shape)!r}."
                )
            if isinstance(left, str) and isinstance(right, str) and left != right:
                self._dimension_equivalences.setdefault(left, set()).add(right)
                self._dimension_equivalences.setdefault(right, set()).add(left)

    def _get_shape_(self, name):
        """Returns a compact shape description for diagnostics."""
        if self.has_shape(name):
            return f"{name}:{self.get_shape(name)}"
        if self.has_rank(name):
            return f"{name}:{('?',) * self.get_rank(name)}"
        return f"{name}:?"

    def _torch_sym_int_to_str(self, value):
        """Converts a Torch symbolic integer into an integer or expression string."""
        if isinstance(value, str):
            return value
        if isinstance(value, TracingInt):
            return value.value
        if hasattr(value, "node") and isinstance(value.node, str):
            return value.node
        from torch.fx.experimental.sym_node import SymNode

        if hasattr(value, "node") and isinstance(value.node, SymNode):
            return str(value.node._expr).replace(" ", "")
        try:
            return int(value)
        except (
            TypeError,
            ValueError,
            AttributeError,
            self.torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode,
        ) as exc:
            raise AssertionError(f"Unable to convert {value!r} into a dimension.") from exc

    def add_dynamic_object(self, key, value, name=None, dim=None, parse=False, check_tokens=True):
        """Registers a symbolic dimension or dimension expression."""
        key = self._dimension_name(key)
        if key is None:
            raise TypeError(f"Unexpected dynamic object key {key!r}.")
        self.dynamic_objects[key] = value
        value_key = self._dynamic_value_key(value)
        self.dynamic_objects_rev.setdefault(value_key, [])
        if not any(
            existing_key == key for existing_key, _ in self.dynamic_objects_rev[value_key]
        ):
            self.dynamic_objects_rev[value_key].append((key, value))
        if name is not None and dim is not None:
            self.dynamic_dimensions_source.setdefault(key, []).append(
                {"input_name": name, "axis": dim}
            )
        tokens = self._expression_names(key)
        if (parse or check_tokens) and tokens - {key}:
            missing = {
                token
                for token in tokens - {key}
                if token not in self.dynamic_objects and not self._dimension_is_declared(token)
            }
            if missing:
                raise AssertionError(
                    f"Dynamic expression {key!r} uses unknown dimensions {sorted(missing)!r}."
                )

    def _dynamic_value_key(self, value):
        if isinstance(value, WrapSym):
            return value.name or str(value)
        if isinstance(value, WrapDim):
            return value.name_as_string
        try:
            return str(self._torch_sym_int_to_str(value))
        except AssertionError:
            return str(value)

    @staticmethod
    def _expression_names(expression):
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError:
            return {expression}
        return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    def _dimension_is_declared(self, dimension):
        return any(
            self.has_shape(name) and dimension in self.get_shape(name)
            for name in self.input_names
        )

    def add_stat(self, kind, name):
        """Increments a Torch conversion statistic."""
        statistics = self.statistics_.setdefault(kind, {})
        statistics[name] = statistics.get(name, 0) + 1

    def extract_input_names_from_args(self, args):
        """Extracts known graph values from nested Torch converter arguments."""
        names = []

        def visit(value):
            if isinstance(value, str) and self.has_name(value):
                if value not in names:
                    names.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)
            elif isinstance(value, slice):
                visit(value.start)
                visit(value.stop)
                visit(value.step)

        visit(args)
        return names

    def get_input_dynamic_shape(
        self, name, input_index, example_shape, dynamic_shapes=None, example_value=None
    ):
        """Combines example tensor dimensions with a Torch dynamic-shape specification."""
        specification = self.dynamic_shapes if dynamic_shapes is None else dynamic_shapes
        if isinstance(example_value, list):
            if not example_value:
                raise ValueError("A tensor sequence example cannot be empty.")
            item_specification = specification
            if isinstance(specification, tuple):
                item_specification = (
                    specification[input_index] if input_index < len(specification) else None
                )
            elif isinstance(specification, dict):
                item_specification = specification.get(name)
            return [
                self.get_input_dynamic_shape(
                    None, 0, example_value[0].shape, dynamic_shapes=(item_specification,)
                )
            ]
        if example_shape is None:
            if self.as_function:
                return None
            raise ValueError(f"No example shape is available for input {name!r}.")
        if specification is None:
            return self.verify_dynamic_shape(example_shape, name=name)
        if isinstance(specification, tuple):
            info = specification[input_index] if input_index < len(specification) else None
        elif isinstance(specification, dict):
            info = specification.get(name)
        else:
            raise TypeError(f"Unexpected dynamic_shapes type {type(specification)!r}.")
        result = list(example_shape)
        if info is None:
            return self.verify_dynamic_shape(result, name=name)
        info_name = self._dimension_name(info)
        if info_name is not None:
            if len(result) != 1:
                raise ValueError(
                    f"A scalar dynamic-shape specification requires rank one, not {result!r}."
                )
            info = {0: info}
        items = info.items() if isinstance(info, dict) else enumerate(info)
        for axis, dimension in items:
            if axis >= len(result):
                continue
            dimension_name = self._dimension_name(dimension)
            if dimension_name is not None:
                result[axis] = dimension_name
                self.add_dynamic_object(
                    dimension_name, dimension_name, name=name, dim=axis, check_tokens=False
                )
        return self.verify_dynamic_shape(result, name=name)

    def get_local_function(self, name, domain="", builder=False):
        """Returns a registered native local function or its source builder."""
        if builder:
            return self.functions_builder[domain, name]
        return self.functions[domain, name]

    def get_local_function_outputs(self, name, domain=""):
        """Returns the output names declared by a native local function."""
        return tuple(str(value) for value in self.get_local_function(name, domain).output)

    def get_type_known(self, name, exc=False):
        """Returns the ONNX dtype recorded from Torch FX metadata."""
        value = self._known_torch_value.get(name)
        if value is None:
            return None
        candidates = []

        def visit(item):
            if isinstance(item, tuple):
                if len(item) == 3:
                    candidates.append(item)
                for element in item:
                    visit(element)

        visit(value)
        for candidate in reversed(candidates):
            dtype = candidate[1]
            if isinstance(dtype, self.torch.dtype):
                return torch_dtype_to_onnx_dtype(dtype)
        if exc:
            raise AssertionError(f"No valid Torch dtype metadata is available for {name!r}.")
        return None

    def has_local_function(self, name, domain="", builder=False):
        """Returns whether a native local function or source builder is registered."""
        if builder:
            return (domain, name) in self.functions_builder
        return (domain, name) in self.functions

    def is_dynamic_dimension(
        self, dim, verify=True, allow_none=False, allow_new_dynamic_dimension=False
    ):
        """Returns whether one dimension is symbolic."""
        if dim is None:
            if allow_none:
                return True
            if verify:
                raise AssertionError("None is not an allowed dynamic dimension.")
            return False
        normalized = self._normalize_dimension(dim, add=allow_new_dynamic_dimension)
        if isinstance(normalized, int):
            return False
        if verify and (
            normalized not in self.dynamic_objects
            and not self._dimension_is_declared(normalized)
            and not self.has_name(normalized)
        ):
            raise AssertionError(f"Dynamic dimension {normalized!r} is not registered.")
        return True

    def is_dynamic_shape(
        self, shape, verify=True, allow_none=False, allow_new_dynamic_dimension=False
    ):
        """Returns whether a shape contains at least one symbolic dimension."""
        dynamic = False
        for dimension in shape:
            if self.is_dynamic_dimension(
                dimension,
                verify=verify,
                allow_none=allow_none,
                allow_new_dynamic_dimension=allow_new_dynamic_dimension,
            ):
                dynamic = True
        return dynamic

    def make_dynamic_object(self, name, value, shape_as_input=False, input_name=None, axis=None):
        """Creates a symbolic dimension and optionally exposes it as a scalar input."""
        if name in self.dynamic_objects:
            if input_name is not None:
                self.dynamic_dimensions_source.setdefault(name, []).append(
                    {"input_name": input_name, "axis": axis}
                )
            return name if shape_as_input and self.has_name(name) else None
        self.add_dynamic_object(
            name, value, name=input_name, dim=axis, parse=True, check_tokens=False
        )
        if shape_as_input:
            if not self.has_name(name):
                dtype = (
                    onnx.TensorProto.FLOAT
                    if isinstance(value, self.torch.SymFloat)
                    else onnx.TensorProto.INT64
                )
                super().make_tensor_input(name, dtype, tuple())
                self.set_value_shape(name, (name,))
            return name
        return name

    def make_new_dynamic_shape(self, rank, prefix="d"):
        """Creates a shape containing fresh Torch symbolic dimensions."""
        if not isinstance(rank, int) or rank < 0:
            raise ValueError(f"Invalid dynamic rank {rank!r}.")
        dimensions = []
        for index in range(rank):
            name = self.unique_dimension_name(f"{prefix}_d{index}")
            value = WrapDim(name)
            self.add_dynamic_object(name, value, check_tokens=False)
            dimensions.append(value)
        return tuple(dimensions)

    def make_shape_from_results(self, shape, name=""):
        """Creates a rank-one INT64 shape tensor from static and symbolic dimensions."""
        if not isinstance(shape, (list, tuple)):
            raise TypeError(f"Unexpected shape type {type(shape)!r}.")
        normalized = self.verify_dynamic_shape(shape)
        if normalized is None:
            raise AssertionError("A concrete shape cannot normalize to None.")
        cache_key = tuple(normalized)
        if cache_key in self._cache_shape:
            return self._cache_shape[cache_key]
        if all(isinstance(dimension, int) for dimension in normalized):
            result = self.make_initializer("", numpy.asarray(normalized, dtype=numpy.int64))
            self._cache_shape[cache_key] = result
            return result
        parts = []
        for dimension in normalized:
            if isinstance(dimension, int):
                parts.append(numpy.asarray([dimension], dtype=numpy.int64))
                continue
            scalar = self._dimension_expression_result(dimension)
            if not self.has_rank(scalar):
                raise ValueError(f"No rank is known for dynamic dimension {dimension!r}.")
            if self.get_rank(scalar) == 0:
                scalar = self.op.UnsqueezeAnyOpset(
                    scalar, self.ZERO, name=f"_mkshape_{name or dimension}"
                )
            elif self.get_rank(scalar) != 1:
                raise ValueError(
                    f"Dynamic dimension result {scalar!r} has rank {self.get_rank(scalar)}."
                )
            parts.append(scalar)
        result = (
            self.op.Concat(*parts, axis=0, name=f"_mkshape_{name}")
            if len(parts) > 1
            else self.op.Identity(parts[0], name=f"_mkshape_{name}")
        )
        self.set_value_shape(result, normalized)
        self._cache_shape[cache_key] = result
        return result

    def _dimension_expression_result(self, expression):
        if self.has_name(expression):
            return expression
        if expression.isidentifier():
            return self.get_dimension_as_result(expression)
        tree = ast.parse(expression, mode="eval")

        def build(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                return numpy.asarray(node.value, dtype=numpy.int64)
            if isinstance(node, ast.Name):
                return self.get_dimension_as_result(node.id)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return self.op.Neg(build(node.operand))
            if isinstance(node, ast.BinOp):
                op_type = (
                    "Add"
                    if isinstance(node.op, ast.Add)
                    else (
                        "Sub"
                        if isinstance(node.op, ast.Sub)
                        else (
                            "Mul"
                            if isinstance(node.op, ast.Mult)
                            else (
                                "Div"
                                if isinstance(node.op, ast.FloorDiv)
                                else "Mod" if isinstance(node.op, ast.Mod) else None
                            )
                        )
                    )
                )
                if op_type is not None:
                    return self.make_node(op_type, [build(node.left), build(node.right)])
            raise ValueError(f"Unsupported dynamic dimension expression {expression!r}.")

        return build(tree.body)

    def make_tensor_sequence_input(self, name, elem_type, shape, marker=""):
        """Declares a tensor-sequence input with legacy input-name bookkeeping."""
        del marker
        input_name = (
            self._requested_input_names[self.current_input]
            if self.current_input < len(self._requested_input_names)
            else name
        )
        self.current_input += 1
        if self.has_name(input_name):
            raise ValueError(f"Input name {input_name!r} already exists.")
        shape = self.verify_dynamic_shape(shape, name=input_name)
        dtype = _get_type(elem_type)
        value = helper.make_tensor_sequence_value_info(input_name, dtype, shape)
        self._inner.make_input(value)
        metadata = {
            "dtype": dtype,
            "shapes": (shape,) if shape is not None else None,
            "ranks": (len(shape),) if shape is not None else None,
            "unknown": False,
        }
        self._sequence_metadata[input_name] = metadata
        if input_name != name:
            self._make_sequence_identity(input_name, name)
            self._sequence_metadata[name] = metadata.copy()
        return name

    def is_sequence(self, name):
        """Returns whether a tensor-sequence input is registered."""
        return name in self._sequence_metadata or super().is_sequence(name)

    def get_sequence(self, name):
        """Returns registered tensor-sequence element metadata."""
        if name not in self._sequence_metadata:
            raise AssertionError(f"Sequence {name!r} is not known{self.get_debug_msg()}")
        return self._sequence_metadata[name]

    def _make_sequence_identity(self, input_name, output_name):
        """Creates a sequence Identity without unsupported native shape inference."""
        self._make_sequence_node("Identity", [input_name], [output_name], "sequence_identity")

    def _make_sequence_node(self, op_type, inputs, outputs, name):
        """Creates a native sequence node while bypassing tensor-only ShapesContext."""
        for output in outputs:
            if self.has_name(output):
                raise ValueError(f"Output name {output!r} already exists.")
        self._synchronize_annotations()
        node_name = str(name) if name else self.unique_name(op_type)
        base_name = node_name
        index = 2
        while node_name in self._node_names:
            node_name = f"{base_name}_{index}"
            index += 1
        self._inner.make_node(op_type, list(inputs), list(outputs), "", node_name, [])
        self._node_names.add(node_name)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def _make_sequence_at(self, inputs, outputs, name):
        """Creates SequenceAt and applies explicit element tensor metadata."""
        normalized_inputs = [
            value if isinstance(value, str) else self.make_initializer("", value)
            for value in inputs
        ]
        if isinstance(outputs, int):
            outputs = [self.unique_name("sequence_at") for _ in range(outputs)]
        elif isinstance(outputs, str):
            outputs = [outputs]
        else:
            outputs = list(outputs)
        if len(outputs) != 1:
            raise ValueError("SequenceAt requires exactly one output.")
        metadata = self.get_sequence(normalized_inputs[0])
        if metadata["unknown"]:
            raise NotImplementedError(
                "SequenceAt requires known sequence element type and rank metadata."
            )
        position = self.get_constant(normalized_inputs[1], exc=False)
        index = int(numpy.asarray(position).item()) if position is not None else 0
        result = self._make_sequence_node(
            "SequenceAt", normalized_inputs, outputs, name or "SequenceAt"
        )
        dtype = metadata["dtype"]
        if isinstance(dtype, tuple):
            dtype = dtype[min(index, len(dtype) - 1)]
        self.set_type(result, dtype)
        shapes = metadata["shapes"]
        ranks = metadata["ranks"]
        if shapes is not None:
            shape = shapes[min(index, len(shapes) - 1)]
            self.set_shape(result, shape, allow_zero=True)
        elif ranks is not None:
            rank = ranks if isinstance(ranks, int) else ranks[min(index, len(ranks) - 1)]
            self.set_rank(result, rank)
        else:
            raise NotImplementedError(
                "SequenceAt requires sequence element shape or rank metadata."
            )
        return result

    def make_local_function(self, builder, function_options, optimize=False, metadata_props=None):
        """Registers a native local function and remembers its Torch source builder."""
        initializers, key = super().make_local_function(
            builder, function_options, optimize=optimize
        )
        self.functions_builder[key] = builder
        if metadata_props:
            model = self._native_model()
            function = next(
                value for value in model.functions if (str(value.domain), str(value.name)) == key
            )
            function.metadata_props.extend(
                onnx.StringStringEntryProto(key=str(k), value=str(v))
                for k, v in metadata_props.items()
            )
            self._inner = NativeGraphBuilder(model)
        return initializers, key

    def make_nodes(
        self,
        builder,
        input_names,
        output_names,
        prefix="",
        function_options=None,
        optimize=False,
        force_rename_with_prefix=None,
    ):
        """Appends a Torch child graph inline or as a native local function."""
        if not isinstance(builder, TorchOnnxLightGraphBuilder):
            raise TypeError(f"Expected a Torch child builder, not {type(builder)!r}.")
        if len(input_names) != len(builder.input_names):
            raise ValueError(
                f"Input count mismatch: {len(input_names)} != {len(builder.input_names)}."
            )
        if len(output_names) != len(builder.output_names):
            raise ValueError(
                f"Output count mismatch: {len(output_names)} != {len(builder.output_names)}."
            )
        for domain, version in builder.opsets.items():
            if self.has_opset(domain) and self.get_opset(domain) != version:
                raise ValueError(f"Conflicting opset {domain!r}.")
            if not self.has_opset(domain):
                self.set_opset(domain, version)
        if function_options is not None and function_options.export_as_function:
            initializers, (domain, function_name) = self.make_local_function(
                builder, function_options, optimize=optimize
            )
            self.make_node(
                function_name,
                [*input_names, *initializers],
                output_names,
                domain=domain,
                name=function_name,
            )
            self._copy_output_metadata(builder, output_names)
            return output_names[0] if len(output_names) == 1 else tuple(output_names)

        source = (
            type(builder)(
                builder.to_native(optimize=True, inline=False),
                optimization_options=builder.optimization_options,
                convert_options=builder.convert_options,
                verbose=builder.verbose,
                as_function=True,
            )
            if optimize
            else builder
        )
        self._import_local_functions(source)
        renaming = dict(zip(source.input_names, input_names))
        rename_prefix = force_rename_with_prefix or prefix
        for initializer_name, initializer in source.initializers_dict.items():
            target = self.unique_name(f"{rename_prefix}{initializer_name}")
            renaming[initializer_name] = self.make_initializer(target, initializer)
        for node in source.nodes:
            node_inputs = [renaming.get(str(value), str(value)) for value in node.input]
            node_outputs = [
                ("" if not value else self.unique_name(f"{rename_prefix}{str(value)}"))
                for value in node.output
            ]
            for old, new in zip(node.output, node_outputs):
                if old:
                    renaming[str(old)] = new
            self.make_node(
                str(node.op_type),
                node_inputs,
                node_outputs,
                domain=str(node.domain),
                attributes=list(node.attribute),
                name=str(node.name) or None,
            )
            for old, new in zip(node.output, node_outputs):
                old = str(old)
                if not old:
                    continue
                if source.has_type(old):
                    self.set_type(new, source.get_type(old))
                if source.has_shape(old):
                    self.set_shape(new, source.get_shape(old), allow_zero=True)
                if source.has_device(old):
                    self.set_device(new, source.get_device(old))
        for source_name, target_name in zip(source.output_names, output_names):
            self.make_node("Identity", [renaming[source_name]], [target_name], name=".make_nodes")
        self._copy_output_metadata(source, output_names)
        return output_names[0] if len(output_names) == 1 else tuple(output_names)

    def _import_local_functions(self, builder):
        missing = [
            function for key, function in builder.functions.items() if key not in self.functions
        ]
        if missing:
            model = self._native_model()
            model.functions.extend(missing)
            self._inner = NativeGraphBuilder(model)
        self.functions_builder.update(builder.functions_builder)

    def _copy_output_metadata(self, builder, output_names):
        for source_name, output_name in zip(builder.output_names, output_names):
            if builder.has_type(source_name):
                self.set_type(output_name, builder.get_type(source_name))
            if builder.has_shape(source_name):
                self.set_shape(output_name, builder.get_shape(source_name), allow_zero=True)
            if builder.has_device(source_name):
                self.set_device(output_name, builder.get_device(source_name))

    def process(self, graph_module, interpreter, source_lines=None):
        """Runs the Torch FX interpreter over a graph module."""
        self.graph_module = graph_module
        self._debug_msg["process.graph_module"] = graph_module
        self._debug_msg["process.graph_module.graph"] = getattr(graph_module, "graph", None)
        dispatcher = getattr(interpreter, "dispatcher", None)
        converter = (
            dispatcher.find_function(type(graph_module))
            if isinstance(graph_module, self.torch.nn.Module) and dispatcher is not None
            else None
        )
        if converter is not None:
            args = [
                interpreter.placeholder(value) if hasattr(value, "name") else value
                for value in self.input_args or ()
            ]
            kwargs = {
                key: (interpreter.placeholder(value) if hasattr(value, "name") else value)
                for key, value in (self.input_kwargs or {}).items()
            }
            outputs = converter(self, {}, None, *args, **kwargs)
            outputs = [outputs] if isinstance(outputs, str) else outputs
            for output in outputs:
                self.make_tensor_output(output, allow_untyped_output=True)
            return
        graph = graph_module.graph
        interpreter.start_graph(graph)
        placeholders = [node for node in graph.nodes if node.op == "placeholder"]
        removable = set()
        for node in reversed(placeholders):
            value = node.meta.get("val") if hasattr(node, "meta") else None
            if node.users or not isinstance(value, (int, bool, float, type(None))):
                break
            removable.add(node.name)
        self._debug_msg["process.inputs_to_remove"] = removable
        try:
            for index, node in enumerate(graph.nodes):
                if node.op == "placeholder" and node.name in removable and not node.users:
                    continue
                self._debug_msg["process.progress"] = (
                    f"node {index}/{len(graph.nodes)} target={node.target}"
                )
                interpreter.run_node(node, source_lines=source_lines)
        finally:
            interpreter.end_graph(graph)

    def rank(self, name):
        """Returns the known tensor rank."""
        if not isinstance(name, str):
            raise TypeError(f"Unexpected tensor name type {type(name)!r}.")
        return self.get_rank(name)

    def register_dynamic_objects_from_shape(self, shape):
        """Registers symbolic dimensions used by a shape."""
        for dimension in shape:
            normalized = self._normalize_dimension(dimension, add=True)
            if isinstance(normalized, str) and normalized not in self.dynamic_objects:
                self.add_dynamic_object(normalized, normalized, parse=True, check_tokens=False)

    def register_users(self, name, users):
        """Registers FX consumers for conversion validation."""
        if name in self._registered_users:
            raise AssertionError(f"Users for {name!r} are already registered.")
        self._registered_users[name] = set(users)

    def same_shape(self, x, y):
        """Returns whether two tensor shapes are equal modulo symbolic aliases."""
        if not self.has_shape(x) or not self.has_shape(y):
            raise AssertionError(f"Missing shape for {x!r} or {y!r}.")
        left, right = self.get_shape(x), self.get_shape(y)
        return len(left) == len(right) and all(
            self._same_dimension(a, b) for a, b in zip(left, right)
        )

    def _same_dimension(self, left, right):
        left = self._normalize_dimension(left, add=False)
        right = self._normalize_dimension(right, add=False)
        if left == right:
            return True
        if not isinstance(left, str) or not isinstance(right, str):
            return False
        pending = [left]
        visited = set()
        while pending:
            current = pending.pop()
            if current == right:
                return True
            if current in visited:
                continue
            visited.add(current)
            pending.extend(self._dimension_equivalences.get(current, ()))
        return False

    def set_shapes_types(self, name, where, value):
        """Records Torch FX type and shape metadata."""
        if hasattr(name, "name"):
            name = name.name
        self._known_torch_value[name] = (where, value)

    def unique_node_name(self, name):
        """Returns a unique native node name."""
        candidate = name
        index = 2
        while candidate in self._node_names or candidate in self._torch_unique_node_names:
            candidate = f"{name}{index}"
            index += 1
        self._torch_unique_node_names.add(candidate)
        return candidate

    def verify_dynamic_shape(self, shape, name=None, add=True):
        """Normalizes a Torch shape into integer and symbolic ONNX dimensions."""
        if shape is None:
            return None
        normalized = tuple(self._normalize_dimension(dimension, add=add) for dimension in shape)
        if any(dimension is None for dimension in normalized):
            raise AssertionError(f"Unexpected None dimension in shape {shape!r}.")
        if name is not None:
            for axis, dimension in enumerate(normalized):
                if isinstance(dimension, str):
                    self.dynamic_dimensions_source.setdefault(dimension, []).append(
                        {"input_name": name, "axis": axis}
                    )
        return normalized

    def _normalize_dimension(self, dimension, add):
        if isinstance(dimension, (int, numpy.integer)):
            return int(dimension)
        name = self._dimension_name(dimension)
        if name is None and isinstance(
            dimension, (self.torch.SymInt, self.torch.SymFloat, TracingInt)
        ):
            name = self._torch_sym_int_to_str(dimension)
        if isinstance(name, int):
            return name
        if name is None:
            raise TypeError(f"Unexpected dynamic dimension {dimension!r}.")
        name = str(name).replace(" ", "")
        name = self._dynamic_alias.get(name, name)
        if add and name not in self.dynamic_objects:
            self.add_dynamic_object(name, dimension, parse=True, check_tokens=False)
        return name

    def pretty_text(self, add_fx_graph=False, recursive=True):
        """Returns native graph diagnostics and optional FX graph text."""
        del recursive
        text = super().pretty_text()
        if add_fx_graph and self.graph_module is not None:
            text += f"\n\nFX graph:\n{self.graph_module.graph}"
        return text

"""Exposes native symbolic inference without a Python shape-engine fallback."""

from onnx_light import onnx
from onnx_light.onnx import helper, numpy_helper
from onnx_light.onnx_core.shape_inference import ShapesContext, SymTensor

from .shape_builder_impl import InferenceMode
from .shape_builder import ShapeBuilder
from ._shape_helper import ONNX_SHAPE


class NativeShapeInference(ShapeBuilder):
    """Infers ONNX tensor descriptors with a persistent native ``ShapesContext``.

    ``TYPE`` uses the same native engine but exposes only element types. Unknown
    operators and incompatible constraints raise the native exception, including
    when the legacy ``exc=False`` argument is supplied. Custom inference can be
    registered directly on :attr:`context`.
    """

    def __init__(self, verbose=0, opset=None):
        self.verbose = verbose
        self.opsets = {"": opset} if isinstance(opset, int) else dict(opset or {"": 18})
        self.context = ShapesContext()
        self._input_names = []
        self._output_names = []
        self._inference = InferenceMode.SHAPE
        self._model = None
        self._devices = {}
        self._constants = {}
        self._unshaped = set()
        for domain, version in self.opsets.items():
            self.context.set_opset_version(domain, version)

    @property
    def input_names(self):
        """Returns the model input names."""
        return self._input_names

    @property
    def output_names(self):
        """Returns the model output names."""
        return self._output_names

    @property
    def main_opset(self):
        """Returns the default-domain opset."""
        return self.opsets.get("", self.opsets.get("ai.onnx", 18))

    @property
    def _known_types(self):
        return {name: self.get_type(name) for name in self.context.names() if self.has_type(name)}

    @property
    def _known_shapes(self):
        return {
            name: self.get_shape(name) for name in self.context.names() if self.has_shape(name)
        }

    @property
    def _known_ranks(self):
        return {name: len(shape) for name, shape in self._known_shapes.items()}

    @property
    def _known_value_shape(self):
        return {
            name: self.value_as_shape(name)
            for name in self.context.names()
            if self.context.get(name).has_value_as_shape()
        }

    def reset_types_and_shapes(self):
        """Clears inferred descriptors and native callbacks."""
        self.context.clear()
        self._input_names = []
        self._output_names = []
        self._model = None
        self._devices.clear()
        self._constants.clear()
        self._unshaped.clear()
        self._inference = InferenceMode.SHAPE
        for domain, version in self.opsets.items():
            self.context.set_opset_version(domain, version)

    def has_type(self, name):
        """Returns whether a native element type is available."""
        return self.context.has(str(name)) and self.context.get(str(name)).dtype != 0

    def get_type(self, name):
        """Returns the native element type, or zero when unavailable."""
        return self.context.get(str(name)).dtype if self.context.has(str(name)) else 0

    def has_shape(self, name, full=False):
        """Returns whether a native shape is available."""
        if self._inference in (InferenceMode.TYPE, InferenceMode.NOTHING):
            return False
        if str(name) in self._unshaped or not self.context.has(str(name)):
            return False
        return not full or all(isinstance(d, int) and d >= 0 for d in self.get_shape(name))

    def get_shape(self, name: str) -> ONNX_SHAPE:
        """Returns native dimensions, preserving anonymous dimensions as ``None``."""
        if not self.has_shape(name):
            raise KeyError(f"Shape is unknown for {name!r}.")
        return tuple(d if d != "" else None for d in self.context.get(str(name)).shape.dims())

    def has_rank(self, name):
        """Returns whether a native shape is available."""
        return self.has_shape(name)

    def get_rank(self, name):
        """Returns the inferred rank."""
        return len(self.get_shape(name))

    def set_type(self, name: str, itype: int, exc: bool = True) -> bool:
        """Seeds a native descriptor with an element type."""
        previous = self.context.get(str(name)) if self.context.has(str(name)) else None
        if previous is None:
            self._unshaped.add(str(name))
        tensor = SymTensor(int(itype), previous.shape.dims() if previous is not None else [])
        if previous is not None and previous.has_value_as_shape():
            tensor.set_value_as_shape(previous.value_as_shape())
        self.context.set(str(name), tensor)
        return True

    def set_shape(self, name, shape, exc=False, **kwargs):
        """Seeds a native descriptor with dimensions."""
        tensor = SymTensor(self.get_type(name), ["" if d is None else d for d in shape])
        if self.context.has(str(name)):
            previous = self.context.get(str(name))
            if previous.has_value_as_shape():
                tensor.set_value_as_shape(previous.value_as_shape())
        self.context.set(str(name), tensor)
        self._unshaped.discard(str(name))
        return True

    def set_rank(self, name, rank):
        """Seeds a rank with anonymous native dimensions."""
        return self.set_shape(name, (None,) * rank)

    def has_device(self, name):
        """Returns whether optional device metadata was supplied."""
        return str(name) in self._devices

    def get_device(self, name):
        """Returns caller-supplied device metadata."""
        return self._devices[str(name)]

    def set_device(self, name: str, device: int, **kwargs) -> None:
        """Stores optional device metadata without performing inference."""
        self._devices[str(name)] = device

    def set_constant(self, name, value):
        """Registers a literal through native Constant inference."""
        if not isinstance(value, onnx.TensorProto):
            raise TypeError("set_constant expects a native TensorProto.")
        context = ShapesContext() if self.context.has(str(name)) else self.context
        for domain, version in self.opsets.items():
            context.set_opset_version(domain, version)
        context.compute_shape_node(helper.make_node("Constant", [], [str(name)], value=value))
        if context is not self.context:
            self.context.set(str(name), context.get(str(name)))
        self._constants[str(name)] = value
        self._unshaped.discard(str(name))

    def is_constant(self, name):
        """Returns whether a literal tensor is registered."""
        return str(name) in self._constants

    def get_constant(self, name, exc=True, computed_value=False, as_shape=False, **kwargs):
        """Returns a registered literal without evaluating Python operators."""
        if as_shape:
            value = self.value_as_shape(name)
            if value is not None:
                return value
        if not self.is_constant(name):
            if exc:
                raise KeyError(f"No literal is registered for {name!r}.")
            return None
        value = numpy_helper.to_array(self._constants[str(name)])
        return tuple(value.tolist()) if as_shape and value.ndim == 1 else value

    def has_opset(self, name):
        """Returns whether the domain has an opset."""
        return self.context.has_opset_version(str(name))

    def get_opset(self, name):
        """Returns the native domain opset."""
        return self.context.opset_version(str(name))

    def set_opset(self, name: str, version: int) -> None:
        """Registers a domain opset with the native engine."""
        self.opsets[str(name)] = version
        self.context.set_opset_version(str(name), version)

    def set_value_shape(self, name, value, equal_to=None):
        """Seeds native shape-tensor values."""
        scalar = isinstance(value, (int, str))
        tensor = (
            self.context.get(str(name))
            if self.context.has(str(name))
            else SymTensor(onnx.TensorProto.INT64, [] if scalar else [len(value)])
        )
        tensor.set_value_as_shape([value] if scalar else list(value))
        self.context.set(str(name), tensor)
        self._unshaped.discard(str(name))
        if equal_to:
            self.context.add_constraint(str(equal_to[0]), str(equal_to[1]))

    def value_as_shape(self, name: str) -> ONNX_SHAPE | None:
        """Returns native shape-tensor values, or ``None`` when unavailable."""
        if not self.context.has(str(name)):
            return None
        tensor = self.context.get(str(name))
        if not tensor.has_value_as_shape():
            return None
        return tuple(d if d != "" else None for d in tensor.value_as_shape().dims())

    def run_node(self, node, exc=False, cost=True):
        """Infers a node natively and optionally estimates its arithmetic cost."""
        unknown_rank = [str(name) for name in node.input if str(name) in self._unshaped]
        if unknown_rank:
            raise ValueError(
                "The native ShapesContext cannot distinguish unknown rank from a "
                f"scalar; provide input ranks for {unknown_rank}."
            )
        self.context.compute_shape_node(node)
        return self.estimate_node_flops(node) if cost else None

    def run_model(self, model, functions=None, exc=False, inference=InferenceMode.SHAPE):
        """Runs native inference and optionally returns per-node cost statistics."""
        if isinstance(inference, str):
            if inference.upper() not in InferenceMode.__members__:
                raise ValueError(f"Unsupported inference mode {inference!r}.")
            inference = InferenceMode[inference.upper()]
        self._inference = InferenceMode(inference)
        if isinstance(model, onnx.GraphProto):
            model = helper.make_model(
                model,
                opset_imports=[helper.make_opsetid(d, v) for d, v in self.opsets.items()],
                functions=list((functions or {}).values()),
            )
        elif not isinstance(model, onnx.ModelProto):
            raise TypeError(f"Expected a native ModelProto or GraphProto, got {type(model)}.")
        self._model = model
        self._input_names = [str(i.name) for i in model.graph.input]
        self._output_names = [str(i.name) for i in model.graph.output]
        self.opsets = {str(o.domain): o.version for o in model.opset_import}
        self._constants = {str(t.name): t for t in model.graph.initializer}
        if self._inference == InferenceMode.NOTHING:
            self.context.clear()
            return None
        if self._inference != InferenceMode.TYPE:
            unknown_rank = [
                str(info.name)
                for info in model.graph.input
                if info.type.HasField("tensor_type")
                and not info.type.tensor_type.HasField("shape")
            ]
            if unknown_rank:
                raise ValueError(
                    "The native ShapesContext cannot distinguish unknown rank from a "
                    f"scalar; provide input ranks for {unknown_rank}."
                )
        self.context.compute_shape_model(model)
        self._unshaped.clear()
        if self._inference == InferenceMode.COST:
            return [
                (
                    str(node.op_type),
                    self.estimate_node_flops(node),
                    tuple(
                        self.get_shape(i) if i and self.has_shape(i) else "?" for i in node.input
                    ),
                )
                for node in model.graph.node
            ]
        return None

    def update_shapes(self, model):
        """Writes native inferred descriptors to a model or graph in place."""
        if self._inference in (InferenceMode.NOTHING, InferenceMode.TYPE):
            return model
        if isinstance(model, onnx.GraphProto):
            self.context.apply_inferred_shapes_to_graph(model)
        else:
            self.context.apply_inferred_shapes_to_model(model)
        return model

    def to_onnx(self):
        """Returns an inferred copy of the most recently processed model."""
        if self._model is None:
            raise ValueError("run_model must be called before to_onnx.")
        model = onnx.ModelProto()
        model.ParseFromString(self._model.SerializeToString())
        return self.update_shapes(model)

    def get_registered_constraints(self) -> dict[str, set[int | str]]:
        """Returns equality constraints recorded by the native engine."""
        result: dict[str, set[int | str]] = {}
        for left, right in self.context.constraints():
            result.setdefault(left, set()).add(
                int(right) if right.lstrip("-").isdigit() else right
            )
            result.setdefault(right, set()).add(int(left) if left.lstrip("-").isdigit() else left)
        return result

    def register_constraint_dimension(
        self, dim_name: str, value: int | str | set[int | str]
    ) -> None:
        """Registers one or several native dimension equalities."""
        self.add_to_constraints(dim_name, value)

    def add_to_constraints(self, dim_name: str, value: int | str | set[int | str]) -> None:
        """Registers one or several native dimension equalities."""
        for item in value if isinstance(value, set) else (value,):
            self.context.add_constraint(str(dim_name), str(item))

    def get_shape_renamed(self, name):
        """Returns dimensions as resolved by the native engine."""
        return self.get_shape(name)

    def estimate_node_flops(self, node):
        """Estimates arithmetic cost using only native descriptors and values."""
        from .cost_inference import estimate_node_flops

        def shape(name: str) -> ONNX_SHAPE | None:
            if not name or not self.has_shape(name):
                return None
            return self.get_shape(name)

        return estimate_node_flops(node, shape, self.value_as_shape)

    def evaluate_cost_with_true_inputs(self, feeds, cost, exc=False):
        """Evaluates symbolic costs against actual input dimensions."""
        from onnx_light.onnx_core.expressions import evaluate_expression

        context = {}
        for name, array in feeds.items():
            if self.has_shape(name):
                context.update(
                    (dim, int(value))
                    for dim, value in zip(self.get_shape(name), array.shape)
                    if isinstance(dim, str)
                )
        result = []
        for op, flops, shapes in cost:
            if isinstance(flops, str):
                try:
                    flops = evaluate_expression(flops, context)
                except RuntimeError:
                    if exc:
                        raise
                    flops = None
            result.append((op, flops, shapes))
        return result

    def evaluate_shape(self, name: str, context: dict[str, int]) -> tuple[int | None, ...]:
        """Evaluates native expressions and preserves anonymous unknown dimensions."""
        from onnx_light.onnx_core.expressions import evaluate_expression

        return tuple(
            evaluate_expression(dim, context) if isinstance(dim, str) else dim
            for dim in self.get_shape(name)
        )

    def get_debug_msg(self, limit=1000):
        """Returns a compact native inference diagnostic."""
        return f"--SHAPE--\n{self._known_shapes}\n--TYPE--\n{self._known_types}"[:limit]

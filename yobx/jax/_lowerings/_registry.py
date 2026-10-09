"""Defines JAX primitive registration and lowering state."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from jax.extend.core import Literal
import numpy as np

from ...helpers.onnx_helper import np_dtype_to_tensor_dtype

DynamicSources = Dict[int, Optional[Tuple[str, int, str]]]
LoweringResult = Union[str, Sequence[str]]
LoweringFunction = Callable[["LoweringState", Any, Sequence[str]], LoweringResult]

_LOWERINGS: Dict[str, LoweringFunction] = {}


def register_lowering(name: str, lowering: LoweringFunction) -> None:
    """Registers one lowering function for a JAX primitive name."""
    if name in _LOWERINGS:
        raise ValueError(f"A lowering is already registered for JAX primitive {name!r}.")
    _LOWERINGS[name] = lowering


def jax_shape(var) -> tuple:
    """Returns the concrete shape of a jaxpr variable."""
    return tuple(var.aval.shape)


def jax_dtype(var) -> int:
    """Returns the ONNX dtype of a jaxpr variable."""
    return np_dtype_to_tensor_dtype(np.dtype(var.aval.dtype))


@dataclass
class LoweringState:
    """Stores the graph and values visible while lowering one jaxpr."""

    builder: Any
    names: Dict[Any, str]
    dynamic_sources: DynamicSources

    def constant(self, value) -> str:
        """Registers a scalar or array literal as an ONNX initializer."""
        return self.builder.make_initializer(
            self.builder.unique_name("constant"), np.asarray(value)
        )

    def value(self, var) -> str:
        """Resolves a jaxpr variable or literal to a graph value."""
        if isinstance(var, Literal):
            return self.constant(var.val)
        return self.names[var]

    def record(self, eqn, values: LoweringResult) -> None:
        """Records lowered outputs and their JAX dtype annotations."""
        if isinstance(values, str):
            values = [values]
        if len(values) != len(eqn.outvars):
            raise ValueError(f"{eqn.primitive.name} produced an unexpected number of outputs.")
        for var, value in zip(eqn.outvars, values):
            self.names[var] = value
            if not self.builder.has_type(value):
                self.builder.set_type(value, jax_dtype(var))


def lower_jaxpr(builder, closed, inputs, dynamic_sources: DynamicSources) -> list[str]:
    """Lowers a closed jaxpr and returns its graph output names."""
    jaxpr = closed.jaxpr
    if len(inputs) != len(jaxpr.invars):
        raise ValueError("Incorrect number of inputs for nested JAX jaxpr.")
    state = LoweringState(builder, dict(zip(jaxpr.invars, inputs)), dynamic_sources)
    for var, const in zip(jaxpr.constvars, closed.consts):
        state.names[var] = state.constant(np.asarray(const))
    for eqn in jaxpr.eqns:
        primitive = eqn.primitive.name
        lowering = _LOWERINGS.get(primitive)
        if lowering is None:
            raise NotImplementedError(
                f"JAX primitive {primitive!r} is not supported by the direct converter."
            )
        values = lowering(state, eqn, [state.value(var) for var in eqn.invars])
        state.record(eqn, values)
    return [state.value(var) for var in jaxpr.outvars]

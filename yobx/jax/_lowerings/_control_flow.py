"""Lowers JAX primitives that contain nested jaxprs."""

from jax.extend.core import ClosedJaxpr

from ._registry import lower_jaxpr, register_lowering


def _lower_nested_jaxpr(state, eqn, args):
    """Lowers a primitive containing a nested jaxpr."""
    primitive = eqn.primitive.name
    inner = eqn.params.get("jaxpr") or eqn.params.get("call_jaxpr") or eqn.params.get("fun_jaxpr")
    if inner is None:
        raise NotImplementedError(f"JAX primitive {primitive!r} has no nested jaxpr.")
    if not isinstance(inner, ClosedJaxpr):
        inner = ClosedJaxpr(inner, ())
    return lower_jaxpr(state.builder, inner, args, state.dynamic_sources)


register_lowering("jit", _lower_nested_jaxpr)
register_lowering("custom_jvp_call", _lower_nested_jaxpr)
register_lowering("custom_vjp_call", _lower_nested_jaxpr)

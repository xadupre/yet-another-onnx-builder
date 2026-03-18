"""Op converter registry for the LiteRT→ONNX converter.

The registry maps :class:`~yobx.litert.litert_helper.BuiltinOperator`
integer codes to converter callables.  Use the
:func:`register_litert_op_converter` decorator to register a new converter:

.. code-block:: python

    from yobx.litert.register import register_litert_op_converter
    from yobx.litert.litert_helper import BuiltinOperator

    @register_litert_op_converter(BuiltinOperator.RELU)
    def convert_relu(g, sts, outputs, op):
        return g.op.Relu(outputs[0] + "_input", outputs=outputs, name="relu")
"""

from typing import Callable, Dict, Optional, Tuple, Union

# Maps BuiltinOperator int (or CUSTOM string) to a converter function.
LITERT_OP_CONVERTERS: Dict[Union[int, str], Callable] = {}


def register_litert_op_converter(op_code: Union[int, str, Tuple[Union[int, str], ...]]):
    """Decorator that registers a converter for one or more TFLite op codes.

    :param op_code: a single :class:`~yobx.litert.litert_helper.BuiltinOperator`
        integer, a custom-op name string, or a tuple of those.
    :raises TypeError: if an op code already has a registered converter.
    """

    def decorator(fct: Callable) -> Callable:
        global LITERT_OP_CONVERTERS
        codes: Tuple[Union[int, str], ...] = op_code if isinstance(op_code, tuple) else (op_code,)
        for code in codes:
            if code in LITERT_OP_CONVERTERS:
                raise TypeError(f"A converter is already registered for LiteRT op code {code!r}.")
            LITERT_OP_CONVERTERS[code] = fct
        return fct

    return decorator


def get_litert_op_converter(op_code: Union[int, str]) -> Optional[Callable]:
    """Return the registered converter for *op_code*, or ``None`` if absent."""
    return LITERT_OP_CONVERTERS.get(op_code)


def get_litert_op_converters() -> Dict[Union[int, str], Callable]:
    """Return a copy of the full registry dictionary."""
    return dict(LITERT_OP_CONVERTERS)

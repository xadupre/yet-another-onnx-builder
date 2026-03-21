"""
Factory for wrapping skl2onnx-style converter functions for use with
:func:`yobx.sklearn.to_onnx`.

This module contains **no** :epkg:`sklearn-onnx` imports.  All interface
objects required by a skl2onnx converter function
(``scope``, ``operator``, ``container``) are provided as lightweight
pure-Python mocks defined in this file.  Only :mod:`onnx` and :mod:`numpy`
(both core :mod:`yobx` dependencies) are imported.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import onnx

# Mapping from ONNX TensorProto element-type integers to NumPy dtypes.
# Defined at module level to avoid rebuilding it on every add_initializer call.
_ONNX_DTYPE_TO_NUMPY: Dict[int, type] = {
    1: np.float32,  # FLOAT
    2: np.uint8,  # UINT8
    3: np.int8,  # INT8
    4: np.uint16,  # UINT16
    5: np.int16,  # INT16
    6: np.int32,  # INT32
    7: np.int64,  # INT64
    10: np.float16,  # FLOAT16
    11: np.float64,  # DOUBLE
    12: np.uint32,  # UINT32
    13: np.uint64,  # UINT64
    14: np.complex64,  # COMPLEX64
    15: np.complex128,  # COMPLEX128
}


# ---------------------------------------------------------------------------
# Pure-Python mocks for the skl2onnx internal interface
# ---------------------------------------------------------------------------


class MockScope:
    """
    Minimal mock for skl2onnx's ``Scope``.

    Provides :meth:`get_unique_variable_name`, the only method called by
    typical skl2onnx converter functions.  Name uniqueness is guaranteed by
    delegating to the :class:`~yobx.xbuilder.GraphBuilder` instance (*g*).
    """

    def __init__(self, target_opset: int, g: object) -> None:
        self.target_opset = target_opset
        self._g = g

    def get_unique_variable_name(self, seed: str) -> str:
        """Return a unique result name via the GraphBuilder."""
        return self._g.unique_name(seed)  # type: ignore[attr-defined]

    def get_unique_operator_name(self, seed: str) -> str:
        """Return a unique node name via the GraphBuilder."""
        return self._g.unique_node_name(seed)  # type: ignore[attr-defined]


class MockVariable:
    """
    Minimal mock for skl2onnx's ``Variable``.

    Stores the tensor name that will appear in emitted ``NodeProto``
    inputs / outputs.
    """

    def __init__(self, raw_name: str, onnx_name: str) -> None:
        self.raw_name = raw_name
        self.onnx_name = onnx_name
        self.type: Optional[object] = None
        self._is_fed: Optional[bool] = None

    @property
    def full_name(self) -> str:
        """Alias for ``onnx_name`` (used by some skl2onnx converters)."""
        return self.onnx_name

    @property
    def is_fed(self) -> Optional[bool]:
        return self._is_fed

    def init_status(
        self,
        is_fed: Optional[bool] = None,
        is_root: Optional[bool] = None,
        is_leaf: Optional[bool] = None,
    ) -> None:
        if is_fed is not None:
            self._is_fed = is_fed


class MockOperator:
    """
    Minimal mock for skl2onnx's ``Operator``.

    Holds the fitted sklearn estimator and the input / output
    :class:`MockVariable` objects so a skl2onnx converter can read
    tensor names via ``operator.inputs[i].onnx_name`` and
    ``operator.output_full_names``.
    """

    def __init__(
        self,
        raw_operator: object,
        op_type: str,
        onnx_name: str,
        target_opset: int,
        scope: MockScope,
    ) -> None:
        self.raw_operator = raw_operator
        self.type = op_type
        self.onnx_name = onnx_name
        self.target_opset = target_opset
        self.scope = scope
        self.inputs: List[MockVariable] = []
        self.outputs: List[MockVariable] = []

    @property
    def input_full_names(self) -> List[str]:
        return [v.onnx_name for v in self.inputs]

    @property
    def output_full_names(self) -> List[str]:
        return [v.onnx_name for v in self.outputs]


class MockContainer:
    """
    Minimal mock for skl2onnx's ``ModelComponentContainer``.

    Instead of accumulating nodes and initializers, this class delegates
    every :meth:`add_node` / :meth:`add_initializer` call directly to the
    :class:`~yobx.xbuilder.GraphBuilder` (*g*) that was provided at
    construction time.
    """

    def __init__(self, target_opset: int, g: object) -> None:
        self.target_opset = target_opset
        self._g = g
        # Some converters query per-domain opsets.
        self.target_opset_all: Dict[str, int] = {"": target_opset, "ai.onnx.ml": 5}
        self.options: Dict = {}

    @property
    def main_opset(self) -> int:
        return self.target_opset

    # ---- option-related helpers (called by skl2onnx's RegisteredConverter) ----

    def _get_allowed_options(self, model: object) -> Dict:
        return {}

    def get_options(
        self, model: object, default_values: Optional[Dict] = None, fail: bool = True
    ) -> Dict:
        return default_values if default_values is not None else {}

    def validate_options(self, operator: object) -> None:
        pass

    def is_allowed(self, op_set: object) -> bool:
        return True

    # ---- node and initializer delegation ----

    def add_node(
        self,
        op_type: str,
        inputs: object,
        outputs: object,
        op_domain: str = "",
        op_version: Optional[int] = None,
        name: Optional[str] = None,
        **attrs: object,
    ) -> None:
        """Delegate a node directly to the GraphBuilder."""
        # Some skl2onnx helpers pass a bare string instead of a list.
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]

        clean: Dict = {}
        for k, v in attrs.items():
            if v is None:
                continue
            if isinstance(v, np.ndarray):
                clean[k] = v.tolist()
            elif isinstance(v, np.integer):
                clean[k] = int(v)
            elif isinstance(v, np.floating):
                clean[k] = float(v)
            else:
                clean[k] = v

        node_name = self._g.unique_node_name(name or "skl2onnx_node")  # type: ignore[attr-defined]
        self._g.make_node(  # type: ignore
            op_type,
            list(inputs),  # type: ignore
            list(outputs),  # type: ignore
            domain=op_domain or "",
            name=node_name,
            **clean,
        )

    def add_onnx_node(self, node: object) -> None:
        """Delegate a pre-built ``NodeProto`` directly to the GraphBuilder."""
        self._g.make_node(  # type: ignore[attr-defined]
            node.op_type,
            list(node.input),
            list(node.output),
            domain=node.domain,
            name=node.name,
            attributes=list(node.attribute),
        )

    def add_initializer(self, name: str, onnx_type: int, shape: object, content: object) -> None:
        """Forward content directly to the GraphBuilder as a new initializer."""
        if isinstance(content, onnx.TensorProto):
            self._g.make_initializer(name, content)  # type: ignore[attr-defined]
        else:
            np_dtype = _ONNX_DTYPE_TO_NUMPY.get(onnx_type, np.float32)
            arr = np.array(content, dtype=np_dtype)
            if shape is not None and arr.shape != tuple(shape):
                arr = arr.reshape(shape)  # type: ignore
            self._g.make_initializer(name, arr)  # type: ignore[attr-defined]

    def add_onnx_initializer(self, tensor: object) -> None:
        """Forward a pre-built ``TensorProto`` directly to the GraphBuilder."""
        self._g.make_initializer(tensor.name, tensor)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_skl2onnx_converter(skl2onnx_op_converter: Callable, input_type: object) -> Callable:
    """
    Wrap a skl2onnx-style converter function so it can be used with
    :func:`yobx.sklearn.to_onnx` via the ``extra_converters`` parameter.

    A *skl2onnx converter function* has the signature::

        converter(scope, operator, container) -> None

    It emits ONNX nodes and initializers into *container*.  This factory
    bridges that convention to the yobx converter convention::

        converter(g, sts, outputs, estimator, X, name) -> str | tuple[str, ...]

    by supplying lightweight pure-Python mock objects for *scope*,
    *operator*, and *container* (all defined in this module — no skl2onnx
    imports required) and then injecting the collected nodes and initializers
    directly into the enclosing :class:`~yobx.xbuilder.GraphBuilder`.

    The converter function and input type can be obtained from skl2onnx::

        from skl2onnx._supported_operators import sklearn_operator_name_map
        from skl2onnx.common._registration import get_converter
        from skl2onnx.common.data_types import FloatTensorType
        from sklearn.neural_network import MLPClassifier
        from yobx.sklearn import to_onnx, make_skl2onnx_converter

        skl2onnx_fn = get_converter(sklearn_operator_name_map[MLPClassifier])
        converter = make_skl2onnx_converter(skl2onnx_fn, FloatTensorType([None, None]))
        artifact = to_onnx(mlp, (X,), extra_converters={MLPClassifier: converter})

    :param skl2onnx_op_converter: the skl2onnx converter function for the
        target estimator type (obtained from skl2onnx's registry).
    :param input_type: a skl2onnx type instance (e.g.
        ``FloatTensorType([None, None])`` or ``DoubleTensorType([None, n_features])``)
        that will be assigned to the mock input variable's ``.type`` attribute.
        skl2onnx converter functions typically call ``guess_proto_type()`` /
        ``guess_numpy_type()`` on this value to determine the ONNX element type, so
        the correct skl2onnx type class must be used.
    :return: a yobx-compatible converter function with the signature
        ``(g, sts, outputs, estimator, X, *, name) -> str | tuple[str, ...]``.

    .. note::

        This module contains **no** skl2onnx imports.  Only :mod:`onnx` and
        :mod:`numpy` (both core :mod:`yobx` dependencies) are used inside the
        mock helper classes.  The caller is responsible for importing skl2onnx
        to obtain both *skl2onnx_op_converter* and *input_type*.
    """

    def _converter(
        g: object,
        sts: Dict,
        outputs: List[str],
        estimator: object,
        X: str,
        name: str = "skl2onnx",
    ) -> Tuple:
        scope = MockScope(g.main_opset, g)  # type: ignore[attr-defined]

        input_var = MockVariable(X, X)
        input_var.type = input_type  # set by caller — no skl2onnx import needed here
        output_vars = [MockVariable(out, out) for out in outputs]

        operator = MockOperator(
            estimator,
            type(estimator).__name__,
            f"{name}_{id(estimator)}",
            g.main_opset,  # type: ignore[attr-defined]
            scope,
        )
        operator.inputs.append(input_var)
        for var in output_vars:
            operator.outputs.append(var)

        container = MockContainer(g.main_opset, g)  # type: ignore[attr-defined]
        skl2onnx_op_converter(scope, operator, container)

        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    return _converter

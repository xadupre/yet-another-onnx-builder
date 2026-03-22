from __future__ import annotations
import contextlib
import sys

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import onnx
from sklearn.base import BaseEstimator
from ..typing import GraphBuilderExtendedProtocol

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


class MockTensorType:
    pass


class MockFloatTensorType(MockTensorType):
    pass


class MockDoubleTensorType(MockTensorType):
    pass


class MockInt64TensorType(MockTensorType):
    pass


class MockInt32TensorType(MockTensorType):
    pass


class MockBooleanTensorType(MockTensorType):
    pass


class MockStringTensorType(MockTensorType):
    pass


MockTableType: Dict[int, Callable[[], MockTensorType]] = {
    onnx.TensorProto.BOOL: MockBooleanTensorType,
    onnx.TensorProto.FLOAT: MockFloatTensorType,
    onnx.TensorProto.DOUBLE: MockDoubleTensorType,
    onnx.TensorProto.INT64: MockInt64TensorType,
    onnx.TensorProto.INT32: MockInt32TensorType,
    onnx.TensorProto.STRING: MockStringTensorType,
}


def mock_guess_numpy_type(data_type):
    if data_type in (
        np.float64,
        np.float32,
        np.int8,
        np.uint8,
        np.str_,
        np.bool_,
        np.int32,
        np.int64,
    ):
        return data_type
    if data_type == str:  # noqa: E721
        return np.str_
    if data_type == bool:  # noqa: E721
        return np.bool_
    if isinstance(data_type, MockFloatTensorType):
        return np.float32
    if isinstance(data_type, MockDoubleTensorType):
        return np.float64
    if isinstance(data_type, MockInt32TensorType):
        return np.int32
    if isinstance(data_type, MockInt64TensorType):
        return np.int64
    if isinstance(data_type, MockStringTensorType):
        return np.str_
    if isinstance(data_type, MockBooleanTensorType):
        return np.bool_
    raise NotImplementedError("Unsupported data_type '{}'.".format(data_type))


def mock_guess_proto_type(data_type):
    if isinstance(data_type, MockFloatTensorType):
        return onnx.TensorProto.FLOAT
    if isinstance(data_type, MockDoubleTensorType):
        return onnx.TensorProto.DOUBLE
    if isinstance(data_type, MockInt32TensorType):
        return onnx.TensorProto.INT32
    if isinstance(data_type, MockInt64TensorType):
        return onnx.TensorProto.INT64
    if isinstance(data_type, MockStringTensorType):
        return onnx.TensorProto.STRING
    if isinstance(data_type, MockBooleanTensorType):
        return onnx.TensorProto.BOOL
    raise NotImplementedError("Unsupported data_type '{}'.".format(data_type))


@contextlib.contextmanager
def patch_skl2onnx_functions(skl2onnx_op_converter):
    try:
        import skl2onnx.common_type  # type: ignore

        wrap = True
    except ImportError:
        # maybe it is not needed
        wrap = False

    if skl2onnx_op_converter.__class__.__name__ == "RegisteredConverter":
        module = sys.modules[skl2onnx_op_converter._fct.__module__]
    else:
        module = sys.modules[skl2onnx_op_converter.__module__]

    function_to_patch = {
        "guess_numpy_type": mock_guess_numpy_type,
        "guess_proto_type": mock_guess_proto_type,
    }
    patched = {}
    sklearn_patched = {}
    for name, fct in function_to_patch.items():
        if hasattr(module, name):
            mocked = getattr(module, name)
            patched[name] = mocked
            setattr(module, name, fct)
        if wrap:
            sklearn_patched[name] = skl2onnx.common_type.data_types.guess_numpy_type  # type: ignore
            setattr(skl2onnx.common_type.data_types, name, fct)  # type: ignore

    try:
        yield
    except StopIteration:
        for k, v in patched.items():
            setattr(module, k, v)
        for k, v in sklearn_patched:
            setattr(skl2onnx.common_type, k, v)  # type: ignore


class MockScope:
    """
    Minimal mock for skl2onnx's ``Scope``.

    Provides :meth:`get_unique_variable_name`, the only method called by
    typical skl2onnx converter functions.  Name uniqueness is guaranteed by
    delegating to the :class:`~yobx.xbuilder.GraphBuilder` instance (*g*).
    """

    def __init__(self, g: GraphBuilderExtendedProtocol) -> None:
        self._g = g

    @property
    def target_opset(self) -> int:
        """The main opset version, derived from the GraphBuilder."""
        return self._g.main_opset

    def get_unique_variable_name(self, seed: str) -> str:
        """Return a unique result name via the GraphBuilder."""
        return self._g.unique_name(seed)

    def get_unique_operator_name(self, seed: str) -> str:
        """Return a unique node name via the GraphBuilder."""
        return self._g.unique_node_name(seed)  # type: ignore


class MockVariable:
    """
    Minimal mock for skl2onnx's ``Variable``.

    Stores the tensor name that will appear in emitted ``NodeProto``
    inputs / outputs.
    """

    def __init__(self, raw_name: str, onnx_name: str) -> None:
        self.raw_name = raw_name
        self.onnx_name = onnx_name

    @property
    def full_name(self) -> str:
        """Alias for ``onnx_name`` (used by some skl2onnx converters)."""
        return self.onnx_name

    @property
    def is_fed(self) -> Optional[bool]:
        return False


class MockOperator:
    """
    Minimal mock for skl2onnx's ``Operator``.

    Holds the fitted sklearn estimator and the input / output
    :class:`MockVariable` objects so a skl2onnx converter can read
    tensor names via ``operator.inputs[i].onnx_name`` and
    ``operator.output_full_names``.
    """

    def __init__(
        self, raw_operator, inputs: List[MockVariable], outputs: List[MockVariable]
    ) -> None:
        self.raw_operator = raw_operator
        self.inputs: List[MockVariable] = inputs
        self.outputs: List[MockVariable] = outputs

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

    def __init__(self, g: GraphBuilderExtendedProtocol) -> None:
        self._g = g
        self.options: Dict = {}

    def is_allowed(self, op_types: Set[str]):
        """Tells if this operators are allowed."""
        ai_onnx_ml_op_types = {
            "LinearClassifier",
            "LinearRegressor",
            "TreeEnsemble",
            "TreeEnsembleRegressor",
            "TreeEnsembleClassifier",
            "ArrayFeatureExtraction",
        }
        if ai_onnx_ml_op_types & op_types:
            return self._g.has_opset("ai.onnx.ml")
        return True

    @property
    def target_opset(self) -> int:
        """The main opset version, derived from the GraphBuilder."""
        return self._g.main_opset  # type: ignore[attr-defined]

    @property
    def target_opset_all(self) -> Dict[str, int]:
        """Per-domain opset mapping, derived from the GraphBuilder."""
        return self._g.opsets  # type: ignore[attr-defined]

    @property
    def main_opset(self) -> int:
        return self._g.get_opset("")

    def get_options(
        self, model: BaseEstimator, default_values: Optional[Dict] = None, fail: bool = True
    ) -> Dict:
        return default_values if default_values is not None else {}

    def add_node(
        self,
        op_type: str,
        inputs: Union[str, List[str]],
        outputs: Union[str, List[str]],
        op_domain: str = "",
        op_version: Optional[int] = None,
        name: Optional[str] = None,
        **attrs: Any,
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

    def add_initializer(
        self, name: str, onnx_type: int, shape: Tuple[int, ...], content: Any
    ) -> None:
        """Forward content directly to the GraphBuilder as a new initializer."""
        if isinstance(content, onnx.TensorProto):
            self._g.make_initializer(name, content)
        else:
            np_dtype = _ONNX_DTYPE_TO_NUMPY.get(onnx_type, np.float32)
            arr = np.array(content, dtype=np_dtype)
            if shape is not None and arr.shape != tuple(shape):
                arr = arr.reshape(shape)
            self._g.make_initializer(name, arr)


def wrap_skl2onnx_converter(skl2onnx_op_converter: Callable) -> Callable:
    """
    Wrap a skl2onnx-style converter function so it can be used with
    :func:`yobx.sklearn.to_onnx` via the ``extra_converters`` parameter.

    .. note::

        This module contains **no** skl2onnx imports.  Only :mod:`onnx` and
        :mod:`numpy` (both core :mod:`yobx` dependencies) are used inside the
        mock helper classes.
    """

    def _converter(
        g: GraphBuilderExtendedProtocol,
        sts: Dict,
        outputs: List[str],
        estimator: BaseEstimator,
        *args: str,
        name: str = "wrap_skl2onnx",
        **kwargs,
    ) -> Tuple:
        scope = MockScope(g)

        input_vars = []
        for a in args:
            input_var = MockVariable(a, a)
            input_var.type = MockTableType[g.get_type(a)]()
            input_vars.append(input_var)
        output_vars = []
        for a in outputs:
            out_var = MockVariable(a, a)
            output_vars.append(out_var)

        operator = MockOperator(estimator, input_vars, output_vars)
        container = MockContainer(g)

        with patch_skl2onnx_functions(skl2onnx_op_converter):
            skl2onnx_op_converter(scope, operator, container)

        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    return _converter

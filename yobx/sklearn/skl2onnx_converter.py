"""
Factory for creating :mod:`yobx`-compatible converters that delegate to
:epkg:`sklearn-onnx` (``skl2onnx``).
"""

from typing import Callable, Dict, List, Tuple


def make_skl2onnx_converter() -> Callable:
    """
    Return a :mod:`yobx`-compatible converter function that directly invokes
    the converter registered in :epkg:`sklearn-onnx` (``skl2onnx``) for the
    target estimator.

    The returned converter can be passed directly to
    :func:`yobx.sklearn.to_onnx` via the ``extra_converters`` parameter::

        from sklearn.neural_network import MLPClassifier
        from yobx.sklearn import to_onnx
        from yobx.sklearn.skl2onnx_converter import make_skl2onnx_converter

        converter = make_skl2onnx_converter()
        artifact = to_onnx(
            mlp, (X,), extra_converters={MLPClassifier: converter}
        )

    The returned function:

    1. Looks up the skl2onnx converter registered for the estimator's type via
       :data:`skl2onnx._supported_operators.sklearn_operator_name_map` and
       :func:`skl2onnx.common._registration.get_converter`.
    2. Creates lightweight mock objects (:class:`~skl2onnx.common._topology.Scope`,
       :class:`~skl2onnx.common._topology.Operator`,
       :class:`~skl2onnx.common._topology.Variable`) using the actual input and
       output tensor names already present in the enclosing
       :class:`~yobx.xbuilder.GraphBuilder`.
    3. Creates a :class:`~skl2onnx.common._container.ModelComponentContainer` that
       collects ONNX nodes and initializers emitted by the converter.
    4. Calls the skl2onnx converter function directly:
       ``converter(scope, operator, container)``.
    5. Injects all collected nodes and initializers straight into the enclosing
       :class:`~yobx.xbuilder.GraphBuilder`, without creating an intermediate
       :class:`onnx.ModelProto` or a local function wrapper.

    :return: a converter function with the signature
        ``(g, sts, outputs, estimator, X, *, name) -> str | tuple[str, ...]``
        accepted by :func:`yobx.sklearn.to_onnx`.

    .. note::

        Only ``FLOAT`` (element type 1) and ``DOUBLE`` (element type 11)
        inputs are supported because those are the only types for which
        :mod:`skl2onnx` provides generic tensor-type descriptors.  For other
        element types a ``NotImplementedError`` is raised at conversion time.

        The estimator's class must be registered in skl2onnx's converter
        registry (``sklearn_operator_name_map``).  If it is not, a
        ``KeyError`` is raised.
    """

    def _converter(
        g,
        sts: Dict,
        outputs: List[str],
        estimator,
        X: str,
        name: str = "skl2onnx",
    ) -> Tuple:
        import onnx
        from skl2onnx._supported_operators import sklearn_operator_name_map
        from skl2onnx.common._container import ModelComponentContainer
        from skl2onnx.common._registration import _converter_pool, get_converter
        from skl2onnx.common._topology import Operator, Scope, Variable
        from skl2onnx.common.data_types import DoubleTensorType, FloatTensorType

        # Determine ONNX element type and number of features.
        itype = g.get_type(X) if g.has_type(X) else onnx.TensorProto.FLOAT
        if hasattr(estimator, "n_features_in_"):
            n_features = int(estimator.n_features_in_)
        elif g.has_shape(X):
            shape = g.get_shape(X)
            n_features = int(shape[1]) if len(shape) > 1 and isinstance(shape[1], int) else None
        else:
            n_features = None

        # Map ONNX element type to a skl2onnx tensor-type descriptor.
        if itype == onnx.TensorProto.FLOAT:
            skl2onnx_type = FloatTensorType([None, n_features])
        elif itype == onnx.TensorProto.DOUBLE:
            skl2onnx_type = DoubleTensorType([None, n_features])
        else:
            raise NotImplementedError(
                f"make_skl2onnx_converter: unsupported elem_type {itype!r}. "
                "Only FLOAT (1) and DOUBLE (11) are currently supported."
            )

        # Look up the registered skl2onnx converter for this estimator type.
        cls = type(estimator)
        if cls not in sklearn_operator_name_map:
            raise KeyError(
                f"make_skl2onnx_converter: no skl2onnx converter is registered for "
                f"{cls.__qualname__!r}. Only estimator types listed in "
                "'skl2onnx._supported_operators.sklearn_operator_name_map' are supported."
            )
        op_name = sklearn_operator_name_map[cls]
        converter = get_converter(op_name)

        # registered_models is required so that ModelComponentContainer can
        # validate and retrieve converter options.
        registered_models = {
            "aliases": sklearn_operator_name_map,
            "conv": _converter_pool,
        }

        # Build mock Scope/Operator/Variable objects.
        # The input Variable uses the actual tensor name already in yobx's
        # graph; output Variables use the caller-supplied output names.
        # All of these names flow directly into the emitted NodeProtos.
        scope = Scope("root", target_opset=g.main_opset)

        input_var = Variable(X, X, scope="root", type=skl2onnx_type)
        input_var.init_status(is_fed=True, is_root=True, is_leaf=False)

        output_vars = []
        for out_name in outputs:
            var = Variable(out_name, out_name, scope="root")
            var.init_status(is_fed=False, is_root=False, is_leaf=True)
            output_vars.append(var)

        operator = Operator(
            f"{name}_{id(estimator)}_op", "root", op_name, estimator, g.main_opset, scope
        )
        operator.inputs.append(input_var)
        for var in output_vars:
            operator.outputs.append(var)

        # Collect nodes and initializers emitted by the skl2onnx converter.
        container = ModelComponentContainer(g.main_opset, registered_models=registered_models)
        converter(scope, operator, container)

        # Inject initializers and nodes directly into yobx's GraphBuilder.
        for init in container.initializers:
            g.make_initializer(init.name, init)
        for node in container.nodes:
            g.make_node(
                node.op_type,
                list(node.input),
                list(node.output),
                domain=node.domain,
                name=node.name,
                attributes=list(node.attribute),
            )

        # Return the output name(s) to the yobx conversion framework.
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    return _converter

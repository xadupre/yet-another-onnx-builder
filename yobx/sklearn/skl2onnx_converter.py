"""
Factory for creating :mod:`yobx`-compatible converters that delegate to
:epkg:`sklearn-onnx` (``skl2onnx``).
"""

from typing import Callable, Dict, List, Optional, Tuple


def make_skl2onnx_converter(
    options: Optional[Dict] = None,
    domain: str = "sklearn_onnx_functions",
) -> Callable:
    """
    Return a :mod:`yobx`-compatible converter function that uses
    :func:`skl2onnx.convert_sklearn` as the conversion back-end.

    The returned converter can be passed directly to
    :func:`yobx.sklearn.to_onnx` via the ``extra_converters`` parameter::

        from sklearn.neural_network import MLPClassifier
        from yobx.sklearn import to_onnx
        from yobx.sklearn.skl2onnx_converter import make_skl2onnx_converter

        converter = make_skl2onnx_converter(options={"zipmap": False})
        artifact = to_onnx(
            mlp, (X,), extra_converters={MLPClassifier: converter}
        )

    The returned function:

    1. Determines the ONNX element type and the number of input features from
       the graph builder and the fitted estimator, then calls
       :func:`skl2onnx.convert_sklearn` to produce a stand-alone
       :class:`onnx.ModelProto`.
    2. Normalises the opset of the sub-model to match the enclosing graph.
    3. Loads the sub-model into a temporary
       :class:`~yobx.xbuilder.GraphBuilder` and registers it as a local
       ONNX function (via
       :meth:`~yobx.xbuilder.GraphBuilder.make_local_function`).
    4. Emits a single call-node for that function, connecting the main
       graph's input tensor to the function.

    The :mod:`yobx` optimiser inlines local functions when
    :func:`~yobx.sklearn.to_onnx` finalises the model, so the returned
    :class:`onnx.ModelProto` contains the individual nodes rather than a
    single call-node.

    :param options: mapping passed to :func:`skl2onnx.convert_sklearn` as the
        ``options`` argument (e.g. ``{"zipmap": False}`` to get a plain float
        probability tensor from classifiers instead of the default
        sequence-of-maps representation).
    :param domain: ONNX domain name used when registering the local function
        in the enclosing graph builder.  Defaults to
        ``"sklearn_onnx_functions"``.
    :return: a converter function with the signature
        ``(g, sts, outputs, estimator, X, *, name) -> str | tuple[str, ...]``
        accepted by :func:`yobx.sklearn.to_onnx`.

    .. note::

        Only ``FLOAT`` (element type 1) and ``DOUBLE`` (element type 11)
        inputs are supported because those are the only types for which
        :mod:`skl2onnx` provides generic tensor descriptors.  For other
        element types, raise ``NotImplementedError`` at conversion time.
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
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import DoubleTensorType, FloatTensorType

        from ..xbuilder.function_options import FunctionOptions

        # Determine the ONNX element type of the input tensor.
        itype = g.get_type(X) if g.has_type(X) else onnx.TensorProto.FLOAT

        # Determine the number of input features.
        # sklearn >=1.0 exposes n_features_in_ on all fitted estimators.
        if hasattr(estimator, "n_features_in_"):
            n_features = int(estimator.n_features_in_)
        elif g.has_shape(X):
            shape = g.get_shape(X)
            n_features = int(shape[1]) if len(shape) > 1 and isinstance(shape[1], int) else None
        else:
            n_features = None

        # Map ONNX element type to a skl2onnx tensor descriptor.
        if itype == onnx.TensorProto.FLOAT:
            skl2onnx_type = FloatTensorType([None, n_features])
        elif itype == onnx.TensorProto.DOUBLE:
            skl2onnx_type = DoubleTensorType([None, n_features])
        else:
            raise NotImplementedError(
                f"make_skl2onnx_converter: unsupported elem_type {itype!r}. "
                "Only FLOAT (1) and DOUBLE (11) are currently supported."
            )

        # Convert the estimator to a stand-alone ONNX model.
        onx = convert_sklearn(
            estimator,
            initial_types=[("X", skl2onnx_type)],
            options=options,
            target_opset=g.main_opset,
        )

        # skl2onnx picks the lowest opset compatible with the nodes it emits.
        # Overwrite it so the sub-model is compatible with the enclosing graph.
        del onx.opset_import[:]
        d = onx.opset_import.add()
        d.domain = ""
        d.version = g.main_opset

        # Load the sub-model into a fresh GraphBuilder and register it as a
        # local ONNX function inside the enclosing graph.
        builder = g.__class__(onx)
        f_options = FunctionOptions(
            export_as_function=True,
            name=g.unique_function_name(type(estimator).__name__),
            domain=domain,
            move_initializer_to_constant=True,
        )
        g.make_local_function(builder, f_options)

        # Emit a call-node for the registered function.
        return g.make_node(f_options.name, [X], outputs, domain=f_options.domain, name=name)

    return _converter

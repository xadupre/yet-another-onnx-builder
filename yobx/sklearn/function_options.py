class SklearnFunctionOptions:
    """
    Options for wrapping sklearn estimators as ONNX local functions
    when calling :func:`yobx.sklearn.to_onnx`.

    When provided to :func:`to_onnx`, each estimator (except
    :class:`sklearn.pipeline.Pipeline` and
    :class:`sklearn.compose.ColumnTransformer`) is exported as
    a separate ONNX local function in the specified *domain*.
    :class:`~sklearn.pipeline.Pipeline` and
    :class:`~sklearn.compose.ColumnTransformer` are kept as
    orchestrators — their individual steps or sub-transformers
    are each wrapped as functions instead.

    :param domain: ONNX domain used for the local functions
    :param move_initializer_to_constant: when *True* (default) every
        weight tensor is embedded inside the function body as a
        ``Constant`` node instead of being passed as an extra input
    """

    def __init__(self, domain: str = "sklearn_local", move_initializer_to_constant: bool = True):
        if not domain:
            raise ValueError("domain must be a non-empty string")
        self.domain = domain
        self.move_initializer_to_constant = move_initializer_to_constant

    def __repr__(self) -> str:
        return (
            f"SklearnFunctionOptions(domain={self.domain!r}, "
            f"move_initializer_to_constant={self.move_initializer_to_constant!r})"
        )

"""
:class:`DataFrameTransformer` — a scikit-learn transformer that wraps a
function operating on a :class:`pandas.DataFrame` (or a
:class:`~yobx.sql.TracedDataFrame` proxy) and can be exported to ONNX via
:func:`yobx.sklearn.to_onnx`.

The function is traced with the same dataframe-tracing infrastructure used by
:func:`yobx.sql.dataframe_to_onnx`.  Each column of the input frame becomes a
separate ONNX input tensor; each column in the returned frame becomes a
separate ONNX output tensor.

Call :meth:`DataFrameTransformer.onnx_args` to obtain the tuple of
``(name, dtype, shape)`` descriptors to pass to :func:`~yobx.sklearn.to_onnx`
as the *args* parameter.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol


class DataFrameTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer that wraps a user-supplied function operating on
    a :class:`pandas.DataFrame` and supports ONNX export via
    :func:`yobx.sklearn.to_onnx`.

    The function is traced during :meth:`fit` using
    :func:`~yobx.sql.dataframe_trace.trace_dataframe` so that the output
    column names are known ahead of time and
    :meth:`get_feature_names_out` can be implemented.  During
    :meth:`transform` the real pandas function is applied to the input
    frame.

    For ONNX export, use :meth:`onnx_args` to obtain the input descriptors::

        from yobx.sklearn import to_onnx
        onx = to_onnx(transformer, transformer.onnx_args())

    The resulting ONNX model has **one input tensor per column** (with the
    same name as the column) and **one output tensor per output column**.

    .. note::

        Embedding a :class:`DataFrameTransformer` inside a
        :class:`sklearn.pipeline.Pipeline` is not supported because sklearn
        pipelines pass a single ``X`` tensor between steps, whereas this
        transformer uses one tensor per named column.

    :param func: callable that accepts a :class:`pandas.DataFrame` (or
        :class:`~yobx.sql.TracedDataFrame`) and returns a
        :class:`pandas.DataFrame`.  The function must contain at least one
        explicit column selection (e.g. via
        :meth:`~yobx.sql.TracedDataFrame.select` or
        :meth:`~yobx.sql.TracedDataFrame.assign`) so that the output column
        names can be determined during tracing.
    :param input_dtypes: mapping from column name to numpy dtype (or any
        value accepted by ``np.dtype(...)``).  Determines the ONNX input
        tensor types.

    Example::

        import numpy as np
        import pandas as pd
        from yobx.sklearn.preprocessing import DataFrameTransformer
        from yobx.sklearn import to_onnx
        from yobx.reference import ExtendedReferenceEvaluator

        def my_transform(df):
            return df.select([
                (df["a"] + df["b"]).alias("total"),
                (df["a"] * 2.0).alias("a_doubled"),
            ])

        dtypes = {"a": np.float32, "b": np.float32}
        t = DataFrameTransformer(my_transform, dtypes)
        t.fit()

        # ONNX export
        onx = to_onnx(t, t.onnx_args())

        # Run with the reference evaluator
        ref = ExtendedReferenceEvaluator(onx)
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        total, a_doubled = ref.run(None, {"a": a, "b": b})
    """

    def __init__(
        self,
        func: Callable,
        input_dtypes: Dict[str, Union[type, np.dtype, str]],
    ):
        self.func = func
        self.input_dtypes = input_dtypes

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X=None, y=None):
        """
        Fit the transformer.

        Traces *func* with :class:`~yobx.sql.TracedDataFrame` proxies to
        discover the output column names and build the internal ONNX model
        used by :meth:`transform`.

        :param X: ignored; accepts ``None`` or a :class:`pandas.DataFrame`
            (only the column structure is used if provided).
        :param y: ignored.
        :return: ``self``
        """
        from ...sql.dataframe_trace import trace_dataframe
        from ...sql.parse import SelectOp
        from ...sql.sql_convert import parsed_query_to_onnx

        self.input_dtypes_: Dict[str, np.dtype] = {
            k: np.dtype(v) for k, v in self.input_dtypes.items()
        }
        pq = trace_dataframe(self.func, self.input_dtypes_)
        select_op = next(
            (op for op in pq.operations if isinstance(op, SelectOp)), None
        )
        if select_op is None:
            raise ValueError(
                "DataFrameTransformer: the traced function must contain an "
                "explicit column selection (e.g. df.select([...]) or "
                "df.assign(...)) so that output column names can be "
                "determined at fit time."
            )
        self.output_names_: List[str] = [item.output_name() for item in select_op.items]
        # Number of ONNX output tensors — used by get_n_expected_outputs()
        # in yobx.sklearn.sklearn_helper to correctly size output_names.
        self.n_onnx_outputs_: int = len(self.output_names_)
        # Cache a compiled ONNX model so that transform() can run inference
        # without recompiling on each call.
        artifact = parsed_query_to_onnx(pq, self.input_dtypes_)
        self.onnx_model_ = artifact.proto
        return self

    def transform(self, X):
        """
        Transform a :class:`pandas.DataFrame` using the compiled ONNX model.

        Each column in *input_dtypes* is extracted from *X* as a 1-D numpy
        array and passed to the ONNX model.  The output tensors are assembled
        into a :class:`pandas.DataFrame` whose column names match
        *output_names_*.

        :param X: a :class:`pandas.DataFrame` whose columns match
            *input_dtypes*.
        :return: a :class:`pandas.DataFrame` with the output columns.
        :raises TypeError: if *X* is not a :class:`pandas.DataFrame`.
        :raises ImportError: if :mod:`onnxruntime` is not installed.
        """
        check_is_fitted(self, ["output_names_", "onnx_model_"])
        import pandas as pd

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"DataFrameTransformer.transform expects a pandas DataFrame, "
                f"got {type(X).__name__!r}."
            )
        try:
            from onnxruntime import InferenceSession
        except ImportError as exc:
            raise ImportError(
                "DataFrameTransformer.transform requires 'onnxruntime'. "
                "Install it with: pip install onnxruntime"
            ) from exc

        sess = InferenceSession(
            self.onnx_model_.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        feeds = {
            col: X[col].values.astype(dtype)
            for col, dtype in self.input_dtypes_.items()
        }
        results = sess.run(None, feeds)
        return pd.DataFrame(dict(zip(self.output_names_, results)))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return the output column names determined during :meth:`fit`.

        :param input_features: ignored.
        :return: 1-D array of output column name strings.
        """
        check_is_fitted(self, "output_names_")
        return np.array(self.output_names_, dtype=object)

    # ------------------------------------------------------------------
    # ONNX export helper
    # ------------------------------------------------------------------

    def onnx_args(self) -> Tuple:
        """
        Return a tuple of ``(name, dtype, shape)`` input descriptors suitable
        for passing as the *args* parameter of :func:`yobx.sklearn.to_onnx`.

        Each element corresponds to one input column: the name is the column
        name, the dtype is taken from *input_dtypes*, and the shape is
        ``('N',)`` (one-dimensional, dynamic batch size).

        :return: tuple of ``(column_name, np.dtype, ('N',))`` triples.
        :raises sklearn.exceptions.NotFittedError: if :meth:`fit` has not been
            called.

        Example::

            onx = to_onnx(transformer, transformer.onnx_args())
        """
        check_is_fitted(self, "input_dtypes_")
        return tuple(
            (col, dtype, ("N",)) for col, dtype in self.input_dtypes_.items()
        )


# ---------------------------------------------------------------------------
# ONNX converter
# ---------------------------------------------------------------------------


@register_sklearn_converter(DataFrameTransformer)
def sklearn_dataframe_function_transformer(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: DataFrameTransformer,
    *column_inputs: str,
    name: str = "dataframe_transformer",
) -> str:
    """
    Converts a :class:`DataFrameTransformer` into ONNX.

    The user function stored in ``estimator.func`` is traced with
    :func:`~yobx.sql.dataframe_trace.trace_dataframe` and the resulting
    :class:`~yobx.sql.parse.ParsedQuery` is compiled into *g* via
    :func:`~yobx.sql.sql_convert.parsed_query_to_onnx_graph`.

    Input tensors (one per column) must already be registered in *g* before
    this converter is called — which is the case when the caller is
    :func:`yobx.sklearn.to_onnx` and the *args* tuple was built with
    :meth:`DataFrameTransformer.onnx_args`.

    :param g: graph builder to add nodes to.
    :param sts: shared state dict (may carry ``"custom_functions"``).
    :param outputs: desired output tensor names (one per output column).
    :param estimator: a fitted :class:`DataFrameTransformer` instance.
    :param column_inputs: names of the already-registered input tensors,
        one per column in the order of ``estimator.input_dtypes_``.
    :param name: prefix for node names added to *g*.
    :return: name of the first output tensor.
    """
    assert isinstance(estimator, DataFrameTransformer), (
        f"sklearn_dataframe_function_transformer: expected a DataFrameTransformer "
        f"but received {type(estimator).__name__!r}. Make sure the correct converter "
        f"is registered for this estimator type."
    )

    from ...sql.dataframe_trace import trace_dataframe
    from ...sql.sql_convert import parsed_query_to_onnx_graph

    pq = trace_dataframe(estimator.func, estimator.input_dtypes_)
    out_names = parsed_query_to_onnx_graph(
        g,
        sts,
        list(outputs) if outputs else [],
        pq,
        estimator.input_dtypes_,
        _finalize=False,
    )
    return out_names[0] if out_names else (column_inputs[0] if column_inputs else "")

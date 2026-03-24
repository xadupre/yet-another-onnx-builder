"""
:class:`DataFrameTransformer` — a scikit-learn transformer backed by a
:class:`~yobx.sql.TracedDataFrame`-API function with automatic ONNX export.

The transformer wraps a tracing function and the associated column schema so
that:

* :func:`yobx.sklearn.to_onnx` can convert it to ONNX **without** the caller
  having to supply ``extra_converters``; and
* :meth:`DataFrameTransformer.transform` can execute the same transformation
  via the ONNX reference evaluator, keeping a single source of truth for both
  the sklearn and ONNX paths.

.. note::

    The ONNX converter for :class:`DataFrameTransformer` is registered lazily
    by :func:`yobx.sklearn.register_sklearn_converters`.  Importing this class
    directly (``from yobx.sklearn.preprocessing import DataFrameTransformer``)
    does **not** register the converter, so it is safe to import before calling
    :func:`~yobx.sklearn.to_onnx`.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..sklearn_helper import NoKnownOutputMixin


class DataFrameTransformer(BaseEstimator, TransformerMixin, NoKnownOutputMixin):
    """
    Scikit-learn transformer backed by a :class:`~yobx.sql.TracedDataFrame`-API
    function with automatic ONNX export support.

    :class:`DataFrameTransformer` eliminates the boilerplate of writing a
    separate ``extra_converters`` entry: the tracing function *func* is used
    both as the ONNX blueprint and as the execution engine (via the ONNX
    reference evaluator) when :meth:`transform` is called.

    :param func: a callable that accepts a :class:`~yobx.sql.TracedDataFrame`
        and returns a :class:`~yobx.sql.TracedDataFrame`.  This function
        defines the transformation and must be written using the
        :class:`~yobx.sql.TracedDataFrame` API (``select``, ``filter``,
        arithmetic operators, etc.).
    :param input_dtypes: mapping from column name to dtype (e.g.
        ``{"a": np.float32, "b": np.float32}``).  This describes the columns
        that the transformer expects as input.

    Example::

        import numpy as np
        from yobx.sklearn.preprocessing import DataFrameTransformer
        from yobx.sklearn import to_onnx

        def _add_cols(df):
            return df.select([(df["a"] + df["b"]).alias("total")])

        t = DataFrameTransformer(func=_add_cols, input_dtypes={"a": np.float32, "b": np.float32})
        t.fit()

        # Export to ONNX — no extra_converters needed
        onx = to_onnx(t, t.onnx_args())
    """

    def __init__(
        self,
        func,
        input_dtypes: Dict[str, Union[np.dtype, type, str]],
    ) -> None:
        self.func = func
        self.input_dtypes = input_dtypes

    # ------------------------------------------------------------------
    # Sklearn API
    # ------------------------------------------------------------------

    def fit(self, X=None, y=None) -> "DataFrameTransformer":
        """
        Fit the transformer (records *input_dtypes* as the fitted attribute).

        :param X: ignored.
        :param y: ignored.
        :return: ``self``.
        """
        self.input_dtypes_: Dict[str, np.dtype] = {
            k: np.dtype(v) for k, v in self.input_dtypes.items()
        }
        # Invalidate any cached ONNX artifact from a previous fit.
        if hasattr(self, "_onnx_artifact_"):
            del self._onnx_artifact_
        return self

    def transform(
        self,
        X,
    ) -> np.ndarray:
        """
        Apply the transformation to *X*.

        The tracing function :attr:`func` is compiled to ONNX on the first
        call (the compiled model is cached as ``_onnx_artifact_``) and
        evaluated via :class:`~yobx.reference.ExtendedReferenceEvaluator`.

        :param X: either a :class:`pandas.DataFrame` (columns are matched by
            name) or a :class:`dict` mapping column names to 1-D numpy arrays.
        :return: a 2-D numpy array whose columns correspond to the output
            columns of :attr:`func`.
        """
        check_is_fitted(self, "input_dtypes_")
        if not hasattr(self, "_onnx_artifact_"):
            from yobx.sql.dataframe_trace import dataframe_to_onnx

            self._onnx_artifact_ = dataframe_to_onnx(self.func, self.input_dtypes_)

        from yobx.reference import ExtendedReferenceEvaluator

        ref = ExtendedReferenceEvaluator(self._onnx_artifact_)

        # Build the feeds dict from the input.
        try:
            import pandas as pd

            if isinstance(X, pd.DataFrame):
                feeds = {col: X[col].to_numpy() for col in self.input_dtypes_}
            else:
                feeds = dict(X)
        except ImportError:
            feeds = dict(X)

        outputs = ref.run(None, feeds)
        if len(outputs) == 1:
            return outputs[0].reshape(-1, 1)
        return np.column_stack(outputs)

    # ------------------------------------------------------------------
    # ONNX export helpers
    # ------------------------------------------------------------------

    def onnx_args(
        self,
    ) -> Tuple[Tuple[str, np.dtype, Tuple[str]], ...]:
        """
        Return a tuple of ``(name, dtype, shape)`` descriptors suitable for
        passing to :func:`yobx.sklearn.to_onnx` as the *args* argument.

        Each input column is mapped to a 1-D tensor with a dynamic batch
        dimension ``"N"``.

        :return: a tuple of ``(column_name, dtype, ("N",))`` triples.

        Example::

            t = DataFrameTransformer(func=f, input_dtypes={"a": np.float32})
            t.fit()
            onx = to_onnx(t, t.onnx_args())
        """
        check_is_fitted(self, "input_dtypes_")
        return tuple((col, dtype, ("N",)) for col, dtype in self.input_dtypes_.items())

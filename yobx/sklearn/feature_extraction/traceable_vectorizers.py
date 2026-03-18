"""
Traceable wrappers for text vectorizers.

:class:`TraceableCountVectorizer` and :class:`TraceableTfIdfVectorizer` are
thin sklearn subclasses of their standard counterparts that carry an ONNX
converter accepting **raw text documents** (a 1-D string tensor) as input.

The ONNX graph emitted by the converters mirrors the sklearn preprocessing
pipeline using standard ONNX operators:

1. :onnxop:`StringNormalizer` — lower-cases each document
   (emitted only when ``lowercase=True``).
2. :onnxop:`StringSplit` — splits each document on whitespace into a
   2-D padded token matrix (requires opset ≥ 20).
3. :onnxop:`TfIdfVectorizer` — maps the token matrix to a float
   feature vector using the fitted vocabulary.

For :class:`TraceableTfIdfVectorizer` the graph also applies the standard
TF-IDF arithmetic (sublinear-TF scaling, IDF weighting, L1/L2 normalisation)
after the count step, mirroring
:func:`~yobx.sklearn.feature_extraction.tfidf_transformer.sklearn_tfidf_transformer`.

**Limitations**

* Only ``analyzer='word'`` is supported.
* Tokenisation uses the ONNX ``StringSplit`` operator which splits on
  whitespace.  sklearn's default token pattern ``(?u)\\b\\w\\w+\\b``
  additionally filters out single-character tokens; tokens of length 1 that
  survive the vocabulary look-up are simply ignored by the ONNX
  ``TfIdfVectorizer`` because they do not appear in ``pool_strings``.
* The ``token_pattern`` and ``preprocessor`` / ``tokenizer`` sklearn
  parameters are not reflected in the ONNX graph.
* Stop words are implicitly handled: they are absent from the fitted
  ``vocabulary_`` and are therefore not counted even when they appear in
  the tokenised output.
"""

from typing import Dict, List, Optional

import numpy as np
import onnx
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from ...typing import GraphBuilderExtendedProtocol
from ..register import register_sklearn_converter
from .count_vectorizer import _build_tfidf_vectorizer_attrs


class TraceableCountVectorizer(CountVectorizer):
    """
    A :class:`~sklearn.feature_extraction.text.CountVectorizer` subclass
    whose ONNX converter accepts **raw text documents** as input.

    This class is fully compatible with ``CountVectorizer``; it can be
    fitted and called in exactly the same way.  The only difference is the
    ONNX model produced by ``to_onnx``: instead of requiring a
    pre-tokenised 2-D string tensor, the model accepts a 1-D string tensor
    of raw documents and internally applies ``StringNormalizer`` and
    ``StringSplit`` before the term-frequency counting step.

    :param args: positional arguments forwarded to ``CountVectorizer``
    :param kwargs: keyword arguments forwarded to ``CountVectorizer``

    Example::

        import numpy as np
        from yobx.sklearn import to_onnx
        from yobx.sklearn.feature_extraction import TraceableCountVectorizer

        cv = TraceableCountVectorizer(ngram_range=(1, 2))
        texts = ["hello world", "world peace", "hello peace"]
        cv.fit(texts)
        # to_onnx accepts a 1-D string array (raw documents)
        onx = to_onnx(cv, (np.array(texts, dtype=object),))
    """

    pass


class TraceableTfIdfVectorizer(TfidfVectorizer):
    """
    A :class:`~sklearn.feature_extraction.text.TfidfVectorizer` subclass
    whose ONNX converter accepts **raw text documents** as input.

    Identical to :class:`TraceableCountVectorizer` in its input/output
    contract; in addition it applies TF-IDF scaling (sublinear-TF, IDF
    weighting, L1/L2 normalisation) after the count step.

    :param args: positional arguments forwarded to ``TfidfVectorizer``
    :param kwargs: keyword arguments forwarded to ``TfidfVectorizer``

    Example::

        import numpy as np
        from yobx.sklearn import to_onnx
        from yobx.sklearn.feature_extraction import TraceableTfIdfVectorizer

        tv = TraceableTfIdfVectorizer(sublinear_tf=True, norm="l2")
        texts = ["hello world", "world peace", "hello peace"]
        tv.fit(texts)
        # to_onnx accepts a 1-D string array (raw documents)
        onx = to_onnx(tv, (np.array(texts, dtype=object),))
    """

    pass


def _emit_tokenization(
    g: GraphBuilderExtendedProtocol,
    X: str,
    lowercase: bool,
    name: str,
) -> str:
    """
    Emits ``StringNormalizer`` (optional) and ``StringSplit`` nodes.

    :param g: graph builder
    :param X: name of the 1-D input STRING tensor ``(N,)``
    :param lowercase: whether to emit a ``StringNormalizer`` lowercase node
    :param name: node-name prefix
    :returns: name of the 2-D padded token tensor ``(N, max_tokens)``
    :raises NotImplementedError: if the graph opset is < 20 (``StringSplit``
        was introduced in opset 20)
    """
    opset = g.get_opset("")
    if opset < 20:
        raise NotImplementedError(
            f"TraceableCountVectorizer / TraceableTfIdfVectorizer require "
            f"opset >= 20 for StringSplit (graph opset is {opset})."
        )

    current = X

    if lowercase:
        x_lower = g.make_node(
            "StringNormalizer",
            [current],
            outputs=1,
            domain="",
            name=f"{name}_lowercase",
            case_change_action="LOWER",
        )
        x_lower = x_lower if isinstance(x_lower, str) else x_lower[0]
        g.set_type(x_lower, onnx.TensorProto.STRING)
        if g.has_rank(current):
            g.set_rank(x_lower, g.get_rank(current))
        current = x_lower

    split_result = g.make_node(
        "StringSplit",
        [current],
        outputs=2,
        domain="",
        name=f"{name}_split",
        delimiter=" ",
    )
    x_tokens, x_counts = split_result
    g.set_type(x_tokens, onnx.TensorProto.STRING)
    g.set_rank(x_tokens, 2)
    g.set_type(x_counts, onnx.TensorProto.INT64)
    g.set_rank(x_counts, 1)
    return x_tokens


@register_sklearn_converter(TraceableCountVectorizer)
def sklearn_traceable_count_vectorizer(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: TraceableCountVectorizer,
    X: str,
    name: str = "traceable_count_vectorizer",
) -> str:
    """
    Converts a :class:`TraceableCountVectorizer` into ONNX.

    The input is a **1-D string tensor** ``(N,)`` of raw documents; the
    graph applies ``StringNormalizer`` (optional) and ``StringSplit`` to
    tokenise each document before passing the result to
    ``TfIdfVectorizer``.

    **Graph layout**

    .. code-block:: text

        X  (N,) STRING
        │
        ├── StringNormalizer(LOWER)   # only when lowercase=True
        │      └── X_lower  (N,) STRING
        ├── StringSplit(delimiter=' ')
        │      └── X_tokens  (N, max_tokens) STRING
        └── TfIdfVectorizer(pool_strings, …)
               └── output  (N, n_features) FLOAT

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``TraceableCountVectorizer``
    :param outputs: desired output names
    :param X: input tensor name — a 1-D ``STRING`` tensor of raw documents
    :param name: prefix for added node names
    :return: output tensor name
    :raises NotImplementedError: if ``analyzer`` is not ``'word'`` or the
        graph opset is < 20
    """
    assert isinstance(estimator, TraceableCountVectorizer), (
        f"Unexpected type {type(estimator)} for estimator."
    )
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    if itype != onnx.TensorProto.STRING:
        raise NotImplementedError(
            f"TraceableCountVectorizer conversion requires a STRING input tensor "
            f"(got ONNX type {itype}).  Pass a 1-D string array of raw documents."
        )

    analyzer = estimator.analyzer
    if analyzer != "word":
        raise NotImplementedError(
            f"TraceableCountVectorizer converter only supports analyzer='word' "
            f"(got {analyzer!r})."
        )

    vocabulary = estimator.vocabulary_
    if not vocabulary:
        raise ValueError(
            "TraceableCountVectorizer has an empty vocabulary; ensure it was fitted."
        )

    n_features = max(vocabulary.values()) + 1

    # Tokenization: StringNormalizer + StringSplit
    x_tokens = _emit_tokenization(g, X, estimator.lowercase, name)

    # Count step via ONNX TfIdfVectorizer
    mode, pool_strings, ngram_counts, ngram_indexes, weights = _build_tfidf_vectorizer_attrs(
        vocabulary, estimator.ngram_range, estimator.binary
    )
    kwargs = dict(
        mode=mode,
        min_gram_length=estimator.ngram_range[0],
        max_gram_length=estimator.ngram_range[1],
        max_skip_count=0,
        ngram_counts=ngram_counts,
        ngram_indexes=ngram_indexes,
        pool_strings=pool_strings,
    )
    if weights:
        kwargs["weights"] = weights

    res = g.make_node(
        "TfIdfVectorizer",
        [x_tokens],
        outputs=outputs,
        domain="",
        name=name,
        **kwargs,
    )
    res_name = res if isinstance(res, str) else res[0]
    g.set_type(res_name, onnx.TensorProto.FLOAT)
    if g.has_shape(X):
        batch_dim = g.get_shape(X)[0]
        g.set_shape(res_name, (batch_dim, n_features))
    elif g.has_rank(X):
        g.set_rank(res_name, 2)
    return res_name


@register_sklearn_converter(TraceableTfIdfVectorizer)
def sklearn_traceable_tfidf_vectorizer(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: TraceableTfIdfVectorizer,
    X: str,
    name: str = "traceable_tfidf_vectorizer",
) -> str:
    """
    Converts a :class:`TraceableTfIdfVectorizer` into ONNX.

    The input is a **1-D string tensor** ``(N,)`` of raw documents.  The
    graph applies ``StringNormalizer`` (optional), ``StringSplit``,
    ``TfIdfVectorizer`` for raw counts, and then the TF-IDF arithmetic
    (optional sublinear-TF scaling, IDF weighting, L1/L2 normalisation).

    **Graph layout (all options active)**

    .. code-block:: text

        X  (N,) STRING
        │
        ├── StringNormalizer(LOWER)           # lowercase=True only
        ├── StringSplit(delimiter=' ')
        ├── TfIdfVectorizer(mode=TF, …)       # raw counts
        ├── Greater/Log/Add/Where             # sublinear_tf (optional)
        ├── Mul(idf_)                         # use_idf (optional)
        └── ReduceL2/Div                      # norm (optional)
               └── output  (N, n_features) FLOAT

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``TraceableTfIdfVectorizer``
    :param outputs: desired output names
    :param X: input tensor name — a 1-D ``STRING`` tensor of raw documents
    :param name: prefix for added node names
    :return: output tensor name
    :raises NotImplementedError: if ``analyzer`` is not ``'word'``, the
        graph opset is < 20, or ``norm`` is ``'l1'``/``'l2'`` and opset < 18
    """
    assert isinstance(estimator, TraceableTfIdfVectorizer), (
        f"Unexpected type {type(estimator)} for estimator."
    )
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    if itype != onnx.TensorProto.STRING:
        raise NotImplementedError(
            f"TraceableTfIdfVectorizer conversion requires a STRING input tensor "
            f"(got ONNX type {itype}).  Pass a 1-D string array of raw documents."
        )

    analyzer = estimator.analyzer
    if analyzer != "word":
        raise NotImplementedError(
            f"TraceableTfIdfVectorizer converter only supports analyzer='word' "
            f"(got {analyzer!r})."
        )

    vocabulary = estimator.vocabulary_
    if not vocabulary:
        raise ValueError(
            "TraceableTfIdfVectorizer has an empty vocabulary; ensure it was fitted."
        )

    n_features = max(vocabulary.values()) + 1

    # Tokenization: StringNormalizer + StringSplit
    x_tokens = _emit_tokenization(g, X, estimator.lowercase, name)

    # Count step
    mode, pool_strings, ngram_counts, ngram_indexes, weights = _build_tfidf_vectorizer_attrs(
        vocabulary, estimator.ngram_range, binary=False
    )
    tf_name = f"{name}_tf"
    kwargs = dict(
        mode=mode,
        min_gram_length=estimator.ngram_range[0],
        max_gram_length=estimator.ngram_range[1],
        max_skip_count=0,
        ngram_counts=ngram_counts,
        ngram_indexes=ngram_indexes,
        pool_strings=pool_strings,
    )
    if weights:
        kwargs["weights"] = weights

    tf = g.make_node(
        "TfIdfVectorizer",
        [x_tokens],
        outputs=[tf_name],
        domain="",
        name=f"{name}_tfvec",
        **kwargs,
    )
    tf = tf if isinstance(tf, str) else tf[0]
    g.set_type(tf, onnx.TensorProto.FLOAT)
    if g.has_shape(X):
        batch_dim = g.get_shape(X)[0]
        g.set_shape(tf, (batch_dim, n_features))
    elif g.has_rank(X):
        g.set_rank(tf, 2)

    # TF-IDF arithmetic (float32; TfIdfVectorizer always outputs float32)
    dtype = np.float32
    result: str = tf

    if estimator.sublinear_tf:
        zero = np.array(0, dtype=dtype)
        one = np.array(1, dtype=dtype)
        gt_zero = g.op.Greater(result, zero, name=f"{name}_gt_zero")
        log_tf = g.op.Log(result, name=f"{name}_log")
        log1p_tf = g.op.Add(log_tf, one, name=f"{name}_log1p")
        result = g.op.Where(gt_zero, log1p_tf, zero, name=f"{name}_sublinear_tf")

    if estimator.use_idf:
        idf = estimator.idf_.astype(dtype)
        result = g.op.Mul(result, idf, name=f"{name}_idf_mul")

    norm: Optional[str] = estimator.norm
    axes = np.array([1], dtype=np.int64)

    if norm in ("l2", "l1"):
        opset = g.get_opset("")
        if opset < 18:
            raise NotImplementedError(
                f"TraceableTfIdfVectorizer converter with norm={norm!r} requires "
                f"opset >= 18 (ReduceL1/ReduceL2 with axes as input was added in "
                f"opset 18), but the graph builder has opset {opset}."
            )
        if norm == "l2":
            norms = g.op.ReduceL2(result, axes, keepdims=1, name=f"{name}_l2norm")
        else:
            norms = g.op.ReduceL1(result, axes, keepdims=1, name=f"{name}_l1norm")
        zero_n = np.array([0], dtype=dtype)
        one_n = np.array([1], dtype=dtype)
        is_zero = g.op.Equal(norms, zero_n, name=f"{name}_is_zero")
        safe_norms = g.op.Where(is_zero, one_n, norms, name=f"{name}_safe_norm")
        res = g.op.Div(result, safe_norms, name=name, outputs=outputs)
    elif norm is None:
        res = g.op.Identity(result, name=name, outputs=outputs)
    else:
        raise ValueError(
            f"Unknown norm={norm!r} for TraceableTfIdfVectorizer, "
            f"expected 'l1', 'l2', or None."
        )

    assert isinstance(res, str)
    if not sts:
        g.set_type_shape_unary_op(res, tf)
    return res

from typing import Dict, List, Optional

import numpy as np
import onnx
from sklearn.feature_extraction.text import TfidfVectorizer

from ...typing import GraphBuilderExtendedProtocol
from ..register import register_sklearn_converter
from .count_vectorizer import _build_tfidf_vectorizer_attrs


@register_sklearn_converter(TfidfVectorizer)
def sklearn_tfidf_vectorizer(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: TfidfVectorizer,
    X: str,
    name: str = "tfidf_vectorizer",
) -> str:
    """
    Converts a :class:`sklearn.feature_extraction.text.TfidfVectorizer`
    into ONNX.

    :class:`~sklearn.feature_extraction.text.TfidfVectorizer` combines a
    :class:`~sklearn.feature_extraction.text.CountVectorizer` with a
    :class:`~sklearn.feature_extraction.text.TfidfTransformer`.  This
    converter reproduces that two-step pipeline:

    1. **Count step** — the ONNX ``TfIdfVectorizer`` operator (opset 9+)
       maps a pre-tokenised string tensor to raw term-frequency counts,
       exactly as ``CountVectorizer.transform`` does.

    2. **TF-IDF step** — the same sublinear-TF scaling, IDF weighting, and
       L1/L2 row normalisation implemented by the
       :class:`~sklearn.feature_extraction.text.TfidfTransformer` converter.

    The input tensor *X* must already be **tokenised** — a 2-D string tensor
    of shape ``(N, max_tokens_per_doc)`` where shorter rows are padded with
    empty strings ``""``.  Raw text documents are not accepted because ONNX
    lacks a standard tokeniser.

    **Supported options**

    * ``analyzer='word'`` only (character-level tokenisation has no ONNX
      equivalent).
    * ``ngram_range``, ``sublinear_tf``, ``use_idf``, ``norm`` ('l1', 'l2',
      or ``None``): all supported.
    * ``smooth_idf``, ``binary`` — reflected via the fitted ``idf_`` values
      and the count mode respectively.

    **Graph layout (all options active)**

    .. code-block:: text

        X  (N, seq_len) STRING
        │
        └── TfIdfVectorizer(mode=TF, …)       # raw counts
               │
               ├── Greater(0) / Log / Add(1) / Where  # sublinear_tf (optional)
               ├── Mul(idf_)                           # use_idf (optional)
               └── ReduceL2 / Div                      # norm (optional)
                      └── output  (N, n_features) float

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``TfidfVectorizer``
    :param outputs: desired output names
    :param X: input tensor name — a ``STRING`` tensor of shape
        ``(N, max_tokens_per_doc)`` (rows padded with ``""`` as needed)
    :param name: prefix for added node names
    :return: output tensor name
    :raises NotImplementedError: if ``analyzer`` is not ``'word'``, or if
        ``norm`` is ``'l1'`` or ``'l2'`` and the graph opset is < 18
    """
    assert isinstance(
        estimator, TfidfVectorizer
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    if itype != onnx.TensorProto.STRING:
        raise NotImplementedError(
            f"TfidfVectorizer conversion requires a STRING input tensor (got ONNX "
            f"type {itype}). Pass a 2-D padded string array of pre-tokenised words "
            f'as input X (shorter rows padded with empty string "").'
        )

    analyzer = estimator.analyzer
    if analyzer != "word":
        raise NotImplementedError(
            f"TfidfVectorizer converter only supports analyzer='word' (got {analyzer!r}). "
            f"For character-level analyzers, tokenise inputs to individual characters "
            f"before calling the converter."
        )

    vocabulary = estimator.vocabulary_
    if not vocabulary:
        raise ValueError("TfidfVectorizer has an empty vocabulary; ensure it was fitted.")

    n_features = max(vocabulary.values()) + 1

    # ------------------------------------------------------------------
    # Step 1 – raw term-frequency counts via ONNX TfIdfVectorizer
    # ------------------------------------------------------------------
    # Always use TF mode here; IDF weighting is handled manually below
    # so that sublinear_tf and norm can be applied consistently.
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
        kwargs["weights"] = weights  # type: ignore

    tf = g.make_node(
        "TfIdfVectorizer", [X], outputs=[tf_name], domain="", name=f"{name}_tfvec", **kwargs
    )
    tf = tf if isinstance(tf, str) else tf[0]

    # TfIdfVectorizer always outputs float32
    g.set_type(tf, onnx.TensorProto.FLOAT)
    if g.has_shape(X):
        batch_dim = g.get_shape(X)[0]
        g.set_shape(tf, (batch_dim, n_features))
    elif g.has_rank(X):
        g.set_rank(tf, 2)

    # ------------------------------------------------------------------
    # Step 2 – TF-IDF arithmetic (mirrors TfidfTransformer converter)
    # ------------------------------------------------------------------
    # The TfIdfVectorizer always returns float32; we work in float32.
    dtype = np.float32

    result: str = tf

    # 2a – sublinear TF scaling: replace count with 1 + log(count) where count > 0
    if estimator.sublinear_tf:
        zero = np.array(0, dtype=dtype)
        one = np.array(1, dtype=dtype)
        gt_zero = g.op.Greater(result, zero, name=f"{name}_gt_zero")
        log_tf = g.op.Log(result, name=f"{name}_log")
        log1p_tf = g.op.Add(log_tf, one, name=f"{name}_log1p")
        result = g.op.Where(gt_zero, log1p_tf, zero, name=f"{name}_sublinear_tf")

    # 2b – IDF weighting
    if estimator.use_idf:
        idf = estimator.idf_.astype(dtype)
        result = g.op.Mul(result, idf, name=f"{name}_idf_mul")

    # 2c – row normalisation
    norm: Optional[str] = estimator.norm
    axes = np.array([1], dtype=np.int64)

    if norm in ("l2", "l1"):
        opset = g.get_opset("")
        if opset < 18:
            raise NotImplementedError(
                f"TfidfVectorizer converter with norm={norm!r} requires opset >= 18 "
                f"(ReduceL1/ReduceL2 with axes as input was added in opset 18), "
                f"but the graph builder has opset {opset}."
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
            f"Unknown norm={norm!r} for TfidfVectorizer, expected 'l1', 'l2', or None."
        )

    assert isinstance(res, str)
    if not sts:
        g.set_type_shape_unary_op(res, tf)
    return res

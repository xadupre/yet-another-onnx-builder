from typing import Dict, List

import onnx
from sklearn.feature_extraction.text import CountVectorizer

from ...typing import GraphBuilderExtendedProtocol
from ..register import register_sklearn_converter


def _build_tfidf_vectorizer_attrs(vocabulary: Dict[str, int], ngram_range, binary: bool):
    """
    Converts a fitted sklearn vocabulary into the attribute lists required by
    the ONNX ``TfIdfVectorizer`` operator.

    For a word-level ``CountVectorizer``, each vocabulary entry is a
    space-joined sequence of words.  The ONNX operator expects them split back
    into individual tokens inside ``pool_strings``, grouped by n-gram length.

    The ``ngram_counts`` attribute in the ONNX spec always counts from
    1-grams upward: ``ngram_counts[k]`` is the starting index in
    ``pool_strings`` for ``(k+1)``-grams.  For gram lengths below
    ``min_gram_length`` the entry is set to ``0`` (an empty slice at the
    beginning of the pool).

    :param vocabulary: mapping ``{token_string: output_column_index}``.
    :param ngram_range: ``(min_n, max_n)`` tuple from the estimator.
    :param binary: when ``True`` return ``mode='IDF'`` with unit weights so
        that the operator clips repeated occurrences to 1.
    :returns: tuple ``(mode, pool_strings, ngram_counts, ngram_indexes, weights)``
        ready to be passed as ONNX node attributes.
    """
    min_n, max_n = ngram_range
    pool_strings: List[str] = []
    # ngram_counts must cover 1-grams through max_n-grams (ONNX spec).
    # Entries for n < min_n are 0 (empty groups at the start of the pool).
    ngram_counts: List[int] = [0] * (min_n - 1)
    ngram_indexes: List[int] = []

    for n in range(min_n, max_n + 1):
        ngram_counts.append(len(pool_strings))
        ngrams_n = [(k, v) for k, v in vocabulary.items() if len(k.split()) == n]
        ngrams_n.sort(key=lambda kv: kv[1])  # stable ordering by output index
        for token, idx in ngrams_n:
            pool_strings.extend(token.split())
            ngram_indexes.append(idx)

    if binary:
        mode = "IDF"
        weights = [1.0] * len(ngram_indexes)
    else:
        mode = "TF"
        weights = []

    return mode, pool_strings, ngram_counts, ngram_indexes, weights


@register_sklearn_converter(CountVectorizer)
def sklearn_count_vectorizer(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: CountVectorizer,
    X: str,
    name: str = "count_vectorizer",
) -> str:
    """
    Converts a :class:`sklearn.feature_extraction.text.CountVectorizer`
    into ONNX using the ``TfIdfVectorizer`` operator (opset 9+).

    The input tensor *X* must already be **tokenized** — it should be a 2-D
    string tensor of shape ``(N, max_tokens_per_doc)`` where each row contains
    the pre-split tokens of one document and shorter rows are padded with
    empty strings ``""`` (which the operator ignores).  This matches the
    behaviour expected by the ONNX ``TfIdfVectorizer`` operator; raw text
    documents cannot be accepted because ONNX lacks a standard tokeniser.

    **Supported options**

    * ``analyzer='word'`` — the only supported value; ``'char'`` and
      ``'char_wb'`` require character-level input tokenisation that has no
      ONNX equivalent.
    * ``ngram_range=(min_n, max_n)`` — any combination of positive integers.
    * ``binary=False`` (default, TF counts) or ``binary=True`` (binary
      presence, 0 or 1 per feature).
    * ``vocabulary_`` — any fitted vocabulary; no restriction on size.

    **Graph layout**

    .. code-block:: text

        X  (N, seq_len) STRING
        │
        └── TfIdfVectorizer(mode=TF/IDF, pool_strings=vocab, …)
               └── output  (N, n_features) FLOAT

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``CountVectorizer``
    :param outputs: desired output names
    :param X: input tensor name — a ``STRING`` tensor of shape
        ``(N, max_tokens_per_doc)`` (rows padded with ``""`` as needed)
    :param name: prefix for added node names
    :return: output tensor name
    :raises NotImplementedError: if ``analyzer`` is not ``'word'``
    """
    assert isinstance(
        estimator, CountVectorizer
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    if itype != onnx.TensorProto.STRING:
        raise NotImplementedError(
            f"CountVectorizer conversion requires a STRING input tensor (got ONNX "
            f"type {itype}). Pass a 2-D padded string array of pre-tokenised words "
            f"as input X (shorter rows padded with empty string \"\")."
        )

    analyzer = estimator.analyzer
    if analyzer != "word":
        raise NotImplementedError(
            f"CountVectorizer converter only supports analyzer='word' (got {analyzer!r}). "
            f"For character-level analyzers, tokenise inputs to individual characters "
            f"before calling the converter."
        )

    vocabulary = estimator.vocabulary_
    if not vocabulary:
        raise ValueError("CountVectorizer has an empty vocabulary; ensure it was fitted.")

    n_features = max(vocabulary.values()) + 1
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
        [X],
        outputs=outputs,
        domain="",
        name=name,
        **kwargs,
    )

    res_name = res if isinstance(res, str) else res[0]
    # TfIdfVectorizer always outputs float32
    g.set_type(res_name, onnx.TensorProto.FLOAT)
    if g.has_shape(X):
        batch_dim = g.get_shape(X)[0]
        g.set_shape(res_name, (batch_dim, n_features))
    elif g.has_rank(X):
        g.set_rank(res_name, 2)
    return res_name

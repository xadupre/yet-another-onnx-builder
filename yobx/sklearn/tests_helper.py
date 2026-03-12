import numpy as np
import onnx
from sklearn.datasets import (
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from ..reference import ExtendedReferenceEvaluator


def _has_predict_proba(model):
    """Check whether the model exposes a ``predict_proba`` method.

    Voting classifiers configured with ``voting='hard'`` do not produce
    probability estimates even though the attribute may be present, so they
    are treated as not having ``predict_proba``.

    :param model: A fitted scikit-learn estimator.
    :return: ``True`` when the model can produce probability estimates,
        ``False`` otherwise.
    """
    if hasattr(model, "voting") and model.voting == "hard":
        return False
    return hasattr(model, "predict_proba")


def _has_decision_function(model):
    """Check whether the model exposes a ``decision_function`` method.

    Voting classifiers and :class:`~sklearn.compose.TransformedTargetRegressor`
    with a ``dtype`` attribute (e.g. ``CastRegressor``) are excluded because
    their ``decision_function`` output is not comparable to a plain model's.

    :param model: A fitted scikit-learn estimator.
    :return: ``True`` when the model provides a ``decision_function``,
        ``False`` otherwise.
    """
    if hasattr(model, "voting"):
        return False
    if hasattr(model, "dtype"):  # CastRegressor
        return False
    return hasattr(model, "decision_function")


def _has_transform_model(model):
    """Check whether the model exposes a ``transform`` method.

    Voting classifiers are excluded because their ``transform`` output does not
    have the same semantics as a standard transformer.

    :param model: A fitted scikit-learn estimator.
    :return: ``True`` when the model has a ``transform`` method,
        ``False`` otherwise.
    """
    if hasattr(model, "voting"):
        return False
    return hasattr(model, "transform")


def fit_classification_model(
    model,
    n_classes,
    is_int=False,
    pos_features=False,
    label_string=False,
    random_state=42,
    is_bool=False,
    n_features=20,
    n_redundant=None,
    n_repeated=None,
    cls_dtype=None,
    is_double=False,
    n_samples=250,
):
    """Fit a classification model on a synthetic dataset and return it with test data.

    Generates a classification dataset using
    :func:`sklearn.datasets.make_classification`, fits *model* on the training
    split, and returns the fitted model together with the held-out test set.

    :param model: An unfitted scikit-learn classifier.
    :param n_classes: Number of target classes.
    :param is_int: When ``True``, cast features to ``np.int64``.
    :param pos_features: When ``True``, take the absolute value of all features
        so that every entry is non-negative.
    :param label_string: When ``True``, convert integer labels to string labels
        of the form ``"cl<i>"``.
    :param random_state: Random seed forwarded to :func:`make_classification`
        and :func:`train_test_split`.
    :param is_bool: When ``True``, cast features to ``bool`` (implies ``is_int``
        for the intermediate cast).
    :param n_features: Total number of features in the generated dataset.
    :param n_redundant: Number of redundant features.  Defaults to
        ``min(2, n_features - min(7, n_features))`` — at most 2, reduced when
        *n_features* is small.
    :param n_repeated: Number of repeated features.  Defaults to ``0``.
    :param cls_dtype: If provided, cast the label array to this dtype before
        fitting.
    :param is_double: When ``True``, cast features to ``np.float64`` after the
        integer/bool cast step.
    :param n_samples: Number of samples in the generated dataset.
    :return: A tuple ``(fitted_model, X_test)`` where *X_test* is the
        held-out feature matrix.
    """
    X, y = make_classification(
        n_classes=n_classes,
        n_features=n_features,
        n_samples=n_samples,
        random_state=random_state,
        n_informative=min(7, n_features),
        n_redundant=n_redundant or min(2, n_features - min(7, n_features)),
        n_repeated=n_repeated or 0,
    )
    if cls_dtype is not None:
        y = y.astype(cls_dtype)
    if label_string:
        y = np.array(["cl%d" % cl for cl in y])
    X = X.astype(np.int64) if is_int or is_bool else X.astype(np.float32)
    X = X.astype(np.double) if is_double else X
    if pos_features:
        X = np.abs(X)
    if is_bool:
        X = X.astype(bool)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test


def fit_clustering_model(
    model,
    n_classes,
    is_int=False,
    pos_features=False,
    label_string=False,
    random_state=42,
    is_bool=False,
    n_features=20,
    n_redundant=None,
    n_repeated=None,
):
    """Fit a clustering model on a synthetic dataset and return it with test data.

    Generates a classification dataset (used purely for the feature matrix)
    using :func:`sklearn.datasets.make_classification`, fits *model* on the
    training split (labels are not used), and returns the fitted model together
    with the held-out test set.

    :param model: An unfitted scikit-learn clustering estimator (e.g.
        :class:`sklearn.cluster.KMeans`).
    :param n_classes: Number of classes used when generating the dataset (acts
        as a proxy for the number of natural clusters in the feature space).
    :param is_int: When ``True``, cast features to ``np.int64``.
    :param pos_features: When ``True``, take the absolute value of all features
        so that every entry is non-negative.
    :param label_string: When ``True``, convert integer labels to string labels
        of the form ``"cl<i>"`` (not used for fitting but kept for API symmetry).
    :param random_state: Random seed forwarded to :func:`make_classification`
        and :func:`train_test_split`.
    :param is_bool: When ``True``, cast features to ``bool`` (implies ``is_int``
        for the intermediate cast).
    :param n_features: Total number of features in the generated dataset.
    :param n_redundant: Number of redundant features.  Defaults to
        ``min(2, n_features - min(7, n_features))`` — at most 2, reduced when
        *n_features* is small.
    :param n_repeated: Number of repeated features.  Defaults to ``0``.
    :return: A tuple ``(fitted_model, X_test)`` where *X_test* is the
        held-out feature matrix.
    """
    X, y = make_classification(
        n_classes=n_classes,
        n_features=n_features,
        n_samples=250,
        random_state=random_state,
        n_informative=min(7, n_features),
        n_redundant=n_redundant or min(2, n_features - min(7, n_features)),
        n_repeated=n_repeated or 0,
    )
    if label_string:
        y = np.array(["cl%d" % cl for cl in y])
    X = X.astype(np.int64) if is_int or is_bool else X.astype(np.float32)
    if pos_features:
        X = np.abs(X)
    if is_bool:
        X = X.astype(bool)
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=42)
    model.fit(X_train)
    return model, X_test


def fit_multilabel_classification_model(
    model, n_classes=5, n_labels=2, n_samples=200, n_features=20, is_int=False
):
    """Fit a multilabel classification model on a synthetic dataset.

    Generates a multilabel dataset using
    :func:`sklearn.datasets.make_multilabel_classification`, fits *model* on
    the training split, and returns the fitted model together with the held-out
    test set.

    :param model: An unfitted scikit-learn multilabel classifier.
    :param n_classes: Number of classes (output labels).
    :param n_labels: Average number of labels per sample.
    :param n_samples: Total number of samples in the generated dataset.
    :param n_features: Number of features in the generated dataset.
    :param is_int: When ``True``, cast features to ``np.int64``; otherwise
        cast to ``np.float32``.
    :return: A tuple ``(fitted_model, X_test)`` where *X_test* is the
        held-out feature matrix.
    """
    X, y = make_multilabel_classification(
        n_classes=n_classes,
        n_labels=n_labels,
        n_features=n_features,
        n_samples=n_samples,
        random_state=42,
    )
    X = X.astype(np.int64) if is_int else X.astype(np.float32)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test


def fit_multi_output_classification_model(
    model,
    n_classes=3,
    n_samples=100,
    n_features=4,
    n_informative=5,
    n_outputs=2,
):
    """Fit a multi-output classification model on a synthetic integer dataset.

    Generates a random integer feature matrix and a multi-column integer label
    matrix, fits a :class:`~sklearn.ensemble.RandomForestClassifier` (the
    *model* parameter is accepted for API consistency but is replaced
    internally), and returns the fitted model together with a small test set.

    :param model: Accepted for API consistency; internally replaced by a
        :class:`~sklearn.ensemble.RandomForestClassifier`.
    :param n_classes: Number of distinct integer classes per output column.
    :param n_samples: Number of training samples.
    :param n_features: Number of feature columns.
    :param n_informative: Upper bound for the random integer values in the
        feature matrix.
    :param n_outputs: Number of output columns (targets).
    :return: A tuple ``(fitted_model, X_test)`` where *X_test* contains 10
        samples drawn from the same integer distribution as the training set.
    """
    np.random.seed(0)
    X_train = np.random.randint(0, n_informative, size=(n_samples, n_features))
    y_train = np.random.randint(0, n_classes, size=(n_samples, n_outputs))
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    X_test = np.random.randint(0, n_informative, size=(10, n_features))
    return model, X_test


def fit_regression_model(
    model,
    is_int=False,
    n_targets=1,
    is_bool=False,
    factor=1.0,
    n_features=10,
    n_samples=250,
    n_informative=10,
):
    """Fit a regression model on a synthetic dataset and return it with test data.

    Generates a regression dataset using :func:`sklearn.datasets.make_regression`,
    fits *model* on the training split, and returns the fitted model together
    with the held-out test set.

    :param model: An unfitted scikit-learn regressor.
    :param is_int: When ``True``, cast features to ``np.int64``.
    :param n_targets: Number of regression targets.
    :param is_bool: When ``True``, cast features to ``bool`` (implies ``is_int``
        for the intermediate cast).
    :param factor: Multiplicative scaling factor applied to the target values
        after generation.
    :param n_features: Total number of features in the generated dataset.
    :param n_samples: Number of samples in the generated dataset.
    :param n_informative: Number of informative features used by
        :func:`make_regression`.
    :return: A tuple ``(fitted_model, X_test)`` where *X_test* is the
        held-out feature matrix.
    """
    X, y = make_regression(
        n_features=n_features,
        n_samples=n_samples,
        n_targets=n_targets,
        random_state=42,
        n_informative=n_informative,
    )
    y *= factor
    X = X.astype(np.int64) if is_int or is_bool else X.astype(np.float32)
    if is_bool:
        X = X.astype(bool)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test


def _assert_close(expected: np.ndarray, value: np.ndarray, name: str, atol=1e-5):
    """Assert that two arrays are element-wise close within an absolute tolerance.

    Checks that *expected* and *value* share the same dtype and shape, then
    verifies that the maximum absolute difference does not exceed *atol*.

    :param expected: Reference array produced by scikit-learn.
    :param value: Array to compare against *expected* (e.g. ONNX output).
    :param name: Human-readable test name included in assertion messages to
        help identify the failing comparison.
    :param atol: Absolute tolerance for numerical comparison.  Defaults to
        ``1e-5``.
    :raises AssertionError: If the dtypes differ, the shapes differ, or the
        maximum absolute difference exceeds *atol*.
    """
    assert (
        expected.dtype == value.dtype
    ), f"Type mismatch between {expected.dtype} and {value.dtype} for test name {name!r}"
    assert (
        expected.shape == value.shape
    ), f"Type mismatch between {expected.shape} and {value.shape} for test name {name!r}"
    diff = np.abs(expected - value).max()
    assert diff <= atol, f"discrepancies {diff} for test name={name!r}"


def dump_data_and_model(
    data: np.ndarray, model: BaseEstimator, model_onnx: onnx.ModelProto, basename: str
):
    """Validate an ONNX model against the original scikit-learn model.

    Runs *data* through both :mod:`onnxruntime` and
    :class:`~yobx.reference.ExtendedReferenceEvaluator` and compares both
    outputs to the predictions produced by *model*:

    * If the model has :meth:`predict_proba`, both labels and probabilities
      are compared (outputs ``0`` and ``1`` respectively).
    * Else if the model has :meth:`transform`, the transformed output is
      compared (output ``0``).
    * Otherwise the :meth:`predict` output is compared (output ``0``).

    All comparisons use :func:`_assert_close` with default tolerance.

    :param data: Input feature matrix (``np.ndarray``) passed to all
        evaluators.
    :param model: The fitted scikit-learn estimator used as the reference.
    :param model_onnx: The ONNX representation of *model* to validate.
    :param basename: Short identifier used as a prefix in assertion messages
        to make failures easier to locate.
    :raises AssertionError: If any output differs beyond the tolerance defined
        in :func:`_assert_close`.
    """
    import onnxruntime

    feeds = {model_onnx.graph.input[0].name: data}
    sess = onnxruntime.InferenceSession(
        model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    ort_value = sess.run(None, feeds)
    py_value = ExtendedReferenceEvaluator(model_onnx).run(None, feeds)
    if _has_predict_proba(model):
        expected_labels = model.predict(data)
        expected_proba = model.predict_proba(data)
        _assert_close(expected_proba, ort_value[1], name=f"{basename}/ort1")
        _assert_close(expected_labels, ort_value[0], name=f"{basename}/ort1")
        _assert_close(expected_proba, py_value[1], name=f"{basename}/py1")
        _assert_close(expected_labels, py_value[0], name=f"{basename}/py1")
    elif _has_transform_model(model):
        expected = model.transform(data)
        _assert_close(expected, ort_value[0], name=f"{basename}/ort2")
        _assert_close(expected, py_value[0], name=f"{basename}/py2")
    else:
        expected = model.predict(data)
        _assert_close(expected, ort_value[0], name=f"{basename}/ort2")
        _assert_close(expected, py_value[0], name=f"{basename}/py2")

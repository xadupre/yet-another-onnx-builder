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
    if hasattr(model, "voting") and model.voting == "hard":
        return False
    return hasattr(model, "predict_proba")


def _has_decision_function(model):
    if hasattr(model, "voting"):
        return False
    if hasattr(model, "dtype"):  # CastRegressor
        return False
    return hasattr(model, "decision_function")


def _has_transform_model(model):
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

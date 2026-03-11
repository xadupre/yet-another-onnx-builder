# SPDX-License-Identifier: Apache-2.0
"""
sklearn_onnx_conftest.py – patches skl2onnx.convert_sklearn so that the
sklearn-onnx test suite exercises yobx.sklearn.to_onnx instead of
skl2onnx's own converter.

This file is copied into the ``tests/`` directory of a cloned sklearn-onnx
repository before running its test suite.  pytest loads ``conftest.py``
before importing any test module, so every
``from skl2onnx import convert_sklearn`` in the test files picks up the
patched implementation automatically.

Usage (from the CI workflow):
    cp _tools/sklearn_onnx_conftest.py /tmp/sklearn-onnx/tests/conftest.py
    pytest /tmp/sklearn-onnx/tests/ ...
"""

import numpy as np
import skl2onnx

try:
    from skl2onnx.common.data_types import (
        FloatTensorType,
        DoubleTensorType,
        Int64TensorType,
    )
except ImportError:
    FloatTensorType = DoubleTensorType = Int64TensorType = None

from yobx.sklearn import to_onnx
from yobx import DEFAULT_TARGET_OPSET


def _make_dummy_input(type_obj):
    """Create a small concrete numpy array from a skl2onnx type descriptor.

    Shape dimensions that are None or 0 are replaced with 3 so that
    the resulting array is valid for fitting a scikit-learn estimator.
    """
    if type_obj is None:
        return np.ones((3, 1), dtype=np.float32)
    shape = getattr(type_obj, "shape", [None, 1])
    concrete = []
    for d in shape:
        if d is None or d == 0:
            concrete.append(3)
        else:
            try:
                concrete.append(int(d))
            except (TypeError, ValueError):
                concrete.append(3)
    if DoubleTensorType is not None and isinstance(type_obj, DoubleTensorType):
        return np.ones(concrete, dtype=np.float64)
    if Int64TensorType is not None and isinstance(type_obj, Int64TensorType):
        return np.zeros(concrete, dtype=np.int64)
    return np.ones(concrete, dtype=np.float32)


def _yobx_convert_sklearn(
    model,
    name=None,
    initial_types=None,
    doc_string="",
    target_opset=None,
    custom_conversion_functions=None,
    custom_shape_calculators=None,
    custom_parsers=None,
    options=None,
    intermediate=False,
    white_op=None,
    black_op=None,
    final_types=None,
    dtype=None,
    naming=None,
    model_optim=True,
    verbose=0,
):
    """Drop-in replacement for skl2onnx.convert_sklearn that delegates to yobx."""
    if not initial_types:
        initial_types = [("X", None)]
    args = tuple(_make_dummy_input(t) for _, t in initial_types)
    input_names = [n for n, _ in initial_types]
    return to_onnx(
        model,
        args,
        input_names=input_names,
        target_opset=target_opset if target_opset is not None else DEFAULT_TARGET_OPSET,
    )


# Patch at module level so that all subsequent
# ``from skl2onnx import convert_sklearn`` statements in test files
# receive the yobx implementation.
skl2onnx.convert_sklearn = _yobx_convert_sklearn

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# run_sklearn_onnx_tests.sh – run the upstream sklearn-onnx test suite locally
# with yobx.sklearn.to_onnx substituted in for skl2onnx.convert_sklearn.
#
# Usage:
#   bash _tools/run_sklearn_onnx_tests.sh [CLONE_DIR] [RESULTS_FILE]
#
#   CLONE_DIR     directory to clone sklearn-onnx into (default: /tmp/sklearn-onnx)
#   RESULTS_FILE  file to write test output to   (default: /tmp/sklearn-onnx-test-results.txt)
#
# The script must be run from the root of the yobx repository so that the
# relative path _tools/sklearn_onnx_conftest.py can be resolved.

if [ -z "${BASH_VERSION:-}" ]; then
    echo "ERROR: this script must be run with bash, not sh/dash." >&2
    exit 1
fi

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLONE_DIR="${1:-/tmp/sklearn-onnx}"
RESULTS_FILE="${2:-/tmp/sklearn-onnx-test-results.txt}"
CONFTEST_SRC="${REPO_ROOT}/_tools/sklearn_onnx_conftest.py"

if [ ! -f "${CONFTEST_SRC}" ]; then
    echo "ERROR: conftest source not found: ${CONFTEST_SRC}" >&2
    echo "       Run this script from the root of the yobx repository." >&2
    exit 1
fi

echo "=== yobx sklearn-onnx compatibility runner ==="
echo "  yobx root  : ${REPO_ROOT}"
echo "  clone dir  : ${CLONE_DIR}"
echo "  results    : ${RESULTS_FILE}"
echo ""

# ------------------------------------------------------------------
# Clone the upstream sklearn-onnx repository (skip if already cloned)
# ------------------------------------------------------------------
if [ -d "${CLONE_DIR}/.git" ]; then
    echo "[1/3] sklearn-onnx already cloned at ${CLONE_DIR} – skipping clone."
else
    echo "[1/3] Cloning sklearn-onnx into ${CLONE_DIR} ..."
    git clone --depth 1 https://github.com/onnx/sklearn-onnx.git "${CLONE_DIR}"
fi

# ------------------------------------------------------------------
# Copy the conftest.py that patches skl2onnx.convert_sklearn
# ------------------------------------------------------------------
echo "[2/3] Installing conftest.py into ${CLONE_DIR}/tests/ ..."
cp "${CONFTEST_SRC}" "${CLONE_DIR}/tests/conftest.py"

# ------------------------------------------------------------------
# Run the upstream test suite (failures are expected for unsupported models)
# ------------------------------------------------------------------
echo "[3/3] Running sklearn-onnx tests (output → ${RESULTS_FILE}) ..."
echo "      Failures for models not yet supported by yobx are expected."
echo ""

set +e   # allow pytest to return non-zero without aborting the script
pytest "${CLONE_DIR}/tests/" \
    --ignore="${CLONE_DIR}/tests/test_algebra_cascade.py" \
    --ignore="${CLONE_DIR}/tests/test_algebra_complex.py" \
    --ignore="${CLONE_DIR}/tests/test_algebra_converters.py" \
    --ignore="${CLONE_DIR}/tests/test_algebra_custom_model.py" \
    --ignore="${CLONE_DIR}/tests/test_algebra_custom_model_sub_estimator.py" \
    --ignore="${CLONE_DIR}/tests/test_algebra_deprecation.py" \
    --ignore="${CLONE_DIR}/tests/test_algebra_double.py" \
    --ignore="${CLONE_DIR}/tests/test_algebra_onnx_doc.py" \
    --ignore="${CLONE_DIR}/tests/test_algebra_onnx_operator_mixin_syntax.py" \
    --ignore="${CLONE_DIR}/tests/test_algebra_onnx_operators.py" \
    --ignore="${CLONE_DIR}/tests/test_algebra_onnx_operators_if.py" \
    --ignore="${CLONE_DIR}/tests/test_algebra_onnx_operators_opset.py" \
    --ignore="${CLONE_DIR}/tests/test_algebra_onnx_operators_scan.py" \
    --ignore="${CLONE_DIR}/tests/test_algebra_onnx_operators_sparse.py" \
    --ignore="${CLONE_DIR}/tests/test_algebra_onnx_operators_sub_estimator.py" \
    --ignore="${CLONE_DIR}/tests/test_algebra_onnx_operators_wrapped.py" \
    --ignore="${CLONE_DIR}/tests/test_algebra_symbolic.py" \
    --ignore="${CLONE_DIR}/tests/test_algebra_test_helper.py" \
    --ignore="${CLONE_DIR}/tests/test_utils_sklearn.py" \
    --durations=20 \
    -q \
    2>&1 | tee "${RESULTS_FILE}"
EXIT_CODE=${PIPESTATUS[0]}
set -e

echo ""
echo "=== Done – results saved to ${RESULTS_FILE} (exit code: ${EXIT_CODE}) ==="
exit "${EXIT_CODE}"

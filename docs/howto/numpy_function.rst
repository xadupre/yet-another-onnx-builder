.. _l-howto-numpy-function:

NumPy function
==============

This page answers common *"how do I…"* questions for exporting a Python
function that processes :epkg:`numpy` arrays into ONNX.

----

How to export a function processing numpy arrays
------------------------------------------------

Use :func:`yobx.sql.trace_numpy_to_onnx` with a representative sample input.
Only the input dtype and shape are used during export.

.. runpython::
    :showcode:

    import numpy as np
    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.sql import trace_numpy_to_onnx

    def normalize_rows(X):
        norms = np.sqrt(np.sum(X ** np.float32(2), axis=1, keepdims=True))
        safe_norms = np.where(norms < np.float32(1e-8), np.float32(1), norms)
        return X / safe_norms

    X_sample = np.zeros((3, 4), dtype=np.float32)
    onx = trace_numpy_to_onnx(normalize_rows, X_sample)
    print(pretty_onnx(onx))

----

How to export a function with multiple inputs and outputs
---------------------------------------------------------

Use ``input_names`` and ``output_names`` when your function takes more than one
input array and returns multiple arrays.

.. runpython::
    :showcode:

    import numpy as np
    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.sql import trace_numpy_to_onnx

    def combine(A, B):
        return A + B, A - B

    A = np.random.randn(2, 3).astype(np.float32)
    B = np.random.randn(2, 3).astype(np.float32)
    onx = trace_numpy_to_onnx(
        combine,
        A,
        B,
        input_names=["A", "B"],
        output_names=["sum", "diff"],
    )
    print(pretty_onnx(onx))

----

How to validate ONNX outputs against numpy
------------------------------------------

Run the exported graph with :epkg:`onnxruntime` and compare with the original
numpy implementation.

.. runpython::
    :showcode:

    import numpy as np
    import onnxruntime
    from yobx.sql import trace_numpy_to_onnx

    def normalize_rows(X):
        norms = np.sqrt(np.sum(X ** np.float32(2), axis=1, keepdims=True))
        safe_norms = np.where(norms < np.float32(1e-8), np.float32(1), norms)
        return X / safe_norms

    rng = np.random.default_rng(0)
    X_sample = rng.standard_normal((8, 4)).astype(np.float32)
    X_test = rng.standard_normal((5, 4)).astype(np.float32)

    onx = trace_numpy_to_onnx(normalize_rows, X_sample)
    sess = onnxruntime.InferenceSession(
        onx.SerializeToString(), providers=["CPUExecutionProvider"]
    )

    (got,) = sess.run(None, {"X": X_test})
    expected = normalize_rows(X_test)
    print(f"max|diff|={np.abs(got - expected).max():.2e}")
    assert np.allclose(got, expected, atol=1e-5)
    print("Outputs match ✓")

.. seealso::

    :ref:`l-design-function-transformer-tracing` — implementation details on
    numpy tracing and supported operators.

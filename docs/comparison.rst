Comparison with Existing Tools
==============================

Design choices `yobx`
+++++++++++++++++++++

* **Single entry point** — `yobx.to_onnx` dispatches to the right backend automatically; no need to learn a different API for every framework.
* **Pluggable graph-builder** — the intermediate ONNX graph uses the native :epkg:`onnx-light` builder through a common converter API. Custom implementations must satisfy the protocol and return native onnx-light protobufs; the former Spox and onnxscript bridges are no longer supported.
* **Transparent names** — node names, initializer names and result names are preserved as-is (unless they are not unique); what the builder writes is what ends up in the ONNX file.
* **Built-in optimizer** — pattern-based graph rewrites (constant folding, fused ops, …) can be run before serialization.
* **ORT-specific targets** — passing `target_opset={"": 22, "com.microsoft": 1}` enables `com.microsoft` domain operators consumed directly by [onnxruntime](https://onnxruntime.ai/).

Comparison with existing tools
++++++++++++++++++++++++++++++

The main new features is the possibility to trace functions written with NumPy, functions operating on DataFrames, and SQL queries.
User can now convert `FunctionTransformer` from scikit-learn or preprocessing through SQL queries or DataFrames.

The implementation was simplified to only handle recent versions of scikit-learn, TensorFlow/Keras, LiteRT.
It was extended to other famous packages such `category_encoders`.

One single package for one single repository, one possible source of issues, making it easier for contributors to answer.

:func:`torch.onnx.export` is the official PyTorch exporter.
``yobx`` instead lowers :func:`torch.export.export` graphs through onnx-light's
native builder, optimizer, and shape inference engine, without delegating to
onnxscript or requiring the reference ``onnx`` package.

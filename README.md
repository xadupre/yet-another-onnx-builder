# yet-another-onnx-builder

A Python package that provides a **pipe operator** (`|`) for composing ONNX
models and operators, inspired by Unix pipes.

## Installation

```bash
pip install onnx_pipe
```

## Usage

### Chain single operators with `|`

```python
import numpy as np
import onnxruntime as rt
from onnx_pipe import op

# Abs followed by Relu followed by Sigmoid
pipe = op("Abs") | op("Relu") | op("Sigmoid")

model = pipe.to_onnx()

sess = rt.InferenceSession(model.SerializeToString())
x = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
(result,) = sess.run(None, {"X": x})
print(result)  # [0.7310586 0.880797  0.7310586]
```

### Wrap an existing `ModelProto`

```python
import onnx
from onnx_pipe import OnnxPipe, op

existing = onnx.load("my_model.onnx")
pipe = OnnxPipe(existing) | op("Sigmoid")
model = pipe.to_onnx()
```

### `op()` with custom operator attributes

```python
from onnx_pipe import op

# Clip operator with min/max attributes
clip_pipe = op("Clip", input_names=["X", "min", "max"], output_names=["Y"])
```

## API

### `op(op_type, *, input_names=None, output_names=None, ...)`

Creates an `OnnxPipe` wrapping a single ONNX operator.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `op_type` | — | ONNX operator name, e.g. `"Abs"` |
| `input_names` | `["X"]` | Graph input names |
| `output_names` | `["Y"]` | Graph output names |
| `input_type` | `FLOAT` | ONNX element type for inputs |
| `output_type` | `FLOAT` | ONNX element type for outputs |
| `domain` | `""` | Operator domain |
| `opset_version` | `20` | Opset version |

### `OnnxPipe(model)`

Wraps an `onnx.ModelProto`.

| Method / property | Description |
|-------------------|-------------|
| `pipe1 \| pipe2` | Returns a new `OnnxPipe` chaining the two models |
| `.to_onnx()` | Returns the underlying `onnx.ModelProto` |
| `.input_names` | List of graph input names |
| `.output_names` | List of graph output names |


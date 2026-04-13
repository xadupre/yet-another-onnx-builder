"""Tests that verify yobx can export models that use ``torch.autocast``."""

import unittest
from onnx import TensorProto
from onnx.inliner import inline_local_functions
from yobx.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_torch,
    skipif_ci_windows,
    hide_stdout,
)
from yobx.torch.interpreter import to_onnx


def _all_graph_nodes(proto):
    """Yields every NodeProto from the main graph and all local functions of the model."""
    yield from proto.graph.node
    for f in proto.functions:
        yield from f.node


def _cast_targets(proto):
    """Returns the set of ``to`` attribute values from all Cast nodes in the model."""
    result = set()
    for node in _all_graph_nodes(proto):
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to":
                    result.add(attr.i)
    return result


@requires_torch("2.4")
class TestOnnxExportAutocast(ExtTestCase):
    """Tests for exporting models that use ``torch.autocast``."""

    @skipif_ci_windows("torch dynamo not supported on windows")
    @ignore_warnings((UserWarning, DeprecationWarning, FutureWarning))
    def test_autocast_disabled(self):
        """Exports a model whose forward uses ``torch.autocast(enabled=False)``."""
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                with torch.autocast(device_type="cpu", dtype=torch.float32, enabled=False):
                    return torch.mm(x, x.T)

        model = Model()
        x = torch.randn(3, 4)
        inputs = (x,)
        expected = model(x)
        onx = to_onnx(model, inputs)
        self.assert_conversion_with_ort_on_cpu(onx, expected, inputs, atol=1e-5)

    @skipif_ci_windows("torch dynamo not supported on windows")
    @ignore_warnings((UserWarning, DeprecationWarning, FutureWarning))
    def test_autocast_float16_enabled(self):
        """Exports a model whose forward uses ``torch.autocast(enabled=True)`` with float16.

        This exercises the ``aten_wrap_with_autocast`` path where ``enabled=True``
        and verifies that the resulting ONNX model runs without error and
        produces numerically close results.
        """
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                with torch.autocast(device_type="cpu", dtype=torch.float16):
                    return torch.mm(x, x.T)

        model = Model()
        x = torch.randn(3, 4)
        inputs = (x,)
        expected = model(x).float()
        onx = to_onnx(model, inputs)
        # The ONNX output is float16; cast to float32 for comparison.
        import numpy as np
        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)
        got_f32 = got[0].astype(np.float32)
        self.assertEqualArray(expected.detach().numpy(), got_f32, atol=1e-1)

    @skipif_ci_windows("torch dynamo not supported on windows")
    @ignore_warnings((UserWarning, DeprecationWarning, FutureWarning))
    def test_autocast_linear_float16(self):
        """Exports a model with a Linear layer under ``torch.autocast(float16)``."""
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 3)

            def forward(self, x):
                with torch.autocast(device_type="cpu", dtype=torch.float16):
                    return self.linear(x)

        model = Model()
        x = torch.randn(2, 4)
        inputs = (x,)
        expected = model(x).float()
        onx = to_onnx(model, inputs)
        import numpy as np
        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)
        got_f32 = got[0].astype(np.float32)
        self.assertEqualArray(expected.detach().numpy(), got_f32, atol=1e-1)

    @skipif_ci_windows("torch dynamo not supported on windows")
    @ignore_warnings((UserWarning, DeprecationWarning, FutureWarning))
    @hide_stdout()
    def test_autocast_float16_cast_nodes_present(self):
        """Verifies that Cast nodes to float16 are present in the ONNX graph.

        When a model uses ``torch.autocast(dtype=float16)``, the exported ONNX
        graph must contain at least one ``Cast`` node whose ``to`` attribute is
        ``TensorProto.FLOAT16``.  This test catches the regression where the
        cast operations were missing from the FX graph and therefore absent from
        the ONNX output.
        """
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                with torch.autocast(device_type="cpu", dtype=torch.float16):
                    # Under float16 autocast the input tensor must be cast to
                    # float16 before the matmul; that Cast must appear in the
                    # exported ONNX graph.
                    return torch.mm(x, x.T)

        model = Model()
        x = torch.randn(3, 4)
        inputs = (x,)
        onx = to_onnx(model, inputs, verbose=10)

        # Inline local functions so every Cast node is visible in the flat graph.
        flat = inline_local_functions(onx)
        cast_tos = _cast_targets(flat)

        self.assertIn(
            TensorProto.FLOAT16,
            cast_tos,
            "Expected a Cast node with to=FLOAT16 in the ONNX graph, "
            f"but only found casts to dtypes: {cast_tos}. "
            "This likely means the autocast dtype-promotion casts are missing "
            "from the FX graph.",
        )

    @skipif_ci_windows("torch dynamo not supported on windows")
    @ignore_warnings((UserWarning, DeprecationWarning, FutureWarning))
    def test_autocast_linear_float16_cast_nodes_and_output_dtype(self):
        """Verifies Cast nodes and correct output dtype for a Linear model under autocast.

        Checks two things:
        1. The ONNX graph (after inlining) contains a Cast to ``FLOAT16``.
        2. The model's graph output is declared as ``FLOAT16`` in the ONNX proto.
        """
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 3)

            def forward(self, x):
                with torch.autocast(device_type="cpu", dtype=torch.float16):
                    return self.linear(x)

        model = Model()
        x = torch.randn(2, 4)
        inputs = (x,)
        onx = to_onnx(model, inputs)

        # 1. Check that a Cast to float16 is present anywhere in the model.
        flat = inline_local_functions(onx)
        cast_tos = _cast_targets(flat)
        self.assertIn(
            TensorProto.FLOAT16,
            cast_tos,
            "Expected a Cast node with to=FLOAT16 in the ONNX graph, "
            f"but only found casts to dtypes: {cast_tos}. "
            "This likely means the autocast dtype-promotion casts are missing.",
        )

        # 2. The graph-level output should be typed as float16.
        output_elem_types = [o.type.tensor_type.elem_type for o in flat.graph.output]
        self.assertIn(
            TensorProto.FLOAT16,
            output_elem_types,
            "Expected the ONNX graph output to have elem_type=FLOAT16, "
            f"but got elem_types={output_elem_types}.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

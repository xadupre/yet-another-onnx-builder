import unittest
from typing import Any, Dict, Optional, Tuple
import numpy as np
import ml_dtypes
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from yobx.ext_test_case import ExtTestCase, hide_stdout
from yobx.helpers.onnx_helper import tensor_dtype_to_np_dtype
from yobx.reference._inference_session import investigate_onnxruntime_issue
from yobx.reference._inference_session_numpy import InferenceSessionForNumpy

TFLOAT = onnx.TensorProto.FLOAT


class TestInferenceSession(ExtTestCase):
    @classmethod
    def _range(cls, *shape, bias: Optional[float] = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    @classmethod
    def _get_model(
        cls,
    ) -> Tuple[onnx.ModelProto, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]:
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["x", "y"], ["gggg"]),
                    oh.make_node("Add", ["gggg", "z"], ["final"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("x", TFLOAT, [None, None]),
                    oh.make_tensor_value_info("y", TFLOAT, [None, None]),
                    oh.make_tensor_value_info("z", TFLOAT, [None, None]),
                ],
                [oh.make_tensor_value_info("final", TFLOAT, [None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        onnx.checker.check_model(model)
        feeds = {"x": cls._range(5, 6), "y": cls._range(5, 6), "z": cls._range(5, 6)}
        return model, feeds, (feeds["x"] + feeds["y"] + feeds["z"],)

    def test_ort_value_dlpack_numpy(self):
        import onnxruntime as onnxrt
        from onnxruntime.capi import _pybind_state as C
        from onnxruntime.capi.onnxruntime_pybind11_state import OrtValue as C_OrtValue

        numpy_arr_input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(numpy_arr_input)
        self.assertEqual(numpy_arr_input.shape, tuple(ortvalue.shape()))
        ptr = ortvalue._ortvalue.data_ptr()

        dlp = ortvalue._ortvalue.to_dlpack()
        self.assertFalse(C.is_dlpack_uint8_tensor(dlp))
        ortvalue2 = C_OrtValue.from_dlpack(dlp, False)
        self.assertEqual(ptr, ortvalue2.data_ptr())
        new_array = ortvalue2.numpy()
        self.assertEqualArray(numpy_arr_input, new_array)

        dlp = ortvalue._ortvalue.__dlpack__()
        self.assertFalse(C.is_dlpack_uint8_tensor(dlp))
        ortvalue2 = C_OrtValue.from_dlpack(dlp, False)
        self.assertEqual(ptr, ortvalue2.data_ptr())
        new_array = ortvalue2.numpy()
        self.assertEqualArray(numpy_arr_input, new_array)

        device = ortvalue._ortvalue.__dlpack_device__()
        self.assertEqual((1, 0), device)

    def test_numpy(self):
        model, feeds, expected = self._get_model()
        wrap = InferenceSessionForNumpy(model, providers="cpu")
        got = wrap.run(None, feeds)
        self.assertIsInstance(got[0], np.ndarray)
        self.assertEqualArray(expected[0], got[0])

    def test_numpy_no_optimization(self):
        model, feeds, expected = self._get_model()
        wrap = InferenceSessionForNumpy(model, providers="cpu", graph_optimization_level=False)
        got = wrap.run(None, feeds)
        self.assertIsInstance(got[0], np.ndarray)
        self.assertEqualArray(expected[0], got[0])

    @hide_stdout()
    def test_investigate_onnxruntime_issue_numpy(self):
        model, feeds, _expected = self._get_model()
        investigate_onnxruntime_issue(
            model,
            feeds=feeds,
            verbose=10,
            dump_filename="test_investigate_onnxruntime_issue_numpy.onnx",
        )

    @hide_stdout()
    def test_investigate_onnxruntime_issue_callable(self):
        import onnxruntime

        model, feeds, _expected = self._get_model()
        investigate_onnxruntime_issue(
            model,
            feeds=feeds,
            verbose=10,
            dump_filename="test_investigate_onnxruntime_issue_callable.onnx",
            onnx_to_session=lambda model: onnxruntime.InferenceSession(
                model.SerializeToString(), providers=["CPUExecutionProvider"]
            ),
        )

    @hide_stdout()
    def test_investigate_onnxruntime_issue_callable_str(self):
        model, feeds, _expected = self._get_model()
        investigate_onnxruntime_issue(
            model,
            feeds=feeds,
            verbose=10,
            dump_filename="test_investigate_onnxruntime_issue_callable.onnx",
            onnx_to_session="cpu_session",
        )

    @classmethod
    def _get_model_init(cls, itype) -> Tuple[onnx.ModelProto, Dict[str, Any], Tuple[Any, ...]]:
        dtype = tensor_dtype_to_np_dtype(itype)
        cst = np.arange(6).astype(dtype)
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("IsNaN", ["x"], ["xi"]),
                    oh.make_node("IsNaN", ["y"], ["yi"]),
                    oh.make_node("Cast", ["xi"], ["xii"], to=onnx.TensorProto.INT64),
                    oh.make_node("Cast", ["yi"], ["yii"], to=onnx.TensorProto.INT64),
                    oh.make_node("Add", ["xii", "yii"], ["gggg"]),
                    oh.make_node("Cast", ["gggg"], ["final"], to=itype),
                ],
                "dummy",
                [oh.make_tensor_value_info("x", itype, [None, None])],
                [oh.make_tensor_value_info("final", itype, [None, None])],
                [onh.from_array(cst, name="y")],
            ),
            opset_imports=[oh.make_opsetid("", 20)],
            ir_version=10,
        )
        onnx.checker.check_model(model)
        feeds = {"x": cls._range(5, 6).astype(dtype)}
        expected = np.isnan(feeds["x"]).astype(int) + np.isnan(cst.astype(float)).astype(int)
        return (model, feeds, (expected.astype(dtype),))

    def test_init_numpy_afloat32(self):
        model, feeds, expected = self._get_model_init(onnx.TensorProto.FLOAT)
        wrap = InferenceSessionForNumpy(model, providers="cpu", graph_optimization_level=False)
        got = wrap.run(None, feeds)
        self.assertIsInstance(got[0], np.ndarray)
        self.assertEqualArray(expected[0], got[0])

    def test_init_numpy_bfloat16(self):
        model, feeds, expected = self._get_model_init(onnx.TensorProto.BFLOAT16)
        wrap = InferenceSessionForNumpy(model, providers="cpu", graph_optimization_level=False)
        got = wrap.run(
            None, {k: v.astype(np.float32).astype(ml_dtypes.bfloat16) for k, v in feeds.items()}
        )
        self.assertIsInstance(got[0], np.ndarray)
        self.assertEqualArray(expected[0], got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)

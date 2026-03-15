from typing import Dict, List, Optional, Tuple, Union
import onnx
import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as ORTC
from ..helpers.helper import size_type
from ..helpers.onnx_helper import (
    np_dtype_to_tensor_dtype,
    onnx_dtype_name,
    tensor_dtype_to_np_dtype,
)
from ._inference_session import _InferenceSession, TensorLike


class InferenceSessionForNumpy(_InferenceSession):
    """
    Wraps an `onnxruntime.InferenceSession` to overload method `run`
    to support :class:`numpy.ndarray`.

    :param sess: model or inference session
    :param session_options: options
    :param providers: providers
    :param nvtx: enable nvidia events
    :param providers: `None`, `"CPU"`, `"CUDA"` or a list of providers
    :param graph_optimization_level: see :class:`onnxruntime.SessionOptions`
    :param log_severity_level: see :class:`onnxruntime.SessionOptions`
    :param log_verbosity_level: see :class:`onnxruntime.SessionOptions`
    :param optimized_model_filepath:  see :class:`onnxruntime.SessionOptions`
    :param disable_aot_function_inlining:  see :class:`onnxruntime.SessionOptions`
    :param use_training_api: use onnxruntime-training API
    """

    def __init__(
        self,
        sess: Union[onnx.ModelProto, str, onnxruntime.InferenceSession],
        session_options: Optional[onnxruntime.SessionOptions] = None,
        providers: Optional[Union[str, List[str]]] = None,
        nvtx: bool = False,
        enable_profiling: bool = False,
        graph_optimization_level: Union[onnxruntime.GraphOptimizationLevel, bool] = None,
        log_severity_level: Optional[int] = None,
        log_verbosity_level: Optional[int] = None,
        optimized_model_filepath: Optional[str] = None,
        disable_aot_function_inlining: Optional[bool] = None,
        use_training_api: Optional[bool] = None,
        cpu_outputs: bool = True,
    ):
        assert cpu_outputs, f"The execution can only happen on CPU but {cpu_outputs=}."
        super().__init__(
            sess,
            session_options=session_options,
            providers=providers,
            nvtx=nvtx,
            enable_profiling=enable_profiling,
            graph_optimization_level=graph_optimization_level,
            log_severity_level=log_severity_level,
            log_verbosity_level=log_verbosity_level,
            optimized_model_filepath=optimized_model_filepath,
            disable_aot_function_inlining=disable_aot_function_inlining,
            use_training_api=use_training_api,
        )
        try:
            import torch

            self._has_torch = hasattr(torch, "__version__")
            self.torch = torch
        except ImportError:
            self._has_torch = False

    def run(
        self,
        output_names: Optional[List[str]],
        feeds: Union[Dict[str, np.ndarray], Dict[str, ORTC.OrtValue]],  # type: ignore
    ) -> List[Optional[TensorLike]]:
        """Calls :meth:`onnxruntime.InferenceSession.run`."""
        # sess.run does not support bfloat16
        # res = self.sess.run(output_names, feeds)
        return self._post_process_inplace(list(self.run_dlpack(output_names, feeds)))  # type: ignore

    def run_dlpack(
        self, output_names: Optional[List[str]], feeds: Dict[str, TensorLike]
    ) -> Tuple[Optional[TensorLike], ...]:
        """
        Same as :meth:`onnxruntime.InferenceSession.run` except that
        feeds is a dictionary of :class:`np.ndarray`.
        The output device is CPU even if the outputs are on CUDA.
        """
        memory = []
        new_feeds = {}
        for k, v in feeds.items():
            if not k:
                continue
            if isinstance(v, np.ndarray):
                new_feeds[k] = ORTC.OrtValue.ortvalue_from_numpy_with_onnx_type(  # type: ignore
                    v, np_dtype_to_tensor_dtype(v.dtype)
                )
            elif v.dtype == np.bool_:
                memory.append(v)
                new_feeds[k] = ORTC.OrtValue.ortvalue_from_numpy_with_onnx_type(  # type: ignore
                    v, onnx.TensorProto.BOOL
                )
            else:
                new_feeds[k] = ORTC.OrtValue.from_dlpack(v.__dlpack__(), False)  # type: ignore

        if self.nvtx:
            self.torch.cuda.nvtx.range_push("run_with_ort_values")  # type: ignore
        ort_outputs = self.sess._sess.run_with_ort_values(  # type: ignore
            new_feeds, output_names or self.output_names, self.run_options
        )
        if self.nvtx:
            self.torch.cuda.nvtx.range_pop()  # type: ignore
        pth_outputs = self._ortvalues_to_numpy_tensor(ort_outputs)
        return pth_outputs

    def _ortvalues_to_numpy_tensor(
        self, ortvalues: Union[List[ORTC.OrtValue], ORTC.OrtValueVector]  # type: ignore
    ) -> Tuple[Optional[TensorLike], ...]:
        if len(ortvalues) == 0:
            return tuple()

        res: List[Optional[TensorLike]] = []  # noqa: F823
        for i in range(len(ortvalues)):
            if not ortvalues[i].has_value():
                res.append(None)
                continue

            el_type = ortvalues[i].element_type()
            if el_type < onnx.TensorProto.BFLOAT16:
                try:
                    a = np.from_dlpack(ortvalues[i])
                except RuntimeError as e:
                    assert "ORT only supports contiguous tensor for now." in str(e), (
                        f"As it says, non-contiguous OrtValue are not supported "
                        f"though DLPack, i={i}, the error is different {e}"
                    )
                    # We make a copy in that case.
                    a = ortvalues[i].numpy()
                res.append(a)
                continue

            assert self._has_torch, f"Type {el_type} is not handled without torch."
            tch = self.torch.from_dlpack(ortvalues[i].to_dlpack())
            size = size_type(el_type)
            assert size == 2, f"Not implemented for type {onnx_dtype_name(el_type)}"
            it = self.torch.uint16
            itch = tch.view(it)
            npt = itch.numpy()

            dtype = tensor_dtype_to_np_dtype(el_type)
            res.append(npt.view(dtype))
        return tuple(res)

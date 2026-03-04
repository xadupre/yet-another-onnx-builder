from typing import Dict, List, Optional, Tuple, Union
import onnx
import numpy as np
import torch
from torch._C import _from_dlpack
import onnxruntime
from onnxruntime.capi import _pybind_state as ORTC
from ..helpers.helper import string_type, size_type
from ..helpers.onnx_helper import tensor_dtype_to_np_dtype, onnx_dtype_name
from ..torch.torch_helper import torch_dtype_to_onnx_dtype
from ._inference_session import _InferenceSession, DEVICES, TensorLike


class InferenceSessionForTorch(_InferenceSession):
    """
    Wraps an `onnxruntime.InferenceSession` to overload method `run`
    to support :class:`torch.Tensor`.

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
    :param cpu_output: if True, force the outputs to be on CPU
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
        cpu_outputs: bool = False,
    ):
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
        self.cpu_outputs = cpu_outputs
        self._torch_from_dlpack = _from_dlpack  # type: ignore
        self.torch = torch
        if torch.cuda.device_count() > 0:  # type: ignore
            for i in range(torch.cuda.device_count()):  # type: ignore
                DEVICES[i] = ORTC.OrtDevice(  # type: ignore
                    ORTC.OrtDevice.cuda(), ORTC.OrtDevice.default_memory(), i  # type: ignore
                )

    def _get_ortvalues_from_torch_tensors(
        self, tensors: Tuple[torch.Tensor, ...], n_outputs: int
    ) -> Tuple[ORTC.OrtValueVector, List[onnxruntime.OrtDevice]]:  # type: ignore
        assert tensors is not None, "tensors cannot be None"
        ortvalues = ORTC.OrtValueVector()  # type: ignore
        ortvalues.reserve(len(tensors))
        dtypes = []
        shapes = []
        data_ptrs = []
        devices = []

        if self.nvtx:
            self.torch.cuda.nvtx.range_push("_get_ortvalues_from_torch_tensors.1")  # type: ignore
        max_device = -1
        new_tensors = []
        for tensor in tensors:
            assert isinstance(tensor, self.torch.Tensor), f"Unexpected type {type(tensor)}"
            dtypes.append(tensor_dtype_to_np_dtype(torch_dtype_to_onnx_dtype(tensor.dtype)))
            shapes.append(tensor.size())
            data_ptrs.append(tensor.data_ptr())
            d = tensor.get_device()
            devices.append(DEVICES[d])
            new_tensors.append(tensor)
            max_device = max(max_device, d)

        if self.nvtx:
            self.torch.cuda.nvtx.range_pop()  # type: ignore
            self.torch.cuda.nvtx.range_push("_get_ortvalues_from_torch_tensors.2")  # type: ignore

        assert isinstance(max_device, int), f"unexpected type for device={max_device!r}"
        ortvalues.push_back_batch(new_tensors, data_ptrs, dtypes, shapes, devices)
        output_devices = []
        for _ in range(n_outputs):
            dev = DEVICES[max_device]
            output_devices.append(dev)

        if self.nvtx:
            self.torch.cuda.nvtx.range_pop()  # type: ignore
        return ortvalues, output_devices

    def _ortvalues_to_torch_tensor(
        self,
        ortvalues: Union[List[ORTC.OrtValue], ORTC.OrtValueVector],  # type: ignore
    ) -> Tuple[torch.Tensor, ...]:
        if len(ortvalues) == 0:
            return tuple()

        if all(ortvalues[i].has_value() for i in range(len(ortvalues))):
            if self.nvtx:
                self.torch.cuda.nvtx.range_push("_ortvalues_to_torch_tensor.1")  # type: ignore
            res = ortvalues.to_dlpacks(_from_dlpack)  # type: ignore
            if self.nvtx:
                self.torch.cuda.nvtx.range_pop()  # type: ignore
        else:
            if self.nvtx:
                self.torch.cuda.nvtx.range_push("_ortvalues_to_torch_tensor.2")  # type: ignore
            res = []
            for i in range(len(ortvalues)):
                res.append(
                    self._torch_from_dlpack(ortvalues[i].to_dlpack())  # type: ignore
                    if ortvalues[i].has_value()
                    else None
                )
            if self.nvtx:
                self.torch.cuda.nvtx.range_pop()  # type: ignore
        return tuple(res)  # type: ignore

    def run(  # type: ignore
        self, output_names: Optional[List[str]], feeds: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Same as :meth:`onnxruntime.InferenceSession.run` except that
        feeds is a dictionary of :class:`torch.Tensor`.
        """
        if self.use_training_api:
            inputs = [feeds[i] for i in self.input_names]
            return self.run_training_api(*inputs, output_names=output_names)
        return self._post_process_inplace(list(self.run_dlpack(output_names, feeds)))

    def run_training_api(
        self, *inputs, output_names: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Calls the former training API now implemented in onnxruntime as well.

        :param inputs: list of :class:`torch.Tensor`
        :param output_names: requested outputs or None for all
        :return: tuple of :class:`torch.Tensor`
        """
        if output_names is None:
            output_names = self.output_names
        ortvalues, output_devices = self._get_ortvalues_from_torch_tensors(
            inputs, len(output_names)
        )

        if self.nvtx:
            self.torch.cuda.nvtx.range_push("run_with_ortvaluevector")  # type: ignore

        ort_outputs = ORTC.OrtValueVector()  # type: ignore
        self.sess.run_with_ortvaluevector(
            self.run_options,
            self.input_names,
            ortvalues,
            output_names,
            ort_outputs,
            output_devices,
        )

        if self.nvtx:
            self.torch.cuda.nvtx.range_pop()  # type: ignore

        pth_outputs = self._ortvalues_to_torch_tensor(ort_outputs)
        return pth_outputs

    def run_dlpack(
        self, output_names: Optional[List[str]], feeds: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Same as :meth:`onnxruntime.InferenceSession.run` except that
        feeds is a dictionary of :class:`torch.Tensor`.
        The output device is CPU even if the outputs are on CUDA.
        """
        input_names = []
        values = ORTC.OrtValueVector()  # type: ignore
        device = -1
        for k, v in feeds.items():
            assert k != "", f"Input cannot be empty but feeds names={list(feeds)}"
            assert hasattr(v, "device"), (
                f"Unexpected class {type(v)} for input {k!r}, "
                f"feeds={string_type(feeds, with_shape=True)}"
            )
            device = max(device, v.get_device())
            assert hasattr(v, "__dlpack__"), f"class {type(v)} should be serialized"
            if not v.is_contiguous():
                v = v.contiguous()
            if v.dtype == torch.bool:
                v = v.to(torch.uint8)
                v = ORTC.OrtValue.from_dlpack(v.__dlpack__(), True)  # type: ignore
            else:
                v = ORTC.OrtValue.from_dlpack(v.detach().__dlpack__(), False)  # type: ignore
            input_names.append(k)
            values.push_back(v)
        if self.nvtx:
            self.torch.cuda.nvtx.range_push("run_with_ortvaluevector")  # type: ignore

        # ort_outputs = self.sess._sess.run_with_ort_values(
        #    new_feeds, output_names or self.output_names, self.run_options
        # )
        ort_outputs = ORTC.OrtValueVector()  # type: ignore
        out_names = output_names or self.output_names
        self.sess._sess.run_with_ortvaluevector(  # type: ignore
            self.run_options,
            input_names,
            values,
            out_names,
            ort_outputs,
            [DEVICES[-1 if self.cpu_outputs else device] for o in out_names],
        )
        if self.nvtx:
            self.torch.cuda.nvtx.range_pop()  # type: ignore
        pth_outputs = self._ortvalues_to_torch_tensor(ort_outputs)
        return pth_outputs

    def _ortvalues_to_numpy_tensor(
        self,
        ortvalues: Union[List[ORTC.OrtValue], ORTC.OrtValueVector],  # type: ignore
    ) -> Tuple[Optional[TensorLike], ...]:
        if len(ortvalues) == 0:
            return tuple()

        if self.nvtx:
            self.torch.cuda.nvtx.range_push("_ortvalues_to_numpy_tensor")  # type: ignore
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

            tch = torch.from_dlpack(ortvalues[i].to_dlpack())
            size = size_type(el_type)
            assert size == 2, f"Not implemented for type {onnx_dtype_name(el_type)}"
            it = torch.uint16
            itch = tch.view(it)
            npt = itch.numpy()

            dtype = tensor_dtype_to_np_dtype(el_type)
            res.append(npt.view(dtype))

        if self.nvtx:
            self.torch.cuda.nvtx.range_pop()  # type: ignore
        return tuple(res)

    def _post_process_inplace(self, outputs):
        for i in range(len(outputs)):
            o = outputs[i]
            if self.sess_bool_outputs[i]:
                if isinstance(o, np.ndarray):
                    if o.dtype != np.bool_:
                        outputs[i] = o.astype(np.bool_)
                else:
                    if o.dtype != torch.bool:
                        outputs[i] = o.to(torch.bool)
        return outputs

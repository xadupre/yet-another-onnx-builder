from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import onnx
from .helper import flatten_object, string_type


def make_feeds(
    proto: Union[onnx.ModelProto, Sequence[str]],
    inputs: Any,
    use_numpy: bool = False,
    copy: bool = False,
    is_modelbuilder: bool = False,
) -> Dict[str, Union["torch.Tensor", np.ndarray]]:  # type: ignore[name-defined] # noqa: F821
    """
    Serializes the inputs to produce feeds expected
    by :class:`onnxruntime.InferenceSession`.

    :param proto: onnx model or list of names
    :param inputs: any kind of inputs
    :param use_numpy: if True, converts torch tensors into numpy arrays
    :param copy: a copy is made, this should be the case if the inputs is ingested
        by ``OrtValue``
    :param is_modelbuilder: if True, the exporter is ModelBuilder, and we need to reorder
        the past_key_values inputs to match the expected order, and get rid of position_ids.
    :return: feeds dictionary
    """
    # NOTE: position_ids is a special case because ModelBuilder does not usually use it,
    # because it's fued into rotary embedding in GQA.
    if is_modelbuilder and isinstance(inputs, dict) and "position_ids" in inputs:
        import torch

        position_ids = inputs["position_ids"]  # type: ignore[valid-type]
        # We just check position_ids are contiguous.
        assert isinstance(position_ids, torch.Tensor) and (
            (
                (position_ids - position_ids.min())
                == torch.tensor(list(range(position_ids.shape[-1]))).unsqueeze(0)
            )
            .max()
            .item()
        ), (
            f"ModelBuilder does not support position_ids={position_ids}, "
            f"inputs={string_type(inputs, with_shape=True)}"
        )
        inputs.pop("position_ids", None)  # Ensure 'position_ids' absent before removing.

    flat = flatten_object(inputs, drop_keys=True)
    if use_numpy:
        import torch
        from ..torch.torch_helper import to_numpy

        flat = [to_numpy(t) if isinstance(t, torch.Tensor) else t for t in flat]
    names = (
        [i.name for i in proto.graph.input]
        if isinstance(proto, onnx.ModelProto)
        else (
            [i.name for i in proto.get_inputs()]
            if hasattr(proto, "get_inputs")
            else (proto.input_names if hasattr(proto, "input_names") else proto)
        )
    )
    assert (
        isinstance(names, list)
        and len(names) <= len(flat)
        and (
            len(names) == len(flat)
            or isinstance(proto, onnx.ModelProto)
            or hasattr(proto, "get_inputs")
        )
    ), (
        f"Not the same number of given inputs {len(flat)} "
        f"and the number of model inputs {len(names)}, "
        f"type(names)={type(names)}, type(proto)={type(proto)}"
        f"\n-- inputs={string_type(inputs, with_shape=True)}"
        f"\n-- names={names}"
    )

    if copy:
        flat = [t.copy() if hasattr(t, "copy") else t.clone() for t in flat]
    # bool, int, float, onnxruntime does not support float, bool, int
    new_flat = []
    for i in flat:
        if isinstance(i, bool):
            i = np.array(i, dtype=np.bool_)
        elif isinstance(i, int):
            i = np.array(i, dtype=np.int64)
        elif isinstance(i, float):
            i = np.array(i, dtype=np.float32)
        new_flat.append(i)
    return dict(zip(names, new_flat))


# ---------------------------------------------------------------------------
# OnnxRuntime type-string → numpy dtype mapping
# ---------------------------------------------------------------------------

_ORT_TYPE_TO_NUMPY: Dict[str, type] = {
    "tensor(float)": np.float32,
    "tensor(float32)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(float64)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int16)": np.int16,
    "tensor(int8)": np.int8,
    "tensor(uint64)": np.uint64,
    "tensor(uint32)": np.uint32,
    "tensor(uint16)": np.uint16,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


def _ort_type_to_numpy_dtype(ort_type: str) -> type:
    """Converts an OnnxRuntime type string (e.g. ``"tensor(float)"``).

    :param ort_type: type string returned by ``NodeArg.type``
    :return: corresponding NumPy dtype
    :raises ValueError: when the type is unknown
    """
    try:
        return _ORT_TYPE_TO_NUMPY[ort_type]
    except KeyError:
        # bfloat16 requires ml_dtypes
        if "bfloat16" in ort_type:
            try:
                import ml_dtypes  # type: ignore[import]

                return ml_dtypes.bfloat16
            except ImportError:
                pass
        raise ValueError(
            f"Unknown OnnxRuntime type string {ort_type!r}. "
            f"Known types: {sorted(_ORT_TYPE_TO_NUMPY)}"
        ) from None


def _get_dim(i: int, s: Union[str, int], batch: int = 1) -> int:
    if isinstance(s, int):
        return s
    if s == "batch" or i == 0:
        return batch
    # Everything else is cache length or sequence length.
    return 0


def _make_empty_cache(
    batch: int,
    onnx_input_names: List[str],
    onnx_input_shapes: List[Tuple[Union[int, str], ...]],
    onnx_input_types: List[str],
    _ORT_TYPE_TO_TORCH,
) -> Dict[str, "torch.Tensor"]:  # noqa: F821
    """Creates an empty cache."""
    import torch

    assert batch > 0, f"batch size = {batch} must be positive"
    feeds = {}
    for name, shape, dtype in zip(onnx_input_names, onnx_input_shapes, onnx_input_types):
        new_shape = tuple(_get_dim(i, s, batch=batch) for i, s in enumerate(shape))
        assert (
            new_shape and new_shape[0] > 0
        ), f"new_shape={new_shape} cannot have a null batch size, name={name!r}, shape={shape}"
        feeds[name] = torch.empty(new_shape, dtype=_ORT_TYPE_TO_TORCH[dtype])
    return feeds


def onnx_generate(
    model_or_path: Union[str, onnx.ModelProto],
    input_ids: Union[np.ndarray, "torch.Tensor"],  # type: ignore[name-defined] # noqa: F821
    attention_mask: Optional[
        Union[np.ndarray, "torch.Tensor"]  # type: ignore[name-defined] # noqa: F821
    ] = None,
    eos_token_id: Optional[int] = None,
    max_new_tokens: int = 20,
    do_sample: bool = False,
    return_session: bool = False,
    verbose: int = 0,
) -> np.ndarray:
    """
    Performs auto-regressive token generation using an exported ONNX model.

    The function mimics the ``generate`` method of HuggingFace *transformers*
    models.  It calls the ONNX forward pass in a loop, appending the most
    likely next token at each step (greedy decoding by default), and feeds the
    updated *past key/value* tensors back into the model on each subsequent
    call.

    Models that do **not** expose past-key-value inputs/outputs are also
    supported: in that case the full ``input_ids`` sequence is fed on every
    step (simpler but less efficient).

    :param model_or_path: path to an ``.onnx`` file **or** a
        :class:`onnx.ModelProto` loaded into memory.
    :param input_ids: initial prompt token IDs, integer array/tensor of
        shape ``[batch, seq_len]``.
    :param attention_mask: optional attention mask of shape
        ``[batch, seq_len]``.  When *None*, an all-ones mask matching
        ``input_ids`` is created automatically.
    :param eos_token_id: when set, generation stops as soon as *all* batch
        items have produced this token.
    :param max_new_tokens: upper bound on the number of tokens to generate
        (not counting the original ``input_ids``).
    :param do_sample: when *True* sample the next token from the softmax
        distribution; when *False* (default) use greedy argmax.
    :param verbose: verbosity level (0 = silent).
    :param return_session: return the session and the feeds
    :return: integer array of shape ``[batch, seq_len + generated_tokens]``
        containing the original prompt followed by the generated tokens.

    Example with a tiny synthetic ONNX decoder (no KV cache)::

        import numpy as np
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh
        from yobx.helpers.rt_helper import onnx_generate

        TINT64 = onnx.TensorProto.INT64
        TFLOAT = onnx.TensorProto.FLOAT
        VOCAB  = 8

        # A minimal "LM head": returns a fixed logits matrix so that the
        # argmax always picks token 3.
        fixed_logits = np.zeros((1, 1, VOCAB), dtype=np.float32)
        fixed_logits[0, 0, 3] = 10.0   # token 3 always wins

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["input_ids"], ["ids_shape"]),
                    oh.make_node(
                        "ConstantOfShape",
                        ["ids_shape"],
                        ["logits"],
                        value=onh.from_array(fixed_logits.reshape(VOCAB)),
                    ),
                ],
                "tiny_lm",
                [oh.make_tensor_value_info("input_ids", TINT64, [1, None])],
                [oh.make_tensor_value_info("logits", TFLOAT, [1, None, VOCAB])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        prompt = np.array([[1, 2]], dtype=np.int64)
        tokens = onnx_generate(model, prompt, max_new_tokens=3, eos_token_id=3)
        # tokens == [[1, 2, 3]]  (stops after the first EOS token)

    .. note::
        When the ONNX model exposes *past key/value* inputs, the function
        automatically creates zero-filled tensors for the initial call and
        feeds back the corresponding outputs on every subsequent step.  The
        expected naming convention is that every model input that is **not**
        one of ``input_ids``, ``attention_mask``, ``position_ids``, or
        ``token_type_ids`` is treated as a KV-cache tensor.  Model outputs
        are mapped back to KV-cache inputs by position (first
        ``len(kv_inputs)`` non-logits outputs → KV inputs).
    """
    import torch
    from ..reference._inference_session_torch import InferenceSessionForTorch

    _ORT_TYPE_TO_TORCH: Dict[str, type] = {
        "tensor(float)": torch.float32,
        "tensor(float32)": torch.float32,
        "tensor(float16)": torch.float16,
        "tensor(double)": torch.float64,
        "tensor(float64)": torch.float64,
        "tensor(int64)": torch.int64,
        "tensor(int32)": torch.int32,
        "tensor(int16)": torch.int16,
        "tensor(int8)": torch.int8,
        "tensor(uint64)": torch.uint64,
        "tensor(uint32)": torch.uint32,
        "tensor(uint16)": torch.uint16,
        "tensor(uint8)": torch.uint8,
        "tensor(bool)": torch.bool,
    }

    return_np = False
    if isinstance(input_ids, np.ndarray):
        return_np = True
        input_ids = torch.from_numpy(input_ids)
        attention_mask = torch.from_numpy(attention_mask) if attention_mask is not None else None

    if not isinstance(model_or_path, InferenceSessionForTorch):
        providers = ["CUDAExecutionProvider"] if input_ids.is_cuda else []
        providers.append("CPUExecutionProvider")
        session = InferenceSessionForTorch(model_or_path, providers=providers)
    else:
        session = model_or_path

    input_shapes = session.input_shapes
    input_names = session.input_names
    input_types = session.input_types
    has_position_ids = "position_ids" in session.input_names
    has_cache_position = "cache_position" in session.input_names

    assert input_names == ["input_ids"] or (
        len(input_names) > 2
        and input_names[:2] == ["input_ids", "attention_mask"]
        and input_names[3 if has_position_ids else 2].startswith("past_key_values")
    ), (
        f"Only text generation is supported but input_names == {input_names}, "
        f"has_position_ids={has_position_ids}"
    )
    assert (
        not has_position_ids or input_names[2] == "position_ids"
    ), f"position_ids must the third input but input_names={input_names}"

    cache_names, cache_shapes, cache_types = [], [], []
    for name, shape, dt in zip(input_names, input_shapes, input_types):
        if name.startswith("past_key_values"):
            cache_names.append(name)
            cache_shapes.append(shape)
            cache_types.append(dt)

    # First call: prefill
    empty_cache = _make_empty_cache(
        input_ids.shape[0], cache_names, cache_shapes, cache_types, _ORT_TYPE_TO_TORCH
    )
    feeds = (
        dict(input_ids=input_ids)
        if len(input_names) == 1
        else dict(
            input_ids=input_ids,
            attention_mask=(
                attention_mask
                if attention_mask is not None
                else torch.ones(input_ids.shape, dtype=input_ids.dtype, device=input_ids.device)
            ),
            **empty_cache,
        )
    )

    if has_position_ids:
        assert input_ids.shape[1] > 0, f"unexpected value for input_ids shape={input_ids.shape}"
        position_ids = torch.unsqueeze(
            torch.arange(input_ids.shape[1], dtype=torch.int64, device=input_ids.device), 0
        )
        feeds["position_ids"] = position_ids

    if has_cache_position:
        assert empty_cache, "no cache means no cache_position"
        first_tensor = next(iter(empty_cache.values()))
        cache_position = torch.arange(
            first_tensor.shape[2],
            input_ids.shape[1] + first_tensor.shape[2],
            dtype=torch.int64,
            device=input_ids.device,
        )
        feeds["cache_position"] = cache_position

    # prefill step
    outputs = session.run(None, feeds)
    batch_size = input_ids.shape[0]

    # Next calls: decode
    for _ in range(max_new_tokens):
        next_token_logits = outputs[0][:, -1, :]

        # The most probable next token is chosen.
        if do_sample:
            shifted = next_token_logits - next_token_logits.max(axis=-1, keepdims=True)
            probs = torch.exp(shifted)
            probs /= probs.sum(axis=-1, keepdims=True)
            next_token_id = np.array(
                [[np.random.choice(probs.shape[-1], p=probs[b])] for b in range(batch_size)],
                dtype=np.int64,
            )  # [batch, 1]
        else:
            next_token_id = torch.argmax(next_token_logits, axis=-1, keepdims=True)

        if next_token_id.item() == eos_token_id:
            break

        input_ids = torch.cat([input_ids, next_token_id.to(input_ids.device)], dim=-1)

        feeds = (
            dict(input_ids=input_ids)
            if len(input_names) == 1
            else dict(
                input_ids=next_token_id,
                attention_mask=torch.ones(
                    input_ids.shape, dtype=input_ids.dtype, device=input_ids.device
                ),
            )
        )
        if has_position_ids:
            feeds["position_ids"] = torch.unsqueeze(
                torch.arange(
                    input_ids.shape[1],
                    input_ids.shape[1] + 1,
                    dtype=torch.int64,
                    device=input_ids.device,
                ),
                0,
            )
        if has_cache_position:
            feeds["cache_position"] = torch.arange(
                input_ids.shape[1],
                input_ids.shape[1] + 1,
                dtype=torch.int64,
                device=input_ids.device,
            )

        feeds.update(
            dict(zip([n for n in input_names if n.startswith("past_key_values")], outputs[1:]))
        )
        outputs = session.run(None, feeds)

    if return_np:
        input_ids = input_ids.detach().numpy()
    if return_session:
        return input_ids, session, feeds
    return input_ids

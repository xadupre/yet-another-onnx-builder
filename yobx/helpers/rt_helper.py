from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union
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


# Inputs that are never treated as KV-cache slots.
_KNOWN_NON_CACHE: FrozenSet[str] = frozenset(
    {"input_ids", "attention_mask", "position_ids", "token_type_ids", "cache_position"}
)


def _make_empty_cache(
    batch: int,
    onnx_input_names: List[str],
    onnx_input_shapes: List[Tuple[Union[int, str], ...]],
    onnx_input_types: List[str],
    ort_type_to_torch: Dict[str, Any],
    device: "torch.device",  # type: ignore # noqa: F821
) -> Dict[str, "torch.Tensor"]:  # type: ignore # noqa: F821
    """Creates zero-filled KV-cache tensors for the first generation step.

    :param batch: batch size
    :param onnx_input_names: names of the KV-cache inputs
    :param onnx_input_shapes: ORT input shapes for those inputs
    :param onnx_input_types: ORT type strings for those inputs
    :param ort_type_to_torch: mapping from ORT type strings to ``torch.dtype``
    :param device: device on which to allocate the tensors (must match the
        device of ``input_ids`` to avoid mixed-device errors)
    :return: dict ``{name: zero tensor}``
    """
    import torch

    assert batch > 0, f"batch size = {batch} must be positive"
    feeds = {}
    for name, shape, dtype in zip(onnx_input_names, onnx_input_shapes, onnx_input_types):
        new_shape = tuple(_get_dim(i, s, batch=batch) for i, s in enumerate(shape))
        assert (
            new_shape and new_shape[0] > 0
        ), f"new_shape={new_shape} cannot have a null batch size, name={name!r}, shape={shape}"
        feeds[name] = torch.zeros(new_shape, dtype=ort_type_to_torch[dtype], device=device)
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
) -> Union[np.ndarray, "torch.Tensor"]:  # type: ignore[name-defined] # noqa: F821
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
    :param return_session: when *True* return a 3-tuple
        ``(tokens, session, last_feeds)`` instead of just the tokens.
    :return: integer array/tensor of shape
        ``[batch, seq_len + generated_tokens]`` containing the original
        prompt followed by the generated tokens.  The type matches
        ``input_ids``: :class:`numpy.ndarray` when the caller passed NumPy
        arrays, :class:`torch.Tensor` otherwise.

    Example with a tiny synthetic ONNX decoder (no KV cache)::

        import numpy as np
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh
        from yobx.helpers.rt_helper import onnx_generate

        TINT64 = onnx.TensorProto.INT64
        TFLOAT = onnx.TensorProto.FLOAT
        VOCAB  = 8

        # A minimal "LM head": always returns the same logits so that the
        # argmax always picks token 3.
        fixed_logits = np.zeros((1, 1, VOCAB), dtype=np.float32)
        fixed_logits[0, 0, 3] = 10.0   # token 3 always wins

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "Constant",
                        [],
                        ["logits"],
                        value=onh.from_array(fixed_logits),
                    ),
                ],
                "tiny_lm",
                [oh.make_tensor_value_info("input_ids", TINT64, [1, None])],
                [oh.make_tensor_value_info("logits", TFLOAT, [1, 1, VOCAB])],
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
        KV-cache heuristic treats any input whose name is **not** in
        ``{input_ids, attention_mask, position_ids, token_type_ids,
        cache_position}`` as a KV-cache slot.  Present-key/value outputs are
        mapped back to past-key/value inputs by position (i.e. ``outputs[1]``
        → ``cache_inputs[0]``, etc.).
    """
    import torch
    from ..reference._inference_session_torch import InferenceSessionForTorch

    _ORT_TYPE_TO_TORCH: Dict[str, Any] = {
        "tensor(float)": torch.float32,
        "tensor(float32)": torch.float32,
        "tensor(float16)": torch.float16,
        "tensor(bfloat16)": torch.bfloat16,
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

    return_np = isinstance(input_ids, np.ndarray)
    if isinstance(input_ids, np.ndarray):
        input_ids = torch.from_numpy(input_ids)
    if isinstance(attention_mask, np.ndarray):
        attention_mask = torch.from_numpy(attention_mask)

    if not isinstance(model_or_path, InferenceSessionForTorch):
        providers = ["CUDAExecutionProvider"] if input_ids.is_cuda else []
        providers.append("CPUExecutionProvider")
        session = InferenceSessionForTorch(model_or_path, providers=providers)
    else:
        session = model_or_path

    input_shapes = session.input_shapes
    input_names = session.input_names
    input_types = session.input_types
    has_position_ids = "position_ids" in input_names
    has_cache_position = "cache_position" in input_names

    device = input_ids.device
    batch_size = input_ids.shape[0]

    # Heuristic KV-cache detection: any input not in the set of known
    # "meta" inputs is treated as a past-key/value slot.
    cache_names = [n for n in input_names if n not in _KNOWN_NON_CACHE]
    cache_shapes = [input_shapes[input_names.index(n)] for n in cache_names]
    cache_types = [input_types[input_names.index(n)] for n in cache_names]

    # Build the initial attention mask (all ones if not provided).
    if attention_mask is None:
        attention_mask = torch.ones(input_ids.shape, dtype=torch.int64, device=device)
    else:
        attention_mask = attention_mask.to(device=device)

    # Bootstrap zero-filled KV-cache tensors on the same device as input_ids.
    empty_cache = _make_empty_cache(
        batch_size, cache_names, cache_shapes, cache_types, _ORT_TYPE_TO_TORCH, device
    )

    # ------------------------------------------------------------------ #
    # Prefill step                                                         #
    # ------------------------------------------------------------------ #
    feeds: Dict[str, Any] = {"input_ids": input_ids}
    if "attention_mask" in input_names:
        feeds["attention_mask"] = attention_mask
    feeds.update(empty_cache)

    if has_position_ids:
        assert input_ids.shape[1] > 0, f"unexpected value for input_ids shape={input_ids.shape}"
        feeds["position_ids"] = (
            torch.arange(input_ids.shape[1], dtype=torch.int64, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

    if has_cache_position:
        past_len = (
            next(iter(empty_cache.values())).shape[2]
            if empty_cache and next(iter(empty_cache.values())).ndim > 2
            else 0
        )
        feeds["cache_position"] = torch.arange(
            past_len,
            input_ids.shape[1] + past_len,
            dtype=torch.int64,
            device=device,
        )

    outputs = session.run(None, feeds)

    # ------------------------------------------------------------------ #
    # Decode loop                                                          #
    # ------------------------------------------------------------------ #
    # Per-batch EOS tracking so that the loop terminates only when *all*
    # sequences have finished (correct for batch_size > 1).
    last_position = 0
    eos_found = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        next_token_logits = outputs[0][:, -1, :]  # [batch, vocab]

        if do_sample:
            # Sample from the probability distribution over the vocabulary.
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)  # [batch, 1]
        else:
            # Greedy decoding: take the argmax token.
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [batch, 1]

        # Update per-batch EOS flags and append the new token.
        if eos_token_id is not None:
            eos_found |= next_token_id.squeeze(-1) == eos_token_id

        input_ids = torch.cat([input_ids, next_token_id.to(device)], dim=-1)

        # Stop once every sequence in the batch has produced EOS.
        if eos_token_id is not None and eos_found.all():
            break

        # Extend the attention mask by one column of ones for the new token.
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device),
            ],
            dim=-1,
        )

        # Build feeds for the next decode step.
        if not cache_names:
            # No KV cache: feed the full growing sequence.
            feeds = {"input_ids": input_ids}
            if "attention_mask" in input_names:
                feeds["attention_mask"] = attention_mask
        else:
            # KV cache: feed only the single new token; map present outputs
            # back to past inputs by position.
            feeds = {"input_ids": next_token_id, "attention_mask": attention_mask}
            for j, name in enumerate(cache_names):
                if 1 + j < len(outputs):
                    feeds[name] = outputs[1 + j]

        if has_position_ids or has_cache_position:
            last_position = input_ids.shape[1] - 1

        if has_position_ids:
            feeds["position_ids"] = torch.full(
                (batch_size, 1),
                last_position,
                dtype=torch.int64,
                device=device,
            )

        if has_cache_position:
            feeds["cache_position"] = torch.arange(
                last_position,
                last_position + 1,
                dtype=torch.int64,
                device=device,
            )

        outputs = session.run(None, feeds)

    result: Union[np.ndarray, torch.Tensor]  # type: ignore[name-defined] # noqa: F821
    if return_np:
        result = input_ids.detach().cpu().numpy()
    else:
        result = input_ids

    if return_session:
        return result, session, feeds  # type: ignore[return-value]
    return result

from typing import Any, Dict, List, Optional, Sequence, Union
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
        )


def onnx_generate(
    proto: Union[str, onnx.ModelProto],
    input_ids: Union[np.ndarray, "torch.Tensor"],  # type: ignore[name-defined] # noqa: F821
    attention_mask: Optional[
        Union[np.ndarray, "torch.Tensor"]  # type: ignore[name-defined] # noqa: F821
    ] = None,
    eos_token_id: Optional[int] = None,
    max_new_tokens: int = 20,
    do_sample: bool = False,
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

    :param proto: path to an ``.onnx`` file **or** a
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
    import onnxruntime

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_numpy(t: Any) -> Optional[np.ndarray]:
        if t is None:
            return None
        if hasattr(t, "detach"):
            return t.detach().cpu().numpy()
        if hasattr(t, "numpy"):
            return t.numpy()
        return np.asarray(t)

    # ------------------------------------------------------------------
    # Prepare initial inputs
    # ------------------------------------------------------------------
    input_ids_np: np.ndarray = _to_numpy(input_ids)  # type: ignore[assignment]
    attn_mask_np: np.ndarray = (
        _to_numpy(attention_mask)
        if attention_mask is not None
        else np.ones_like(input_ids_np)
    )
    batch_size: int = input_ids_np.shape[0]

    # ------------------------------------------------------------------
    # Create ORT session
    # ------------------------------------------------------------------
    if isinstance(proto, str):
        sess = onnxruntime.InferenceSession(proto, providers=["CPUExecutionProvider"])
    else:
        sess = onnxruntime.InferenceSession(
            proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )

    in_infos = sess.get_inputs()
    out_infos = sess.get_outputs()
    in_names: List[str] = [i.name for i in in_infos]
    out_names: List[str] = [i.name for i in out_infos]

    # ------------------------------------------------------------------
    # Identify standard vs KV-cache inputs/outputs
    # ------------------------------------------------------------------
    _STANDARD = {"input_ids", "attention_mask", "position_ids", "token_type_ids"}
    kv_in_names: List[str] = [n for n in in_names if n not in _STANDARD]

    # The logits output is the first output whose name contains "logit",
    # falling back to the very first output.
    logits_out_idx: int = 0
    for _i, _n in enumerate(out_names):
        if "logit" in _n.lower():
            logits_out_idx = _i
            break

    # KV-cache outputs are everything except logits (positional mapping to
    # kv_in_names: output[k] → kv_in_names[k]).
    kv_out_names: List[str] = [n for i, n in enumerate(out_names) if i != logits_out_idx]

    if verbose:
        print(f"[onnx_generate] in_names     = {in_names}")
        print(f"[onnx_generate] out_names    = {out_names}")
        print(f"[onnx_generate] kv_in_names  = {kv_in_names}")
        print(f"[onnx_generate] kv_out_names = {kv_out_names}")

    # ------------------------------------------------------------------
    # Initialise KV cache (empty tensors whose seq dimension = 0)
    # ------------------------------------------------------------------
    # A model is considered to have a KV cache when:
    #   1. there are non-standard inputs (kv_in_names is non-empty),
    #   2. the number of such inputs equals the number of non-logits outputs,
    #   3. at least one of those input names contains a KV-cache keyword.
    _KV_KEYWORDS = {"past", "present", "key", "value", "cache", "kv"}
    _has_kv_keyword = any(
        any(kw in n.lower() for kw in _KV_KEYWORDS) for n in kv_in_names
    )
    has_kv_cache = (
        bool(kv_in_names)
        and len(kv_in_names) == len(kv_out_names)
        and _has_kv_keyword
    )

    kv_cache: Dict[str, np.ndarray] = {}
    if has_kv_cache:
        for inp_info in in_infos:
            if inp_info.name not in _STANDARD:
                dtype = _ort_type_to_numpy_dtype(inp_info.type)
                shape: List[int] = []
                for dim in inp_info.shape:
                    if dim is None or isinstance(dim, str):
                        shape.append(0)
                    else:
                        shape.append(int(dim))
                # Overwrite batch dimension with actual batch size.
                if shape:
                    shape[0] = batch_size
                kv_cache[inp_info.name] = np.zeros(shape, dtype=dtype)

    # ------------------------------------------------------------------
    # Generation loop
    # ------------------------------------------------------------------
    generated: np.ndarray = input_ids_np.copy()

    for step in range(max_new_tokens):
        feeds: Dict[str, np.ndarray] = {}

        # ---- input_ids -----------------------------------------------
        if "input_ids" in in_names:
            if step == 0 or not has_kv_cache:
                feeds["input_ids"] = generated
            else:
                # Only the last token; past context is in the KV cache.
                feeds["input_ids"] = generated[:, -1:]

        # ---- attention_mask ------------------------------------------
        if "attention_mask" in in_names:
            if step == 0 or not has_kv_cache:
                feeds["attention_mask"] = attn_mask_np
            else:
                # Past length is encoded in the KV cache's seq dimension.
                past_len = kv_cache[kv_in_names[0]].shape[2]
                # The model sees `past_len` cached positions + 1 new token.
                feeds["attention_mask"] = np.ones(
                    (batch_size, past_len + 1), dtype=attn_mask_np.dtype
                )

        # ---- position_ids (optional) ---------------------------------
        if "position_ids" in in_names:
            if step == 0 or not has_kv_cache:
                seq_len = generated.shape[1]
                feeds["position_ids"] = np.arange(seq_len, dtype=np.int64)[
                    np.newaxis, :
                ].repeat(batch_size, axis=0)
            else:
                past_len = kv_cache[kv_in_names[0]].shape[2]
                feeds["position_ids"] = np.full(
                    (batch_size, 1), past_len, dtype=np.int64
                )

        # ---- KV cache ------------------------------------------------
        feeds.update(kv_cache)

        if verbose:
            print(
                f"[onnx_generate] step={step} feeds={string_type(feeds, with_shape=True)}"
            )

        # ---- Run model -----------------------------------------------
        results = sess.run(out_names, feeds)
        logits: np.ndarray = results[logits_out_idx]  # [batch, seq, vocab]

        # ---- Select next token ---------------------------------------
        last_logits = logits[:, -1, :]  # [batch, vocab]
        if do_sample:
            last_logits = last_logits.astype(np.float64)
            shifted = last_logits - last_logits.max(axis=-1, keepdims=True)
            probs = np.exp(shifted)
            probs /= probs.sum(axis=-1, keepdims=True)
            next_token = np.array(
                [
                    [np.random.choice(probs.shape[-1], p=probs[b])]
                    for b in range(batch_size)
                ],
                dtype=np.int64,
            )  # [batch, 1]
        else:
            next_token = np.argmax(last_logits, axis=-1, keepdims=True).astype(
                np.int64
            )  # [batch, 1]

        generated = np.concatenate([generated, next_token], axis=1)

        # ---- Update KV cache from model outputs ----------------------
        if has_kv_cache:
            for kv_in, kv_out in zip(kv_in_names, kv_out_names):
                result_dict = dict(zip(out_names, results))
                kv_cache[kv_in] = result_dict[kv_out]

        # ---- Early stopping ------------------------------------------
        if eos_token_id is not None and (next_token == eos_token_id).all():
            if verbose:
                print(f"[onnx_generate] EOS reached at step {step}")
            break

    return generated

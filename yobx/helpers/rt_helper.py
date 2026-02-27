from typing import Any, Dict, Sequence, Union
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
        from .torch.torch_helper import to_numpy

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

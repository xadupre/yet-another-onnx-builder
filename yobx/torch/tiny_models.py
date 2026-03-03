from dataclasses import dataclass
import copy
from typing import Any, Dict, Optional
import torch


class _AddModel(torch.nn.Module):
    """A minimal two-input add model used for testing torch patches."""

    def forward(self, x, y):
        return x + y


class _ScaledBroadcastModel(torch.nn.Module):
    """A model that multiplies a ``(batch, seq, hidden)`` tensor by a
    ``(1, 1, hidden)`` scale vector, exercising shape-broadcasting paths
    through the torch fake-tensor machinery."""

    def forward(self, x, w):
        return x * w


@dataclass
class ModelData:
    """Contains all the necessary information to export a model."""

    model_id: str
    model: torch.nn.Module
    export_inputs: Dict[str, Any]
    dynamic_shapes: Dict[str, Any]


def _update_config(config: Any, mkwargs: Dict[str, Any]):
    """Updates a configuration with different values."""
    for k, v in mkwargs.items():
        if k == "attn_implementation":
            config._attn_implementation = v
            if getattr(config, "_attn_implementation_autoset", False):
                config._attn_implementation_autoset = False
            continue
        if isinstance(v, dict):
            if not hasattr(config, k) or getattr(config, k) is None:
                setattr(config, k, v)
                continue
            existing = getattr(config, k)
            if type(existing) is dict:
                existing.update(v)
            else:
                _update_config(getattr(config, k), v)
            continue
        if type(config) is dict:
            config[k] = v
        else:
            setattr(config, k, v)


def get_tiny_model(model_id, config_updates: Optional[Dict[str, Any]] = None) -> ModelData:
    """
    Creates a tiny models, usually untrained to write tests.
    The `model_id` refers to what exists on HuggingFace.
    This functions is not expected to have fully coverage of architectures.

    Supported model IDs:

    * ``"add"`` — a minimal two-input element-wise add model.
      Exercises :func:`torch.export._trace._get_range_constraints` (kwargs ordering
      patch) and :class:`torch.fx.experimental.symbolic_shapes.DynamicDimConstraintPrinter`
      (constraint-printing patch).  Requires only :epkg:`torch`, no *transformers*.

    * ``"broadcast_multiply"`` — multiplies a ``(batch, seq, hidden)`` tensor by a
      ``(1, 1, hidden)`` scale vector.  Exercises
      :func:`torch._subclasses.fake_impls.infer_size` and
      :func:`torch._refs._broadcast_shapes` (broadcasting-shape patches).
      Requires only :epkg:`torch`, no *transformers*.

    * ``"arnir0/Tiny-LLM"`` — a tiny LLaMA-based causal language model with a
      :class:`transformers.cache_utils.DynamicCache` past-key-value cache.

    :param model_id: model id, see the list of supported values above
    :param config_updates: modification to add to the configuration before creating the model
    :return: the necessary information

    Example::

        import torch
        from yobx.torch import get_tiny_model, apply_patches_for_model

        model_data = get_tiny_model("add")
        result = model_data.model(**model_data.export_inputs)
        with apply_patches_for_model(patch_torch=True):
            ep = torch.export.export(
                model_data.model,
                (),
                kwargs=model_data.export_inputs,
                dynamic_shapes=model_data.dynamic_shapes,
            )

    Example::

        import torch
        from yobx.torch import get_tiny_model
        from yobx.torch.torch_helper import torch_deepcopy

        model_data = get_tiny_model("arnir0/Tiny-LLM")
        # torch_deepcopy is needed because the cache (past_key_values) is modified in-place
        # during the forward pass; reusing the same object would corrupt subsequent calls.
        result = model_data.model(**torch_deepcopy(model_data.export_inputs))
        ep = torch.export.export(
            model_data.model,
            (),
            kwargs=torch_deepcopy(model_data.export_inputs),
            dynamic_shapes=model_data.dynamic_shapes,
        )
    """
    if model_id == "add":
        batch = torch.export.Dim("batch", min=1, max=1024)
        seq = torch.export.Dim("seq", min=1, max=4096)
        return ModelData(
            model_id=model_id,
            model=_AddModel(),
            export_inputs=dict(
                x=torch.randn(2, 3, dtype=torch.float32),
                y=torch.randn(2, 3, dtype=torch.float32),
            ),
            dynamic_shapes=dict(
                x={0: batch, 1: seq},
                y={0: batch, 1: seq},
            ),
        )

    if model_id == "broadcast_multiply":
        batch = torch.export.Dim("batch", min=1, max=1024)
        seq = torch.export.Dim("seq", min=1, max=4096)
        return ModelData(
            model_id=model_id,
            model=_ScaledBroadcastModel(),
            export_inputs=dict(
                x=torch.randn(2, 5, 8, dtype=torch.float32),
                w=torch.randn(1, 1, 8, dtype=torch.float32),
            ),
            dynamic_shapes=dict(
                x={0: batch, 1: seq},
                w={},
            ),
        )

    if model_id == "arnir0/Tiny-LLM":
        import transformers
        from .in_transformers.models import get_cached_configuration
        from .in_transformers.cache_helper import make_dynamic_cache

        config = get_cached_configuration(model_id)
        if config_updates:
            config = copy.deepcopy(config)
            _update_config(config, config_updates)

        bsize, nheads, slen, dim = 2, 1, 30, 96
        return ModelData(
            model_id=model_id,
            model=transformers.AutoModelForCausalLM.from_config(config),
            export_inputs=dict(
                input_ids=torch.randint(15, size=(2, 3), dtype=torch.int64),
                attention_mask=torch.randint(1, size=(2, 33), dtype=torch.int64),
                position_ids=torch.arange(3, dtype=torch.int64).unsqueeze(0).expand((2, -1)),
                past_key_values=make_dynamic_cache(
                    [
                        (
                            torch.randn(bsize, nheads, slen, dim),
                            torch.randn(bsize, nheads, slen, dim),
                        )
                    ]
                ),
            ),
            dynamic_shapes=dict(
                input_ids={0: "batch", 1: "seq_length"},
                attention_mask={
                    0: "batch",
                    1: "past_length+seq_length",
                },
                position_ids={0: "batch", 1: "seq_length"},
                past_key_values=[{0: "batch", 2: "past_length"} for _ in range(2)],
            ),
        )

    raise ValueError(f"Model {model_id} is not supported yet.")

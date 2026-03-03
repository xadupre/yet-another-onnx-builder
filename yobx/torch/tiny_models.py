from dataclasses import dataclass
import copy
from typing import Any, Dict, Optional
import torch


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

    :param model_id: model_id
    :param config_updates: modification to add to the configuration before creating the model
    :return: the necessary information
    """
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

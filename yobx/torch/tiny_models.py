from dataclasses import dataclass
from typing import Any, Dict
import torch


@dataclass
class ModelData:
    """
    Contains all the necessary informations to export a model.
    """

    model_id: str
    model: torch.nn.Module
    export_inputs: Dict[str, Any]
    dynamic_shapes: Dict[str, Any]


def get_tiny_model(model_id) -> ModelData:
    """
    Creates a tiny models, usually untrained to write tests.
    The `model_id` refers to what exists on HuggingFace.
    This functions is not expected to have fully coverage of architectures.

    :param model_id: model_id
    :return: the necessary information
    """
    if model_id == "arnir0/Tiny-LLM":
        import transformers
        from .transformers.models import get_cached_configuration
        from .transformers.cache_helper import make_dynamic_cache

        config = get_cached_configuration(model_id)
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

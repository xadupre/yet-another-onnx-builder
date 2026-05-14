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
    inputs_batch1: Optional[Dict[str, Any]] = None

    @property
    def dynamic_shapes_for_torch_export_export(self) -> Dict[str, Any]:
        """
        The dynamic shapes contains strings.
        :func:`torch.export.export` needs them to be replaced
        by ``torch.export.Dim.DYNAMIC``. This is what
        this property is doing.
        """
        from . import use_dyn_not_str

        return use_dyn_not_str(self.dynamic_shapes)


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


class TinyBroadcastAddModel(torch.nn.Module):
    """
    A model where one output dynamic dimension becomes ``max(d1, d2)`` after a broadcast.

    Inputs ``x`` (shape ``(batch, d1)``) and ``y`` (shape ``(batch, d2)``) are added
    element-wise. Broadcasting rules require that ``d1 == d2`` or one of them equals
    ``1`` at runtime; the symbolic output shape is ``(batch, max(d1, d2))``.

    This model triggers the following when exported.

    .. runpython::
        :showcode:
        :process:

        import torch
        from yobx.helpers import string_type
        from yobx.torch import apply_patches_for_model, use_dyn_not_str
        from yobx.torch.tiny_models import TinyBroadcastAddModel

        model = TinyBroadcastAddModel()
        export_inputs = TinyBroadcastAddModel._export_inputs()
        dynamic_shapes = use_dyn_not_str(TinyBroadcastAddModel._dynamic_shapes())

        print(f"-- inputs: {string_type(export_inputs, with_shape=True)}")
        print(f"-- shapes: {dynamic_shapes}")
        print("--")
        print("-- simple export --")
        print("--")

        try:
            torch.export.export(
                model,
                (),
                kwargs=export_inputs,
                dynamic_shapes=dynamic_shapes,
            )
        except Exception as e:
            print(e)

        print("--")
        print("-- export with backed_size_oblivious=True --")
        print("--")

        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            try:
                torch.export.export(
                    model,
                    (),
                    kwargs=export_inputs,
                    dynamic_shapes=dynamic_shapes,
                )
            except Exception as e:
                print(e)

        print("--")
        print("-- patched export --")
        print("--")

        with (
            torch.fx.experimental._config.patch(backed_size_oblivious=True),
            apply_patches_for_model(patch_torch=True),
        ):
            ep = torch.export.export(
                model,
                (),
                kwargs=export_inputs,
                dynamic_shapes=dynamic_shapes,
            )
            print(ep)
    """

    _export_inputs = lambda: dict(x=torch.randn(2, 5), y=torch.randn(2, 1))  # noqa: E731
    _dynamic_shapes = lambda: dict(x={0: "batch", 1: "d1"}, y={0: "batch", 1: "d2"})  # noqa: E731

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


def get_tiny_model(model_id, config_updates: Optional[Dict[str, Any]] = None) -> ModelData:
    """
    Creates a tiny models, usually untrained to write tests.
    The `model_id` refers to what exists on HuggingFace.
    This functions is not expected to have fully coverage of architectures.

    Supported model IDs:

    * ``"arnir0/Tiny-LLM"`` — a tiny LLaMA-based causal language model with a
      :class:`transformers.cache_utils.DynamicCache` past-key-value cache.
    * ``"local/BroadcastAdd"`` — a minimal two-input model whose output has the
      symbolic shape ``(batch, max(d1, d2))`` due to broadcasting,
      see :class:`yobx.torch.tiny_models.TinyBroadcastAddModel`.

    :param model_id: model id, see the list of supported values above
    :param config_updates: modification to add to the configuration before creating the model
    :return: the necessary information

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
    if model_id == "local/BroadcastAdd":
        return ModelData(
            model_id=model_id,
            model=TinyBroadcastAddModel(),
            export_inputs=TinyBroadcastAddModel._export_inputs(),
            dynamic_shapes=TinyBroadcastAddModel._dynamic_shapes(),
        )

    if model_id == "arnir0/Tiny-LLM":
        from transformers import AutoModelForCausalLM
        from .in_transformers.models import get_cached_configuration
        from .in_transformers.cache_helper import make_dynamic_cache

        config = get_cached_configuration(model_id)
        if config_updates:
            config = copy.deepcopy(config)
            _update_config(config, config_updates)

        bsize, nheads, slen, dim = 2, 1, 30, 96
        return ModelData(
            model_id=model_id,
            model=AutoModelForCausalLM.from_config(config),
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
                attention_mask={0: "batch", 1: "past_length+seq_length"},
                position_ids={0: "batch", 1: "seq_length"},
                past_key_values=[{0: "batch", 2: "past_length"} for _ in range(2)],
            ),
            inputs_batch1=dict(
                input_ids=torch.randint(15, size=(1, 3), dtype=torch.int64),
                attention_mask=torch.randint(1, size=(1, 33), dtype=torch.int64),
                position_ids=torch.arange(3, dtype=torch.int64).unsqueeze(0),
                past_key_values=make_dynamic_cache(
                    [(torch.randn(1, nheads, slen, dim), torch.randn(1, nheads, slen, dim))]
                ),
            ),
        )

    raise ValueError(f"Model {model_id} is not supported yet.")

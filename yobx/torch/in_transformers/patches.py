import contextlib
import sys
import warnings
from typing import Any, Dict, Generator, List, Optional, Tuple
import torch
import transformers
from ...helpers.patch_helper import PatchInfo
from ._patches_model_rope_utils import common_RotaryEmbedding, patched_dynamic_rope_update


def _make_patch_info_for_rotary(submodule_class):
    patch = PatchInfo.make(
        # patched_dynamic_rope_update(submodule.forward.__wrapped__.__wrapped__),
        common_RotaryEmbedding.forward,
        submodule_class,
        "forward",
        family="transformers",
    )
    patch.add_dependency(
        PatchInfo.make(
            patched_dynamic_rope_update,
            transformers.modeling_rope_utils,
            "dynamic_rope_update",
            family="transformers",
            _last_patched_function=transformers.modeling_rope_utils.dynamic_rope_update,
        )
    )
    return patch


def get_patches_for(model: Optional[torch.nn.Module] = None) -> List[PatchInfo]:
    """
    Returns the list of patches for a specific model.
    if model is None, patches everything it can.

    .. note::
        The function detects that ``RotaryEmbedding.forward`` is wrapped by checking
        if can find substring ``transformers/modeling_rope_utils.py`` in
        ``RotaryEmbedding.forward.__wrapped__``. It does not seem to be the case
        with Python 3.10.
    """
    if model is None:
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        return [_make_patch_info_for_rotary(LlamaRotaryEmbedding)]
    patches = []
    for _name, submodule in model.named_modules():
        if (
            hasattr(submodule.forward, "__wrapped__")
            and hasattr(submodule.forward.__wrapped__, "__code__")
            and (
                "transformers/modeling_rope_utils.py"
                in str(submodule.forward.__wrapped__.__code__)
                or "transformers\\modeling_rope_utils.py"
                in str(submodule.forward.__wrapped__.__code__)
            )
        ):
            # RotaryEmbedding is wrapped and this one includes a control-flow.
            if submodule.__class__.__name__.endswith("RotaryEmbedding"):
                patch = _make_patch_info_for_rotary(submodule.__class__)
            elif submodule.__class__.__name__.endswith("RotaryEmbedding") and sys.version_info[
                :2
            ] < (3, 11):
                warnings.warn(
                    "RotaryEmbedding.forward cannot be patched with python<3.11.", UserWarning
                )
                continue
            else:
                raise NotImplementedError(
                    f"Wrapped {submodule.__class__.__name__} is not implemented yet."
                )
            patches.append(patch)
    return patches


@contextlib.contextmanager
def enable_transformers_onnx_export_flags(
    model: Optional[torch.nn.Module] = None, verbose: int = 0
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that enables the ONNX-export flags that are already
    implemented inside the :epkg:`transformers` library and restores them
    on exit.

    Two kinds of flags are handled:

    * ``config.onnx_export`` (and any sub-config that exposes it, such as
      :class:`LongformerConfig` or :class:`LEDConfig`) is set to ``True``.
    * Every submodule that implements ``prepare_for_onnx_export_`` (for
      example ``ProphetNetNgramSelfAttention``) has the method called on it,
      and its ``onnx_trace`` attribute is restored on exit.

    Calling this with ``model=None`` is a no-op and yields an empty report.

    :param model: the model whose flags should be toggled
    :param verbose: prints out a line every time a flag is toggled
    :return: yields a dict with two keys ``configs`` and ``modules`` listing
        the configs and submodules that were touched.
    """
    touched_configs: List[Tuple[Any, bool]] = []
    touched_modules: List[Tuple[Any, bool]] = []
    report: Dict[str, Any] = {"configs": touched_configs, "modules": touched_modules}

    if model is None:
        try:
            yield report
        finally:
            pass
        return

    def _walk_configs(cfg: Any, seen: set) -> Generator[Any, None, None]:
        if cfg is None or id(cfg) in seen:
            return
        seen.add(id(cfg))
        yield cfg
        for attr in ("text_config", "language_config", "vision_config", "audio_config"):
            sub = getattr(cfg, attr, None)
            if sub is not None:
                yield from _walk_configs(sub, seen)

    config = getattr(model, "config", None)
    if config is not None:
        for cfg in _walk_configs(config, set()):
            if hasattr(cfg, "onnx_export"):
                previous = bool(cfg.onnx_export)
                if not previous:
                    cfg.onnx_export = True
                    touched_configs.append((cfg, previous))
                    if verbose:
                        print(
                            f"[enable_transformers_onnx_export_flags] set "
                            f"{type(cfg).__name__}.onnx_export=True"
                        )

    for submodule in model.modules():
        prepare = getattr(submodule, "prepare_for_onnx_export_", None)
        if not callable(prepare):
            continue
        previous = bool(getattr(submodule, "onnx_trace", False))
        if previous:
            continue
        try:
            prepare()
        except Exception as exc:  # pragma: no cover - defensive, transformers-specific
            warnings.warn(
                f"prepare_for_onnx_export_ failed on {type(submodule).__name__}: {exc}",
                UserWarning,
            )
            continue
        touched_modules.append((submodule, previous))
        if verbose:
            print(
                f"[enable_transformers_onnx_export_flags] called "
                f"{type(submodule).__name__}.prepare_for_onnx_export_()"
            )

    try:
        yield report
    finally:
        for cfg, previous in touched_configs:
            with contextlib.suppress(Exception):  # pragma: no cover - defensive
                cfg.onnx_export = previous
        for submodule, previous in touched_modules:
            with contextlib.suppress(Exception):  # pragma: no cover - defensive
                submodule.onnx_trace = previous

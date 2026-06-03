import enum
from typing import Union


class TransformersPatchEnum(enum.Flag):
    """
    Selects which transformers-related patches :func:`apply_patches_for_model`
    should apply. Members can be combined with ``|``.

    * :attr:`NONE` — no transformers patches are applied.
    * :attr:`YOBX_PATCH` — applies yobx's own transformers patches
      (rotary embedding wrappers, flattening helpers, ...).
    * :attr:`TRANSFORMERS_PATCH` — toggles the ONNX-export switches that
      :epkg:`transformers` itself ships (``config.onnx_export``,
      ``module.prepare_for_onnx_export_()``).
    * :attr:`ALL` — shortcut for ``YOBX_PATCH | TRANSFORMERS_PATCH``.
    """

    NONE = 0
    YOBX_PATCH = enum.auto()
    TRANSFORMERS_PATCH = enum.auto()
    ALL = YOBX_PATCH | TRANSFORMERS_PATCH


def coerce_transformers_patch(
    value: Union[bool, TransformersPatchEnum, str],
) -> TransformersPatchEnum:
    """Backward-compat conversion: ``True`` -> ``ALL``, ``False`` -> ``NONE``."""
    if isinstance(value, TransformersPatchEnum):
        return value
    if isinstance(value, bool):
        return TransformersPatchEnum.ALL if value else TransformersPatchEnum.NONE
    if isinstance(value, str):
        if value == "transformers":
            return TransformersPatchEnum.TRANSFORMERS_PATCH
        if value == "yobx":
            return TransformersPatchEnum.YOBX_PATCH
        if value in ("yobx+transformers", "transformers+yobx"):
            return TransformersPatchEnum.YOBX_PATCH | TransformersPatchEnum.TRANSFORMERS_PATCH
        if value in ("none", ""):
            return TransformersPatchEnum.NONE
        raise ValueError(f"Unable to interpreter value={value!r}")
    raise TypeError(
        f"patch_transformers must be a bool or TransformersPatchEnum, got {type(value).__name__}"
    )

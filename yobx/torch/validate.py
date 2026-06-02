"""
Validates an ONNX export for a HuggingFace model using
:class:`InputObserver <yobx.torch.InputObserver>` and a default text prompt.
This module captures real model inputs by running the
model on a default prompt through an :class:`InputObserver`.  The observed inputs and
inferred dynamic shapes are then used for the ONNX export.
"""

import contextlib
from collections import Counter
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple, Union
import os

DEFAULT_PROMPT = "Continue: it rains, what should I do?"
"""Default text prompt used when validating text-generation models."""


@dataclass(init=False)
class ValidateSummary:
    """
    Flat summary dictionary returned by :func:`validate_model`.

    Contains status flags and error messages collected during validation.
    Fields that were not reached (e.g. because an earlier step failed) remain
    ``None``.

    Args:
        model_id: HuggingFace model identifier
        prompt: Text prompt used during validation
        config_from_cache: bundled"`` / ``"local"``
            when config was loaded from cache, ``False`` for network.
        config_overrides: String representation of the config overrides applied.
        error_config: Error message if config loading failed.
        error_tokenizer: Error message if tokenizer loading failed.
        model_from_cache: ``True`` when the model was loaded from the local HF cache
        error_model: Error message if model loading failed.
        n_captured: Number of input sets captured by the :class:`InputObserver`.
        error_observer: Error message if input capture failed.
        export: ``"OK"`` or ``"FAILED"`` depending on whether the ONNX export succeeded.
        error_export: Error message if the ONNX export failed.
        n_nodes: Total number of nodes in the exported ONNX graph.
        top_op_types: Compact summary of the most frequent op types, e.g.
            ``"MatMul:10,Add:7,Mul:5"``.
        discrepancies_ok: Number of input sets where ONNX Runtime results matched PyTorch.
        discrepancies_total: Total number of input sets checked for discrepancies.
        discrepancies: ``"OK"`` or ``"FAILED"`` for the overall discrepancy check.
        discrepancies_max_abs: Maximum absolute discrepancy across all input sets.
        discrepancies_atol: Absolute tolerance threshold used to determine success.
        discrepancies_ratio_001: Fraction of output elements with absolute difference > 0.01,
            aggregated across all input sets.
        discrepancies_ratio_01: Fraction of output elements with absolute difference > 0.1,
            aggregated across all input sets.
        error_discrepancies: Error message if the discrepancy check raised an exception.
    """

    model_id: str
    prompt: str
    config_from_cache: Optional[Union[str, bool]] = None
    config_overrides: Optional[str] = None
    error_config: Optional[str] = None
    error_tokenizer: Optional[str] = None
    model_from_cache: Optional[bool] = None
    error_model: Optional[str] = None
    n_captured: Optional[int] = None
    error_observer: Optional[str] = None
    export: Optional[str] = None
    error_export: Optional[str] = None
    n_nodes: Optional[int] = None
    top_op_types: Optional[str] = None
    discrepancies_ok: Optional[int] = None
    discrepancies_total: Optional[int] = None
    discrepancies: Optional[str] = None
    discrepancies_max_abs: Optional[float] = None
    discrepancies_atol: Optional[float] = None
    discrepancies_ratio_001: Optional[float] = None
    discrepancies_ratio_01: Optional[float] = None
    error_discrepancies: Optional[str] = None

    def __init__(
        self,
        model_id: str,
        prompt: str,
        config_from_cache: Optional[Union[str, bool]] = None,
        config_overrides: Optional[str] = None,
        error_config: Optional[str] = None,
        error_tokenizer: Optional[str] = None,
        model_from_cache: Optional[bool] = None,
        error_model: Optional[str] = None,
        n_captured: Optional[int] = None,
        error_observer: Optional[str] = None,
        export: Optional[str] = None,
        error_export: Optional[str] = None,
        n_nodes: Optional[int] = None,
        top_op_types: Optional[str] = None,
        discrepancies_ok: Optional[int] = None,
        discrepancies_total: Optional[int] = None,
        discrepancies: Optional[str] = None,
        discrepancies_max_abs: Optional[float] = None,
        discrepancies_atol: Optional[float] = None,
        discrepancies_ratio_001: Optional[float] = None,
        discrepancies_ratio_01: Optional[float] = None,
        error_discrepancies: Optional[str] = None,
    ):
        self.model_id = model_id
        self.prompt = prompt
        self.config_from_cache = config_from_cache
        self.config_overrides = config_overrides
        self.error_config = error_config
        self.error_tokenizer = error_tokenizer
        self.model_from_cache = model_from_cache
        self.error_model = error_model
        self.n_captured = n_captured
        self.error_observer = error_observer
        self.export = export
        self.error_export = error_export
        self.n_nodes = n_nodes
        self.top_op_types = top_op_types
        self.discrepancies_ok = discrepancies_ok
        self.discrepancies_total = discrepancies_total
        self.discrepancies = discrepancies
        self.discrepancies_max_abs = discrepancies_max_abs
        self.discrepancies_atol = discrepancies_atol
        self.discrepancies_ratio_001 = discrepancies_ratio_001
        self.discrepancies_ratio_01 = discrepancies_ratio_01
        self.error_discrepancies = error_discrepancies

    def items(self):
        """Yield ``(field_name, value)`` pairs for every non-``None`` field.

        This mirrors ``dict.items()`` so that existing code such as
        ``for k, v in sorted(summary.items())`` keeps working without
        modification.
        """
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None:
                yield f.name, v

    def keys(self):
        """Return an iterator over field names, mirroring ``dict.keys()``."""
        for f in fields(self):
            yield f.name

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key* if present; otherwise *default*."""
        if any(f.name == key for f in fields(self)):
            return getattr(self, key, default)
        return default

    def __getitem__(self, key: str) -> Any:
        """Provide dict-style access to fields by name."""
        for f in fields(self):
            if f.name == key:
                return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Return ``True`` when *key* names a field that has been set (non-``None``)."""
        return getattr(self, key, None) is not None


@dataclass
class ValidateData:
    """Intermediate artefacts collected by :func:`validate_model`.

    All fields default to ``None`` and are populated progressively as
    validation proceeds.
    """

    config: Optional[Any] = None
    """Loaded ``transformers`` config object."""
    model: Optional[Any] = None
    """Loaded (or randomly-initialised) PyTorch model."""
    input_ids: Optional[Any] = None
    """Input token ids tensor used during capture."""
    attention_mask: Optional[Any] = None
    """Attention mask tensor used during capture."""
    observer: Optional[Any] = None
    """:class:`InputObserver` instance after capture."""
    kwargs: Optional[Dict[str, Any]] = None
    """Inferred export keyword arguments."""
    dynamic_shapes: Optional[Any] = None
    """Inferred dynamic shapes passed to the exporter."""
    filename: Optional[str] = None
    """Path to the exported ``.onnx`` file."""
    artifact: Optional[Any] = None
    """:class:`~yobx.container.ExportArtifact` returned by the yobx exporter, or ``None``."""
    discrepancies: Optional[List[Dict[str, Any]]] = None
    """Per-input-set discrepancy records from :meth:`InputObserver.check_discrepancies`."""

    def items(self):
        """Yield ``(field_name, value)`` pairs for every non-``None`` field.

        This mirrors ``dict.items()`` so that existing code such as
        ``for k, v in sorted(data.items())`` keeps working without
        modification.
        """
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None:
                yield f.name, v

    def __contains__(self, key: str) -> bool:
        """Return ``True`` when *key* names a field that has been set (non-``None``)."""
        return getattr(self, key, None) is not None


def _to_onnx(*args, exporter: str = "yobx", **kwargs):
    if exporter == "yobx":
        from ..xbuilder import OptimizationOptions
        from . import to_onnx

        new_kwargs = {}
        for k, v in kwargs.items():
            if k == "opset_version":
                new_kwargs["target_opset"] = v
            elif k == "optimization":
                if v in ("default", "default+onnxruntime"):
                    new_kwargs["options"] = OptimizationOptions(patterns=v)
                else:
                    raise ValueError(f"unexpected value {v!r} for k={k!r}")
            else:
                new_kwargs[k] = v

        return to_onnx(*args, **new_kwargs)
    if exporter in ("dynamo", "onnx-dynamo", "modelbuilder"):
        import torch

        # Build a kwargs dict that torch.onnx.export understands.
        # Keys like "optimization" have no equivalent — skip them.
        # "filename" maps to the "f" positional-style param of older APIs; the
        # newer (2.x dynamo) API returns an ExportOutput that must be saved.
        dynamo_kwargs: dict = {"optimize": False, "dynamo": True}
        for k, v in kwargs.items():
            if k == "filename":
                dynamo_kwargs["f"] = v
            elif k == "optimization":
                dynamo_kwargs["optimize"] = v in (
                    "default",
                    "ir",
                    "default+onnxruntime",
                    "os_ort",
                )
            elif k == "verbose":
                dynamo_kwargs["report"] = True
            else:
                dynamo_kwargs[k] = v

        if exporter == "modelbuilder":
            dynamo_kwargs.setdefault("dynamo", True)

        epo = torch.onnx.export(*args, **dynamo_kwargs)
        if kwargs.get("optimization", "") in ("os_ort", "default+onnxruntime"):
            from onnxscript.rewriter.ort_fusions import optimize_for_ort

            optimize_for_ort(epo)  # type: ignore
        # saving is part of the the export
        return epo
    raise NotImplementedError(f"exporter={exporter!r} not implemented.")


def _apply_config_override(config: Any, key: str, value: Any) -> None:
    """Apply a single config override, with support for dotted paths and
    automatic propagation to nested sub-configs.

    Multimodal HuggingFace configs (e.g. ``Gemma3Config``) keep text-model
    attributes such as ``num_hidden_layers`` on a nested ``text_config`` and
    expose nothing at the top level.  A bare ``setattr(config, key, value)``
    would silently create an unused attribute, leaving the actual model size
    untouched and frequently triggering opaque downstream failures (for
    instance ``"upper bound and lower bound inconsistent with step sign"``
    raised by ``torch.arange`` inside generation).

    Behaviour:

    * Dotted keys (``"text_config.num_hidden_layers"``) walk the attribute
      chain and set the value on the final object.
    * Plain keys are set on the top-level ``config`` only when the attribute
      is already defined there.  Otherwise, the value is forwarded to the
      first conventional language-model sub-config that exposes the
      attribute (``text_config`` then ``language_config``), which matches
      the user intent for multimodal models such as ``Gemma3Config`` where
      ``num_hidden_layers`` lives on ``text_config``.  Vision/audio/image
      sub-configs are intentionally **not** touched, since reducing e.g.
      ``vision_config.num_hidden_layers`` produces a broken vision tower
      whose conv shapes no longer line up.  Use the dotted form to target
      them explicitly.
    """
    if "." in key:
        parts = key.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
        return

    # Top-level config already owns the attribute → set in place.
    if key in vars(config):
        setattr(config, key, value)
        return

    # Multimodal configs: forward to the conventional language sub-config.
    for sub_name in ("text_config", "language_config"):
        sub = getattr(config, sub_name, None)
        if sub is not None and hasattr(sub, key):
            setattr(sub, key, value)
            return

    # Fallback: set at the top level (creates a new attribute if missing).
    setattr(config, key, value)


def _detect_task(config: Any) -> str:
    """Detect the validation task from a HuggingFace config.

    Returns one of:

    * ``"image-classification"`` for vision classifiers such as
      ``BeitForImageClassification`` / ``ViTForImageClassification``.
    * ``"feature-extraction"`` for encoder-only "base" text models such as
      ``FunnelBaseModel`` / ``FunnelModel`` / ``BertModel`` that do not expose
      a ``generate`` method and have no task-specific head.
    * ``"causal-lm"`` (default) for text-generation models.

    Detection prefers the config ``architectures`` attribute when present, and
    falls back to the ``model_type`` membership in the corresponding
    ``transformers`` auto mappings. This keeps the heuristic robust both for
    hub configs (which set ``architectures``) and for configs instantiated
    directly from Python.
    """
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
        MODEL_MAPPING_NAMES,
    )

    archs = getattr(config, "architectures", None) or []
    for arch in archs:
        if isinstance(arch, str) and "ForImageClassification" in arch:
            return "image-classification"

    # Bare encoder architectures (no task head) such as ``FunnelBaseModel``,
    # ``FunnelModel`` or ``BertModel`` are listed as the values of
    # ``MODEL_MAPPING_NAMES`` (some entries are tuples to cover ``*Base``
    # variants). When the config declares one of those architectures we treat
    # it as a feature-extraction task and load it via ``AutoModel``.
    base_arch_names: set = set()
    for value in MODEL_MAPPING_NAMES.values():
        if isinstance(value, str):
            base_arch_names.add(value)
        else:
            base_arch_names.update(v for v in value if isinstance(v, str))
    for arch in archs:
        if isinstance(arch, str) and arch in base_arch_names:
            return "feature-extraction"

    model_type = getattr(config, "model_type", None)
    if model_type and model_type in MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES:
        return "image-classification"

    return "causal-lm"


def _load_config(
    model_id: str,
    config_overrides: Optional[Dict[str, Any]],
    verbose: int,
    quiet: bool,
    summary: "ValidateSummary",
    collected_data: "ValidateData",
):
    """Load the model config (bundled → local HF cache → network) and apply overrides."""
    from transformers import AutoConfig

    if verbose:
        print(f"[validate_model] loading config for {model_id!r}")

    try:
        from .in_transformers.models import get_cached_configuration

        _cached = get_cached_configuration(model_id)
        if _cached is not None:
            config = _cached
            summary.config_from_cache = "bundled"
        else:
            try:
                config = AutoConfig.from_pretrained(model_id, local_files_only=True)
                summary.config_from_cache = "local"
            except OSError:
                config = AutoConfig.from_pretrained(model_id)
                summary.config_from_cache = False
    except Exception as exc:
        summary.error_config = str(exc)
        if not quiet:
            raise
        return None

    if config_overrides:
        for k, v in config_overrides.items():
            _apply_config_override(config, k, v)
        summary.config_overrides = str(config_overrides)

    collected_data.config = config
    return config


def _load_tokenizer(model_id: str, verbose: int, quiet: bool, summary: "ValidateSummary"):
    """Load the tokenizer (local HF cache first, then network)."""
    from transformers import AutoTokenizer

    if verbose:
        print(f"[validate_model] loading tokenizer for {model_id!r}")

    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:
        summary.error_tokenizer = str(exc)
        if not quiet:
            raise
        return None

    return tokenizer


def _load_auto_model(
    auto_cls,
    model_id: str,
    config,
    random_weights: bool,
    dtype,
    torch_device,
    verbose: int,
    quiet: bool,
    summary: "ValidateSummary",
    collected_data: "ValidateData",
    label: str = "model",
):
    """Load (or instantiate with random weights) a model via *auto_cls*.

    *auto_cls* is a HuggingFace ``AutoModelFor*`` class. *label* is a short
    human-readable tag inserted into verbose log lines (e.g. ``"model"`` or
    ``"image model"``).
    """
    import torch

    if verbose:
        if random_weights:
            print(
                f"[validate_model] creating {label} from config (random weights) "
                f"for {model_id!r}"
            )
        else:
            print(f"[validate_model] loading {label} for {model_id!r}")

    dtype_kwargs: Dict[str, Any] = {"dtype": dtype} if dtype is not None else {}
    try:
        if random_weights:
            model: torch.nn.Module = auto_cls.from_config(config, **dtype_kwargs)
        else:
            try:
                model = auto_cls.from_pretrained(
                    model_id, config=config, local_files_only=True, **dtype_kwargs
                )
                summary.model_from_cache = True
            except OSError:
                model = auto_cls.from_pretrained(model_id, config=config, **dtype_kwargs)
                summary.model_from_cache = False
        model = model.to(torch_device)
        model.eval()
    except Exception as exc:
        summary.error_model = str(exc)
        if not quiet:
            raise
        return None

    collected_data.model = model
    return model


def _load_model(
    model_id: str,
    config,
    random_weights: bool,
    dtype,
    torch_device,
    verbose: int,
    quiet: bool,
    summary: "ValidateSummary",
    collected_data: "ValidateData",
):
    """Load (or instantiate with random weights) the CausalLM model."""
    from transformers import AutoModelForCausalLM

    # Multimodal configs (e.g. Gemma3Config) instantiate a *ForConditionalGeneration
    # wrapper with vision/audio towers when fed to AutoModelForCausalLM. Validating
    # the text-only causal LM is what we actually want here, so when the config
    # exposes a text sub-config we instantiate from it directly. This also avoids
    # tracing the vision tower whose symbolic conv shapes can produce confusing
    # "negative output size" errors during export.
    causal_config = config
    for sub_name in ("text_config", "language_config"):
        sub = getattr(config, sub_name, None)
        if sub is not None and hasattr(sub, "num_hidden_layers"):
            causal_config = sub
            if verbose:
                print(
                    f"[validate_model] using {sub_name} from {type(config).__name__} "
                    f"for text-only causal LM instantiation"
                )
            break

    return _load_auto_model(
        AutoModelForCausalLM,
        model_id,
        causal_config,
        random_weights,
        dtype,
        torch_device,
        verbose,
        quiet,
        summary,
        collected_data,
        label="model",
    )


def _load_image_model(
    model_id: str,
    config,
    random_weights: bool,
    dtype,
    torch_device,
    verbose: int,
    quiet: bool,
    summary: "ValidateSummary",
    collected_data: "ValidateData",
):
    """Load (or instantiate with random weights) an image-classification model."""
    from transformers import AutoModelForImageClassification

    return _load_auto_model(
        AutoModelForImageClassification,
        model_id,
        config,
        random_weights,
        dtype,
        torch_device,
        verbose,
        quiet,
        summary,
        collected_data,
        label="image model",
    )


def _load_feature_extraction_model(
    model_id: str,
    config,
    random_weights: bool,
    dtype,
    torch_device,
    verbose: int,
    quiet: bool,
    summary: "ValidateSummary",
    collected_data: "ValidateData",
):
    """Load (or instantiate with random weights) a bare encoder model via ``AutoModel``.

    Used for "feature-extraction" tasks: encoder-only models such as
    ``FunnelBaseModel`` / ``BertModel`` that have no task head and do not
    expose a ``generate`` method.
    """
    from transformers import AutoModel

    return _load_auto_model(
        AutoModel,
        model_id,
        config,
        random_weights,
        dtype,
        torch_device,
        verbose,
        quiet,
        summary,
        collected_data,
        label="encoder model",
    )


def _build_image_inputs(config, dtype, torch_device) -> Dict[str, Any]:
    """Build random ``pixel_values`` inputs sized from the model config."""
    import torch

    image_size = getattr(config, "image_size", 224)
    if isinstance(image_size, (list, tuple)):
        height, width = int(image_size[0]), int(image_size[1])
    else:
        height = width = int(image_size)
    num_channels = int(getattr(config, "num_channels", 3))
    torch_dtype = dtype if dtype is not None else torch.float32
    pixel_values = torch.randn(
        1, num_channels, height, width, dtype=torch_dtype, device=torch_device
    )
    return {"pixel_values": pixel_values}


def _capture_inputs_forward(
    model,
    forward_kwargs: Dict[str, Any],
    patch: bool,
    verbose: int,
    quiet: bool,
    summary: "ValidateSummary",
    collected_data: "ValidateData",
):
    """Run ``model(**forward_kwargs)`` under an InputObserver to capture real inputs.

    Used for non-generative models (e.g. image classification) that do not
    expose ``model.generate``.
    """
    from .flatten import register_flattening_functions
    from .input_observer import InputObserver
    from .patch import apply_patches_for_model

    if verbose:
        print(
            f"[validate_model] capturing inputs with InputObserver "
            f"(forward kwargs: {sorted(forward_kwargs)})"
        )

    observer = InputObserver()

    with (
        register_flattening_functions(patch_transformers=patch),
        (
            apply_patches_for_model(patch_transformers=patch, model=model)
            if patch
            else contextlib.nullcontext()
        ),
        observer(model),
    ):
        model(**forward_kwargs)

    collected_data.observer = observer
    summary.n_captured = len(observer.info or [])

    if verbose:
        print(f"[validate_model] captured {len(observer.info or [])} input set(s)")

    return observer


def _capture_inputs(
    model,
    input_ids,
    attention_mask,
    max_new_tokens: int,
    patch: bool,
    prompt: str,
    verbose: int,
    quiet: bool,
    summary: "ValidateSummary",
    collected_data: "ValidateData",
):
    """Run model.generate under an InputObserver to capture real inputs."""
    from .flatten import register_flattening_functions
    from .input_observer import InputObserver
    from .patch import apply_patches_for_model

    if verbose:
        print(f"[validate_model] capturing inputs with InputObserver (prompt={prompt!r})")

    observer = InputObserver()

    try:
        with (
            register_flattening_functions(patch_transformers=patch),
            (
                apply_patches_for_model(patch_transformers=patch, model=model)
                if patch
                else contextlib.nullcontext()
            ),
            observer(model),
        ):
            generate_kwargs: Dict[str, Any] = dict(
                input_ids=input_ids, do_sample=False, max_new_tokens=max_new_tokens
            )
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask
            model.generate(**generate_kwargs)
    except Exception as exc:
        summary.error_observer = str(exc)
        if not quiet:
            raise
        return None

    collected_data.observer = observer
    summary.n_captured = len(observer.info or [])

    if verbose:
        print(f"[validate_model] captured {len(observer.info or [])} input set(s)")

    return observer


def _infer_shapes(observer, patch: bool, verbose: int, collected_data: "ValidateData"):
    """Infer export kwargs and dynamic shapes from the captured observer."""
    from .flatten import register_flattening_functions

    with register_flattening_functions(patch_transformers=patch):
        kwargs = observer.infer_arguments()
        dynamic_shapes = observer.infer_dynamic_shapes(set_batch_dimension_for=True)

    collected_data.kwargs = kwargs
    collected_data.dynamic_shapes = dynamic_shapes

    if verbose:
        from ..helpers import string_type

        print(f"[validate_model] kwargs: {string_type(kwargs, with_shape=True)}")
        print(f"[validate_model] dynamic_shapes: {dynamic_shapes}")

    return kwargs, dynamic_shapes


def _export(
    model,
    model_id: str,
    kwargs: Dict[str, Any],
    dynamic_shapes,
    exporter: str,
    opset: int,
    optimization: Optional[str],
    patch: bool,
    dump_folder: Optional[str],
    verbose: int,
    quiet: bool,
    summary: "ValidateSummary",
    collected_data: "ValidateData",
):
    """Export the model to ONNX and store the output filename."""
    from .flatten import register_flattening_functions
    from .patch import apply_patches_for_model

    model_name = model_id.replace("/", "-")
    suffix = ".".join([exporter, str(opset), optimization or "", "patch" if patch else "t"])
    filename = f"{model_name}.{suffix}.onnx"
    if dump_folder is not None:
        os.makedirs(dump_folder, exist_ok=True)
        filename = os.path.join(dump_folder, filename)
    else:
        import tempfile

        _tmpdir = tempfile.mkdtemp()
        filename = os.path.join(_tmpdir, filename)

    collected_data.filename = filename

    if verbose:
        print(f"[validate_model] exporting to ONNX ({exporter=}, {opset=}) -> {filename!r}")

    try:
        export_kwargs: Dict[str, Any] = {}
        if optimization is not None:
            export_kwargs["optimization"] = optimization

        with (
            register_flattening_functions(patch_transformers=patch),
            (
                apply_patches_for_model(patch_torch=patch, patch_transformers=patch, model=model)
                if patch
                else contextlib.nullcontext()
            ),
        ):
            artifact = _to_onnx(
                model,
                (),
                kwargs=kwargs,
                dynamic_shapes=dynamic_shapes,
                filename=filename,
                exporter=exporter,
                opset_version=opset,
                verbose=max(0, verbose - 1),
                **export_kwargs,
            )
        collected_data.artifact = artifact
        summary.export = "OK"
    except Exception as exc:
        summary.export = "FAILED"
        summary.error_export = str(exc)
        if not quiet:
            raise
        return False

    # Compute node statistics from the exported ONNX file for the standard output.
    if os.path.exists(filename):
        import onnx

        onx = onnx.load(filename, load_external_data=False)
        counts = Counter(n.op_type for n in onx.graph.node)
        summary.n_nodes = sum(counts.values())
        top = counts.most_common(5)
        summary.top_op_types = ",".join(f"{op}:{cnt}" for op, cnt in top)

    if verbose:
        print(f"[validate_model] export succeeded -> {filename!r}")

    return True


def _check_discrepancies(
    observer,
    filename: str,
    verbose: int,
    quiet: bool,
    summary: "ValidateSummary",
    collected_data: "ValidateData",
):
    """Run ONNX Runtime on every captured input set and compare against PyTorch outputs."""
    if verbose:
        print("[validate_model] checking discrepancies ...")
    atol = 1e-4
    if quiet:
        try:
            disc_data = observer.check_discrepancies(filename, atol=atol)
        except Exception as exc:
            summary.discrepancies = "FAILED"
            summary.error_discrepancies = str(exc)
            return
    else:
        disc_data = observer.check_discrepancies(filename, atol=atol)
    collected_data.discrepancies = disc_data
    n_ok = sum(1 for row in disc_data if row.get("SUCCESS", False))
    n_total = len(disc_data)
    summary.discrepancies_ok = n_ok
    summary.discrepancies_total = n_total
    summary.discrepancies = "OK" if n_ok == n_total else "FAILED"
    summary.discrepancies_atol = atol
    # Aggregate per-element stats across all examples that ran without error.
    numeric_rows = [row for row in disc_data if "abs" in row]
    if numeric_rows:
        summary.discrepancies_max_abs = max(row["abs"] for row in numeric_rows)
        total_n = sum(row.get("n", 0) for row in numeric_rows)
        if total_n > 0:
            total_001 = sum(row.get(">0.01", 0) for row in numeric_rows)
            total_01 = sum(row.get(">0.1", 0) for row in numeric_rows)
            summary.discrepancies_ratio_001 = total_001 / total_n
            summary.discrepancies_ratio_01 = total_01 / total_n
    if verbose:
        print(f"[validate_model] discrepancies: {n_ok}/{n_total} OK")
    if verbose >= 2:
        for row in disc_data:
            idx = row.get("index", "?")
            ok = row.get("SUCCESS", False)
            status = "OK" if ok else "FAILED"
            if "error" in row:
                print(f"  [{idx}] {status}  error={row['error']}")
            else:
                abs_diff = row.get("abs", float("nan"))
                rel_diff = row.get("rel", float("nan"))
                print(f"  [{idx}] {status}  abs={abs_diff:.3g}  rel={rel_diff:.3g}")
                if verbose >= 3:
                    if "inputs" in row:
                        print(f"       inputs:       {row['inputs']}")
                    if "outputs_torch" in row:
                        print(f"       outputs_torch: {row['outputs_torch']}")
                    if "outputs_ort" in row:
                        print(f"       outputs_ort:  {row['outputs_ort']}")


def validate_model(
    model_id: str,
    prompt: str = DEFAULT_PROMPT,
    exporter: str = "yobx",
    optimization: Optional[str] = "default",
    verbose: int = 0,
    dump_folder: Optional[str] = None,
    opset: int = 22,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
    max_new_tokens: int = 10,
    do_run: bool = True,
    patch: bool = True,
    quiet: bool = False,
    tokenized_inputs: Optional[Dict[str, Any]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    random_weights: bool = False,
) -> Tuple["ValidateSummary", "ValidateData"]:
    """
    Validates an ONNX export for any HuggingFace ``model_id`` by capturing real
    model inputs with :class:`InputObserver <yobx.torch.InputObserver>` and a
    text *prompt*, instead of relying on dummy / random tensors.

    The function:

    1. Loads the model config for *model_id* (cached copy preferred).
    2. Optionally applies *config_overrides* to tweak the architecture
       (e.g. reduce ``num_hidden_layers`` for a faster test).
    3. Loads the tokeniser and the pretrained model weights — or, when
       *random_weights* is ``True``, instantiates the model directly from the
       (potentially modified) config without downloading any weights.
    4. Runs :meth:`model.generate` with *prompt* inside an
       :class:`InputObserver <yobx.torch.InputObserver>` context to collect
       real input/output tensors.
    5. Exports the model to ONNX using the observed inputs and the inferred
       dynamic shapes.
    6. Computes discrepancies between the original PyTorch outputs and the
       ONNX runtime outputs for every captured input set.

    :param model_id: HuggingFace model id, e.g. ``"arnir0/Tiny-LLM"``
    :param prompt: Text prompt used to drive the generation step.
        Defaults to :data:`DEFAULT_PROMPT`.
    :param exporter: ONNX exporter to use, e.g. ``"yobx"`` (default),
        ``"modelbuilder"``, or ``"onnx-dynamo"``.
    :param optimization: Optimisation level applied after export.
        Passed directly to :func:`yobx.torch.to_onnx`.  ``None`` means no
        optimisation; ``"default"`` applies the default set.
    :param verbose: Verbosity level.  ``0`` = silent; ``1`` = one-line
        progress messages; ``2`` = per-input-set discrepancy summary
        (index, SUCCESS, abs/rel diff); ``3`` = additionally prints the
        input and output tensor shapes for every discrepancy row.
    :param dump_folder: When given, all artefacts (ONNX file, export logs …)
        are saved under this directory.
    :param opset: ONNX opset version to target (default 22).
    :param dtype: Cast the model (and inputs) to this dtype before exporting,
        e.g. ``"float16"``.  ``None`` keeps the default (``float32``).
    :param device: Run on this device, e.g. ``"cpu"`` or ``"cuda"``.
        ``None`` defaults to CPU.
    :param max_new_tokens: Number of tokens generated by
        :meth:`model.generate` during input capture (default 10).
        Larger values capture more past-key-value shapes.
    :param do_run: When ``True`` (default), checks that the ONNX model can be
        run after export and computes discrepancies.
    :param patch: Apply ``apply_patches_for_model`` and
        ``register_flattening_functions`` during export (default ``True``).
    :param quiet: When ``True``, exceptions are caught and reported in the
        returned summary dictionary rather than re-raised.
    :param tokenized_inputs: Optional pre-tokenized inputs to use instead of
        running the tokenizer on *prompt*.  Should be a dict with at least
        ``"input_ids"`` and optionally ``"attention_mask"`` (mirrors the output
        of a HuggingFace tokenizer).  When provided the tokenizer is not loaded
        and *prompt* is only stored in the summary for reference.
    :param config_overrides: Optional mapping of config attribute names to new
        values applied to the model config before the model is instantiated,
        e.g. ``{"num_hidden_layers": 2}``.  Useful to create a smaller model
        for testing without changing the architecture definition on disk.
    :param random_weights: When ``True``, instantiate the model from the
        (possibly modified) config with random weights instead of downloading
        the pretrained weights.  This avoids any network access for the model
        itself, which is useful for fast unit-testing or CI validation.
    :return: A 2-tuple ``(summary, data)`` where *summary* is a
        :class:`ValidateSummary` instance with status flags and error messages,
        and *data* is a :class:`ValidateData` instance that collects all
        intermediate artefacts.

    Example::

        from yobx.torch.validate import validate_model

        summary, data = validate_model("arnir0/Tiny-LLM", verbose=1)
        for k, v in sorted(summary.items()):
            print(f":{k},{v};")
    """
    import torch

    summary: ValidateSummary = ValidateSummary(model_id=model_id, prompt=prompt)
    collected_data: ValidateData = ValidateData()

    # ------------------------------------------------------------------ device / dtype
    torch_device = torch.device(device or "cpu")
    if dtype is not None:
        dtype = getattr(torch, dtype)

    # ----------------------------------------------------------------- config
    config = _load_config(model_id, config_overrides, verbose, quiet, summary, collected_data)
    if config is None:
        return summary, collected_data

    task = _detect_task(config)
    if verbose:
        print(f"[validate_model] detected task: {task!r}")

    if task == "image-classification":
        # Image classifiers do not use a tokenizer; build pixel_values from config.
        model = _load_image_model(
            model_id,
            config,
            random_weights,
            dtype,
            torch_device,
            verbose,
            quiet,
            summary,
            collected_data,
        )
        if model is None:
            return summary, collected_data

        forward_kwargs = _build_image_inputs(config, dtype, torch_device)
        observer = _capture_inputs_forward(
            model, forward_kwargs, patch, verbose, quiet, summary, collected_data
        )
        if observer is None:
            return summary, collected_data

        kwargs, dynamic_shapes = _infer_shapes(observer, patch, verbose, collected_data)

        ok = _export(
            model,
            model_id,
            kwargs,
            dynamic_shapes,
            exporter,
            opset,
            optimization,
            patch,
            dump_folder,
            verbose,
            quiet,
            summary,
            collected_data,
        )
        if not ok:
            return summary, collected_data

        if do_run:
            assert (
                collected_data.filename
            ), "No filename, this is needed to check for discrepancies."
            assert os.path.exists(
                collected_data.filename
            ), f"{collected_data.filename!r} is missing"
            _check_discrepancies(
                observer, collected_data.filename, verbose, quiet, summary, collected_data
            )

        from ..container import ExportArtifact

        if isinstance(collected_data.artifact, ExportArtifact) and collected_data.filename:
            artifact = collected_data.artifact
            if artifact.report is None:
                from ..container import ExportReport

                artifact.report = ExportReport()
            if collected_data.discrepancies is not None:
                artifact.report.discrepancies = collected_data.discrepancies
            artifact.save_report(collected_data.filename)

        return summary, collected_data

    if task == "feature-extraction":
        # Encoder-only "base" models (e.g. FunnelBaseModel) use a tokenizer
        # but do not expose ``generate``. Capture inputs via a plain forward
        # pass on the tokenized prompt.
        if tokenized_inputs is None:
            tokenizer = _load_tokenizer(model_id, verbose, quiet, summary)
            if tokenizer is None:
                return summary, collected_data
        else:
            tokenizer = None

        model = _load_feature_extraction_model(
            model_id,
            config,
            random_weights,
            dtype,
            torch_device,
            verbose,
            quiet,
            summary,
            collected_data,
        )
        if model is None:
            return summary, collected_data

        if tokenized_inputs is not None:
            forward_kwargs = {
                k: (v.to(torch_device) if hasattr(v, "to") else v)
                for k, v in tokenized_inputs.items()
            }
        else:
            tokenized = tokenizer(prompt, return_tensors="pt")  # type: ignore
            forward_kwargs = {k: v.to(torch_device) for k, v in tokenized.items()}

        collected_data.input_ids = forward_kwargs.get("input_ids")
        collected_data.attention_mask = forward_kwargs.get("attention_mask")

        observer = _capture_inputs_forward(
            model, forward_kwargs, patch, verbose, quiet, summary, collected_data
        )
        if observer is None:
            return summary, collected_data

        kwargs, dynamic_shapes = _infer_shapes(observer, patch, verbose, collected_data)

        ok = _export(
            model,
            model_id,
            kwargs,
            dynamic_shapes,
            exporter,
            opset,
            optimization,
            patch,
            dump_folder,
            verbose,
            quiet,
            summary,
            collected_data,
        )
        if not ok:
            return summary, collected_data

        if do_run:
            assert (
                collected_data.filename
            ), "No filename, this is needed to check for discrepancies."
            assert os.path.exists(
                collected_data.filename
            ), f"{collected_data.filename!r} is missing"
            _check_discrepancies(
                observer, collected_data.filename, verbose, quiet, summary, collected_data
            )

        from ..container import ExportArtifact

        if isinstance(collected_data.artifact, ExportArtifact) and collected_data.filename:
            artifact = collected_data.artifact
            if artifact.report is None:
                from ..container import ExportReport

                artifact.report = ExportReport()
            if collected_data.discrepancies is not None:
                artifact.report.discrepancies = collected_data.discrepancies
            artifact.save_report(collected_data.filename)

        return summary, collected_data

    # --------------------------------------------------------------- tokenizer
    if tokenized_inputs is None:
        tokenizer = _load_tokenizer(model_id, verbose, quiet, summary)
        if tokenizer is None:
            return summary, collected_data
    else:
        tokenizer = None

    # --------------------------------------------------------------- load model
    model = _load_model(
        model_id,
        config,
        random_weights,
        dtype,
        torch_device,
        verbose,
        quiet,
        summary,
        collected_data,
    )
    if model is None:
        return summary, collected_data

    # ---------------------------------------------------------- tokenise prompt / use supplied
    if tokenized_inputs is not None:
        inputs = tokenized_inputs
    else:
        inputs = tokenizer(prompt, return_tensors="pt")  # type: ignore
    input_ids = inputs["input_ids"].to(torch_device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(torch_device)

    collected_data.input_ids = input_ids
    collected_data.attention_mask = attention_mask

    # ------------------------------------------------------- capture with observer
    observer = _capture_inputs(
        model,
        input_ids,
        attention_mask,
        max_new_tokens,
        patch,
        prompt,
        verbose,
        quiet,
        summary,
        collected_data,
    )
    if observer is None:
        return summary, collected_data

    # ---------------------------------------------------- infer args & shapes
    kwargs, dynamic_shapes = _infer_shapes(observer, patch, verbose, collected_data)

    # ------------------------------------------------------------------- export
    ok = _export(
        model,
        model_id,
        kwargs,
        dynamic_shapes,
        exporter,
        opset,
        optimization,
        patch,
        dump_folder,
        verbose,
        quiet,
        summary,
        collected_data,
    )
    if not ok:
        return summary, collected_data

    # --------------------------------------------------------- check discrepancies
    if do_run:
        assert collected_data.filename, "No filename, this is needed to check for discrepancies."
        assert os.path.exists(collected_data.filename), f"{collected_data.filename!r} is missing"
        _check_discrepancies(
            observer, collected_data.filename, verbose, quiet, summary, collected_data
        )

    # --------------------------------- update xlsx report with discrepancies (yobx exporter only)
    from ..container import ExportArtifact

    if isinstance(collected_data.artifact, ExportArtifact) and collected_data.filename:
        artifact = collected_data.artifact
        if artifact.report is None:
            from ..container import ExportReport

            artifact.report = ExportReport()
        if collected_data.discrepancies is not None:
            artifact.report.discrepancies = collected_data.discrepancies
        artifact.save_report(collected_data.filename)

    return summary, collected_data

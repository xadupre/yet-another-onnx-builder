from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.image import PatchExtractor

from ...typing import GraphBuilderExtendedProtocol
from ..register import register_sklearn_converter


@register_sklearn_converter(PatchExtractor)
def sklearn_patch_extractor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: PatchExtractor,
    X: str,
    name: str = "patch_extractor",
) -> str:
    """
    Converts a :class:`sklearn.feature_extraction.image.PatchExtractor` into ONNX.

    The transformer extracts patches from a collection of images using a
    sliding window.  The image dimensions must be known at conversion time
    (i.e. concrete, not symbolic) because the number and positions of the
    patches depend on them.

    Only the deterministic, exhaustive extraction (``max_patches=None``) is
    supported.  Passing ``max_patches`` other than ``None`` raises
    :class:`NotImplementedError`.

    **Supported input shapes**

    * 3-D ``(n_samples, image_height, image_width)`` – grayscale images.
    * 4-D ``(n_samples, image_height, image_width, n_channels)`` – colour
      images.

    **Graph layout**

    For a 3-D input with ``patch_size = (ph, pw)`` and an image grid of
    ``(h, w)`` pixels, the graph contains:

    .. code-block:: text

        For each (i, j) in [0..h-ph] × [0..w-pw]:
            Slice(X, [i,j], [i+ph, j+pw], axes=[1,2])  →  (N, ph, pw)
            Unsqueeze(·, axis=1)                         →  (N, 1, ph, pw)

        Concat(all unsqueezed patches, axis=1)           →  (N, n_p, ph, pw)
        Reshape(·, [-1, ph, pw])                         →  (N*n_p, ph, pw)

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn` (unused; present for
        interface consistency)
    :param estimator: a fitted ``PatchExtractor``
    :param outputs: desired output names
    :param X: input name
    :param name: prefix name for the added nodes
    :return: output name
    """
    assert isinstance(
        estimator, PatchExtractor
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"
    assert g.has_shape(X), (
        f"PatchExtractor conversion requires a known input shape so that "
        f"patch positions can be pre-computed; missing shape for {X!r}"
        f"{g.get_debug_msg()}"
    )

    if estimator.max_patches is not None:
        raise NotImplementedError(
            "PatchExtractor with max_patches != None uses random sampling "
            "and cannot be deterministically exported to ONNX."
        )

    shape: Tuple = g.get_shape(X)
    ndim = len(shape)
    assert ndim in (3, 4), (
        f"PatchExtractor expects a 3-D (N, H, W) or 4-D (N, H, W, C) input; "
        f"got shape {shape}{g.get_debug_msg()}"
    )

    h, w = shape[1], shape[2]
    assert h is not None and w is not None, (
        f"Image height and width must be concrete (not None) at conversion time; "
        f"got shape {shape}{g.get_debug_msg()}"
    )

    ph, pw = estimator.patch_size

    assert (
        ph <= h and pw <= w
    ), f"patch_size {estimator.patch_size} must not exceed image size ({h}, {w})."

    patches: List[str] = []
    axes_hw = np.array([1, 2], dtype=np.int64)

    for i in range(h - ph + 1):
        for j in range(w - pw + 1):
            starts = np.array([i, j], dtype=np.int64)
            ends = np.array([i + ph, j + pw], dtype=np.int64)
            patch = g.op.Slice(X, starts, ends, axes_hw, name=f"{name}_slice_{i}_{j}")
            patch_unsqueezed = g.op.Unsqueeze(
                patch, np.array([1], dtype=np.int64), name=f"{name}_unsqueeze_{i}_{j}"
            )
            patches.append(patch_unsqueezed)

    if len(patches) == 1:
        # Single patch: skip Concat, just remove the extra dim we added.
        all_patches = g.op.Squeeze(
            patches[0], np.array([1], dtype=np.int64), name=f"{name}_squeeze"
        )
    else:
        # Concat along dim 1 → (N, n_patches_per_image, ph, pw[, C])
        all_patches = g.op.Concat(*patches, axis=1, name=f"{name}_concat")

    # Reshape to (N * n_patches_per_image, ph, pw[, C])
    if ndim == 3:
        new_shape = np.array([-1, ph, pw], dtype=np.int64)
    else:
        c = shape[3]
        assert c is not None, (
            f"Channel dimension must be concrete (not None) at conversion time; "
            f"got shape {shape}{g.get_debug_msg()}"
        )
        new_shape = np.array([-1, ph, pw, c], dtype=np.int64)

    result = g.op.Reshape(all_patches, new_shape, name=name, outputs=outputs)
    return result

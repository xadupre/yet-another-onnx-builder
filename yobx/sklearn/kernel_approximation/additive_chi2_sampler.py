import numpy as np
from typing import Dict, List

from sklearn.kernel_approximation import AdditiveChi2Sampler

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype

# Default sample_interval values taken from sklearn source.
# See figure 2 c) of "Efficient additive kernels via explicit feature maps"
# A. Vedaldi and A. Zisserman, Pattern Analysis and Machine Intelligence, 2011.
_DEFAULT_SAMPLE_INTERVALS = {1: 0.8, 2: 0.5, 3: 0.4}


@register_sklearn_converter(AdditiveChi2Sampler)
def sklearn_additive_chi2_sampler(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: AdditiveChi2Sampler,
    X: str,
    name: str = "additive_chi2_sampler",
) -> str:
    """
    Converts a :class:`sklearn.kernel_approximation.AdditiveChi2Sampler` into ONNX.

    The conversion replicates :meth:`AdditiveChi2Sampler.transform`, which
    maps each input feature ``x`` into ``2*sample_steps - 1`` output features:

    * **step 0** (one feature per input feature):

      .. code-block:: text

          sqrt(x * sample_interval)

    * **step j** (j = 1 … sample_steps-1, two features per input feature):

      .. code-block:: text

          factor_j = sqrt(2 * x * sample_interval / cosh(π * j * sample_interval))

          cos_j = factor_j * cos(j * sample_interval * log(x))
          sin_j = factor_j * sin(j * sample_interval * log(x))

    The output columns are arranged as:

    .. code-block:: text

        [sqrt(all F features),
         cos_1(all F features), sin_1(all F features),
         cos_2(all F features), sin_2(all F features), ...]

    giving a total of ``n_features * (2 * sample_steps - 1)`` output columns.

    Zero-valued inputs produce zero outputs for every component.  To avoid
    ``log(0) = -∞`` causing NaN propagation, the logarithm is evaluated on
    ``max(x, tiny)`` where ``tiny`` is the smallest positive normal float for
    the working dtype.  The ``factor_j`` is computed from the original ``x``
    and naturally evaluates to zero when ``x = 0``, so the masked product
    ``factor_j * cos/sin(…)`` is exactly zero for zero inputs.

    :param g: the graph builder to add nodes to
    :param sts: shape/type information already inferred by scikit-learn; when
        non-empty the function skips manual ``set_type``/``set_shape`` calls
        because the caller will handle them
    :param estimator: a fitted (or stateless) ``AdditiveChi2Sampler``
    :param outputs: desired output names
    :param X: input tensor name (non-negative values required)
    :param name: prefix name for the added nodes
    :return: output tensor name
    """
    assert isinstance(
        estimator, AdditiveChi2Sampler
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    sample_steps = int(estimator.sample_steps)

    # Resolve sample_interval (mirrors sklearn logic).
    if estimator.sample_interval is None:
        if sample_steps not in _DEFAULT_SAMPLE_INTERVALS:
            raise ValueError(
                "If sample_steps is not in {1, 2, 3}, you need to provide sample_interval."
            )
        sample_interval = _DEFAULT_SAMPLE_INTERVALS[sample_steps]
    else:
        sample_interval = float(estimator.sample_interval)

    # ── Component 0: sqrt(x * sample_interval) ──────────────────────────────
    sqrt_coeff = np.array([sample_interval], dtype=dtype)
    sqrt_comp = g.op.Sqrt(
        g.op.Mul(X, sqrt_coeff, name=f"{name}_sqrt_scaled"), name=f"{name}_sqrt0"
    )

    components = [sqrt_comp]

    if sample_steps > 1:
        # Protect log(x) from -inf when x=0 by clamping at the smallest
        # representable positive normal float.  factor_j is computed from the
        # original X (not the clamped value), so it equals zero exactly when
        # x=0, making the final product zero despite the clamped log argument.
        eps = np.array([np.finfo(dtype).tiny], dtype=dtype)
        x_safe = g.op.Max(X, eps, name=f"{name}_x_safe")
        log_x = g.op.Log(x_safe, name=f"{name}_log_x")

        for j in range(1, sample_steps):
            # factor_j = sqrt(2 * x * sample_interval / cosh(π * j * sample_interval))
            cosh_val = float(np.cosh(np.pi * j * sample_interval))
            step_coeff = np.array([2.0 * sample_interval / cosh_val], dtype=dtype)
            factor = g.op.Sqrt(
                g.op.Mul(X, step_coeff, name=f"{name}_factor_scaled_{j}"),
                name=f"{name}_factor_{j}",
            )

            # log_step = j * sample_interval * log(x)
            j_si = np.array([float(j) * sample_interval], dtype=dtype)
            log_step = g.op.Mul(log_x, j_si, name=f"{name}_log_step_{j}")

            # cos and sin projections
            cos_comp = g.op.Mul(
                factor, g.op.Cos(log_step, name=f"{name}_cos_{j}"), name=f"{name}_cos_comp_{j}"
            )
            sin_comp = g.op.Mul(
                factor, g.op.Sin(log_step, name=f"{name}_sin_{j}"), name=f"{name}_sin_comp_{j}"
            )

            components.append(cos_comp)
            components.append(sin_comp)

    # ── Concatenate all components along the feature axis ───────────────────
    if len(components) == 1:
        res = g.op.Identity(components[0], name=name, outputs=outputs)
    else:
        res = g.op.Concat(*components, axis=1, name=name, outputs=outputs)

    g.set_type(res, itype)
    if g.has_shape(X):
        batch_dim = g.get_shape(X)[0]
        n_features = g.get_shape(X)[1]
        n_out_features = n_features * (2 * sample_steps - 1)
        g.set_shape(res, (batch_dim, n_out_features))
    return res

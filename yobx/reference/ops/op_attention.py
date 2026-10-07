import numpy as np
import scipy.special as scipy_special
from ._native_op import NativeOpKernel


class Attention(NativeOpKernel):
    op_domain = "com.microsoft"

    def _run(
        self,
        x,
        weights,
        bias,
        mask_index,
        past,
        attention_bias,
        num_heads=None,
        unidirectional=0,
        qkv_hidden_sizes=None,
        past_present_share_buffer=0,
        do_rotary=0,
        rotary_embedding_dim=None,
        mask_filter_value=-10000.0,
        scale=None,
    ):
        if past is not None:
            raise NotImplementedError("Attention with past state is not implemented.")
        if past_present_share_buffer not in (None, 0):
            raise NotImplementedError(
                "Attention with a shared past/present buffer is not implemented."
            )
        if do_rotary not in (None, 0):
            raise NotImplementedError(
                f"Attention with rotary embeddings is not implemented "
                f"(rotary_embedding_dim={rotary_embedding_dim!r})."
            )
        if unidirectional not in (0, 1):
            raise ValueError(f"unidirectional must be 0 or 1, not {unidirectional!r}.")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(f"num_heads must be a positive integer, not {num_heads!r}.")

        if qkv_hidden_sizes is None:
            if weights.shape[1] % 3:
                raise ValueError(
                    f"Cannot split weights shape {weights.shape!r} into equal Q, K, V sizes."
                )
            hidden_sizes = [weights.shape[1] // 3] * 3
        else:
            hidden_sizes = list(qkv_hidden_sizes)
            if len(hidden_sizes) != 3 or sum(hidden_sizes) != weights.shape[1]:
                raise ValueError(
                    f"qkv_hidden_sizes={hidden_sizes!r} is inconsistent with "
                    f"weights shape {weights.shape!r}."
                )
        q_size, k_size, v_size = hidden_sizes
        if q_size != k_size or q_size % num_heads or v_size % num_heads:
            raise ValueError(
                f"Unsupported Q/K/V sizes {hidden_sizes!r} for num_heads={num_heads}."
            )

        q_weights = weights[:, :q_size]
        k_weights = weights[:, q_size : q_size + k_size]
        v_weights = weights[:, q_size + k_size :]
        if bias is None:
            q_bias = k_bias = v_bias = 0
        else:
            if bias.shape != (sum(hidden_sizes),):
                raise ValueError(
                    f"Bias shape {bias.shape!r} does not match Q/K/V sizes {hidden_sizes!r}."
                )
            q_bias = bias[:q_size]
            k_bias = bias[q_size : q_size + k_size]
            v_bias = bias[q_size + k_size :]

        q_shape = (*x.shape[:2], num_heads, q_size // num_heads)
        v_shape = (*x.shape[:2], num_heads, v_size // num_heads)
        xqb_4d = (x @ q_weights + q_bias).reshape(q_shape)
        xkb_4d = (x @ k_weights + k_bias).reshape(q_shape)
        xvb_4d = (x @ v_weights + v_bias).reshape(v_shape)
        rot_xqb = np.transpose(xqb_4d, axes=(0, 2, 1, 3))
        rot_xkb = np.transpose(xkb_4d, axes=(0, 2, 1, 3))
        factor = scale if scale is not None else 1.0 / np.sqrt(q_shape[-1])
        scores = factor * rot_xqb @ np.transpose(rot_xkb, [0, 1, 3, 2])
        if attention_bias is not None:
            scores = scores + attention_bias
        if mask_index is not None:
            if mask_index.ndim == 2 and mask_index.shape[0] == x.shape[0]:
                mask = mask_index[:, None, None, :]
            elif mask_index.ndim == 3 and mask_index.shape[0] == x.shape[0]:
                mask = mask_index[:, None, :, :]
            elif mask_index.ndim == 4:
                mask = mask_index
            else:
                raise NotImplementedError(
                    f"Attention mask shape {mask_index.shape!r} is not implemented."
                )
            scores = np.where(mask != 0, scores, mask_filter_value)
        if unidirectional:
            query_length, key_length = scores.shape[-2:]
            causal_mask = np.arange(key_length) > np.arange(query_length)[:, None]
            scores = np.where(causal_mask, mask_filter_value, scores)

        transpose_3 = np.transpose(xvb_4d, axes=(0, 2, 1, 3))
        softmax = scipy_special.softmax(scores, axis=-1)
        matmul_1 = softmax @ transpose_3
        transpose_5 = np.transpose(matmul_1, axes=(0, 2, 1, 3))
        view_3 = transpose_5.reshape(*x.shape[:2], v_size)
        return (view_3,)

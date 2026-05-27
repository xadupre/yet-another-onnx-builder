import transformers

false = False
null = None
true = True


def _ccached_local_GraniteMoeHybrid():
    "local/GraniteMoeHybrid"
    # Tiny GraniteMoeHybrid configuration mixing one ``mamba`` and one
    # ``attention`` layer. Dimensions are chosen so that
    # ``mamba_expand * hidden_size == mamba_n_heads * mamba_d_head`` (the
    # constraint enforced by :meth:`GraniteMoeHybridConfig.validate_architecture`).
    return transformers.GraniteMoeHybridConfig(
        **{
            "architectures": ["GraniteMoeHybridForCausalLM"],
            "attention_bias": false,
            "attention_dropout": 0.0,
            "attention_multiplier": 1.0,
            "bos_token_id": 1,
            "embedding_multiplier": 1.0,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 64,
            "initializer_range": 0.02,
            "intermediate_size": 128,
            "layer_types": ["mamba", "attention"],
            "logits_scaling": 1.0,
            "mamba_chunk_size": 16,
            "mamba_conv_bias": true,
            "mamba_d_conv": 4,
            "mamba_d_head": 8,
            "mamba_d_state": 16,
            "mamba_expand": 2,
            "mamba_n_groups": 1,
            "mamba_n_heads": 16,
            "mamba_proj_bias": false,
            "max_position_embeddings": 64,
            "model_type": "granitemoehybrid",
            "num_attention_heads": 4,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "num_local_experts": 4,
            "output_router_logits": false,
            "position_embedding_type": null,
            "residual_multiplier": 1.0,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
            "router_aux_loss_coef": 0.001,
            "shared_intermediate_size": 64,
            "tie_word_embeddings": false,
            "use_cache": true,
            "vocab_size": 256,
        }
    )


def _ccached_arnir0_tiny_LLM():
    "arnir0/Tiny-LLM"
    return transformers.LlamaConfig(
        **{
            "architectures": ["LlamaForCausalLM"],
            "attention_bias": false,
            "attention_dropout": 0.0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "head_dim": 96,
            "hidden_act": "silu",
            "hidden_size": 192,
            "initializer_range": 0.02,
            "intermediate_size": 1024,
            "max_position_embeddings": 1024,
            "mlp_bias": false,
            "model_type": "llama",
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": null,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
            "dtype": "float32",
            "transformers_version": "4.51.0.dev0",
            "use_cache": true,
            "vocab_size": 32000,
        }
    )


if not hasattr(transformers, "GraniteMoeHybridConfig"):
    # Older versions of transformers do not ship GraniteMoeHybridConfig; drop
    # the registration so _retrieve_cached_configurations does not expose it.
    del _ccached_local_GraniteMoeHybrid

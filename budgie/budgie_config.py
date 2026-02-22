from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig

class BudgieConfig(PretrainedConfig):
    model_type = "budgie"

    def __init__(
        self,
        vocab_size: int = 32768,
        hidden_size: int = 2048,
        intermediate_size: int = 2048*3,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = 16,
        hidden_act: str = "silu",
        max_position_embeddings: int = 8192,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6, 
        use_cache: bool = False,
        pad_token_id: int = 3,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        pretraining_tp: int = 1,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.01,
        mlp_bias: bool = False,
        qk_rope_dim: int | None = None,
        use_tiny_conv: bool = True,
        tiny_conv_kernel_size: int = 4,
        tiny_conv_bias: bool = False,
        tiny_conv_init_zero: bool = True,
        use_hybrid_layers: bool = True,
        local_attn_implementation: str = "gla_sliding",
        bridge_attn_implementation: str = "gla_landmark",
        bridge_every_n_layers: int = 6,
        bridge_layer_offset: int = 5,
        tiny_conv_on_local_layers: bool = False,
        tiny_conv_on_bridge_layers: bool = True,
        tiny_conv_every_n_bridge_layers: int = 2,
        tiny_conv_bridge_start: int = 2,
        sliding_window: int | None = 1024,
        landmark_every: int | None = 128,
        landmark_token_id: int | None = None,
        use_xformers: bool = False,
        use_liger_kernel: bool = True,
        use_causal_conv1d: bool = True,
        share_all_layers: bool = False,
        num_phases: int = 2,
        use_phase_layer_gates: bool = True,
        _attn_implementation: str = "sdpa",
        **kwargs,
    ):
        legacy_attention_window = kwargs.pop("attention_window", None)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp

        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"`hidden_size` must be divisible by `num_attention_heads`, got hidden_size={hidden_size} and "
                f"num_attention_heads={num_attention_heads}."
            )
        head_dim = hidden_size // num_attention_heads
        self.head_dim = head_dim
        if qk_rope_dim is None:
            # RoPE works on pairs of dims, so the rotary dim must be even. Keep the historical default of head_dim//2
            # but round it down to an even number and clamp to at least 2.
            rope_dim = head_dim // 2
            rope_dim = (rope_dim // 2) * 2
            rope_dim = 2 if rope_dim == 0 else rope_dim
            self.qk_rope_dim = rope_dim
        else:
            if not isinstance(qk_rope_dim, int) or qk_rope_dim <= 0 or (qk_rope_dim % 2) != 0:
                raise ValueError(f"`qk_rope_dim` must be a positive even int, got {qk_rope_dim!r}.")
            self.qk_rope_dim = qk_rope_dim

        self.use_tiny_conv = use_tiny_conv
        self.tiny_conv_kernel_size = tiny_conv_kernel_size
        self.tiny_conv_bias = tiny_conv_bias
        self.tiny_conv_init_zero = tiny_conv_init_zero

        self.use_hybrid_layers = use_hybrid_layers
        self.local_attn_implementation = local_attn_implementation
        self.bridge_attn_implementation = bridge_attn_implementation
        self.bridge_every_n_layers = bridge_every_n_layers
        self.bridge_layer_offset = bridge_layer_offset
        self.tiny_conv_on_local_layers = tiny_conv_on_local_layers
        self.tiny_conv_on_bridge_layers = tiny_conv_on_bridge_layers
        self.tiny_conv_every_n_bridge_layers = tiny_conv_every_n_bridge_layers
        self.tiny_conv_bridge_start = tiny_conv_bridge_start

        self.sliding_window = sliding_window if sliding_window is not None else legacy_attention_window
        self.landmark_every = landmark_every
        self.landmark_token_id = landmark_token_id
        self.use_xformers = use_xformers
        self.use_liger_kernel = use_liger_kernel
        self.use_causal_conv1d = use_causal_conv1d
        self.share_all_layers = share_all_layers
        self.num_phases = num_phases
        self.use_phase_layer_gates = use_phase_layer_gates

        # Used by the decoder layer registry in `budgie/modeling_budgie_GLA.py`.
        self._attn_implementation = _attn_implementation


__all__ = ["BudgieConfig"]

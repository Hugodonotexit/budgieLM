from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig


class BudgieConfig(PretrainedConfig):
    model_type = "budgie"

    def __init__(
        self,
        # -----------------
        # Core architecture
        # -----------------
        vocab_size: int = 32768,
        hidden_size: int = 3584,
        intermediate_size: int = int(3584*3.75),
        num_hidden_layers: int = 40,

        # -----------------
        # Attention (Q/K/V)
        # -----------------
        num_attention_heads: int = 32,
        gla_num_groups: int = 2,
        num_key_value_heads: int | None = None,
        hidden_act: str = "silu",

        # ----------------------------
        # Context length / positions
        # ----------------------------
        max_position_embeddings: int = 1024 * 64,
        # Alias for `max_position_embeddings` (often called "context length" in training code).
        context_length: int | None = None,

        # -------------------------
        # Initialization / misc
        # -------------------------
        initializer_range: float = 0.03,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,

        # ----------
        # Token ids
        # ----------
        pad_token_id: int = 3,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,

        # -------------------------
        # Training compatibility
        # -------------------------
        pretraining_tp: int = 1,

        # ------
        # RoPE
        # ------
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        # Back-compat: allow passing a scalar scaling factor instead of a HF-style rope_scaling dict.
        # For 8k -> 32k extension: set max_position_embeddings=32768 and rope_scaling_factor=4.0.
        rope_scaling_factor: float | None = None,
        rope_scaling_type: str = "yarn",
        # Optional hint for rope_scaling variants that want the original context length.
        original_max_position_embeddings: int | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.01,
        mlp_bias: bool = False,
        qk_rope_dim: int | None = None,

        
        # --------
        # Tiny conv
        # --------
        use_tiny_conv: bool = True,
        tiny_conv_kernel_size: int = 4,
        tiny_conv_bias: bool = False,
        tiny_conv_init_zero: bool = True,

        # -------------
        # Hybrid layers
        # -------------
        use_hybrid_layers: bool = False,
        local_attn_implementation: str = "gla_sliding",
        bridge_attn_implementation: str = "gla_landmark",
        bridge_every_n_layers: int = 6,
        bridge_layer_offset: int = 3,

        # ---------------------
        # Tiny conv (per-layer)
        # ---------------------
        tiny_conv_on_local_layers: bool = False,
        tiny_conv_on_bridge_layers: bool = True,
        tiny_conv_every_n_bridge_layers: int = 1,
        tiny_conv_bridge_start: int = 0,

        # ----------------------
        # GSM token mixer (opt-in)
        # ----------------------
        use_gsm: bool = True,
        gsm_n_groups: int = 8,
        gsm_gate_rank: int = 32,
        gsm_alpha: float = 0.5,
        gsm_dropout: float = 0.0,
        gsm_use_triton: bool = True,
        gsm_w_init_scale: float = 0.02,
        gsm_rms_eps: float = 1e-5,
        gsm_on_local_layers: bool = False,
        gsm_on_bridge_layers: bool = True,
        gsm_every_n_bridge_layers: int = 1,
        gsm_bridge_start: int = 6,

        # ----------------------
        # Sliding-window attention
        # ----------------------
        sliding_window: int | None = 2048,
        swa_dilation: int = 4,
        swa_dilated_window: int | None = 2048,

        # ------------------
        # Landmark attention
        # ------------------
        landmark_every: int | None = 256,
        landmark_token_id: int | None = None,

        # -------------------------
        # Macro structure (opt-in)
        # -------------------------
        use_macro_structure: bool = True,
        macro_structure_pattern: str = "swa,perceiver,swa_dilated,bridge,swa",

        # ---------------------------------
        # Perceiver-style macro-block knobs
        # ---------------------------------
        perceiver_num_latents: int = 128,
        perceiver_heads_tokens: int | None = None,
        perceiver_heads_latents: int | None = None,
        perceiver_mlp_mult: float = 3.0,
        perceiver_dropout: float = 0.0,
        perceiver_droppath: float = 0.05,
        perceiver_attn_backend: str = "sdpa",
        perceiver_latent_process_layers: int = 1,
        perceiver_use_fft_in_latents: bool = False,
        perceiver_gate_writeback: bool = True,
        perceiver_read_is_causal: bool = True,
        perceiver_write_is_causal: bool = True,

        # ----------------
        # Optional backends
        # ----------------
        use_xformers: bool = True,
        use_liger_kernel: bool = True,
        use_causal_conv1d: bool = True,

        # -------------------------
        # Sharing / phases (opt-in)
        # -------------------------
        share_all_layers: bool = False,
        num_phases: int = 1,
        use_phase_layer_gates: bool = False,

        # -------------------------
        # Product Key Memory (PKM)
        # -------------------------
        use_pkm: bool = True,
        pkm_on_non_dilated_swa: bool = True,
        pkm_num_product_keys: int = 8,
        pkm_product_key_size: int = 32,
        pkm_value_size: int = 128,

        # -----------------------
        # Internal / registry knob
        # -----------------------
        _attn_implementation: str = "sdpa",
        **kwargs,
    ):
        legacy_attention_window = kwargs.pop("attention_window", None)

        if context_length is not None:
            max_position_embeddings = int(context_length)
        if not isinstance(max_position_embeddings, int) or max_position_embeddings <= 0:
            raise ValueError(f"`max_position_embeddings` must be a positive int, got {max_position_embeddings!r}.")

        if rope_scaling_factor is not None:
            rope_scaling_factor = float(rope_scaling_factor)
            if rope_scaling_factor <= 0:
                raise ValueError(f"`rope_scaling_factor` must be > 0, got {rope_scaling_factor!r}.")
            if rope_scaling is None and rope_scaling_factor != 1.0:
                rope_scaling = {"type": str(rope_scaling_type), "factor": rope_scaling_factor}

        if original_max_position_embeddings is not None:
            original_max_position_embeddings = int(original_max_position_embeddings)
            if original_max_position_embeddings <= 0:
                raise ValueError(
                    "`original_max_position_embeddings` must be a positive int, "
                    f"got {original_max_position_embeddings!r}."
                )
            if rope_scaling is not None and "original_max_position_embeddings" not in rope_scaling:
                rope_scaling = dict(rope_scaling)
                rope_scaling["original_max_position_embeddings"] = original_max_position_embeddings

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # -----------------
        # Core architecture
        # -----------------
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp

        # -----------------
        # Attention / RoPE
        # -----------------
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.original_max_position_embeddings = original_max_position_embeddings

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

        if not isinstance(gla_num_groups, int) or gla_num_groups <= 0:
            raise ValueError(f"`gla_num_groups` must be a positive int, got {gla_num_groups!r}.")
        self.gla_num_groups = gla_num_groups

        # --------
        # Tiny conv
        # --------
        self.use_tiny_conv = use_tiny_conv
        self.tiny_conv_kernel_size = tiny_conv_kernel_size
        self.tiny_conv_bias = tiny_conv_bias
        self.tiny_conv_init_zero = tiny_conv_init_zero

        # -------------
        # Hybrid layers
        # -------------
        self.use_hybrid_layers = use_hybrid_layers
        self.local_attn_implementation = local_attn_implementation
        self.bridge_attn_implementation = bridge_attn_implementation
        self.bridge_every_n_layers = bridge_every_n_layers
        self.bridge_layer_offset = bridge_layer_offset
        self.tiny_conv_on_local_layers = tiny_conv_on_local_layers
        self.tiny_conv_on_bridge_layers = tiny_conv_on_bridge_layers
        self.tiny_conv_every_n_bridge_layers = tiny_conv_every_n_bridge_layers
        self.tiny_conv_bridge_start = tiny_conv_bridge_start

        # ----------------------
        # GSM token mixer
        # ----------------------
        self.use_gsm = bool(use_gsm)
        self.gsm_n_groups = int(gsm_n_groups)
        self.gsm_gate_rank = int(gsm_gate_rank)
        self.gsm_alpha = float(gsm_alpha)
        self.gsm_dropout = float(gsm_dropout)
        self.gsm_use_triton = bool(gsm_use_triton)
        self.gsm_w_init_scale = float(gsm_w_init_scale)
        self.gsm_rms_eps = float(gsm_rms_eps)
        self.gsm_on_local_layers = bool(gsm_on_local_layers)
        self.gsm_on_bridge_layers = bool(gsm_on_bridge_layers)
        self.gsm_every_n_bridge_layers = int(gsm_every_n_bridge_layers)
        self.gsm_bridge_start = int(gsm_bridge_start)

        if self.gsm_n_groups <= 0:
            raise ValueError(f"`gsm_n_groups` must be a positive int, got {self.gsm_n_groups!r}.")
        if self.gsm_gate_rank <= 0:
            raise ValueError(f"`gsm_gate_rank` must be a positive int, got {self.gsm_gate_rank!r}.")
        if self.gsm_every_n_bridge_layers <= 0:
            raise ValueError(
                f"`gsm_every_n_bridge_layers` must be a positive int, got {self.gsm_every_n_bridge_layers!r}."
            )
        if self.gsm_bridge_start <= 0:
            raise ValueError(f"`gsm_bridge_start` must be a positive int, got {self.gsm_bridge_start!r}.")
        if not (0.0 <= self.gsm_dropout < 1.0):
            raise ValueError(f"`gsm_dropout` must be in [0, 1), got {self.gsm_dropout!r}.")
        if self.gsm_w_init_scale < 0.0:
            raise ValueError(f"`gsm_w_init_scale` must be >= 0, got {self.gsm_w_init_scale!r}.")
        if self.gsm_rms_eps <= 0.0:
            raise ValueError(f"`gsm_rms_eps` must be > 0, got {self.gsm_rms_eps!r}.")
        if self.use_gsm and (self.hidden_size % self.gsm_n_groups) != 0:
            raise ValueError(
                "`hidden_size` must be divisible by `gsm_n_groups` when `use_gsm=True`. "
                f"Got hidden_size={self.hidden_size}, gsm_n_groups={self.gsm_n_groups}."
            )

        # ----------------------
        # Sliding / landmark attn
        # ----------------------
        self.sliding_window = sliding_window if sliding_window is not None else legacy_attention_window
        self.swa_dilation = int(swa_dilation)
        self.swa_dilated_window = int(swa_dilated_window) if swa_dilated_window is not None else None
        self.landmark_every = landmark_every
        self.landmark_token_id = landmark_token_id

        if self.swa_dilation <= 0:
            raise ValueError(f"`swa_dilation` must be a positive int, got {self.swa_dilation!r}.")
        if self.swa_dilated_window is not None and self.swa_dilated_window <= 0:
            raise ValueError(
                f"`swa_dilated_window` must be a positive int when set, got {self.swa_dilated_window!r}."
            )

        # -------------------------
        # Macro structure / Perceiver
        # -------------------------
        self.use_macro_structure = bool(use_macro_structure)
        self.macro_structure_pattern = str(macro_structure_pattern)
        self.perceiver_num_latents = int(perceiver_num_latents)
        self.perceiver_heads_tokens = (
            int(perceiver_heads_tokens) if perceiver_heads_tokens is not None else self.num_attention_heads
        )
        self.perceiver_heads_latents = (
            int(perceiver_heads_latents) if perceiver_heads_latents is not None else self.num_attention_heads // 2
        )
        self.perceiver_mlp_mult = float(perceiver_mlp_mult)
        self.perceiver_dropout = float(perceiver_dropout)
        self.perceiver_droppath = float(perceiver_droppath)
        self.perceiver_attn_backend = str(perceiver_attn_backend).lower()
        self.perceiver_latent_process_layers = int(perceiver_latent_process_layers)
        self.perceiver_use_fft_in_latents = bool(perceiver_use_fft_in_latents)
        self.perceiver_gate_writeback = bool(perceiver_gate_writeback)
        self.perceiver_read_is_causal = bool(perceiver_read_is_causal)
        self.perceiver_write_is_causal = bool(perceiver_write_is_causal)

        if self.perceiver_num_latents <= 0:
            raise ValueError(
                f"`perceiver_num_latents` must be a positive int, got {self.perceiver_num_latents!r}."
            )
        if self.perceiver_heads_tokens <= 0 or (self.hidden_size % self.perceiver_heads_tokens) != 0:
            raise ValueError(
                "`perceiver_heads_tokens` must be a positive divisor of `hidden_size`, got "
                f"{self.perceiver_heads_tokens!r} for hidden_size={self.hidden_size}."
            )
        if self.perceiver_heads_latents <= 0 or (self.hidden_size % self.perceiver_heads_latents) != 0:
            raise ValueError(
                "`perceiver_heads_latents` must be a positive divisor of `hidden_size`, got "
                f"{self.perceiver_heads_latents!r} for hidden_size={self.hidden_size}."
            )
        if self.perceiver_mlp_mult <= 0.0:
            raise ValueError(f"`perceiver_mlp_mult` must be > 0, got {self.perceiver_mlp_mult!r}.")
        if not (0.0 <= self.perceiver_dropout < 1.0):
            raise ValueError(f"`perceiver_dropout` must be in [0, 1), got {self.perceiver_dropout!r}.")
        if not (0.0 <= self.perceiver_droppath < 1.0):
            raise ValueError(f"`perceiver_droppath` must be in [0, 1), got {self.perceiver_droppath!r}.")
        if self.perceiver_attn_backend not in ("sdpa", "xformers", "flash"):
            raise ValueError(
                f"`perceiver_attn_backend` must be one of ('sdpa','xformers','flash'), got {self.perceiver_attn_backend!r}."
            )
        if self.perceiver_latent_process_layers <= 0:
            raise ValueError(
                "`perceiver_latent_process_layers` must be a positive int, got "
                f"{self.perceiver_latent_process_layers!r}."
            )

        # ----------------
        # Optional backends
        # ----------------
        self.use_xformers = use_xformers
        self.use_liger_kernel = use_liger_kernel
        self.use_causal_conv1d = use_causal_conv1d

        # -------------------------
        # Sharing / phases (opt-in)
        # -------------------------
        self.share_all_layers = share_all_layers
        self.num_phases = num_phases
        self.use_phase_layer_gates = use_phase_layer_gates

        # -------------------------
        # Product Key Memory (PKM)
        # -------------------------
        self.use_pkm = bool(use_pkm)
        self.pkm_on_non_dilated_swa = bool(pkm_on_non_dilated_swa)
        self.pkm_num_product_keys = int(pkm_num_product_keys)
        self.pkm_product_key_size = int(pkm_product_key_size)
        self.pkm_value_size = int(pkm_value_size)

        # Used by the decoder layer registry in `budgie/modeling_budgie_GLA.py`.
        self._attn_implementation = _attn_implementation


__all__ = ["BudgieConfig"]

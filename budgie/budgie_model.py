from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

from .budgie_config import BudgieConfig
from .budgie_pretrained_model import BudgiePreTrainedModel
from .modeling_budgie_GLA import (
    BudgieCausalDepthwiseConv1d,
    BudgieGLASharedHybrid,
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
    _prepare_4d_causal_attention_mask_with_cache_position,
    budgie_make_embedding,
    budgie_make_mlp,
    budgie_make_rmsnorm,
)
from .modeling_budgie_gsm import GSM
from .modeling_budgie_latent_bottleneck import LatentBottleneckMacroBlock
from .modeling_budgie_pkm import SWABlockWithPKM

logger = logging.get_logger(__name__)


class BudgieLayerScaffold(nn.Module):
    def __init__(
        self,
        config: BudgieConfig,
        layer_idx: int,
        enable_tiny_conv: bool,
        enable_gsm: bool,
        is_bridge_layer: bool,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = budgie_make_rmsnorm(config, config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = budgie_make_rmsnorm(config, config.hidden_size, config.rms_norm_eps)
        if enable_tiny_conv:
            kernel_size = int(getattr(config, "tiny_conv_kernel_size", 3))
            bias = bool(getattr(config, "tiny_conv_bias", False))
            init_zero = bool(getattr(config, "tiny_conv_init_zero", True))
            use_causal_conv1d = bool(getattr(config, "use_causal_conv1d", True))
            self.tiny_conv = BudgieCausalDepthwiseConv1d(
                config.hidden_size,
                kernel_size=kernel_size,
                bias=bias,
                init_zero=init_zero,
                use_causal_conv1d=use_causal_conv1d,
            )
        else:
            self.tiny_conv = None

        if enable_gsm:
            self.gsm = GSM(
                d_model=config.hidden_size,
                max_seq_len=int(config.max_position_embeddings),
                n_groups=int(getattr(config, "gsm_n_groups", 8)),
                gate_rank=int(getattr(config, "gsm_gate_rank", 32)),
                alpha=float(getattr(config, "gsm_alpha", 0.5)),
                dropout=float(getattr(config, "gsm_dropout", 0.0)),
                use_triton=bool(getattr(config, "gsm_use_triton", True)),
                w_init_scale=float(getattr(config, "gsm_w_init_scale", 0.02)),
                rms_eps=float(getattr(config, "gsm_rms_eps", 1e-6)),
            )
        else:
            self.gsm = None

        self.bridge_mixer_fusion = bool(is_bridge_layer and self.gsm is not None)
        if self.bridge_mixer_fusion:
            self.bridge_alpha_proj = nn.Linear(config.hidden_size, 1, bias=True)
            self.bridge_out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            nn.init.zeros_(self.bridge_alpha_proj.weight)
            nn.init.zeros_(self.bridge_alpha_proj.bias)
            nn.init.eye_(self.bridge_out_proj.weight)
        else:
            self.bridge_alpha_proj = None
            self.bridge_out_proj = None


class BudgiePerceiverMacroLayer(nn.Module):
    """
    Wrapper that adapts a Perceiver-style macro-block to the decoder-layer call signature.
    """

    def __init__(self, config: BudgieConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.seed_latents = nn.Parameter(
            torch.randn(int(getattr(config, "perceiver_num_latents", 128)), config.hidden_size)
            / math.sqrt(float(config.hidden_size))
        )
        self.block = LatentBottleneckMacroBlock(
            config=config,
            heads_tokens=int(getattr(config, "perceiver_heads_tokens", config.num_attention_heads)),
            heads_latents=int(getattr(config, "perceiver_heads_latents", config.num_attention_heads)),
            mlp_mult=float(getattr(config, "perceiver_mlp_mult", 4.0)),
            dropout=float(getattr(config, "perceiver_dropout", 0.0)),
            droppath=float(getattr(config, "perceiver_droppath", 0.0)),
            latent_process_layers=int(getattr(config, "perceiver_latent_process_layers", 1)),
            use_fft_in_latents=bool(getattr(config, "perceiver_use_fft_in_latents", False)),
            gate_writeback=bool(getattr(config, "perceiver_gate_writeback", True)),
            layer_idx=layer_idx,
        )
        self.read_is_causal = bool(getattr(config, "perceiver_read_is_causal", True))
        self.write_is_causal = bool(getattr(config, "perceiver_write_is_causal", True))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_gate: Optional[torch.Tensor] = None,
        mlp_gate: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        token_mask_2d = kwargs.get("attention_mask_2d", None)
        if token_mask_2d is None and attention_mask is not None and attention_mask.dim() == 2:
            token_mask_2d = attention_mask

        del attention_mask, position_ids, cache_position, position_embeddings, kwargs
        residual = hidden_states
        bsz = hidden_states.shape[0]
        latents = self.seed_latents.unsqueeze(0).expand(bsz, -1, -1).contiguous()
        out, _ = self.block(
            hidden_states,
            latents,
            read_is_causal=self.read_is_causal,
            write_is_causal=self.write_is_causal,
            attn_mask_read=token_mask_2d,
        )
        delta = out - residual
        if attn_gate is None:
            attn_gate = 1
        if mlp_gate is None:
            mlp_gate = 1
        hidden_states = residual + (delta * attn_gate * mlp_gate)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (past_key_value,)
        return outputs


class BudgieModel(BudgiePreTrainedModel):
    def __init__(self, config: BudgieConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.num_phases = int(getattr(config, "num_phases", 1))
        self.use_phase_layer_gates = bool(getattr(config, "use_phase_layer_gates", False))

        self.embed_tokens = budgie_make_embedding(config, config.vocab_size, config.hidden_size, self.padding_idx)
        share_all_layers = bool(getattr(config, "share_all_layers", False))
        use_hybrid_layers = bool(getattr(config, "use_hybrid_layers", False))
        use_macro_structure = bool(getattr(config, "use_macro_structure", True))

        local_impl = str(getattr(config, "local_attn_implementation", "gla_sliding"))
        bridge_impl = str(getattr(config, "bridge_attn_implementation", "gla_landmark"))
        bridge_every = int(getattr(config, "bridge_every_n_layers", 6))
        bridge_offset = int(getattr(config, "bridge_layer_offset", 5))

        tiny_conv_on_local = bool(getattr(config, "tiny_conv_on_local_layers", False))
        tiny_conv_on_bridge = bool(getattr(config, "tiny_conv_on_bridge_layers", True))
        tiny_conv_every_n_bridge = int(getattr(config, "tiny_conv_every_n_bridge_layers", 2))
        tiny_conv_bridge_start = int(getattr(config, "tiny_conv_bridge_start", 2))
        use_gsm = bool(getattr(config, "use_gsm", False))
        gsm_on_local = bool(getattr(config, "gsm_on_local_layers", False))
        gsm_on_bridge = bool(getattr(config, "gsm_on_bridge_layers", True))
        gsm_every_n_bridge = int(getattr(config, "gsm_every_n_bridge_layers", 1))
        gsm_bridge_start = int(getattr(config, "gsm_bridge_start", 1))

        macro_pattern_raw = str(getattr(config, "macro_structure_pattern", "swa,perceiver,swa_dilated,bridge,swa"))
        macro_pattern = [p.strip().lower() for p in macro_pattern_raw.split(",") if p.strip()]
        if use_macro_structure:
            if not macro_pattern:
                raise ValueError("`macro_structure_pattern` must contain at least one layer token when enabled.")
            valid_tokens = {"swa", "perceiver", "swa_dilated", "bridge"}
            invalid = [p for p in macro_pattern if p not in valid_tokens]
            if invalid:
                raise ValueError(
                    "`macro_structure_pattern` contains invalid tokens. "
                    f"Allowed tokens: {sorted(valid_tokens)}. Got invalid={invalid!r}."
                )

        def is_bridge(layer_idx: int) -> bool:
            return use_hybrid_layers and layer_idx >= bridge_offset and (layer_idx - bridge_offset) % bridge_every == 0

        def bridge_number(layer_idx: int) -> int:
            return ((layer_idx - bridge_offset) // bridge_every) + 1

        self._layer_is_bridge: list[bool] = [is_bridge(i) for i in range(config.num_hidden_layers)]
        self._layer_type: list[str] | None = None

        def _layer_tiny_conv(bridge: bool, bridge_idx: int) -> bool:
            if not bool(getattr(config, "use_tiny_conv", False)):
                return False
            if not bridge:
                return tiny_conv_on_local
            return (
                tiny_conv_on_bridge
                and bridge_idx >= tiny_conv_bridge_start
                and (bridge_idx - tiny_conv_bridge_start) % tiny_conv_every_n_bridge == 0
            )

        def _layer_gsm(bridge: bool, bridge_idx: int) -> bool:
            if not use_gsm:
                return False
            if not bridge:
                return gsm_on_local
            return (
                gsm_on_bridge
                and bridge_idx >= gsm_bridge_start
                and (bridge_idx - gsm_bridge_start) % gsm_every_n_bridge == 0
            )

        if self.use_phase_layer_gates:
            if self.num_phases <= 0:
                raise ValueError("`config.num_phases` must be a positive int.")
            self.phase_embed = nn.Embedding(self.num_phases, config.hidden_size)
            self.layer_embed = nn.Embedding(config.num_hidden_layers, config.hidden_size)
            self.attn_gate_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            self.mlp_gate_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            nn.init.zeros_(self.attn_gate_proj.weight)
            nn.init.zeros_(self.attn_gate_proj.bias)
            nn.init.zeros_(self.mlp_gate_proj.weight)
            nn.init.zeros_(self.mlp_gate_proj.bias)
        else:
            self.phase_embed = None
            self.layer_embed = None
            self.attn_gate_proj = None
            self.mlp_gate_proj = None

        if not share_all_layers:
            if use_macro_structure:
                self._layer_type = [macro_pattern[i % len(macro_pattern)] for i in range(config.num_hidden_layers)]
                self._layer_is_bridge = [kind == "bridge" for kind in self._layer_type]
                self._needs_4d_causal_mask = any(self._layer_is_bridge)
                layers: list[nn.Module] = []
                bridge_seen = 0
                for layer_idx, kind in enumerate(self._layer_type):
                    if kind == "perceiver":
                        layers.append(BudgiePerceiverMacroLayer(config, layer_idx=layer_idx))
                        continue

                    bridge = kind == "bridge"
                    if bridge:
                        bridge_seen += 1
                        bridge_idx = bridge_seen
                    else:
                        bridge_idx = 0

                    if kind == "swa":
                        attn_impl = "gla_sliding"
                    elif kind == "swa_dilated":
                        attn_impl = "gla_sliding_dilated"
                    elif kind == "bridge":
                        attn_impl = bridge_impl
                    else:
                        raise RuntimeError(f"Unexpected layer type token: {kind!r}.")

                    # Use PKM-augmented SWA block for non-dilated SWA layers if enabled
                    if (
                        kind == "swa"
                        and bool(getattr(config, "use_pkm", False))
                        and bool(getattr(config, "pkm_on_non_dilated_swa", True))
                    ):
                        layers.append(
                            SWABlockWithPKM(
                                config,
                                layer_idx,
                                attn_implementation=attn_impl,
                                num_product_keys=int(getattr(config, "pkm_num_product_keys", 8)),
                                product_key_size=int(getattr(config, "pkm_product_key_size", 32)),
                                value_size=int(getattr(config, "pkm_value_size", 128)),
                                enable_tiny_conv=_layer_tiny_conv(bridge=bridge, bridge_idx=bridge_idx),
                                enable_gsm=_layer_gsm(bridge=bridge, bridge_idx=bridge_idx),
                                is_bridge_layer=bridge,
                            )
                        )
                    else:
                        layers.append(
                            LlamaDecoderLayer(
                                config,
                                layer_idx,
                                attn_implementation=attn_impl,
                                enable_tiny_conv=_layer_tiny_conv(bridge=bridge, bridge_idx=bridge_idx),
                                enable_gsm=_layer_gsm(bridge=bridge, bridge_idx=bridge_idx),
                                is_bridge_layer=bridge,
                            )
                        )
                self.layers = nn.ModuleList(layers)
            elif not use_hybrid_layers:
                self._needs_4d_causal_mask = False
                self.layers = nn.ModuleList(
                    [
                        LlamaDecoderLayer(
                            config,
                            layer_idx,
                            enable_gsm=(use_gsm and gsm_on_local),
                            is_bridge_layer=False,
                        )
                        for layer_idx in range(config.num_hidden_layers)
                    ]
                )
            else:
                self._needs_4d_causal_mask = any(self._layer_is_bridge)
                layers: list[nn.Module] = []
                for layer_idx in range(config.num_hidden_layers):
                    bridge = self._layer_is_bridge[layer_idx]
                    attn_impl = bridge_impl if bridge else local_impl
                    bn = bridge_number(layer_idx) if bridge else 0

                    layers.append(
                        LlamaDecoderLayer(
                            config,
                            layer_idx,
                            attn_implementation=attn_impl,
                            enable_tiny_conv=_layer_tiny_conv(bridge=bridge, bridge_idx=bn),
                            enable_gsm=_layer_gsm(bridge=bridge, bridge_idx=bn),
                            is_bridge_layer=bridge,
                        )
                    )
                self.layers = nn.ModuleList(layers)
        else:
            if use_macro_structure:
                raise ValueError("`use_macro_structure=True` is not supported with `share_all_layers=True` yet.")

            def impl_to_mode(impl: str) -> str:
                if impl == "gla_sliding":
                    return "sliding"
                if impl == "gla_landmark":
                    return "landmark"
                raise ValueError(
                    "With `share_all_layers=True`, only `gla_sliding` and `gla_landmark` are supported as attention implementations."
                )

            local_mode = impl_to_mode(local_impl)
            bridge_mode = impl_to_mode(bridge_impl)

            self._layer_attn_mode: list[str] = [
                (bridge_mode if self._layer_is_bridge[i] else local_mode) for i in range(config.num_hidden_layers)
            ]
            self._needs_4d_causal_mask = any(m == "landmark" for m in self._layer_attn_mode)

            self.shared_attn = BudgieGLASharedHybrid(config=config, layer_idx=0)
            self.shared_mlp = budgie_make_mlp(config)

            scaffolds: list[nn.Module] = []
            for layer_idx in range(config.num_hidden_layers):
                bridge = self._layer_is_bridge[layer_idx]
                bn = bridge_number(layer_idx) if bridge else 0
                scaffolds.append(
                    BudgieLayerScaffold(
                        config,
                        layer_idx=layer_idx,
                        enable_tiny_conv=_layer_tiny_conv(bridge=bridge, bridge_idx=bn),
                        enable_gsm=_layer_gsm(bridge=bridge, bridge_idx=bn),
                        is_bridge_layer=bridge,
                    )
                )

            self.layers = nn.ModuleList(scaffolds)
        self.norm = budgie_make_rmsnorm(config, config.hidden_size, config.rms_norm_eps)

        original_head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        self.config.head_dim = self.config.qk_rope_dim
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
        self.config.head_dim = original_head_dim

        self.gradient_checkpointing = False

        self.post_init()

    def _get_phase_layer_gates(
        self, *, phase_idx: int, layer_idx: int, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_phase_layer_gates:
            ones = hidden_states.new_ones((1, 1, hidden_states.shape[-1]))
            return ones, ones

        cond = self.phase_embed(torch.tensor(phase_idx, device=hidden_states.device)) + self.layer_embed(
            torch.tensor(layer_idx, device=hidden_states.device)
        )
        attn_gate = 1.0 + torch.tanh(self.attn_gate_proj(cond)).to(dtype=hidden_states.dtype)
        mlp_gate = 1.0 + torch.tanh(self.mlp_gate_proj(cond)).to(dtype=hidden_states.dtype)
        return attn_gate.view(1, 1, -1), mlp_gate.view(1, 1, -1)

    def _shared_layer_forward(
        self,
        scaffold: BudgieLayerScaffold,
        phase_idx: int,
        layer_idx: int,
        virtual_layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[Cache],
        output_attentions: bool,
        use_cache: bool,
        cache_position: Optional[torch.LongTensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        *,
        landmark_mask: Optional[torch.Tensor],
        attention_mask_2d: Optional[torch.Tensor],
    ):
        residual = hidden_states
        hs = scaffold.input_layernorm(hidden_states)
        if scaffold.bridge_mixer_fusion and self._layer_attn_mode[layer_idx] == "landmark":
            z = hs
            if scaffold.tiny_conv is not None:
                z = scaffold.tiny_conv(
                    z,
                    use_cache=bool(use_cache),
                    past_key_value=past_key_values,
                    layer_idx=virtual_layer_idx,
                    attention_mask_2d=attention_mask_2d,
                )
        else:
            if scaffold.tiny_conv is not None:
                hs = hs + scaffold.tiny_conv(
                    hs,
                    use_cache=bool(use_cache),
                    past_key_value=past_key_values,
                    layer_idx=virtual_layer_idx,
                    attention_mask_2d=attention_mask_2d,
                )
            z = hs

        attn_gate, mlp_gate = self._get_phase_layer_gates(phase_idx=phase_idx, layer_idx=layer_idx, hidden_states=z)

        attn_out, attn_weights, present = self.shared_attn(
            hidden_states=z,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            landmark_mask=landmark_mask,
            attention_mask_2d=attention_mask_2d,
            layer_idx=virtual_layer_idx,
            attn_mode=self._layer_attn_mode[layer_idx],
        )

        if scaffold.bridge_mixer_fusion and self._layer_attn_mode[layer_idx] == "landmark":
            gsm_out = scaffold.gsm(z)
            alpha = torch.sigmoid(scaffold.bridge_alpha_proj(z.mean(dim=1))).view(z.shape[0], 1, 1)
            mixed = (alpha * attn_out) + ((1.0 - alpha) * gsm_out)
            attn_out = scaffold.bridge_out_proj(mixed)
        elif scaffold.gsm is not None:
            attn_out = attn_out + scaffold.gsm(z)

        hidden_states = residual + (attn_out * attn_gate)
        residual = hidden_states

        hs = scaffold.post_attention_layernorm(hidden_states)
        hs = self.shared_mlp(hs)
        hidden_states = residual + (hs * mlp_gate)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present,)
        return outputs

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of `input_ids` or `inputs_embeds`.")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        landmark_mask = None
        landmark_token_id = getattr(self.config, "landmark_token_id", None)
        if landmark_token_id is not None and input_ids is not None:
            landmark_mask = input_ids.eq(int(landmark_token_id))

        return_legacy_cache = False
        if self.num_phases > 1:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("`StaticCache` is not supported when `num_phases > 1`. Use `DynamicCache` instead.")
            if past_key_values is not None and not isinstance(past_key_values, Cache):
                raise ValueError(
                    "Legacy `past_key_values` (tuple/list) is not supported when `num_phases > 1`. "
                    "Pass a `transformers.cache_utils.Cache` instance (e.g. `DynamicCache`)."
                )
            if use_cache:
                logger.warning_once(
                    "Multi-phase KV-cache enabled (`num_phases > 1`): cache memory scales with `num_phases`."
                )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if past_key_values is not None and not isinstance(past_key_values, Cache) and not self.training:
            # Back-compat for `num_phases==1`: accept legacy tuples/lists and convert to a Cache.
            return_legacy_cache = bool(use_cache)
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "Detected `past_key_values` as a legacy tuple. Prefer passing a `transformers.cache_utils.Cache` instance."
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        num_layers = int(getattr(self.config, "num_hidden_layers", len(self.layers)))
        if num_layers != len(self.layers):
            raise RuntimeError(
                f"Unexpected layer count mismatch: config.num_hidden_layers={num_layers}, len(self.layers)={len(self.layers)}."
            )

        if bool(getattr(self.config, "share_all_layers", False)):
            for phase_idx in range(self.num_phases):
                for layer_idx, scaffold in enumerate(self.layers):
                    virtual_layer_idx = (phase_idx * num_layers) + layer_idx
                    if output_hidden_states:
                        all_hidden_states += (hidden_states,)

                    if self.gradient_checkpointing and self.training:

                        def create_custom_forward(scaffold, phase_idx, layer_idx, virtual_layer_idx):
                            def custom_forward(*inputs):
                                return self._shared_layer_forward(
                                    scaffold,
                                    phase_idx,
                                    layer_idx,
                                    virtual_layer_idx,
                                    *inputs,
                                    landmark_mask=landmark_mask,
                                    attention_mask_2d=attention_mask,
                                )

                            return custom_forward

                        layer_outputs = self._gradient_checkpointing_func(
                            create_custom_forward(scaffold, phase_idx, layer_idx, virtual_layer_idx),
                            hidden_states,
                            causal_mask,
                            position_ids,
                            past_key_values,
                            output_attentions,
                            use_cache,
                            cache_position,
                            position_embeddings,
                        )
                    else:
                        layer_outputs = self._shared_layer_forward(
                            scaffold,
                            phase_idx,
                            layer_idx,
                            virtual_layer_idx,
                            hidden_states,
                            causal_mask,
                            position_ids,
                            past_key_values,
                            output_attentions,
                            use_cache,
                            cache_position,
                            position_embeddings,
                            landmark_mask=landmark_mask,
                            attention_mask_2d=attention_mask,
                        )

                    hidden_states = layer_outputs[0]

                    if use_cache:
                        next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                    if output_attentions:
                        all_self_attns += (layer_outputs[1],)
        else:
            for phase_idx in range(self.num_phases):
                for layer_idx, decoder_layer in enumerate(self.layers):
                    virtual_layer_idx = (phase_idx * num_layers) + layer_idx
                    if output_hidden_states:
                        all_hidden_states += (hidden_states,)

                    if self.gradient_checkpointing and self.training:

                        def create_custom_forward(module, phase_idx, layer_idx, virtual_layer_idx):
                            def custom_forward(*inputs):
                                hs_in = inputs[0]
                                attn_gate, mlp_gate = self._get_phase_layer_gates(
                                    phase_idx=phase_idx, layer_idx=layer_idx, hidden_states=hs_in
                                )
                                return module(
                                    *inputs,
                                    attn_gate=attn_gate,
                                    mlp_gate=mlp_gate,
                                    layer_idx=virtual_layer_idx,
                                    landmark_mask=landmark_mask,
                                    attention_mask_2d=attention_mask,
                                )

                            return custom_forward

                        layer_outputs = self._gradient_checkpointing_func(
                            create_custom_forward(decoder_layer, phase_idx, layer_idx, virtual_layer_idx),
                            hidden_states,
                            causal_mask,
                            position_ids,
                            past_key_values,
                            output_attentions,
                            use_cache,
                            cache_position,
                            position_embeddings,
                        )
                    else:
                        attn_gate, mlp_gate = self._get_phase_layer_gates(
                            phase_idx=phase_idx, layer_idx=layer_idx, hidden_states=hidden_states
                        )
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            attn_gate=attn_gate,
                            mlp_gate=mlp_gate,
                            layer_idx=virtual_layer_idx,
                            landmark_mask=landmark_mask,
                            attention_mask_2d=attention_mask,
                        )

                    hidden_states = layer_outputs[0]

                    if use_cache:
                        next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                    if output_attentions:
                        all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor | None,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache | None,
        output_attentions: bool,
    ):
        if not getattr(self, "_needs_4d_causal_mask", False) and self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if getattr(self, "_needs_4d_causal_mask", False):
                # Bridge layers (e.g. landmark attention) need an explicit additive 4D mask.
                pass
            else:
                # When decoding with a KV-cache (`past_seen_tokens>0`), SDPA's `is_causal=True` path may not be
                # sufficient across PyTorch versions when `q_len != k_len`. In that case, prefer an explicit
                # additive causal mask that accounts for `cache_position`.
                if past_seen_tokens == 0 and AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    is_training=self.training,
                ):
                    return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


__all__ = ["BudgieModel"]

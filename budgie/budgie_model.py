from __future__ import annotations

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
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    _prepare_4d_causal_attention_mask_with_cache_position,
)

logger = logging.get_logger(__name__)


class BudgieModel(BudgiePreTrainedModel):
    def __init__(self, config: BudgieConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        use_hybrid_layers = bool(getattr(config, "use_hybrid_layers", False))
        if not use_hybrid_layers:
            self._needs_4d_causal_mask = False
            self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        else:
            local_impl = str(getattr(config, "local_attn_implementation", "gla_sliding"))
            bridge_impl = str(getattr(config, "bridge_attn_implementation", "gla_landmark"))
            bridge_every = int(getattr(config, "bridge_every_n_layers", 6))
            bridge_offset = int(getattr(config, "bridge_layer_offset", 5))

            tiny_conv_on_local = bool(getattr(config, "tiny_conv_on_local_layers", False))
            tiny_conv_on_bridge = bool(getattr(config, "tiny_conv_on_bridge_layers", True))
            tiny_conv_every_n_bridge = int(getattr(config, "tiny_conv_every_n_bridge_layers", 2))
            tiny_conv_bridge_start = int(getattr(config, "tiny_conv_bridge_start", 2))

            def is_bridge(layer_idx: int) -> bool:
                return layer_idx >= bridge_offset and (layer_idx - bridge_offset) % bridge_every == 0

            def bridge_number(layer_idx: int) -> int:
                return ((layer_idx - bridge_offset) // bridge_every) + 1

            self._needs_4d_causal_mask = any(is_bridge(i) for i in range(config.num_hidden_layers))

            layers: list[nn.Module] = []
            for layer_idx in range(config.num_hidden_layers):
                bridge = is_bridge(layer_idx)
                attn_impl = bridge_impl if bridge else local_impl

                if not bool(getattr(config, "use_tiny_conv", False)):
                    enable_tiny_conv = False
                elif not bridge:
                    enable_tiny_conv = tiny_conv_on_local
                else:
                    bn = bridge_number(layer_idx)
                    enable_tiny_conv = (
                        tiny_conv_on_bridge
                        and bn >= tiny_conv_bridge_start
                        and (bn - tiny_conv_bridge_start) % tiny_conv_every_n_bridge == 0
                    )

                layers.append(
                    LlamaDecoderLayer(
                        config,
                        layer_idx,
                        attn_implementation=attn_impl,
                        enable_tiny_conv=enable_tiny_conv,
                    )
                )

            self.layers = nn.ModuleList(layers)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        original_head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        self.config.head_dim = self.config.qk_rope_dim
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
        self.config.head_dim = original_head_dim

        self.gradient_checkpointing = False

        self.post_init()

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
        if use_cache and not isinstance(past_key_values, Cache) and not self.training:
            return_legacy_cache = True
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

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, landmark_mask=landmark_mask, attention_mask_2d=attention_mask)

                    return custom_forward

                layer_outputs = self._gradient_checkpointing_func(
                    create_custom_forward(decoder_layer),
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
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
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
                if AttentionMaskConverter._ignore_causal_mask_sdpa(
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

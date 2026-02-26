from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.cache_utils import Cache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import is_torchdynamo_compiling, logging

from .budgie_config import BudgieConfig
from .budgie_model import BudgieModel
from .budgie_pretrained_model import BudgiePreTrainedModel
from .modeling_budgie_GLA import (
    _LIGER_CE_LOSS_CLS,
    _LIGER_FUSED_LINEAR_CE_LOSS_CLS,
    _prepare_4d_causal_attention_mask_with_cache_position,
)

logger = logging.get_logger(__name__)


class BudgieForCausalLM(BudgiePreTrainedModel, GenerationMixin):
    config_class = BudgieConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: BudgieConfig):
        super().__init__(config)
        self.model = BudgieModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._liger_ce_loss_fn = None
        self._liger_fused_linear_ce_loss_fn = None
        self._liger_ce_disabled = False
        self._liger_fused_linear_ce_disabled = False

        self.post_init()
        self.tie_weights()

    def tie_weights(self):
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        if getattr(self.config, "tie_word_embeddings", False):
            return self.model.embed_tokens
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        if getattr(self.config, "tie_word_embeddings", False):
            self.model.embed_tokens = new_embeddings
        else:
            self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        use_liger_kernel = bool(getattr(self.config, "use_liger_kernel", False))

        loss = None
        used_fused_linear_ce = False
        if labels is not None and use_liger_kernel and not self._liger_fused_linear_ce_disabled:
            fused_cls = _LIGER_FUSED_LINEAR_CE_LOSS_CLS
            if fused_cls is not None and int(getattr(self.config, "pretraining_tp", 1)) == 1:
                if self._liger_fused_linear_ce_loss_fn is None:
                    try:  # pragma: no cover
                        if isinstance(fused_cls, type):
                            try:
                                self._liger_fused_linear_ce_loss_fn = fused_cls(ignore_index=-100)
                            except TypeError:
                                self._liger_fused_linear_ce_loss_fn = fused_cls()
                        else:
                            self._liger_fused_linear_ce_loss_fn = fused_cls
                    except Exception as exc:
                        self._liger_fused_linear_ce_disabled = True
                        logger.warning_once(
                            f"Failed to initialize LigerFusedLinearCrossEntropyLoss; falling back. Error: {exc}"
                        )

                if self._liger_fused_linear_ce_loss_fn is not None and not self._liger_fused_linear_ce_disabled:
                    hs = hidden_states[:, :-1, :].contiguous().reshape(-1, hidden_states.shape[-1])
                    tgt = labels[:, 1:].contiguous().reshape(-1)
                    w = self.lm_head.weight
                    loss_fn = self._liger_fused_linear_ce_loss_fn
                    try:  # pragma: no cover
                        # Liger fused CE expects flattened hidden states + labels.
                        # Preferred signature is (weight, hidden_states, target).
                        try:
                            loss = loss_fn(w, hs, tgt)
                        except TypeError:
                            loss = loss_fn(hs, w, tgt)
                    except Exception as exc:
                        self._liger_fused_linear_ce_disabled = True
                        logger.warning_once(
                            f"LigerFusedLinearCrossEntropyLoss failed at runtime; falling back. Error: {exc}"
                        )
                        loss = None
                    else:
                        used_fused_linear_ce = True

        # Compute logits (optionally detached under fused CE during training).
        def _compute_logits() -> torch.Tensor:
            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                return torch.cat(logits, dim=-1)
            if labels is None and not is_torchdynamo_compiling():
                logger.warning_once(
                    "Starting from v4.46, the `logits` model output will have the same type as the model (except at "
                    "train time, where it will always be FP32)"
                )
            return self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        if labels is not None and used_fused_linear_ce and self.training:
            with torch.no_grad():
                logits = _compute_logits()
        else:
            logits = _compute_logits()

        if labels is not None and loss is None and use_liger_kernel and not self._liger_ce_disabled:
            ce_cls = _LIGER_CE_LOSS_CLS
            if ce_cls is not None:
                if self._liger_ce_loss_fn is None:
                    try:  # pragma: no cover
                        if isinstance(ce_cls, type):
                            try:
                                self._liger_ce_loss_fn = ce_cls(ignore_index=-100)
                            except TypeError:
                                self._liger_ce_loss_fn = ce_cls()
                        else:
                            self._liger_ce_loss_fn = ce_cls
                    except Exception as exc:
                        self._liger_ce_disabled = True
                        logger.warning_once(f"Failed to initialize LigerCrossEntropyLoss; falling back. Error: {exc}")

                if self._liger_ce_loss_fn is not None and not self._liger_ce_disabled:
                    try:  # pragma: no cover
                        shift_logits = logits.float()[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
                        shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
                        loss = self._liger_ce_loss_fn(shift_logits, shift_labels)
                    except Exception as exc:
                        self._liger_ce_disabled = True
                        logger.warning_once(f"LigerCrossEntropyLoss failed at runtime; falling back. Error: {exc}")
                        loss = None

        if labels is not None and loss is None:
            shift_logits = logits.float()[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if inputs_embeds is not None:
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask is not None and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        if past_key_values is None:
            return past_key_values

        reordered = past_key_values
        if hasattr(past_key_values, "reorder_cache"):
            maybe = past_key_values.reorder_cache(beam_idx)
            if maybe is not None:
                reordered = maybe
        else:
            try:
                reordered = super()._reorder_cache(past_key_values, beam_idx)
            except Exception:
                reordered = past_key_values

        conv_state = getattr(reordered, "_budgie_conv_state", None)
        if isinstance(conv_state, dict):
            for layer_idx, state in list(conv_state.items()):
                if state is not None:
                    conv_state[layer_idx] = state.index_select(0, beam_idx)

        return reordered


__all__ = ["BudgieForCausalLM"]

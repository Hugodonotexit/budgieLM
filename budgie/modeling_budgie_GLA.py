# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import importlib
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import logging
from transformers.models.llama.configuration_llama import LlamaConfig

from .modeling_budgie_gsm import GSM

logger = logging.get_logger(__name__)


_flash_attn_import_error = None
try:
    from hopper.flash_attn_interface import flash_attn_func
except Exception as exc:  # pragma: no cover
    _flash_attn_import_error = exc
    flash_attn_func = None

_xformers_import_error = None
_xformers_memory_efficient_attention = None
_xformers_LowerTriangularMask = None
_xformers_LocalAttentionFromBottomRightMask = None
try:  # pragma: no cover
    import xformers.ops as _xops

    _xformers_memory_efficient_attention = getattr(_xops, "memory_efficient_attention", None)
    try:
        from xformers.ops.fmha.attn_bias import LowerTriangularMask as _xformers_LowerTriangularMask
    except Exception:
        _xformers_LowerTriangularMask = None
    try:
        from xformers.ops.fmha.attn_bias import (
            LocalAttentionFromBottomRightMask as _xformers_LocalAttentionFromBottomRightMask,
        )
    except Exception:
        _xformers_LocalAttentionFromBottomRightMask = None
except Exception as exc:
    _xformers_import_error = exc
    _xops = None

_causal_conv1d_import_error = None
causal_conv1d_fn = None
causal_conv1d_update = None
try:  # pragma: no cover
    from causal_conv1d import causal_conv1d_fn as causal_conv1d_fn
    from causal_conv1d import causal_conv1d_update as causal_conv1d_update
except Exception as exc:  # pragma: no cover
    _causal_conv1d_import_error = exc
    causal_conv1d_fn = None
    causal_conv1d_update = None

_liger_import_error = None
_LIGER_RMSNORM_CLS = None
_LIGER_SWIGLU_MLP_CLS = None
_liger_rotary_pos_emb = None
_LIGER_SOFTMAX = None
_LIGER_CE_LOSS_CLS = None
_LIGER_FUSED_LINEAR_CE_LOSS_CLS = None
_liger_experimental_import_error = None
_LIGER_EMBEDDING_CLS = None
for _liger_mod_name in (
    "liger_kernel.transformers",
    "liger_kernel.transformers.layers",
    "liger_kernel.transformers.modeling",
):
    try:  # pragma: no cover
        _liger_mod = importlib.import_module(_liger_mod_name)
    except Exception as exc:
        _liger_import_error = exc
        continue

    if _LIGER_RMSNORM_CLS is None:
        _LIGER_RMSNORM_CLS = getattr(_liger_mod, "LigerRMSNorm", None)
    if _LIGER_SWIGLU_MLP_CLS is None:
        _LIGER_SWIGLU_MLP_CLS = getattr(_liger_mod, "LigerSwiGLUMLP", None)
    if _liger_rotary_pos_emb is None:
        _liger_rotary_pos_emb = getattr(_liger_mod, "liger_rotary_pos_emb", None)
    if _LIGER_SOFTMAX is None:
        _LIGER_SOFTMAX = getattr(_liger_mod, "LigerSoftmax", None)
    if _LIGER_CE_LOSS_CLS is None:
        _LIGER_CE_LOSS_CLS = getattr(_liger_mod, "LigerCrossEntropyLoss", None)
    if _LIGER_FUSED_LINEAR_CE_LOSS_CLS is None:
        _LIGER_FUSED_LINEAR_CE_LOSS_CLS = getattr(_liger_mod, "LigerFusedLinearCrossEntropyLoss", None)

    if (
        _LIGER_RMSNORM_CLS is not None
        and _LIGER_SWIGLU_MLP_CLS is not None
        and _liger_rotary_pos_emb is not None
        and _LIGER_SOFTMAX is not None
        and _LIGER_CE_LOSS_CLS is not None
        and _LIGER_FUSED_LINEAR_CE_LOSS_CLS is not None
    ):
        break

try:  # pragma: no cover
    _liger_exp_mod = importlib.import_module("liger_kernel.transformers.experimental")
    _LIGER_EMBEDDING_CLS = getattr(_liger_exp_mod, "LigerEmbedding", None)
except Exception as exc:  # pragma: no cover
    _liger_experimental_import_error = exc

if _LIGER_RMSNORM_CLS is not None:  # pragma: no cover
    ALL_LAYERNORM_LAYERS.append(_LIGER_RMSNORM_CLS)


class BudgieCausalDepthwiseConv1d(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 3,
        bias: bool = False,
        init_zero: bool = True,
        use_causal_conv1d: bool = True,
    ):
        super().__init__()
        if kernel_size <= 0 or not isinstance(kernel_size, int):
            raise ValueError(f"`kernel_size` must be a positive int, got {kernel_size!r}.")
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.use_causal_conv1d = bool(use_causal_conv1d)
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=0,
            groups=hidden_size,
            bias=bias,
        )
        if init_zero:
            nn.init.zeros_(self.conv.weight)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, S, H)
        *,
        use_cache: bool = False,
        past_key_value: Optional[Cache] = None,
        layer_idx: Optional[int] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,  # (B, S) with 1=keep,0=pad
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"`hidden_states` must be 3D (B, S, H), got shape {tuple(hidden_states.shape)}.")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"`hidden_states` last dim must be hidden_size={self.hidden_size}, got {hidden_states.shape[-1]}."
            )

        bsz, q_len, _ = hidden_states.shape
        k = self.kernel_size

        def _conv_causal_padded(sequence_bsh: torch.Tensor, *, take_last: int) -> torch.Tensor:
            x = sequence_bsh.transpose(1, 2)  # (B, H, L)
            if k > 1:
                x = F.pad(x, (k - 1, 0))
            y = self.conv(x).transpose(1, 2)  # (B, L, H)
            return y[:, -take_last:, :]

        def _conv_no_pad(context_bsh: torch.Tensor, *, take_last: int) -> torch.Tensor:
            x = context_bsh.transpose(1, 2)  # (B, H, L)
            y = self.conv(x).transpose(1, 2)  # (B, L-k+1, H)
            return y[:, -take_last:, :]

        def _causal_conv1d_full(context_bsh: torch.Tensor, *, take_last: int) -> torch.Tensor:
            x = context_bsh.transpose(1, 2).contiguous()  # (B, H, L)
            weight = self.conv.weight.squeeze(1).contiguous()  # (H, K)
            bias = self.conv.bias
            if causal_conv1d_fn is None:
                return _conv_causal_padded(context_bsh, take_last=take_last)
            y = causal_conv1d_fn(x, weight, bias, None)  # (B, H, L)
            return y.transpose(1, 2)[:, -take_last:, :].contiguous()  # (B, take_last, H)

        if use_cache and past_key_value is not None and k > 1:
            if layer_idx is None:
                raise ValueError("`layer_idx` is required when `use_cache=True` for tiny conv state caching.")

            conv_state = getattr(past_key_value, "_budgie_conv_state", None)
            if conv_state is None:
                conv_state = {}
                setattr(past_key_value, "_budgie_conv_state", conv_state)

            prev = conv_state.get(layer_idx)
            if prev is None:
                prev = hidden_states.new_zeros((bsz, self.hidden_size, k - 1))

            if prev.shape != (bsz, self.hidden_size, k - 1):
                raise ValueError(
                    f"Unexpected conv cache state shape for layer {layer_idx}: expected {(bsz, self.hidden_size, k - 1)}, got {tuple(prev.shape)}."
                )

            if (
                self.use_causal_conv1d
                and causal_conv1d_update is not None
                and hidden_states.shape[1] == 1
            ):
                x_t = hidden_states[:, 0, :].contiguous()  # (B, H)
                weight = self.conv.weight.squeeze(1).contiguous()  # (H, K)
                bias = self.conv.bias
                out = causal_conv1d_update(x_t, prev, weight, bias, None)
                if isinstance(out, tuple):
                    y_t, new_prev = out[0], out[1]
                else:
                    y_t, new_prev = out, prev
                y = y_t[:, None, :]

                if new_prev.shape != (bsz, self.hidden_size, k - 1):
                    raise ValueError(
                        f"`causal_conv1d_update` returned/updated an unexpected state shape {tuple(new_prev.shape)}; expected {(bsz, self.hidden_size, k - 1)}."
                    )
            else:
                prev_bsh = prev.transpose(1, 2)  # (B, k-1, H)
                context = torch.cat([prev_bsh, hidden_states], dim=1)  # (B, k-1+S, H)
                if self.use_causal_conv1d and causal_conv1d_fn is not None:
                    y = _causal_conv1d_full(context, take_last=q_len)
                else:
                    y = _conv_no_pad(context, take_last=q_len)

                context_tail = context[:, -(k - 1) :, :]  # (B, k-1, H)
                new_prev = context_tail.transpose(1, 2).contiguous()  # (B, H, k-1)
            if not self.training:
                new_prev = new_prev.detach()
            conv_state[layer_idx] = new_prev
        else:
            if self.use_causal_conv1d and causal_conv1d_fn is not None and k > 1:
                y = _causal_conv1d_full(hidden_states, take_last=q_len)
            else:
                # Includes k==1 (pointwise) or missing kernel.
                y = _conv_causal_padded(hidden_states, take_last=q_len)

        if attention_mask_2d is not None and attention_mask_2d.dim() == 2:
            if attention_mask_2d.shape[1] != y.shape[1]:
                if attention_mask_2d.shape[1] < y.shape[1]:
                    raise ValueError(
                        f"`attention_mask_2d` sequence length must be >= output length {y.shape[1]}, got {attention_mask_2d.shape[1]}."
                    )
                attention_mask_2d = attention_mask_2d[:, -y.shape[1] :]
            y = y * attention_mask_2d[:, :, None].to(dtype=y.dtype)

        return y




def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask



class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states.to(input_dtype))

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)




def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


_LIGER_ROPE_VARIANT: dict[int, str | None] = {}


def budgie_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
    use_liger_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (use_liger_kernel and _liger_rotary_pos_emb is not None):
        return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    variant = _LIGER_ROPE_VARIANT.get(int(unsqueeze_dim), "__probe__")
    if variant is None:
        return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    rotary_dim = min(q.shape[-1], k.shape[-1], cos.shape[-1], sin.shape[-1])
    q_rot, k_rot = q[..., :rotary_dim], k[..., :rotary_dim]
    cos_rot, sin_rot = cos[..., :rotary_dim], sin[..., :rotary_dim]

    # `liger_rotary_pos_emb` expects Q/K in (B, H, S, D) layout. When callers provide (B, S, H, D)
    # (i.e. `unsqueeze_dim=2`), transpose into the expected layout before invoking Liger, then transpose back.
    if q_rot.ndim != 4 or k_rot.ndim != 4:
        return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
    if int(unsqueeze_dim) == 1:
        q_in, k_in = q_rot, k_rot
        def _postprocess(q_out: torch.Tensor, k_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return q_out, k_out
    elif int(unsqueeze_dim) == 2:
        q_in, k_in = q_rot.transpose(1, 2), k_rot.transpose(1, 2)
        def _postprocess(q_out: torch.Tensor, k_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return q_out.transpose(1, 2), k_out.transpose(1, 2)
    else:
        # Unknown layout for Liger; fall back to the reference implementation.
        return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    liger_unsqueeze_dim = 1  # Q/K are passed to Liger in (B, H, S, D) layout.

    def run_variant(name: str) -> tuple[torch.Tensor, torch.Tensor]:
        if name == "kw_unsqueeze_dim":
            q_out, k_out = _liger_rotary_pos_emb(q_in, k_in, cos_rot, sin_rot, unsqueeze_dim=liger_unsqueeze_dim)
            return _postprocess(q_out, k_out)
        if name == "pos_args":
            q_out, k_out = _liger_rotary_pos_emb(q_in, k_in, cos_rot, sin_rot)
            return _postprocess(q_out, k_out)
        if name == "pos_args_unsqueezed":
            q_out, k_out = _liger_rotary_pos_emb(
                q_in,
                k_in,
                cos_rot.unsqueeze(liger_unsqueeze_dim),
                sin_rot.unsqueeze(liger_unsqueeze_dim),
            )
            return _postprocess(q_out, k_out)
        raise ValueError(f"Unknown Liger RoPE variant {name!r}.")

    try:  # pragma: no cover
        if variant == "__probe__":
            for name in ("kw_unsqueeze_dim", "pos_args", "pos_args_unsqueezed"):
                try:
                    q_embed, k_embed = run_variant(name)
                except TypeError:
                    continue
                except Exception:
                    continue
                if q_embed.shape == q_rot.shape and k_embed.shape == k_rot.shape:
                    _LIGER_ROPE_VARIANT[int(unsqueeze_dim)] = name
                    break
            else:
                _LIGER_ROPE_VARIANT[int(unsqueeze_dim)] = None
                return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        else:
            q_embed, k_embed = run_variant(variant)
    except Exception as exc:
        _LIGER_ROPE_VARIANT[int(unsqueeze_dim)] = None
        logger.warning_once(f"Failed to use liger_rotary_pos_emb; falling back to torch RoPE. Error: {exc}")
        return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    if q.shape[-1] > rotary_dim:
        q_embed = torch.cat([q_embed, q[..., rotary_dim:]], dim=-1)
    if k.shape[-1] > rotary_dim:
        k_embed = torch.cat([k_embed, k[..., rotary_dim:]], dim=-1)
    return q_embed, k_embed


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # Some architectures apply RoPE only to a subset of the head dimension (e.g. `qk_rope_dim`), while other parts of
    # Q/K remain unrotated. In those cases, `cos/sin` can be larger than the incoming `q/k` slice (or vice-versa, if a
    # slice larger than RoPE dim is passed). Be defensive and apply RoPE to the shared prefix.
    if cos.shape[-1] != q.shape[-1] or sin.shape[-1] != q.shape[-1]:
        rotary_dim = min(q.shape[-1], k.shape[-1], cos.shape[-1], sin.shape[-1])
        q_rot, k_rot = q[..., :rotary_dim], k[..., :rotary_dim]
        cos = cos[..., :rotary_dim]
        sin = sin[..., :rotary_dim]

        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

        if q.shape[-1] > rotary_dim:
            q_embed = torch.cat([q_embed, q[..., rotary_dim:]], dim=-1)
        if k.shape[-1] > rotary_dim:
            k_embed = torch.cat([k_embed, k[..., rotary_dim:]], dim=-1)
        return q_embed, k_embed

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def budgie_make_rmsnorm(config, hidden_size: int, eps: float):
    if bool(getattr(config, "use_liger_kernel", False)) and _LIGER_RMSNORM_CLS is not None:
        try:  # pragma: no cover
            return _LIGER_RMSNORM_CLS(hidden_size, eps=eps)
        except TypeError:
            try:
                return _LIGER_RMSNORM_CLS(hidden_size, eps)
            except Exception as exc:
                logger.warning_once(f"Failed to create Liger RMSNorm; falling back to LlamaRMSNorm. Error: {exc}")
        except Exception as exc:
            logger.warning_once(f"Failed to create Liger RMSNorm; falling back to LlamaRMSNorm. Error: {exc}")
    return LlamaRMSNorm(hidden_size, eps=eps)


def budgie_make_mlp(config):
    if bool(getattr(config, "use_liger_kernel", False)) and _LIGER_SWIGLU_MLP_CLS is not None:
        if int(getattr(config, "pretraining_tp", 1)) != 1:
            logger.warning_once("Liger MLP backend is disabled when `config.pretraining_tp > 1`; using LlamaMLP.")
        else:
            try:  # pragma: no cover
                return _LIGER_SWIGLU_MLP_CLS(config)
            except TypeError:
                try:
                    return _LIGER_SWIGLU_MLP_CLS(
                        hidden_size=int(config.hidden_size),
                        intermediate_size=int(config.intermediate_size),
                        bias=bool(getattr(config, "mlp_bias", False)),
                    )
                except TypeError:
                    try:
                        return _LIGER_SWIGLU_MLP_CLS(
                            int(config.hidden_size),
                            int(config.intermediate_size),
                            bool(getattr(config, "mlp_bias", False)),
                        )
                    except Exception as exc:
                        logger.warning_once(f"Failed to create Liger MLP; falling back to LlamaMLP. Error: {exc}")
                except Exception as exc:
                    logger.warning_once(f"Failed to create Liger MLP; falling back to LlamaMLP. Error: {exc}")
            except Exception as exc:
                logger.warning_once(f"Failed to create Liger MLP; falling back to LlamaMLP. Error: {exc}")
    return LlamaMLP(config)


def budgie_make_embedding(config, vocab_size: int, hidden_size: int, padding_idx: int):
    if bool(getattr(config, "use_liger_kernel", False)) and _LIGER_EMBEDDING_CLS is not None:
        try:  # pragma: no cover
            emb = _LIGER_EMBEDDING_CLS(vocab_size, hidden_size, padding_idx=padding_idx)
        except TypeError:
            try:
                emb = _LIGER_EMBEDDING_CLS(vocab_size, hidden_size, padding_idx)
            except TypeError:
                try:
                    emb = _LIGER_EMBEDDING_CLS(vocab_size, hidden_size)
                except Exception as exc:
                    logger.warning_once(f"Failed to create LigerEmbedding; falling back to nn.Embedding. Error: {exc}")
                    emb = None
            except Exception as exc:
                logger.warning_once(f"Failed to create LigerEmbedding; falling back to nn.Embedding. Error: {exc}")
                emb = None
        except Exception as exc:
            logger.warning_once(f"Failed to create LigerEmbedding; falling back to nn.Embedding. Error: {exc}")
            emb = None

        if emb is not None and hasattr(emb, "weight"):
            return emb

    return nn.Embedding(vocab_size, hidden_size, padding_idx)


_LIGER_SOFTMAX_FN = None
_LIGER_SOFTMAX_DISABLED = False


def budgie_softmax(scores: torch.Tensor, *, use_liger_kernel: bool, dim: int = -1, dtype: torch.dtype = torch.float32):
    global _LIGER_SOFTMAX_FN, _LIGER_SOFTMAX_DISABLED
    if not (use_liger_kernel and _LIGER_SOFTMAX is not None and dim == -1) or _LIGER_SOFTMAX_DISABLED:
        return torch.softmax(scores, dim=dim, dtype=dtype)
    if _LIGER_SOFTMAX_FN is None:
        try:
            _LIGER_SOFTMAX_FN = _LIGER_SOFTMAX() if isinstance(_LIGER_SOFTMAX, type) else _LIGER_SOFTMAX
        except Exception as exc:
            _LIGER_SOFTMAX_DISABLED = True
            logger.warning_once(f"Failed to initialize LigerSoftmax; falling back to torch.softmax. Error: {exc}")
            return torch.softmax(scores, dim=dim, dtype=dtype)

    try:  # pragma: no cover
        # Prefer float32 softmax (common for stability), but fall back if the kernel doesn't support it.
        try:
            out = _LIGER_SOFTMAX_FN(scores.float())
            return out.to(dtype=scores.dtype)
        except Exception:
            return _LIGER_SOFTMAX_FN(scores)
    except Exception as exc:
        _LIGER_SOFTMAX_DISABLED = True
        logger.warning_once(f"Failed to run LigerSoftmax; falling back to torch.softmax. Error: {exc}")
        return torch.softmax(scores, dim=dim, dtype=dtype)


def _budgie_xformers_attention(
    *,
    query_states: torch.Tensor,  # (B, S_q, H, D)
    key_states: torch.Tensor,  # (B, S_k, H, D)
    value_states: torch.Tensor,  # (B, S_k, H, D)
    attention_mask_2d: Optional[torch.Tensor],  # (B, S_k) with 1=keep,0=pad
    dropout_p: float,
    is_causal: bool,
    sliding_window: Optional[int] = None,
    sliding_dilation: int = 1,
    allow_xformers_op: bool = True,
    use_liger_kernel: bool = False,
) -> torch.Tensor:
    q = query_states.contiguous()
    k = key_states.contiguous()
    v = value_states.contiguous()

    bsz, q_len, nheads, _ = q.shape
    k_len = k.shape[1]
    qk_dim = q.shape[-1]

    if sliding_window is not None and isinstance(sliding_window, int) and sliding_window <= 0:
        raise ValueError("`sliding_window` must be a positive int when provided.")
    if not isinstance(sliding_dilation, int) or sliding_dilation <= 0:
        raise ValueError(f"`sliding_dilation` must be a positive int, got {sliding_dilation!r}.")

    # Many xFormers CUDA kernels have strict requirements (dtype, head dim multiple-of-8, etc.).
    # If we can confidently predict that dispatch will fail, skip straight to the eager fallback
    # to avoid emitting a huge NotImplementedError message.
    try_xformers_op = bool(allow_xformers_op and _xformers_memory_efficient_attention is not None)
    if q.is_cuda:
        if q.dtype not in (torch.float16, torch.bfloat16):
            try_xformers_op = False
        if qk_dim < 32 or (qk_dim % 8) != 0:
            try_xformers_op = False
        if sliding_dilation != 1:
            # xFormers masks do not provide a direct dilation pattern API for this local-causal layout.
            try_xformers_op = False
        if sliding_window is not None:
            try:
                major, minor = torch.cuda.get_device_capability(q.device)
                if (major, minor) < (8, 0):
                    try_xformers_op = False
            except Exception:
                # If we can't query capability, keep trying.
                pass

    attn_bias = None
    if is_causal and sliding_window is None and _xformers_LowerTriangularMask is not None:
        attn_bias = _xformers_LowerTriangularMask()
    elif is_causal and sliding_window is not None and _xformers_LocalAttentionFromBottomRightMask is not None:
        # xFormers has changed this API across versions.
        try:
            attn_bias = _xformers_LocalAttentionFromBottomRightMask(window_left=sliding_window, window_right=0)
        except TypeError:
            try:
                attn_bias = _xformers_LocalAttentionFromBottomRightMask(sliding_window, 0)
            except TypeError:
                try:
                    attn_bias = _xformers_LocalAttentionFromBottomRightMask(window_size=sliding_window)
                except TypeError:
                    attn_bias = _xformers_LocalAttentionFromBottomRightMask(sliding_window)

    # If there is padding, we must materialize a tensor bias so that masking is correctly applied.
    if attn_bias is not None and attention_mask_2d is not None:
        try:
            if attention_mask_2d.dim() == 2 and attention_mask_2d.shape == (bsz, k_len):
                if attention_mask_2d.to(dtype=torch.bool).all().item() is False:
                    attn_bias = None
        except Exception:
            # If anything goes wrong (e.g. tracing), be conservative and materialize.
            attn_bias = None

    if attn_bias is None:
        min_val = torch.finfo(torch.float32).min
        bias = q.new_zeros((q_len, k_len), dtype=torch.float32)

        i = torch.arange(q_len, device=q.device)[:, None]
        j = torch.arange(k_len, device=q.device)[None, :]
        offset = k_len - q_len

        allowed = j <= (i + offset) if is_causal else torch.ones_like(bias, dtype=torch.bool)
        if sliding_window is not None:
            if not isinstance(sliding_window, int) or sliding_window <= 0:
                raise ValueError("xFormers sliding-window attention requires `sliding_window` to be a positive int.")
            allowed = allowed & (j >= (i + offset - (sliding_window - 1)))
            if sliding_dilation != 1:
                dist = (i + offset) - j
                allowed = allowed & ((dist % sliding_dilation) == 0)

        bias = bias.masked_fill(~allowed, min_val)
        attn_bias = bias[None, None, :, :]  # (1, 1, S_q, S_k)

        if attention_mask_2d is not None:
            if attention_mask_2d.dim() != 2:
                raise ValueError("`attention_mask_2d` must be a 2D padding mask (B, S_k) when using xFormers.")
            key_keep = attention_mask_2d.to(dtype=torch.bool)
            if key_keep.shape[0] != bsz or key_keep.shape[1] != k_len:
                raise ValueError(
                    f"`attention_mask_2d` must have shape {(bsz, k_len)}, got {tuple(key_keep.shape)}."
                )
            if not torch.all(key_keep):
                attn_bias = attn_bias.expand(bsz, 1, q_len, k_len).clone()
                attn_bias = attn_bias.masked_fill(~key_keep[:, None, None, :], min_val)

    if try_xformers_op:
        try:
            out = _xformers_memory_efficient_attention(q, k, v, attn_bias=attn_bias, p=dropout_p)
            return out
        except NotImplementedError as exc:
            logger.warning_once(
                "xFormers `memory_efficient_attention` has no supported CUDA operator for these inputs; falling back to eager attention."
            )
            logger.debug(str(exc))
        except RuntimeError as exc:
            logger.warning_once(
                "xFormers `memory_efficient_attention` failed at runtime; falling back to eager attention."
            )
            logger.debug(str(exc))

    # Eager fallback (supports any dtype/device, but can be slow).
    q_t = q.transpose(1, 2)  # (B, H, S_q, D)
    k_t = k.transpose(1, 2)  # (B, H, S_k, D)
    v_t = v.transpose(1, 2)  # (B, H, S_k, D)

    dim = q_t.shape[-1]
    scale = dim**-0.5
    # Use a mask value representable in the *actual compute dtype* that autocast may select.
    # Using float32's minimum with fp16 tensors can overflow during `masked_fill`.
    min_val = torch.finfo(q.dtype).min if q.is_floating_point() else torch.finfo(torch.float32).min

    if sliding_window is not None:
        if not isinstance(sliding_window, int) or sliding_window <= 0:
            raise ValueError("Eager sliding-window attention requires `sliding_window` to be a positive int.")

        # Chunked sliding-window attention without materializing (c, w, d) windows.
        w = int(sliding_window)
        offset = k_len - q_len
        chunk_q = 128

        key_keep_full = None
        if attention_mask_2d is not None:
            if attention_mask_2d.dim() != 2:
                raise ValueError("`attention_mask_2d` must be a 2D padding mask (B, S_k) for eager attention.")
            key_keep_full = attention_mask_2d.to(dtype=torch.bool)
            if key_keep_full.shape != (bsz, k_len):
                raise ValueError(
                    f"`attention_mask_2d` must have shape {(bsz, k_len)}, got {tuple(key_keep_full.shape)}."
                )

        out_t = q_t.new_empty(q_t.shape)  # (B, H, S_q, D)

        for qs in range(0, q_len, chunk_q):
            qe = min(q_len, qs + chunk_q)
            q_chunk = q_t[:, :, qs:qe, :]  # (B, H, c, D)
            c = qe - qs

            key_start = max(0, offset + qs - (w - 1))
            key_end = offset + qe  # <= k_len when q_len<=k_len
            k_range = k_t[:, :, key_start:key_end, :]  # (B, H, K, D)
            v_range = v_t[:, :, key_start:key_end, :]  # (B, H, K, D)

            q_pos = torch.arange(qs, qe, device=q.device)  # (c,)
            q_abs = (offset + q_pos).to(dtype=torch.long)  # absolute key positions per query
            k_abs = torch.arange(key_start, key_end, device=q.device, dtype=torch.long)  # (K,)

            if is_causal:
                allowed = (k_abs[None, :] <= q_abs[:, None]) & (k_abs[None, :] >= (q_abs[:, None] - (w - 1)))
            else:
                allowed = k_abs[None, :] >= (q_abs[:, None] - (w - 1))
            if sliding_dilation != 1:
                dist = q_abs[:, None] - k_abs[None, :]
                if is_causal:
                    allowed = allowed & ((dist % sliding_dilation) == 0)
                else:
                    allowed = allowed & ((dist.abs() % sliding_dilation) == 0)

            attn_scores = torch.matmul(q_chunk, k_range.transpose(-2, -1)) * scale
            attn_scores = attn_scores.masked_fill(~allowed[None, None, :, :], min_val)

            if key_keep_full is not None:
                keep_range = key_keep_full[:, key_start:key_end]  # (B, K)
                attn_scores = attn_scores.masked_fill(~keep_range[:, None, None, :], min_val)

            attn_probs = budgie_softmax(
                attn_scores, use_liger_kernel=use_liger_kernel, dim=-1, dtype=torch.float32
            ).to(q.dtype)
            if dropout_p and dropout_p > 0:
                attn_probs = nn.functional.dropout(attn_probs, p=dropout_p, training=True)

            out_chunk = torch.matmul(attn_probs, v_range.to(attn_probs.dtype)).to(q.dtype)  # (B, H, c, D)
            if out_chunk.shape[-2] != c:
                raise RuntimeError("Unexpected attention output shape in sliding-window eager path.")
            out_t[:, :, qs:qe, :] = out_chunk

        return out_t.transpose(1, 2).contiguous()

    # Dense eager causal attention (O(S^2) memory/time).
    attn_scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale  # (B,H,Q,K)
    offset = k_len - q_len
    i = torch.arange(q_len, device=q.device)[:, None]
    j = torch.arange(k_len, device=q.device)[None, :]
    allowed = j <= (i + offset) if is_causal else torch.ones((q_len, k_len), device=q.device, dtype=torch.bool)
    attn_scores = attn_scores.masked_fill(~allowed[None, None, :, :], min_val)

    if attention_mask_2d is not None:
        if attention_mask_2d.dim() != 2:
            raise ValueError("`attention_mask_2d` must be a 2D padding mask (B, S_k) for eager attention.")
        key_keep = attention_mask_2d.to(dtype=torch.bool)
        if key_keep.shape != (bsz, k_len):
            raise ValueError(f"`attention_mask_2d` must have shape {(bsz, k_len)}, got {tuple(key_keep.shape)}.")
        attn_scores = attn_scores.masked_fill(~key_keep[:, None, None, :], min_val)

    attn_probs = budgie_softmax(attn_scores, use_liger_kernel=use_liger_kernel, dim=-1, dtype=torch.float32).to(q.dtype)
    if dropout_p and dropout_p > 0:
        attn_probs = nn.functional.dropout(attn_probs, p=dropout_p, training=True)

    out = torch.matmul(attn_probs, v_t.to(attn_probs.dtype)).to(q.dtype)  # (B,H,Q,D)
    return out.transpose(1, 2).contiguous()


def _budgie_landmark_attention(
    *,
    query_states: torch.Tensor,  # (B, S, H, D)
    key_states: torch.Tensor,  # (B, S, H, D)
    value_states: torch.Tensor,  # (B, S, H, D)
    attention_mask_2d: Optional[torch.Tensor],  # (B, S) with 1=keep,0=pad
    landmark_mask: Optional[torch.Tensor],  # (B, S) bool
    landmark_every: Optional[int],  # positional landmarks
    dropout_p: float,
    is_causal: bool,
    use_liger_kernel: bool = False,
) -> torch.Tensor:
    """
    Memory-efficient eager landmark attention.

    Pattern (causal "block-local + landmarks"):
    - Each token attends to all tokens in its current block (delimited by landmark tokens), causally.
    - Each token attends to all *previous* landmark tokens (global), causally.

    This avoids materializing an (S,S) matrix by processing one block at a time.
    """

    if not is_causal:
        raise NotImplementedError("Budgie landmark attention currently only supports causal self-attention.")

    q = query_states.contiguous()
    k = key_states.contiguous()
    v = value_states.contiguous()

    bsz, q_len, nheads, dim = q.shape
    k_len = k.shape[1]
    if q_len != k_len:
        raise ValueError(
            "Budgie landmark attention currently requires q_len == k_len (no KV-cache). "
            f"Got q_len={q_len}, k_len={k_len}."
        )

    key_keep = None
    if attention_mask_2d is not None:
        if attention_mask_2d.dim() != 2:
            raise ValueError("`attention_mask_2d` must be a 2D padding mask (B, S).")
        key_keep = attention_mask_2d.to(dtype=torch.bool)
        if key_keep.shape != (bsz, k_len):
            raise ValueError(f"`attention_mask_2d` must have shape {(bsz, k_len)}, got {tuple(key_keep.shape)}.")

    if landmark_mask is not None:
        if landmark_mask.shape != (bsz, k_len):
            raise ValueError(f"`landmark_mask` must have shape {(bsz, k_len)}, got {tuple(landmark_mask.shape)}.")

    q_t = q.transpose(1, 2)  # (B, H, S, D)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)

    scale = dim**-0.5
    # Use a mask value representable in the actual compute dtype. Under autocast, matmuls may run in fp16/bf16 even
    # if inputs are cast to fp32, so using float32's minimum can overflow during `masked_fill`.
    min_val = torch.finfo(q.dtype).min if q.is_floating_point() else torch.finfo(torch.float32).min
    out_t = q_t.new_empty((bsz, nheads, q_len, dim))  # (B, H, S, D)

    device = q.device

    # Fast path: positional landmarks (uniform blocks across the batch).
    if landmark_mask is None:
        if not isinstance(landmark_every, int) or landmark_every <= 0:
            raise ValueError(
                "Landmark mode requires either a provided `landmark_mask` or `landmark_every` as a positive int."
            )

        block_size = int(landmark_every)
        max_full_landmarks = k_len // block_size
        if max_full_landmarks <= 0:
            landmark_pos = torch.empty((0,), device=device, dtype=torch.long)
        else:
            landmark_pos = torch.arange(
                block_size - 1,
                block_size * max_full_landmarks,
                block_size,
                device=device,
                dtype=torch.long,
            )

        for start in range(0, k_len, block_size):
            end = min(k_len, start + block_size)
            q_block = q_t[:, :, start:end, :]  # (B, H, Lq, D)
            k_local = k_t[:, :, start:end, :]
            v_local = v_t[:, :, start:end, :]
            lq = end - start

            scores_local = torch.matmul(q_block, k_local.transpose(-2, -1)) * scale  # (B, H, Lq, Lq)
            causal = torch.tril(torch.ones((lq, lq), device=device, dtype=torch.bool))
            scores_local = scores_local.masked_fill(~causal[None, None, :, :], min_val)

            if key_keep is not None:
                keep_local = key_keep[:, start:end]
                # Avoid `.item()` syncs on CUDA; masking with an all-True keep mask is a cheap no-op.
                scores_local = scores_local.masked_fill(~keep_local[:, None, None, :], min_val)

            prev_landmarks = start // block_size
            if prev_landmarks > 0:
                lm_idx = landmark_pos[:prev_landmarks]
                k_lm = k_t[:, :, lm_idx, :]  # (B, H, P, D)
                v_lm = v_t[:, :, lm_idx, :]
                scores_lm = torch.matmul(q_block, k_lm.transpose(-2, -1)) * scale  # (B, H, Lq, P)

                if key_keep is not None:
                    keep_lm = key_keep[:, lm_idx]
                    scores_lm = scores_lm.masked_fill(~keep_lm[:, None, None, :], min_val)

                scores = torch.cat([scores_local, scores_lm], dim=-1)
                probs = budgie_softmax(scores, use_liger_kernel=use_liger_kernel, dim=-1, dtype=torch.float32).to(
                    dtype=q.dtype
                )
                if dropout_p and dropout_p > 0:
                    probs = nn.functional.dropout(probs, p=dropout_p, training=True)

                probs_local = probs[..., :lq]
                probs_lm = probs[..., lq:]
                out_block = torch.matmul(probs_local, v_local.to(dtype=probs.dtype)).to(dtype=q.dtype)
                out_block = out_block + torch.matmul(probs_lm, v_lm.to(dtype=probs.dtype)).to(dtype=q.dtype)
            else:
                probs = budgie_softmax(
                    scores_local, use_liger_kernel=use_liger_kernel, dim=-1, dtype=torch.float32
                ).to(dtype=q.dtype)
                if dropout_p and dropout_p > 0:
                    probs = nn.functional.dropout(probs, p=dropout_p, training=True)
                out_block = torch.matmul(probs, v_local.to(dtype=probs.dtype)).to(dtype=q.dtype)

            if key_keep is not None:
                query_keep = key_keep[:, start:end]
                out_block = out_block * query_keep[:, None, :, None].to(dtype=out_block.dtype)

            out_t[:, :, start:end, :] = out_block

        return out_t.transpose(1, 2).contiguous()

    # Token-id landmarks: per-sample block boundaries.
    for b in range(bsz):
        is_landmark = landmark_mask[b].to(dtype=torch.bool, device=device)  # (S,)
        landmark_pos = torch.nonzero(is_landmark, as_tuple=False).flatten()  # (N,)

        # Blocks end right after each landmark token (so the landmark is included in the preceding block).
        block_ends = (landmark_pos + 1).tolist()
        if not block_ends or block_ends[-1] != k_len:
            block_ends.append(k_len)
        block_starts = [0] + block_ends[:-1]

        prev_landmarks = 0
        for start, end in zip(block_starts, block_ends):
            if start >= end:
                continue

            q_block = q_t[b, :, start:end, :]  # (H, Lq, D)
            k_local = k_t[b, :, start:end, :]  # (H, Lk, D)
            v_local = v_t[b, :, start:end, :]
            lq = end - start

            query_keep = None
            if key_keep is not None:
                query_keep = key_keep[b, start:end]
                if not bool(torch.any(query_keep).item()):
                    out_t[b, :, start:end, :] = 0
                    if bool(is_landmark[end - 1].item()):
                        prev_landmarks += 1
                    continue

                valid_q = torch.nonzero(query_keep, as_tuple=False).flatten()
                q_block = q_block.index_select(-2, valid_q)
            else:
                valid_q = None

            scores_local = torch.matmul(q_block, k_local.transpose(-2, -1)) * scale

            if valid_q is None:
                causal = torch.tril(torch.ones((lq, lq), device=device, dtype=torch.bool))
                scores_local = scores_local.masked_fill(~causal[None, :, :], min_val)
            else:
                j = torch.arange(lq, device=device, dtype=torch.long)[None, :]  # (1, Lk)
                allowed = j <= valid_q[:, None]  # (Lq_valid, Lk)
                scores_local = scores_local.masked_fill(~allowed[None, :, :], min_val)

            if key_keep is not None:
                keep_local = key_keep[b, start:end]
                if not bool(torch.all(keep_local).item()):
                    scores_local = scores_local.masked_fill(~keep_local[None, None, :], min_val)

            if prev_landmarks > 0:
                lm_idx = landmark_pos[:prev_landmarks]
                k_lm = k_t[b, :, lm_idx, :]
                v_lm = v_t[b, :, lm_idx, :]
                scores_lm = torch.matmul(q_block, k_lm.transpose(-2, -1)) * scale
                if key_keep is not None:
                    keep_lm = key_keep[b, lm_idx]
                    if not bool(torch.all(keep_lm).item()):
                        scores_lm = scores_lm.masked_fill(~keep_lm[None, None, :], min_val)

                scores = torch.cat([scores_local, scores_lm], dim=-1)
                probs = budgie_softmax(scores, use_liger_kernel=use_liger_kernel, dim=-1, dtype=torch.float32).to(
                    dtype=q.dtype
                )
                if dropout_p and dropout_p > 0:
                    probs = nn.functional.dropout(probs, p=dropout_p, training=True)

                probs_local = probs[..., :lq]
                probs_lm = probs[..., lq:]
                out_block = torch.matmul(probs_local, v_local.to(dtype=probs.dtype)).to(dtype=q.dtype)
                out_block = out_block + torch.matmul(probs_lm, v_lm.to(dtype=probs.dtype)).to(dtype=q.dtype)
            else:
                probs = budgie_softmax(
                    scores_local, use_liger_kernel=use_liger_kernel, dim=-1, dtype=torch.float32
                ).to(dtype=q.dtype)
                if dropout_p and dropout_p > 0:
                    probs = nn.functional.dropout(probs, p=dropout_p, training=True)
                out_block = torch.matmul(probs, v_local.to(dtype=probs.dtype)).to(dtype=q.dtype)

            if valid_q is None:
                out_t[b, :, start:end, :] = out_block
            else:
                out_full = out_t[b, :, start:end, :]
                out_full.zero_()
                out_full.index_copy_(-2, valid_q, out_block)

            if bool(is_landmark[end - 1].item()):
                prev_landmarks += 1

    return out_t.transpose(1, 2).contiguous()



class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True


        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rope_dim = self.config.qk_rope_dim 


        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        #self.config.head_dim = self.rope_dim # 64 // 2 = 32
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config) 
        #self.config.head_dim = self.head_dim


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = budgie_apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            unsqueeze_dim=1,
            use_liger_kernel=bool(getattr(self.config, "use_liger_kernel", False)),
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            effective_layer_idx = kwargs.get("layer_idx", self.layer_idx)
            key_states, value_states = past_key_value.update(key_states, value_states, effective_layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value



class LlamaGLA(LlamaAttention):
    

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
        self.gla_num_groups = int(getattr(self.config, "gla_num_groups", 2))
        if self.gla_num_groups <= 0:
            raise ValueError(f"`config.gla_num_groups` must be a positive int, got {self.gla_num_groups}.")
        if self.num_heads % self.gla_num_groups != 0:
            raise ValueError(
                "`config.gla_num_groups` must divide `config.num_attention_heads`. "
                f"Got gla_num_groups={self.gla_num_groups}, num_attention_heads={self.num_heads}."
            )
        self.heads_per_group = self.num_heads // self.gla_num_groups

        # Keep the historical design: each group has a latent size of 2*head_dim (and is then expanded to
        # heads_per_group * 2*head_dim to produce per-head K and V).
        self.latent_dim_per_group = 2 * self.head_dim

        self.kv_proj_dim = self.gla_num_groups * self.latent_dim_per_group
        self.q_proj_dim = 8 * self.head_dim

        self.q_norm = budgie_make_rmsnorm(self.config, self.q_proj_dim, self.config.rms_norm_eps)


        self.kv_norms = nn.ModuleList(
            [budgie_make_rmsnorm(self.config, self.latent_dim_per_group, self.config.rms_norm_eps) for _ in range(self.gla_num_groups)]
        )


        #Query
        self.W_dQ = nn.Linear(self.hidden_size, self.q_proj_dim)         
        self.W_uQ_rope = nn.Linear(self.q_proj_dim, self.num_heads * (self.head_dim + self.rope_dim)) 


        self.W_dKV = nn.Linear(self.hidden_size, self.kv_proj_dim + self.rope_dim) 
        
        
        self.W_ukvs = nn.ModuleList(
            [
                nn.Linear(self.latent_dim_per_group, self.heads_per_group * (2 * self.head_dim))
                for _ in range(self.gla_num_groups)
            ]
        )




    def _flash_attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        landmark_mask: Optional[torch.Tensor] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dropout_p = float(self.attention_dropout) if self.training else 0.0

        q_len = query_states.shape[1]
        k_len = key_states.shape[1]

        q = query_states.transpose(1, 2)  # (B, H, S_q, D)
        k = key_states.transpose(1, 2)
        v = value_states.transpose(1, 2)

        attn_mask = attention_mask
        is_causal = bool(self.is_causal)
        if attn_mask is not None:
            is_causal = False
            # Common HF shape: (B, 1, S_q, S_k). SDPA typically expects (B, S_q, S_k) or broadcastable.
            if attn_mask.dim() == 4 and attn_mask.shape[1] == 1:
                attn_mask = attn_mask[:, 0, :, :]
            if attn_mask.dim() == 3:
                attn_mask = attn_mask[:, :q_len, :k_len]

        if hasattr(nn.functional, "scaled_dot_product_attention"):
            try:
                out = nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
                )
                return out.transpose(1, 2).contiguous()
            except (RuntimeError, TypeError):
                pass

        # Fallback: use xFormers if enabled/available, otherwise eager matmul/softmax.
        out = _budgie_xformers_attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask_2d=attention_mask_2d,
            dropout_p=dropout_p,
            is_causal=bool(self.is_causal),
            sliding_window=None,
            allow_xformers_op=bool(getattr(self.config, "use_xformers", False)),
            use_liger_kernel=bool(getattr(self.config, "use_liger_kernel", False)),
        )
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        

        bsz, q_len, _ = hidden_states.size()

        q_proj = self.W_dQ(hidden_states)

        query_states = self.q_norm(q_proj)
        query_states = self.W_uQ_rope(query_states).view(bsz, q_len, self.num_heads, self.head_dim + self.rope_dim)
        query_rope = query_states[..., self.head_dim:]

         
        KV_compressed_cache = self.W_dKV(hidden_states)
        compressed_kv, key_rope = torch.split(KV_compressed_cache, [self.kv_proj_dim, self.rope_dim], dim=-1)
        key_rope = key_rope.view(bsz, q_len, 1, self.rope_dim)


        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` to using externally computed `position_embeddings`."
            )
            cos, sin = self.rotary_emb(query_rope, position_ids) 
        else:
            cos, sin = position_embeddings

        query_rope, key_rope = budgie_apply_rotary_pos_emb(
            query_rope,
            key_rope,
            cos,
            sin,
            unsqueeze_dim=2,
            use_liger_kernel=bool(getattr(self.config, "use_liger_kernel", False)),
        )
        query_states[..., self.head_dim:].copy_(query_rope)

        # Cache
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            effective_layer_idx = kwargs.get("layer_idx", self.layer_idx)
            compressed_kv_cache, key_rope_cache = past_key_value.update(
                compressed_kv.unsqueeze(1),
                key_rope.transpose(1, 2),
                effective_layer_idx,
                cache_kwargs,
            )
            compressed_kv = compressed_kv_cache.squeeze(1)
            key_rope_full = key_rope_cache.squeeze(1)
        else:
            key_rope_full = key_rope.squeeze(2)

        kv_len = compressed_kv.shape[1]
        key_states = query_states.new_zeros((bsz, kv_len, self.num_heads, self.head_dim + self.rope_dim))
        value_states = query_states.new_zeros((bsz, kv_len, self.num_heads, self.head_dim + self.rope_dim))

        
        latent_dim = self.latent_dim_per_group
        heads_per_group = self.heads_per_group
        for g in range(self.gla_num_groups):
            head_start = g * heads_per_group
            head_end = head_start + heads_per_group

            kv_g = compressed_kv[..., (g * latent_dim) : ((g + 1) * latent_dim)]
            kv_g = self.kv_norms[g](kv_g)
            kv_out = self.W_ukvs[g](kv_g).view(bsz, kv_len, heads_per_group, 2 * self.head_dim)

            key_states[:, :, head_start:head_end, :self.head_dim] = kv_out[..., : self.head_dim]
            value_states[:, :, head_start:head_end, :self.head_dim] = kv_out[..., self.head_dim :]

        key_states[..., self.head_dim:].copy_(
            key_rope_full[:, :, None, :].expand(bsz, kv_len, self.num_heads, self.rope_dim)
        )


        attn_output = self._flash_attn(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            landmark_mask=kwargs.get("landmark_mask"),
            attention_mask_2d=kwargs.get("attention_mask_2d"),
        )


        attn_output = attn_output[..., :self.head_dim].contiguous() 
        attn_output = attn_output.view(bsz, q_len, -1) 
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class BudgieGLASlidingWindowAttention(LlamaGLA):
    """
    GLA attention with a causal sliding window.

    Configure with:
    - `config.sliding_window` as a positive int.
    - `config._attn_implementation = "gla_sliding"`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sliding_window = getattr(self.config, "sliding_window", None)
        self.sliding_dilation = 1

        if not isinstance(self.sliding_window, int) or self.sliding_window <= 0:
            raise ValueError(
                "BudgieGLASlidingWindowAttention requires `config.sliding_window` to be a positive int."
            )

    def _flash_attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        landmark_mask: Optional[torch.Tensor] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if flash_attn_func is None:
            dropout_p = float(self.attention_dropout) if self.training else 0.0
            return _budgie_xformers_attention(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask_2d=attention_mask_2d,
                dropout_p=dropout_p,
                is_causal=bool(self.is_causal),
                sliding_window=int(self.sliding_window),
                sliding_dilation=int(self.sliding_dilation),
                allow_xformers_op=bool(getattr(self.config, "use_xformers", False)),
                use_liger_kernel=bool(getattr(self.config, "use_liger_kernel", False)),
            )

        try:
            attn_output, _ = flash_attn_func(
                q=query_states,
                k=key_states,
                v=value_states,
                causal=self.is_causal,
                window_size=(self.sliding_window, 0),
            )
        except TypeError:
            try:
                attn_output, _ = flash_attn_func(
                    q=query_states,
                    k=key_states,
                    v=value_states,
                    causal=self.is_causal,
                    window_size=self.sliding_window,
                )
            except TypeError:
                try:
                    attn_output, _ = flash_attn_func(
                        q=query_states,
                        k=key_states,
                        v=value_states,
                        causal=self.is_causal,
                        window_size_left=self.sliding_window,
                        window_size_right=0,
                    )
                except TypeError as exc:
                    raise TypeError(
                        "`flash_attn_func` does not appear to support any of these sliding-window signatures: "
                        "`window_size=(left, right)`, `window_size=int`, or `window_size_left/window_size_right`."
                    ) from exc

        return attn_output


class BudgieGLADilatedSlidingWindowAttention(LlamaGLA):
    """
    GLA attention with a causal sliding window and dilation.

    Configure with:
    - `config.swa_dilation` as a positive int.
    - `config.swa_dilated_window` or `config.sliding_window` as a positive int.
    - `config._attn_implementation = "gla_sliding_dilated"`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sliding_window = getattr(self.config, "swa_dilated_window", None)
        if self.sliding_window is None:
            self.sliding_window = getattr(self.config, "sliding_window", None)
        self.sliding_dilation = int(getattr(self.config, "swa_dilation", 2))
        if not isinstance(self.sliding_window, int) or self.sliding_window <= 0:
            raise ValueError(
                "BudgieGLADilatedSlidingWindowAttention requires `config.swa_dilated_window` (or `config.sliding_window`) to be a positive int."
            )
        if self.sliding_dilation <= 0:
            raise ValueError(
                f"BudgieGLADilatedSlidingWindowAttention requires `config.swa_dilation` > 0, got {self.sliding_dilation!r}."
            )

    def _flash_attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        landmark_mask: Optional[torch.Tensor] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dropout_p = float(self.attention_dropout) if self.training else 0.0
        return _budgie_xformers_attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask_2d=attention_mask_2d,
            dropout_p=dropout_p,
            is_causal=bool(self.is_causal),
            sliding_window=int(self.sliding_window),
            sliding_dilation=int(self.sliding_dilation),
            allow_xformers_op=bool(getattr(self.config, "use_xformers", False)),
            use_liger_kernel=bool(getattr(self.config, "use_liger_kernel", False)),
        )


class BudgieGLALandmarkAttention(LlamaGLA):
    """
    Landmark attention implemented on top of GLA projections.

    This implements a causal "block-local + landmark" pattern:
    - Every token attends to all previous landmark tokens.
    - Every token attends to all tokens within its current block (delimited by landmark tokens).

    Configure with:
    - Either `config.landmark_token_id` to mark landmarks in `input_ids`, or `config.landmark_every` as a positive int
      (e.g. 128) to assume landmarks are every Nth token (indices `N-1, 2N-1, ...`).
    - `config._attn_implementation = "gla_landmark"`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.landmark_every = getattr(self.config, "landmark_every", None)
        if self.landmark_every is None and getattr(self.config, "landmark_token_id", None) is None:
            raise ValueError(
                "BudgieGLALandmarkAttention requires either `config.landmark_every` (positional landmarks) or "
                "`config.landmark_token_id` (token-id landmarks)."
            )
        if self.landmark_every is not None and (not isinstance(self.landmark_every, int) or self.landmark_every <= 0):
            raise ValueError("If set, `config.landmark_every` must be a positive int.")

    def _flash_attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        landmark_mask: Optional[torch.Tensor] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dropout_p = float(self.attention_dropout) if self.training else 0.0

        # Landmark attention is implemented eagerly (no FlashAttention kernel for this sparsity pattern).
        # Prefer `attention_mask_2d` (padding) to avoid relying on a dense 4D additive mask.
        return _budgie_landmark_attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask_2d=attention_mask_2d if attention_mask_2d is not None else (attention_mask if attention_mask is not None and attention_mask.dim() == 2 else None),
            landmark_mask=landmark_mask,
            landmark_every=self.landmark_every,
            dropout_p=dropout_p,
            is_causal=bool(self.is_causal),
            use_liger_kernel=bool(getattr(self.config, "use_liger_kernel", False)),
        )



class BudgieGLASharedHybrid(LlamaGLA):
    """
    Shared GLA attention that can switch between algorithms per call.

    Use with `attn_mode`:
    - `"sliding"`: causal sliding-window attention (uses xFormers if enabled/supported, otherwise eager).
    - `"landmark"`: eager landmark attention (block-local + global landmarks).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sliding_window = getattr(self.config, "sliding_window", None)
        self.landmark_every = getattr(self.config, "landmark_every", None)
        if self.landmark_every is not None and (not isinstance(self.landmark_every, int) or self.landmark_every <= 0):
            raise ValueError("If set, `config.landmark_every` must be a positive int.")

    def _flash_attn_sliding(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if flash_attn_func is None:
            dropout_p = float(self.attention_dropout) if self.training else 0.0
            return _budgie_xformers_attention(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask_2d=attention_mask_2d,
                dropout_p=dropout_p,
                is_causal=bool(self.is_causal),
                sliding_window=int(self.sliding_window) if self.sliding_window is not None else None,
                allow_xformers_op=bool(getattr(self.config, "use_xformers", False)),
                use_liger_kernel=bool(getattr(self.config, "use_liger_kernel", False)),
            )

        if not isinstance(self.sliding_window, int) or self.sliding_window <= 0:
            raise ValueError("Sliding-window mode requires `config.sliding_window` to be a positive int.")

        try:
            attn_output, _ = flash_attn_func(
                q=query_states,
                k=key_states,
                v=value_states,
                causal=self.is_causal,
                window_size=(self.sliding_window, 0),
            )
        except TypeError:
            try:
                attn_output, _ = flash_attn_func(
                    q=query_states,
                    k=key_states,
                    v=value_states,
                    causal=self.is_causal,
                    window_size=self.sliding_window,
                )
            except TypeError:
                attn_output, _ = flash_attn_func(
                    q=query_states,
                    k=key_states,
                    v=value_states,
                    causal=self.is_causal,
                    window_size_left=self.sliding_window,
                    window_size_right=0,
                )
        return attn_output

    def _flash_attn_landmark(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor],
        landmark_mask: Optional[torch.Tensor],
        attention_mask_2d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dropout_p = float(self.attention_dropout) if self.training else 0.0
        return _budgie_landmark_attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask_2d=attention_mask_2d if attention_mask_2d is not None else (attention_mask if attention_mask is not None and attention_mask.dim() == 2 else None),
            landmark_mask=landmark_mask,
            landmark_every=self.landmark_every,
            dropout_p=dropout_p,
            is_causal=bool(self.is_causal),
            use_liger_kernel=bool(getattr(self.config, "use_liger_kernel", False)),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        attn_mode = kwargs.get("attn_mode", None)
        if attn_mode not in ("sliding", "landmark"):
            raise ValueError("BudgieGLASharedHybrid requires `attn_mode` to be either 'sliding' or 'landmark'.")

        bsz, q_len, _ = hidden_states.size()

        q_proj = self.W_dQ(hidden_states)
        query_states = self.q_norm(q_proj)
        query_states = self.W_uQ_rope(query_states).view(bsz, q_len, self.num_heads, self.head_dim + self.rope_dim)
        query_rope = query_states[..., self.head_dim:]

        KV_compressed_cache = self.W_dKV(hidden_states)
        compressed_kv, key_rope = torch.split(KV_compressed_cache, [self.kv_proj_dim, self.rope_dim], dim=-1)
        key_rope = key_rope.view(bsz, q_len, 1, self.rope_dim)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(query_rope, position_ids)
        else:
            cos, sin = position_embeddings

        query_rope, key_rope = budgie_apply_rotary_pos_emb(
            query_rope,
            key_rope,
            cos,
            sin,
            unsqueeze_dim=2,
            use_liger_kernel=bool(getattr(self.config, "use_liger_kernel", False)),
        )
        query_states[..., self.head_dim:].copy_(query_rope)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            effective_layer_idx = kwargs.get("layer_idx", self.layer_idx)
            compressed_kv_cache, key_rope_cache = past_key_value.update(
                compressed_kv.unsqueeze(1),
                key_rope.transpose(1, 2),
                effective_layer_idx,
                cache_kwargs,
            )
            compressed_kv = compressed_kv_cache.squeeze(1)
            key_rope_full = key_rope_cache.squeeze(1)
        else:
            key_rope_full = key_rope.squeeze(2)

        kv_len = compressed_kv.shape[1]
        key_states = query_states.new_zeros((bsz, kv_len, self.num_heads, self.head_dim + self.rope_dim))
        value_states = query_states.new_zeros((bsz, kv_len, self.num_heads, self.head_dim + self.rope_dim))

        latent_dim = self.latent_dim_per_group
        heads_per_group = self.heads_per_group
        for g in range(self.gla_num_groups):
            head_start = g * heads_per_group
            head_end = head_start + heads_per_group

            kv_g = compressed_kv[..., (g * latent_dim) : ((g + 1) * latent_dim)]
            kv_g = self.kv_norms[g](kv_g)
            kv_out = self.W_ukvs[g](kv_g).view(bsz, kv_len, heads_per_group, 2 * self.head_dim)

            key_states[:, :, head_start:head_end, :self.head_dim] = kv_out[..., : self.head_dim]
            value_states[:, :, head_start:head_end, :self.head_dim] = kv_out[..., self.head_dim :]

        key_states[..., self.head_dim:].copy_(
            key_rope_full[:, :, None, :].expand(bsz, kv_len, self.num_heads, self.rope_dim)
        )

        if attn_mode == "sliding":
            attn_output = self._flash_attn_sliding(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask_2d=kwargs.get("attention_mask_2d"),
            )
        else:
            attn_output = self._flash_attn_landmark(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                landmark_mask=kwargs.get("landmark_mask"),
                attention_mask_2d=kwargs.get("attention_mask_2d"),
            )

        attn_output = attn_output[..., :self.head_dim].contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            return attn_output, None, past_key_value
        return attn_output, None, past_key_value



LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "sdpa": LlamaGLA,
    "gla_sliding": BudgieGLASlidingWindowAttention,
    "gla_sliding_dilated": BudgieGLADilatedSlidingWindowAttention,
    "gla_landmark": BudgieGLALandmarkAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        attn_implementation: Optional[str] = None,
        enable_tiny_conv: Optional[bool] = None,
        enable_gsm: Optional[bool] = None,
        is_bridge_layer: bool = False,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        impl = attn_implementation if attn_implementation is not None else config._attn_implementation
        self.self_attn = LLAMA_ATTENTION_CLASSES[impl](config=config, layer_idx=layer_idx)

        self.mlp = budgie_make_mlp(config)
        self.input_layernorm = budgie_make_rmsnorm(config, config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = budgie_make_rmsnorm(config, config.hidden_size, config.rms_norm_eps)

        if enable_tiny_conv is None:
            enable_tiny_conv = bool(getattr(config, "use_tiny_conv", False))

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

        if enable_gsm is None:
            enable_gsm = bool(getattr(config, "use_gsm", False))

        if enable_gsm:
            self.gsm = GSM(
                d_model=config.hidden_size,
                max_seq_len=int(getattr(config, "max_position_embeddings", 0)),
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

        self.bridge_mixer_fusion = bool(is_bridge_layer and self.gsm is not None and impl == "gla_landmark")
        if self.bridge_mixer_fusion:
            self.bridge_alpha_proj = nn.Linear(config.hidden_size, 1, bias=True)
            self.bridge_out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            nn.init.zeros_(self.bridge_alpha_proj.weight)
            nn.init.zeros_(self.bridge_alpha_proj.bias)
            nn.init.eye_(self.bridge_out_proj.weight)
        else:
            self.bridge_alpha_proj = None
            self.bridge_out_proj = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        attn_gate: Optional[torch.Tensor] = None,
        mlp_gate: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hs = self.input_layernorm(hidden_states)
        if self.bridge_mixer_fusion:
            z = hs
            if self.tiny_conv is not None:
                z = self.tiny_conv(
                    z,
                    use_cache=bool(use_cache),
                    past_key_value=past_key_value,
                    layer_idx=kwargs.get("layer_idx", self.layer_idx),
                    attention_mask_2d=kwargs.get("attention_mask_2d"),
                )
        else:
            if self.tiny_conv is not None:
                hs = hs + self.tiny_conv(
                    hs,
                    use_cache=bool(use_cache),
                    past_key_value=past_key_value,
                    layer_idx=kwargs.get("layer_idx", self.layer_idx),
                    attention_mask_2d=kwargs.get("attention_mask_2d"),
                )
            z = hs

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=z,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        if self.bridge_mixer_fusion:
            y_gsm = self.gsm(z)
            alpha = torch.sigmoid(self.bridge_alpha_proj(z.mean(dim=1))).view(z.shape[0], 1, 1)
            y = (alpha * hidden_states) + ((1.0 - alpha) * y_gsm)
            hidden_states = self.bridge_out_proj(y)
        elif self.gsm is not None:
            hidden_states = hidden_states + self.gsm(z)


        if attn_gate is None:
            attn_gate = 1
        hidden_states = residual + (hidden_states * attn_gate)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if mlp_gate is None:
            mlp_gate = 1
        hidden_states = residual + (hidden_states * mlp_gate)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

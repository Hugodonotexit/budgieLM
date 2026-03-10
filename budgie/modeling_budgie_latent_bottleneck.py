"""
Perceiver-style Macro-block (Read -> Latent-process -> Write -> Token-MLP)
integrated with Budgie internals:
- Uses GLA-style projections/attention kernels for cross- and self-attention.
- Reuses shared Budgie factories for RMSNorm/MLP (no duplicate norm/MLP classes).
"""

from __future__ import annotations

import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .budgie_config import BudgieConfig
from .modeling_budgie_GLA import (
    LlamaGLA,
    _budgie_xformers_attention,
    budgie_make_mlp,
    budgie_make_rmsnorm,
)


def _clone_cfg_for_heads(config: BudgieConfig, *, heads: int, dropout: float) -> BudgieConfig:
    heads = int(heads)
    if heads <= 0 or (config.hidden_size % heads) != 0:
        raise ValueError(
            "`heads` must be a positive divisor of hidden_size. "
            f"Got heads={heads}, hidden_size={config.hidden_size}."
        )

    cfg = copy.deepcopy(config)
    cfg.num_attention_heads = heads
    cfg.num_key_value_heads = heads
    cfg.attention_dropout = float(dropout)

    gla_groups = int(getattr(cfg, "gla_num_groups", 1))
    if heads % gla_groups != 0:
        cfg.gla_num_groups = max(1, math.gcd(heads, gla_groups))
    return cfg


def _clone_cfg_for_mlp(config: BudgieConfig, *, mlp_mult: float) -> BudgieConfig:
    if mlp_mult <= 0.0:
        raise ValueError(f"`mlp_mult` must be > 0, got {mlp_mult!r}.")
    cfg = copy.deepcopy(config)
    cfg.intermediate_size = int(cfg.hidden_size * float(mlp_mult))
    return cfg


class GLACrossAttention(nn.Module):
    """
    Cross-attention with GLA projections:
    - Q built from x_q using `LlamaGLA` query path.
    - K/V built from x_kv using `LlamaGLA` compressed-KV path.
    - Attention kernel uses Budgie's optimized helper (`_budgie_xformers_attention`).
    """

    def __init__(
        self,
        config: BudgieConfig,
        *,
        heads: int,
        dropout: float = 0.0,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.cfg = _clone_cfg_for_heads(config, heads=heads, dropout=dropout)
        self.gla = LlamaGLA(config=self.cfg, layer_idx=layer_idx)

        self.dropout = float(dropout)
        self.num_heads = int(self.cfg.num_attention_heads)
        self.head_dim = int(self.gla.head_dim)
        self.backend = str(getattr(config, "perceiver_attn_backend", "sdpa")).lower()
        if self.backend not in ("sdpa", "xformers", "flash"):
            self.backend = "sdpa"
        self.use_xformers = bool(getattr(self.cfg, "use_xformers", False) and self.backend == "xformers")
        self.use_liger_kernel = bool(getattr(self.cfg, "use_liger_kernel", False))

    def _build_q(self, x_q: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = x_q.shape
        q_proj = self.gla.W_dQ(x_q)
        q = self.gla.q_norm(q_proj)
        q = self.gla.W_uQ_rope(q).view(bsz, q_len, self.num_heads, self.head_dim + self.gla.rope_dim)
        return q[..., : self.head_dim].contiguous()

    def _build_kv(self, x_kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, kv_len, _ = x_kv.shape
        compressed_kv = self.gla.W_dKV(x_kv)[..., : self.gla.kv_proj_dim]

        key_states = x_kv.new_zeros((bsz, kv_len, self.num_heads, self.head_dim))
        value_states = x_kv.new_zeros((bsz, kv_len, self.num_heads, self.head_dim))

        latent_dim = self.gla.latent_dim_per_group
        heads_per_group = self.gla.heads_per_group
        for g in range(self.gla.gla_num_groups):
            head_start = g * heads_per_group
            head_end = head_start + heads_per_group

            kv_g = compressed_kv[..., (g * latent_dim) : ((g + 1) * latent_dim)]
            kv_g = self.gla.kv_norms[g](kv_g)
            kv_out = self.gla.W_ukvs[g](kv_g).view(bsz, kv_len, heads_per_group, 2 * self.head_dim)

            key_states[:, :, head_start:head_end, :] = kv_out[..., : self.head_dim]
            value_states[:, :, head_start:head_end, :] = kv_out[..., self.head_dim :]

        return key_states, value_states

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        q = self._build_q(x_q)
        k, v = self._build_kv(x_kv)

        if self.backend == "xformers":
            out = _budgie_xformers_attention(
                query_states=q,
                key_states=k,
                value_states=v,
                attention_mask_2d=attention_mask_2d,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=bool(is_causal),
                sliding_window=None,
                sliding_dilation=1,
                allow_xformers_op=self.use_xformers,
                use_liger_kernel=self.use_liger_kernel,
            )
        else:
            qh = q.transpose(1, 2).contiguous()
            kh = k.transpose(1, 2).contiguous()
            vh = v.transpose(1, 2).contiguous()
            attn_bias = None
            if attention_mask_2d is not None:
                key_keep = attention_mask_2d.to(dtype=torch.bool)
                if key_keep.dim() != 2 or key_keep.shape != (x_q.shape[0], x_kv.shape[1]):
                    raise ValueError(
                        "`attention_mask_2d` must have shape "
                        f"{(x_q.shape[0], x_kv.shape[1])}, got {tuple(key_keep.shape)}."
                    )
                min_val = torch.finfo(torch.float32).min
                attn_bias = qh.new_zeros((x_q.shape[0], 1, x_q.shape[1], x_kv.shape[1]), dtype=torch.float32)
                attn_bias = attn_bias.masked_fill(~key_keep[:, None, None, :], min_val)

            out_h = F.scaled_dot_product_attention(
                qh,
                kh,
                vh,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=bool(is_causal),
            )
            out = out_h.transpose(1, 2).contiguous()

        bsz, q_len, _, _ = out.shape
        out = out.view(bsz, q_len, self.num_heads * self.head_dim)
        return self.gla.o_proj(out)


class FFTTokenMixer(nn.Module):
    """
    Cheap global token mixing using FFT along sequence dimension.
    Uses torch.fft (cuFFT on CUDA).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = torch.fft.rfft(x, dim=1)
        xf = xf * self.gain.view(1, 1, -1)
        y = torch.fft.irfft(xf, n=x.size(1), dim=1)
        return y


class LatentBottleneckMacroBlock(nn.Module):
    """
    Read -> Latent process -> Write -> Token MLP
    Tokens: X [B, N, D]
    Latents: L [B, M, D]
    """

    def __init__(
        self,
        config: BudgieConfig,
        heads_tokens: int,
        heads_latents: int,
        mlp_mult: float = 4.0,
        dropout: float = 0.0,
        droppath: float = 0.0,
        latent_process_layers: int = 1,
        use_fft_in_latents: bool = False,
        gate_writeback: bool = True,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.dim = int(config.hidden_size)
        self.droppath = float(droppath)
        self.gate_writeback = bool(gate_writeback)

        mlp_cfg = _clone_cfg_for_mlp(config, mlp_mult=mlp_mult)

        self.norm_x1 = budgie_make_rmsnorm(config, self.dim, config.rms_norm_eps)
        self.norm_l1 = budgie_make_rmsnorm(config, self.dim, config.rms_norm_eps)
        self.read = GLACrossAttention(
            config,
            heads=int(heads_tokens),
            dropout=dropout,
            layer_idx=(layer_idx * 100) + 1,
        )

        self.latent_layers = nn.ModuleList()
        for i in range(int(latent_process_layers)):
            layer = nn.ModuleDict(
                {
                    "norm1": budgie_make_rmsnorm(config, self.dim, config.rms_norm_eps),
                    "self_attn": GLACrossAttention(
                        config,
                        heads=int(heads_latents),
                        dropout=dropout,
                        layer_idx=(layer_idx * 100) + 10 + i,
                    ),
                    "norm2": budgie_make_rmsnorm(config, self.dim, config.rms_norm_eps),
                    "mlp": budgie_make_mlp(mlp_cfg),
                }
            )
            self.latent_layers.append(layer)

        self.latent_fft = FFTTokenMixer(self.dim) if use_fft_in_latents else None

        self.norm_x2 = budgie_make_rmsnorm(config, self.dim, config.rms_norm_eps)
        self.norm_l2 = budgie_make_rmsnorm(config, self.dim, config.rms_norm_eps)
        self.write = GLACrossAttention(
            config,
            heads=int(heads_tokens),
            dropout=dropout,
            layer_idx=(layer_idx * 100) + 2,
        )

        self.write_gate = nn.Linear(self.dim, 1, bias=True) if self.gate_writeback else None

        self.norm_x3 = budgie_make_rmsnorm(config, self.dim, config.rms_norm_eps)
        self.mlp_x = budgie_make_mlp(mlp_cfg)

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.droppath <= 0.0:
            return x
        keep = 1.0 - self.droppath
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

    def forward(
        self,
        x: torch.Tensor,
        l: torch.Tensor,
        *,
        read_is_causal: bool = False,
        write_is_causal: bool = False,
        attn_mask_read: Optional[torch.Tensor] = None,
        attn_mask_write: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # In causal mode, latent processing must also be causal to avoid leaking
        # future token information through latent-to-latent mixing.
        latent_is_causal = bool(read_is_causal or write_is_causal)

        l = l + self._drop_path(
            self.read(
                self.norm_l1(l),
                self.norm_x1(x),
                attention_mask_2d=attn_mask_read,
                is_causal=read_is_causal,
            )
        )

        for layer in self.latent_layers:
            l_n = layer["norm1"](l)
            l = l + self._drop_path(layer["self_attn"](l_n, l_n, is_causal=latent_is_causal))
            if self.latent_fft is not None and not latent_is_causal:
                # FFT latent mixing is global/non-causal; skip it in causal mode.
                l = l + self._drop_path(self.latent_fft(l))
            l = l + self._drop_path(layer["mlp"](layer["norm2"](l)))

        w = self.write(
            self.norm_x2(x),
            self.norm_l2(l),
            attention_mask_2d=attn_mask_write,
            is_causal=write_is_causal,
        )
        if self.write_gate is not None:
            w = w * torch.sigmoid(self.write_gate(x))
        x = x + self._drop_path(w)

        x = x + self._drop_path(self.mlp_x(self.norm_x3(x)))
        return x, l


class LatentBottleneckBackbone(nn.Module):
    """
    Minimal standalone demo backbone (kept for smoke tests).
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        n_latents: int = 128,
        heads_tokens: int = 8,
        heads_latents: int = 8,
        mlp_mult: float = 4.0,
        dropout: float = 0.0,
        droppath: float = 0.0,
        latent_process_layers: int = 1,
        use_fft_in_latents: bool = False,
    ):
        super().__init__()
        cfg = BudgieConfig(
            vocab_size=int(vocab_size),
            hidden_size=int(dim),
            intermediate_size=int(dim * mlp_mult),
            num_hidden_layers=int(depth),
            num_attention_heads=int(max(heads_tokens, heads_latents)),
            num_key_value_heads=int(max(heads_tokens, heads_latents)),
            gla_num_groups=1,
            use_hybrid_layers=False,
            use_macro_structure=False,
            use_tiny_conv=False,
            use_gsm=False,
            use_liger_kernel=False,
            max_position_embeddings=4096,
        )

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.latents = nn.Parameter(torch.randn(n_latents, dim) / math.sqrt(dim))
        self.blocks = nn.ModuleList(
            [
                LatentBottleneckMacroBlock(
                    config=cfg,
                    heads_tokens=heads_tokens,
                    heads_latents=heads_latents,
                    mlp_mult=mlp_mult,
                    dropout=dropout,
                    droppath=(droppath * (i / max(depth - 1, 1))),
                    latent_process_layers=latent_process_layers,
                    use_fft_in_latents=use_fft_in_latents,
                    gate_writeback=True,
                    layer_idx=i,
                )
                for i in range(depth)
            ]
        )
        self.norm_out = budgie_make_rmsnorm(cfg, dim, cfg.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(input_ids)
        bsz = x.size(0)
        l = self.latents.unsqueeze(0).expand(bsz, -1, -1).contiguous()
        for blk in self.blocks:
            x, l = blk(x, l, read_is_causal=False, write_is_causal=False)
        return self.norm_out(x)


def maybe_compile(model: nn.Module) -> nn.Module:
    if hasattr(torch, "compile"):
        return torch.compile(model, mode="max-autotune", fullgraph=False)
    return model


__all__ = [
    "GLACrossAttention",
    "MultiHeadCrossAttention",
    "FFTTokenMixer",
    "LatentBottleneckMacroBlock",
    "LatentBottleneckBackbone",
    "maybe_compile",
]

# Back-compat alias; prefer `GLACrossAttention`.
MultiHeadCrossAttention = GLACrossAttention

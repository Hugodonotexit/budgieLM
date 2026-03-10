# budgieLM 1.0

`budgieLM 1.0` is a decoder-only long-context model stack built around a macro layer schedule:

`SWA -> Perceiver macro-block -> SWA(dilated) -> Bridge block -> SWA`

This repo contains the model package (config, decoder, LM head, attention variants, GSM mixer, Perceiver block).

## Architecture

![budgieLM graph](asset/BudgieLMgraph.png?v=2)
(For reference only, detail missing)
Default macro pattern (enabled by default):

- `use_macro_structure=True`
- `macro_structure_pattern="swa,perceiver,swa_dilated,bridge,swa"`

Layer tokens map to:

- `swa`: local GLA sliding-window attention (`gla_sliding`)
- `perceiver`: Perceiver-style latent bottleneck macro layer
- `swa_dilated`: dilated local GLA sliding-window attention (`gla_sliding_dilated`)
- `bridge`: landmark GLA bridge layer (`gla_landmark`)

## Core Concepts

### What is GSM?

`GSM` (Gated Spectral Mixer) is a global token mixer that works in the frequency domain.

At a high level:

1. Normalize and split channels into groups.
2. Run `rFFT` across the sequence axis.
3. Apply a learned complex frequency response plus a content-conditioned gate.
4. Run `irFFT` back to token space.
5. Project and return to the residual stream.

Why use it:

- It gives long-range mixing at roughly `O(N log N)` token cost instead of `O(N^2)`.
- It complements attention: attention handles selective routing, GSM handles smooth/global mixing.
- In this repo it is AMP-safe (FFT path in fp32, output cast back to model dtype).

### What is a Perceiver-style latent bottleneck?

A Perceiver-style latent bottleneck introduces a small latent array `L` (length `M`) that is much shorter than token length `N`.

The block does:

1. `Read`: latents cross-attend to tokens (`L <- X`).
2. `Latent process`: self-attn + MLP (and optional FFT mixer) in latent space only.
3. `Write`: tokens cross-attend back to latents (`X <- L`).
4. Token MLP residual.

Why use it:

- Global interaction cost becomes closer to `O(NM + M^2)` instead of full-token `O(N^2)`.
- You can keep strong global modeling while scaling to longer contexts by choosing moderate `M`.
- Tradeoff: `M` too small can bottleneck fine-grained token details.

## Bridge Block (1.0)

On bridge layers with GSM enabled, Budgie fuses landmark attention and GSM with a learned gate:

1. `Z = TinyConv(RMSNorm(X))` (or `RMSNorm(X)` if tiny conv is off)
2. `Y_lm = LandmarkGLA(Z)`, `Y_gsm = GSM(Z)`
3. `alpha = sigmoid(Linear(Pool(Z)))`
4. `Y = alpha * Y_lm + (1 - alpha) * Y_gsm`
5. `X' = X + Proj(Y)`

If GSM is enabled on non-bridge layers, it runs as an additional residual mixer branch (`attn_out + GSM(z)`).

## What Changed in 1.0

- Macro-structure scheduler in the core model.
- Perceiver macro layer integrated directly into decoder depth.
- Dilated SWA attention path (`gla_sliding_dilated`).
- Bridge fusion between Landmark GLA and GSM (with learned alpha gate).
- GSM refactor: no separate `GSMConfig`; `GSM` takes direct constructor args.
- Perceiver refactor to reuse Budgie internals (GLA projections, Budgie RMSNorm/MLP factories).
- `GLACrossAttention` naming for latent bottleneck cross-attention (alias kept: `MultiHeadCrossAttention`).

## Repository Layout

- `budgie/budgie_config.py`: `BudgieConfig`
- `budgie/budgie_model.py`: decoder stack assembly and macro scheduler
- `budgie/budgie_for_causal_lm.py`: Causal LM wrapper + loss
- `budgie/modeling_budgie_GLA.py`: GLA attention variants, tiny conv, decoder layer
- `budgie/modeling_budgie_gsm.py`: AMP-safe GSM token mixer + optional Triton kernel
- `budgie/modeling_budgie_latent_bottleneck.py`: Perceiver macro-block implementation

## Requirements

Required:

- Python 3.10+
- `torch`
- `transformers`

Optional acceleration:

- `xformers`
- `flash-attn` (or compatible flash attention backend in your environment)
- `liger-kernel`
- `causal-conv1d`
- `triton` (for fused GSM spectral modulation on CUDA)

## Quick Start

```python
import torch
from budgie import BudgieConfig, BudgieForCausalLM

cfg = BudgieConfig(
    vocab_size=32768,
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=20,
    num_attention_heads=16,
    max_position_embeddings=32768,
    # 1.0 macro stack
    use_macro_structure=True,
    macro_structure_pattern="swa,perceiver,swa_dilated,bridge,swa",
    # keep hybrid scheduler off when using macro pattern
    use_hybrid_layers=False,
    # bridge conv + GSM
    use_tiny_conv=True,
    tiny_conv_on_bridge_layers=True,
    use_gsm=True,
    gsm_on_local_layers=False,
    gsm_on_bridge_layers=True,
    gsm_bridge_start=1,  # bridge indexing is 1-based
)

model = BudgieForCausalLM(cfg).eval()
input_ids = torch.randint(0, cfg.vocab_size, (2, 256))
with torch.no_grad():
    out = model(input_ids=input_ids)
print(out.logits.shape)  # [2, 256, vocab_size]
```

## Core Config Guide

### 1) Macro structure (recommended for 1.0)

- `use_macro_structure`: enable pattern-based layer assignment.
- `macro_structure_pattern`: comma-separated tokens from `swa, perceiver, swa_dilated, bridge`.
- `swa_dilation`: dilation factor for `swa_dilated` layers.
- `swa_dilated_window`: optional window for dilated SWA (falls back to `sliding_window`).

### 2) Legacy hybrid scheduler

Use this only when `use_macro_structure=False`.

- `use_hybrid_layers`
- `bridge_every_n_layers`
- `bridge_layer_offset`
- `local_attn_implementation`
- `bridge_attn_implementation`

### 3) Tiny conv in bridge/local layers

- Global switch: `use_tiny_conv`

Per-layer placement knobs:

- `tiny_conv_on_local_layers`
- `tiny_conv_on_bridge_layers`
- `tiny_conv_every_n_bridge_layers`
- `tiny_conv_bridge_start`

### 4) GSM token mixer

- Global switch: `use_gsm`

Placement knobs:

- `gsm_on_local_layers`
- `gsm_on_bridge_layers`
- `gsm_every_n_bridge_layers`
- `gsm_bridge_start` (set `>= 1` for bridge index schedule)

Core hyperparameters:

- `gsm_n_groups`
- `gsm_gate_rank`
- `gsm_alpha`
- `gsm_dropout`
- `gsm_use_triton`
- `gsm_w_init_scale`
- `gsm_rms_eps`

### 5) Perceiver macro-block

- `perceiver_num_latents`
- `perceiver_heads_tokens`
- `perceiver_heads_latents`
- `perceiver_mlp_mult`
- `perceiver_dropout`
- `perceiver_droppath`
- `perceiver_attn_backend` (`sdpa`, `xformers`, `flash`)
- `perceiver_latent_process_layers`
- `perceiver_use_fft_in_latents`
- `perceiver_gate_writeback`
- `perceiver_read_is_causal`
- `perceiver_write_is_causal`

Guidance:

- `perceiver_read_is_causal=True` is recommended for strict autoregressive causality.
- `perceiver_write_is_causal=True` is recommended for strict autoregressive causality.
- `perceiver_use_fft_in_latents=True` adds a cheap global latent mixer path (auto-skipped when Perceiver is in causal mode).
- `perceiver_attn_backend="flash"` is accepted by config, but current Perceiver path runs via SDPA/xFormers helper.

### 6) Shared depth and multi-phase reuse

- `share_all_layers=True` shares one attention+MLP core across depth.
- `num_phases > 1` reruns the stack multiple times.
- `use_phase_layer_gates=True` adds learned residual gates per `(phase, layer)`.

## Recommended 1.0 Preset

For your requested structure:

- `use_macro_structure=True`
- `macro_structure_pattern="swa,perceiver,swa_dilated,bridge,swa"`
- `use_hybrid_layers=False`
- `use_tiny_conv=True`
- `tiny_conv_on_bridge_layers=True`
- `use_gsm=True`
- `gsm_on_bridge_layers=True`
- `gsm_on_local_layers=False`
- `perceiver_use_fft_in_latents=True` (optional but useful)

## Performance Notes

- Use `bf16` or `fp16` on CUDA.
- Compile when possible (`torch.compile`).
- SDPA is the default backend; optional kernels auto-fallback when unavailable.
- GSM forces FFT path compute in fp32 and casts outputs back to input dtype for AMP stability.

## License

MIT (see `LICENSE`).

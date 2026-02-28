# budgieLM1.0

`budgieLM1.0` is a compact decoder-only language model focused on long-context research.

It combines local sliding-window attention with periodic landmark bridge layers, plus optional tiny convolution, stack reuse, and shared-depth variants.

## Architecture

![budgieLM graph](asset/BudgieLMgraph.png?v=2)

## What is in 1.0

- Hybrid attention schedule with two layer types:
  - local layers: `gla_sliding`
  - bridge layers: `gla_landmark`
- Optional tiny causal depthwise convolution.
- Optional multi-phase stack reuse (`num_phases > 1`).
- Optional ALBERT-style depth sharing (`share_all_layers=True`).
- PyTorch SDPA-first implementation with fallbacks and optional acceleration (`xformers`, `flash-attn`, `liger-kernel`, `causal_conv1d`).

## Repository layout

- `budgie/budgie_config.py`: `BudgieConfig`
- `budgie/budgie_model.py`: core decoder model
- `budgie/budgie_for_causal_lm.py`: causal LM head
- `budgie/modeling_budgie_GLA.py`: attention and conv implementations

## Requirements

- Python 3.10+
- `torch`
- `transformers`

Optional dependencies:

- `xformers`
- `flash-attn`
- `liger-kernel`
- `causal-conv1d`

## Quick start

```bash
cd ai/budgie
python - <<'PY'
import torch
from budgie import BudgieConfig, BudgieForCausalLM

cfg = BudgieConfig(
    vocab_size=32768,
    hidden_size=3072,
    intermediate_size=12288,
    num_hidden_layers=30,
    num_attention_heads=32,
    max_position_embeddings=65536,
)

model = BudgieForCausalLM(cfg).eval()
input_ids = torch.randint(0, cfg.vocab_size, (1, 128))
out = model(input_ids=input_ids)
print(out.logits.shape)
PY
```

## Core configuration knobs

### Attention backends

- `_attn_implementation="sdpa"`
- `local_attn_implementation="gla_sliding"`
- `bridge_attn_implementation="gla_landmark"`

### Hybrid layer schedule

Enable with:

- `use_hybrid_layers=True`
- `bridge_every_n_layers`
- `bridge_layer_offset`

Bridge layer indices follow:

`bridge_layer_offset + k * bridge_every_n_layers`

### Sliding window and landmarks

- `sliding_window`: window size for local layers.
- `landmark_every`: positional landmarks (`N-1, 2N-1, ...`).
- `landmark_token_id`: token-driven landmarks (`input_ids == landmark_token_id`).

### Tiny convolution

- `use_tiny_conv=True`
- `use_causal_conv1d=True` to use `causal_conv1d` when available.
- `tiny_conv_on_local_layers` / `tiny_conv_on_bridge_layers` to control placement.

### Multi-phase stack reuse

- `num_phases > 1` reruns the full decoder stack.
- `use_phase_layer_gates=True` enables per-(phase, layer) residual gates.
- KV-cache works with dynamic cache; memory scales with `num_phases`.

### Depth sharing

- `share_all_layers=True` shares one attention+MLP core across all layers.

### GLA grouping

- `gla_num_groups` must divide `num_attention_heads`.

## GPU compatibility

- Designed to run without Hopper-only FlashAttention requirements.
- SDPA and eager paths are supported.
- On older GPUs, optional kernels may not always activate; implementation falls back to PyTorch paths.

## Training note

This repository is the model package. The training pipeline in this workspace lives outside this folder (for example `ai/train_pipe.py`).

## License

MIT (see `LICENSE`).

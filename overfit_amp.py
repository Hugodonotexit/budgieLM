#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass

import torch

from budgie import BudgieConfig, BudgieForCausalLM


@dataclass(frozen=True)
class ByteTokenizer:
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    vocab_size: int = 260  # 4 specials + 256 bytes

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [4 + b for b in text.encode("utf-8", errors="replace")]
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        out = []
        for token_id in ids:
            if 4 <= token_id < 260:
                out.append(token_id - 4)
        return bytes(out).decode("utf-8", errors="replace")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown --device {requested!r}. Expected: auto|cuda|cpu.")


@torch.no_grad()
def greedy_decode(
    model: BudgieForCausalLM,
    input_ids: torch.Tensor,  # (1, S)
    *,
    max_new_tokens: int,
    amp: bool,
) -> torch.Tensor:
    model.eval()
    ids = input_ids
    for _ in range(max_new_tokens):
        if amp and ids.is_cuda:
            with torch.amp.autocast(dtype=torch.float16, enabled=True, device_type="cuda"):
                out = model(ids, use_cache=False)
        else:
            out = model(ids, use_cache=False)
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Budgie tiny AMP overfit smoke test (byte-level tokenizer).")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Tiny model knobs (defaults follow the plan).
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--intermediate_size", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv_heads", type=int, default=4)
    parser.add_argument("--num_phases", type=int, default=2)
    parser.add_argument("--sliding_window", type=int, default=64)
    parser.add_argument("--landmark_every", type=int, default=16)
    parser.add_argument("--use_xformers", action=argparse.BooleanOptionalAction, default=True)

    # Sampling / generation
    parser.add_argument("--prompt_len", type=int, default=32)
    parser.add_argument("--gen_len", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=300, help="How many times to repeat the tiny corpus.")
    args = parser.parse_args()

    _set_seed(args.seed)
    device = _pick_device(args.device)
    amp = device.type == "cuda"
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        major, minor = torch.cuda.get_device_capability(device)
        print(f"device: cuda ({props.name}) sm{major}{minor}  amp={amp}")
    else:
        print(f"device: {device.type}  amp={amp}")

    tok = ByteTokenizer()

    corpus = "BUDGIE OVERFIT TEST\n" + ("hello budgie!\n" * 50) + "END\n"
    train_text = corpus * int(args.repeat)
    data_ids = tok.encode(train_text, add_bos=True, add_eos=True)
    data = torch.tensor(data_ids, dtype=torch.long)  # keep on CPU

    if data.numel() <= args.seq_len + 1:
        raise ValueError("Training text too small for requested --seq_len.")

    config = BudgieConfig(
        vocab_size=tok.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        max_position_embeddings=max(256, args.seq_len + args.gen_len + 8),
        attention_dropout=0.0,
        use_cache=False,
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        use_xformers=bool(args.use_xformers),
        use_liger_kernel=False,
        _attn_implementation="sdpa",
        use_hybrid_layers=True,
        local_attn_implementation="gla_sliding",
        bridge_attn_implementation="gla_landmark",
        bridge_every_n_layers=2,
        bridge_layer_offset=1,
        use_tiny_conv=True,
        tiny_conv_kernel_size=4,
        tiny_conv_on_local_layers=False,
        tiny_conv_on_bridge_layers=True,
        tiny_conv_bridge_start=2,
        tiny_conv_every_n_bridge_layers=2,
        use_causal_conv1d=True,
        sliding_window=int(args.sliding_window),
        landmark_every=int(args.landmark_every),
        share_all_layers=True,
        num_phases=int(args.num_phases),
        use_phase_layer_gates=True,
    )

    model = BudgieForCausalLM(config).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=0.0)
    scaler = torch.amp.GradScaler(enabled=amp)

    seq = int(args.seq_len)
    batch = int(args.batch_size)
    arange = torch.arange(seq, dtype=torch.long)

    interval_t0 = time.perf_counter()
    interval_tokens = 0

    for step in range(1, int(args.steps) + 1):
        max_start = data.shape[0] - seq + 1
        starts = torch.randint(0, max_start, (batch,), dtype=torch.long)
        idx = starts[:, None] + arange[None, :]
        x = data[idx].to(device, non_blocking=True)

        if amp:
            with torch.amp.autocast(dtype=torch.float16, enabled=True, device_type="cuda"):
                out = model(x, labels=x, use_cache=False)
                loss = out.loss
            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            out = model(x, labels=x, use_cache=False)
            loss = out.loss
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        interval_tokens += batch * seq

        if step == 1 or (args.log_every and step % int(args.log_every) == 0) or step == int(args.steps):
            if amp:
                torch.cuda.synchronize()
            now = time.perf_counter()
            dt = max(1e-9, now - interval_t0)
            tps = interval_tokens / dt
            interval_t0 = now
            interval_tokens = 0
            print(f"step {step:5d}/{int(args.steps)}  loss {float(loss):.4f}  tok/s {tps:,.0f}")

            prompt_len = min(int(args.prompt_len), data.shape[0] - 1)
            prompt = data[:prompt_len].unsqueeze(0).to(device)
            gen = greedy_decode(model, prompt, max_new_tokens=int(args.gen_len), amp=amp)
            cont = tok.decode(gen[0, prompt_len:].tolist())
            print("sample:", repr(cont[:200]))
            model.train()


if __name__ == "__main__":
    main()

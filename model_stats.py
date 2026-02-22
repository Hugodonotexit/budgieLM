#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

from budgie import BudgieConfig, BudgieForCausalLM


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


def _parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("fp32", "float32"):
        return torch.float32
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError("Unknown --dtype. Expected: fp32|fp16|bf16.")


def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 0:
        return f"{num_bytes} B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(num_bytes)
    u = 0
    while size >= 1024.0 and u < len(units) - 1:
        size /= 1024.0
        u += 1
    if u == 0:
        return f"{int(size)} {units[u]}"
    return f"{size:.2f} {units[u]}"


@dataclass(frozen=True)
class CudaMem:
    allocated: int
    reserved: int
    max_allocated: int
    max_reserved: int


def _cuda_mem(device: torch.device) -> CudaMem:
    return CudaMem(
        allocated=int(torch.cuda.memory_allocated(device)),
        reserved=int(torch.cuda.memory_reserved(device)),
        max_allocated=int(torch.cuda.max_memory_allocated(device)),
        max_reserved=int(torch.cuda.max_memory_reserved(device)),
    )


def _print_cuda_mem(prefix: str, mem: CudaMem) -> None:
    print(
        f"{prefix}"
        f"  alloc={_format_bytes(mem.allocated)}"
        f"  reserv={_format_bytes(mem.reserved)}"
        f"  peak_alloc={_format_bytes(mem.max_allocated)}"
        f"  peak_reserv={_format_bytes(mem.max_reserved)}"
    )


def _param_stats(model: torch.nn.Module) -> tuple[int, int, int, int]:
    total = 0
    trainable = 0
    nbytes = 0
    trainable_nbytes = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            trainable_nbytes += n * p.element_size()
        nbytes += n * p.element_size()
    return total, trainable, nbytes, trainable_nbytes


def _optimizer_state_multiplier(name: str) -> int:
    name = name.lower()
    if name == "adamw":
        return 2  # exp_avg + exp_avg_sq
    if name == "sgdm":
        return 1  # momentum buffer
    if name == "sgd":
        return 0
    raise ValueError(f"Unknown optimizer {name!r}. Expected: adamw|sgd|sgdm.")


def _print_config_summary(config: BudgieConfig) -> None:
    # Keep this resilient to config field changes.
    keys = [
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "use_cache",
        "use_xformers",
        "use_hybrid_layers",
        "sliding_window",
        "landmark_every",
        "share_all_layers",
        "num_phases",
        "use_phase_layer_gates",
        "use_tiny_conv",
        "use_causal_conv1d",
    ]
    parts = []
    for k in keys:
        if hasattr(config, k):
            parts.append(f"{k}={getattr(config, k)}")
    print("config:", "  ".join(parts))


def _load_config(config_path: str | None) -> BudgieConfig:
    if not config_path:
        return BudgieConfig()  # <-- defaults
    if str(config_path).endswith(".json"):
        return BudgieConfig.from_json_file(config_path)
    return BudgieConfig.from_pretrained(config_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print Budgie parameter counts + CUDA VRAM usage (uses default BudgieConfig unless --config_path is set)."
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--grad_ckpt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing for train steps (lower VRAM, slower).",
    )

    parser.add_argument("--run_forward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_train_step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", type=int, default=1, help="Number of train steps when --run_train_step is enabled.")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--use_cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "sgd", "sgdm"],
        help="Optimizer to use when --run_train_step is enabled (affects VRAM usage a lot).",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for --optimizer sgdm (ignored otherwise).",
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Optional path to a Budgie config (directory with config.json, or a config.json file).",
    )

    args = parser.parse_args()

    device = _pick_device(args.device)
    dtype = _parse_dtype(args.dtype)

    # CPU safety: fp16/bf16 are often unsupported or slow on CPU.
    if device.type != "cuda":
        args.amp = False
        if dtype != torch.float32:
            print("CPU device: forcing --dtype fp32 (fp16/bf16 often unsupported on CPU).")
            dtype = torch.float32
    else:
        # bf16 requires SM80+ typically.
        if dtype == torch.bfloat16:
            try:
                major, minor = torch.cuda.get_device_capability(device)
                if (major, minor) < (8, 0):
                    print("CUDA SM<80: forcing --dtype fp16 (bf16 typically unsupported on this GPU).")
                    dtype = torch.float16
            except Exception:
                pass

    config = _load_config(args.config_path)

    # Allow runtime override of use_cache without rebuilding config
    if hasattr(config, "use_cache"):
        config.use_cache = bool(args.use_cache)

    _print_config_summary(config)
    if not bool(getattr(config, "share_all_layers", False)) and int(getattr(config, "num_phases", 1)) > 1:
        print("note: `num_phases>1` only applies when `share_all_layers=True` (phases are ignored otherwise).")

    if device.type == "cuda":
        try:
            props = torch.cuda.get_device_properties(device)
            major, minor = torch.cuda.get_device_capability(device)
            print(
                f"cuda device: {props.name}  capability=sm{major}{minor}  total_vram={_format_bytes(int(props.total_memory))}"
            )
        except Exception:
            pass

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        _print_cuda_mem("cuda baseline:", _cuda_mem(device))

    model = BudgieForCausalLM(config)
    total_params, trainable_params, param_bytes_cpu, trainable_bytes_cpu = _param_stats(model)
    print(f"params: total={total_params:,}  trainable={trainable_params:,}  (CPU bytes={_format_bytes(param_bytes_cpu)})")

    model = model.to(device=device, dtype=dtype)
    _, _, param_bytes, trainable_bytes = _param_stats(model)
    print(
        f"params: bytes_on_device≈{_format_bytes(param_bytes)}  trainable≈{_format_bytes(trainable_bytes)}  dtype={dtype}"
    )
    model_param_dtype = next(model.parameters()).dtype

    if device.type == "cuda":
        _print_cuda_mem("after model.to:", _cuda_mem(device))

    if args.run_train_step:
        opt_mult = _optimizer_state_multiplier(str(args.optimizer))
        train_lower_bound = int(param_bytes + trainable_bytes + (opt_mult * trainable_bytes))
        print(
            "train lower bound (weights+grads+optimizer state; excludes activations/temp buffers): "
            f"≥{_format_bytes(train_lower_bound)}  optimizer={args.optimizer}  param_dtype={model_param_dtype}"
        )
        if device.type == "cuda":
            try:
                total_vram = int(torch.cuda.get_device_properties(device).total_memory)
                if train_lower_bound > total_vram:
                    print(
                        "warning: lower bound already exceeds total VRAM; expect OOM unless you change dtype/optimizer or use sharding/offload."
                    )
            except Exception:
                pass

    if args.grad_ckpt:
        try:
            model.gradient_checkpointing_enable()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
            print("grad_ckpt: enabled (use_cache forced to False)")
        except Exception as exc:
            print(f"grad_ckpt: failed to enable ({exc}); continuing without checkpointing.")

    batch_size = int(args.batch_size)
    seq_len = int(args.seq_len)

    # Input ids use config.vocab_size; if missing, fall back to something sane.
    vocab_size = int(getattr(config, "vocab_size", 32000))
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)

    if args.run_forward:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            if args.amp and device.type == "cuda":
                with torch.amp.autocast(dtype=torch.float16, enabled=True, device_type="cuda"):
                    _ = model(input_ids, attention_mask=attention_mask, use_cache=bool(args.use_cache))
            else:
                _ = model(input_ids, attention_mask=attention_mask, use_cache=bool(args.use_cache))
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = max(1e-9, time.perf_counter() - t0)
        tps = (batch_size * seq_len) / dt
        print(f"forward: {dt*1e3:.1f} ms  tok/s={tps:,.0f}")
        if device.type == "cuda":
            _print_cuda_mem("after forward:", _cuda_mem(device))

    if args.run_train_step:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        if str(args.optimizer) == "adamw":
            import bitsandbytes as bnb
            optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=float(args.lr), weight_decay=0.0)
        elif str(args.optimizer) == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=float(args.lr), momentum=0.0, weight_decay=0.0)
        elif str(args.optimizer) == "sgdm":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=float(args.lr), momentum=float(args.momentum), weight_decay=0.0
            )
        else:
            raise ValueError(f"Unknown optimizer {args.optimizer!r}.")
        scaler_enabled = bool(args.amp and device.type == "cuda" and model_param_dtype == torch.float32)
        if args.amp and device.type == "cuda" and not scaler_enabled:
            print("amp: GradScaler disabled (requires FP32 params; you are running FP16 params)")
        scaler = torch.amp.GradScaler(enabled=scaler_enabled)

        steps = int(args.steps)
        try:
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                if args.amp and device.type == "cuda":
                    with torch.amp.autocast(dtype=torch.float16, enabled=True, device_type="cuda"):
                        out = model(input_ids, attention_mask=attention_mask, labels=input_ids, use_cache=False)
                        loss = out.loss
                    if scaler_enabled:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                else:
                    out = model(input_ids, attention_mask=attention_mask, labels=input_ids, use_cache=False)
                    loss = out.loss
                    loss.backward()
                    optimizer.step()
        except torch.OutOfMemoryError as exc:
            if device.type == "cuda":
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                _print_cuda_mem("after OOM:", _cuda_mem(device))
            print(
                "OOM during train step. This usually happens when optimizer state (e.g. AdamW exp_avg/exp_avg_sq) "
                "does not fit in VRAM. Try `--dtype fp16`, `--optimizer sgd/sgdm`, smaller model/seq_len, or sharding/offload."
            )
            raise SystemExit(1) from exc

        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"train_step: loss={float(loss):.4f}  steps={steps}")
        if device.type == "cuda":
            _print_cuda_mem("after train_step:", _cuda_mem(device))

if __name__ == "__main__":
    main()

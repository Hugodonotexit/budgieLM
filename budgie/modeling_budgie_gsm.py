from __future__ import annotations

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl

    TRITON_OK = True
except Exception:
    TRITON_OK = False


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return x * weight


if TRITON_OK:

    @triton.jit
    def spec_modulate_kernel(
        Fr_ptr,
        Fi_ptr,
        Wr_ptr,
        Wi_ptr,
        M_ptr,
        Or_ptr,
        Oi_ptr,
        FREQ: tl.constexpr,
        DG: tl.constexpr,
        alpha: tl.constexpr,
        stride_fr_bg,
        stride_fr_f,
        stride_fr_d,
        stride_w_f,
        stride_w_d,
        stride_m_bg,
        stride_m_f,
        stride_o_bg,
        stride_o_f,
        stride_o_d,
        BLOCK_D: tl.constexpr,
    ):
        pid_bg = tl.program_id(0)
        pid_f = tl.program_id(1)

        mask_f = pid_f < FREQ
        d = tl.arange(0, BLOCK_D)
        mask_d = d < DG
        mask = mask_f & mask_d

        m = tl.load(M_ptr + pid_bg * stride_m_bg + pid_f * stride_m_f, mask=mask_f, other=0.0).to(tl.float32)
        scale = 1.0 + alpha * m

        fr = tl.load(
            Fr_ptr + pid_bg * stride_fr_bg + pid_f * stride_fr_f + d * stride_fr_d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        fi = tl.load(
            Fi_ptr + pid_bg * stride_fr_bg + pid_f * stride_fr_f + d * stride_fr_d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        wr = tl.load(Wr_ptr + pid_f * stride_w_f + d * stride_w_d, mask=mask, other=0.0).to(tl.float32)
        wi = tl.load(Wi_ptr + pid_f * stride_w_f + d * stride_w_d, mask=mask, other=0.0).to(tl.float32)

        or_ = (fr * wr - fi * wi) * scale
        oi_ = (fr * wi + fi * wr) * scale

        tl.store(Or_ptr + pid_bg * stride_o_bg + pid_f * stride_o_f + d * stride_o_d, or_, mask=mask)
        tl.store(Oi_ptr + pid_bg * stride_o_bg + pid_f * stride_o_f + d * stride_o_d, oi_, mask=mask)


class GSM(nn.Module):
    """
    Gated Spectral Mixer (GSM)
    - Group channels into n_groups
    - rFFT across sequence (fp32)
    - Apply learned complex per-frequency response + content-conditioned gate
    - irFFT back (fp32), cast to input dtype
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        *,
        n_groups: int = 8,
        gate_rank: int = 32,
        alpha: float = 0.5,
        dropout: float = 0.0,
        use_triton: bool = True,
        w_init_scale: float = 0.02,
        rms_eps: float = 1e-6,
    ):
        super().__init__()
        if d_model % n_groups != 0:
            raise ValueError("`d_model` must be divisible by `n_groups`.")

        self.d_model = int(d_model)
        self.n_groups = int(n_groups)
        self.gate_rank = int(gate_rank)
        self.alpha = float(alpha)
        self.use_triton = bool(use_triton)
        self.w_init_scale = float(w_init_scale)
        self.rms_eps = float(rms_eps)
        self.max_seq_len = int(max_seq_len)
        self.dg = d_model // n_groups
        self.Fmax = max_seq_len // 2 + 1

        self.norm_w = nn.Parameter(torch.ones(d_model))
        self.Wr = nn.Parameter(torch.empty(self.Fmax, self.dg))
        self.Wi = nn.Parameter(torch.empty(self.Fmax, self.dg))
        self.pool_proj = nn.Linear(d_model, n_groups * gate_rank, bias=True)
        self.U = nn.Parameter(torch.empty(n_groups, gate_rank, self.Fmax))
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.Wr, mean=1.0, std=self.w_init_scale)
        nn.init.normal_(self.Wi, mean=0.0, std=self.w_init_scale)
        nn.init.normal_(self.U, mean=0.0, std=0.02)
        nn.init.zeros_(self.pool_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected hidden size {self.d_model}, got {d_model}.")
        if seqlen > self.max_seq_len:
            raise ValueError(
                f"N={seqlen} > max_seq_len={self.max_seq_len}. Add interpolation for Wr/Wi/U."
            )

        x_n = rmsnorm(x, self.norm_w, eps=self.rms_eps)

        g = self.n_groups
        dg = self.dg
        xg = x_n.view(bsz, seqlen, g, dg).permute(0, 2, 1, 3).contiguous().view(bsz * g, seqlen, dg)

        xg_f = xg.float()
        fg = torch.fft.rfft(xg_f, n=seqlen, dim=1)
        fr = fg.real.contiguous()
        fi = fg.imag.contiguous()

        s = x_n.mean(dim=1).float()
        gate_rank = self.gate_rank
        gate = torch.sigmoid(self.pool_proj(s)).view(bsz, g, gate_rank)

        freq = seqlen // 2 + 1
        m = torch.einsum("bgr,grf->bgf", gate, self.U[:, :, :freq]).reshape(bsz * g, freq).contiguous()

        wr = self.Wr[:freq]
        wi = self.Wi[:freq]

        if self.use_triton and TRITON_OK and x.is_cuda:
            bg = bsz * g
            out_r = torch.empty((bg, freq, dg), device=x.device, dtype=torch.float32)
            out_i = torch.empty((bg, freq, dg), device=x.device, dtype=torch.float32)
            grid = (bg, freq)

            if dg >= 128:
                block_d = 128
                num_warps = 4
            elif dg >= 64:
                block_d = 64
                num_warps = 4
            else:
                block_d = 32
                num_warps = 2

            spec_modulate_kernel[grid](
                fr,
                fi,
                wr,
                wi,
                m,
                out_r,
                out_i,
                FREQ=freq,
                DG=dg,
                alpha=self.alpha,
                stride_fr_bg=fr.stride(0),
                stride_fr_f=fr.stride(1),
                stride_fr_d=fr.stride(2),
                stride_w_f=wr.stride(0),
                stride_w_d=wr.stride(1),
                stride_m_bg=m.stride(0),
                stride_m_f=m.stride(1),
                stride_o_bg=out_r.stride(0),
                stride_o_f=out_r.stride(1),
                stride_o_d=out_r.stride(2),
                BLOCK_D=block_d,
                num_warps=num_warps,
            )

            fmod = torch.complex(out_r, out_i)
        else:
            scale = (1.0 + self.alpha * m).unsqueeze(-1)
            wr = wr.unsqueeze(0)
            wi = wi.unsqueeze(0)
            out_r = (fr * wr - fi * wi) * scale
            out_i = (fr * wi + fi * wr) * scale
            fmod = torch.complex(out_r, out_i)

        yg = torch.fft.irfft(fmod, n=seqlen, dim=1)
        y = yg.view(bsz, g, seqlen, dg).permute(0, 2, 1, 3).contiguous().view(bsz, seqlen, d_model)
        y = y.to(dtype=x.dtype)
        y = self.drop(self.proj(y))
        return y


class GSMBlock(nn.Module):
    """Residual wrapper: x + GSM(x)."""

    def __init__(self, d_model: int, max_seq_len: int, **gsm_kwargs):
        super().__init__()
        self.gsm = GSM(d_model=d_model, max_seq_len=max_seq_len, **gsm_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gsm(x)


__all__ = ["GSM", "GSMBlock", "TRITON_OK"]

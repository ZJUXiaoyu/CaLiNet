"""Plain DL baselines: CNNBaseline (1D U-Net) and TransformerBaseline.

Distinguished from 1D U-Net w/ anchor / CaLiNet-E by NOT having the W_global linear
anchor — they learn the full 3-lead -> 12-lead mapping from scratch.
This isolates the contribution of the linear anchor (1D U-Net w/ anchor = CNN +
anchor) and the calibration framework (CaLiNet-E = 1D U-Net w/ anchor + per-patient
calibration + FiLM) in the ablation table.

Loss / val flow are shared with 1D U-Net w/ anchor (MSE on reconstructed leads only,
val_score = 0.3 PCC + 0.4 exp(-nrmse/tau_n) + 0.3 exp(-morph/tau_m)).

Parameter budget chosen to roughly match 1D U-Net w/ anchor backbone (~3M):
  CNNBaseline       channels=(32,64,128,256)            ~2.9M
  TransformerBaseline d=192, 6 layers, 8 heads, ffn=768 ~2.7M
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Building blocks (similar to ResidualUNet ConvBlock; kept local so
# baselines.py is self-contained and the ablation diff is easy to read.)
# ----------------------------------------------------------------------
class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================================================================
# CNN baseline — plain 1D U-Net, no linear anchor, no zero-init head.
# ======================================================================
class CNNBaseline(nn.Module):
    """3-lead -> 12-lead via plain 1D U-Net (random init).

    Forward signature matches 1D U-Net w/ anchor for drop-in compatibility with the
    07-style train script: input (B, T, n_in), output (B, T, n_out).
    """

    def __init__(
        self,
        n_in: int = 3,
        n_out: int = 12,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        pad_to_multiple: int = 16,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.pad_to_multiple = pad_to_multiple

        channels = tuple(channels)
        self.encoders = nn.ModuleList()
        self.downsample = nn.ModuleList()
        c_prev = n_in
        for c in channels:
            self.encoders.append(_ConvBlock(c_prev, c))
            self.downsample.append(nn.Conv1d(c, c, kernel_size=2, stride=2))
            c_prev = c

        self.bottleneck = _ConvBlock(channels[-1], channels[-1])

        self.upsample = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for c in reversed(channels):
            self.upsample.append(
                nn.ConvTranspose1d(c_prev, c, kernel_size=2, stride=2)
            )
            self.decoders.append(_ConvBlock(c * 2, c))
            c_prev = c

        # Standard init head — NOT zero-init (no linear baseline to fall to).
        self.head = nn.Conv1d(channels[0], n_out, kernel_size=1)

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        L = x.shape[-1]
        m = self.pad_to_multiple
        pad = (m - L % m) % m
        if pad:
            x = F.pad(x, (0, pad), mode="reflect")
        return x, pad

    def forward(self, Xt: torch.Tensor) -> torch.Tensor:
        # Xt: (B, T, n_in) -> (B, T, n_out)
        x = Xt.transpose(1, 2)                      # (B, n_in, T)
        x, pad = self._pad(x)
        skips = []
        h = x
        for enc, down in zip(self.encoders, self.downsample):
            h = enc(h)
            skips.append(h)
            h = down(h)
        h = self.bottleneck(h)
        for up, dec, skip in zip(self.upsample, self.decoders, reversed(skips)):
            h = up(h)
            if h.shape[-1] != skip.shape[-1]:
                h = h[..., : skip.shape[-1]]
            h = torch.cat([h, skip], dim=1)
            h = dec(h)
        y = self.head(h)
        if pad:
            y = y[..., :-pad]
        return y.transpose(1, 2)                    # (B, T, n_out)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
# Transformer baseline — linear proj -> TransformerEncoder -> linear proj.
# ======================================================================
class _SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        return x + self.pe[:, : x.shape[1]]


class TransformerBaseline(nn.Module):
    """3-lead -> 12-lead via transformer encoder over time.

    Standard pre-norm encoder, sinusoidal positional encoding.
    Default config: d_model=192, 6 layers, 8 heads, ffn=768 -> ~2.7M params.
    """

    def __init__(
        self,
        n_in: int = 3,
        n_out: int = 12,
        d_model: int = 192,
        n_layers: int = 6,
        n_heads: int = 8,
        ffn: int = 768,
        dropout: float = 0.1,
        max_len: int = 4096,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.in_proj = nn.Linear(n_in, d_model)
        self.pos = _SinusoidalPE(d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, n_out)

    def forward(self, Xt: torch.Tensor) -> torch.Tensor:
        # Xt: (B, T, n_in) -> (B, T, n_out)
        h = self.in_proj(Xt)
        h = self.pos(h)
        h = self.encoder(h)
        h = self.out_norm(h)
        return self.out_proj(h)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

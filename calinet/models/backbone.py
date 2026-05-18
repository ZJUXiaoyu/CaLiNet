"""Shared 1D U-Net backbone for 1D U-Net w/ anchor and CaLiNet-E.

Locked design rules (v1.2):
  1. Single nn.Module class. FiLM conditioning is gated by a flag so
     1D U-Net w/ anchor (use_film_conditioning=False) and CaLiNet-E (=True) share the
     EXACT same architecture, channel counts, and kernel sizes. This
     keeps the '1D U-Net w/ anchor = CaLiNet-E without calibration' ablation clean.
  2. Final conv layer is ZERO-INITIALIZED so that an untrained backbone
     outputs ~0 → predict_residual=True wrappers degrade to their
     linear baseline at epoch 0 (1D U-Net w/ anchor→GL, CaLiNet-E→PCM). Training
     curves then start from a known baseline and can only go up.
  3. Reflect-pad to next multiple of `pad_to_multiple` so 1000-sample
     windows survive the U-Net's 4 downsample/upsample stages.
  4. FiLM uses (1 + γ) * h + β so γ=β=0 at init is identity modulation.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Conv → GroupNorm → GELU → Conv → GroupNorm → GELU."""
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


class FiLMBlock(nn.Module):
    """Feature-wise linear modulation: h ↦ (1 + γ) * h + β.

    γ = β = 0 at init → identity, so adding FiLM does not perturb
    a pretrained backbone.
    """
    def __init__(self, channels: int, embedding_dim: int):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        # h: [B, C, T], e: [B, embedding_dim]
        gb = self.proj(e)                   # [B, 2C]
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)         # [B, C, 1]
        beta  = beta.unsqueeze(-1)
        return (1.0 + gamma) * h + beta


# ──────────────────────────────────────────────────────────────────────
# Main backbone
# ──────────────────────────────────────────────────────────────────────
class ResidualUNet(nn.Module):
    """1D U-Net producing a 12-lead correction signal from 3-lead input.

    Output shape matches input length (after de-padding).

    Parameters
    ----------
    n_in            : input lead count (default 3 for I, II, V2)
    n_out           : output lead count (default 12)
    channels        : encoder channel widths, e.g. (32, 64, 128, 256)
    embedding_dim   : FiLM conditioning dimension (only used when
                      use_film_conditioning=True)
    use_film_conditioning : if True, expects e_i in forward and applies
                      FiLM at bottleneck and the last 2 encoder blocks.
    pad_to_multiple : reflect-pad the time axis up to a multiple of this
                      (must equal 2 ** len(channels)).
    film_layers     : which named blocks to insert FiLM at; default
                      ('bottleneck', 'enc_3', 'enc_4').
    """

    def __init__(
        self,
        n_in: int = 3,
        n_out: int = 12,
        channels: Sequence[int] = (32, 64, 128, 256),
        embedding_dim: int = 128,
        use_film_conditioning: bool = False,
        pad_to_multiple: int = 16,
        film_layers: Sequence[str] = ("bottleneck", "enc_3", "enc_4"),
    ):
        super().__init__()
        channels = tuple(channels)
        self.use_film = use_film_conditioning
        self.pad_to_multiple = pad_to_multiple
        self.film_layers = set(film_layers) if use_film_conditioning else set()

        # Encoder
        self.encoders = nn.ModuleList()
        self.downsample = nn.ModuleList()
        c_prev = n_in
        for c in channels:
            self.encoders.append(ConvBlock(c_prev, c))
            self.downsample.append(nn.Conv1d(c, c, kernel_size=2, stride=2))
            c_prev = c

        # Bottleneck
        self.bottleneck = ConvBlock(channels[-1], channels[-1])

        # Decoder
        self.upsample = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for c in reversed(channels):
            self.upsample.append(
                nn.ConvTranspose1d(c_prev, c, kernel_size=2, stride=2)
            )
            self.decoders.append(ConvBlock(c * 2, c))
            c_prev = c

        # Final 1x1 conv → 12 leads, ZERO-INITIALIZED
        self.head = nn.Conv1d(channels[0], n_out, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # FiLM blocks
        if use_film_conditioning:
            self.film_blocks = nn.ModuleDict()
            self.embed = nn.Sequential(
                nn.Linear(n_in * n_out + n_out, embedding_dim),
                nn.GELU(),
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, embedding_dim),
            )
            for name in film_layers:
                if name == "bottleneck":
                    ch = channels[-1]
                elif name.startswith("enc_"):
                    idx = int(name.split("_")[1]) - 1
                    ch = channels[idx]
                else:
                    raise ValueError(f"unknown film layer: {name}")
                self.film_blocks[name] = FiLMBlock(ch, embedding_dim)

    # ------------------------------------------------------------------
    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        L = x.shape[-1]
        m = self.pad_to_multiple
        pad = (m - L % m) % m
        if pad:
            x = F.pad(x, (0, pad), mode="reflect")
        return x, pad

    def forward(
        self,
        x: torch.Tensor,                   # (B, n_in, T)
        e_i: torch.Tensor | None = None,   # (B, embedding_dim) or None
    ) -> torch.Tensor:
        x, pad = self._pad(x)

        # Encoder with optional FiLM after each enc block
        skips = []
        h = x
        for i, (enc, down) in enumerate(zip(self.encoders, self.downsample)):
            h = enc(h)
            name = f"enc_{i+1}"
            if self.use_film and name in self.film_layers and e_i is not None:
                h = self.film_blocks[name](h, e_i)
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.bottleneck(h)
        if self.use_film and "bottleneck" in self.film_layers and e_i is not None:
            h = self.film_blocks["bottleneck"](h, e_i)

        # Decoder
        for up, dec, skip in zip(self.upsample, self.decoders, reversed(skips)):
            h = up(h)
            # Crop in case of odd length
            if h.shape[-1] != skip.shape[-1]:
                h = h[..., : skip.shape[-1]]
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        y = self.head(h)
        if pad:
            y = y[..., :-pad]
        return y

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

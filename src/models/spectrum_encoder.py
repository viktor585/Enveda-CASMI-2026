"""
src/models/spectrum_encoder.py

The Transformer Encoder stack ("the reader"): turns one batch of filtered,
padded MS2 spectra + precursor-ion metadata into a dense latent memory
matrix the decoder can cross-attend over.

Pipeline:
    1. Continuous peak embeddings — `SinusoidalMZEmbedding` (from
       src/tokenizer.py) turns each (m/z, intensity) peak pair into a
       d_model-dim vector. Padded peak slots embed too, but are masked out
       of attention via the padding mask, not filtered before embedding.
    2. Metadata embeddings — precursor m/z (via the same sinusoidal m/z
       embedding, so it shares the fine-grained resolution described in
       tokenizer.py), adduct (categorical, `nn.Embedding` over
       `ADDUCT_VOCAB`), and collision energy (continuous eV, via a small
       sinusoidal scalar embedding tailored to that range). Each becomes one
       extra "metadata token" prepended to the peak sequence — the standard
       way to inject global conditioning into a Transformer encoder without
       a separate fusion mechanism (the self-attention layers do the
       fusing).
    3. Absolute positional encoding is added on top of every token
       (metadata + peaks). This is deliberately in addition to the
       continuous m/z embedding, not instead of it: the m/z embedding
       encodes each peak's *absolute mass*, while this classic index-based
       sinusoidal PE encodes each peak's *position in the m/z-ascending
       sequence built by dataset.py* — i.e. its rank relative to its
       neighbors — which self-attention otherwise has no way to recover
       from an unordered set of vectors.
    4. The combined, position-encoded token sequence passes through a stack
       of `nn.TransformerEncoderLayer`s (multi-head self-attention +
       feedforward, pre/post-norm per PyTorch defaults) sized from
       config.py (D_MODEL, N_HEADS, NUM_ENCODER_LAYERS, DIM_FEEDFORWARD,
       DROPOUT).
    5. Output: a (batch_size, seq_len, d_model) latent memory matrix, plus
       the matching padding mask, for the decoder's cross-attention.

Requires: torch
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from src import config
from src.dataset import ADDUCT_VOCAB
from src.tokenizer import SinusoidalMZEmbedding

# Number of prepended metadata tokens: precursor_mz, adduct, collision_energy.
NUM_METADATA_TOKENS = 3


class SinusoidalScalarEmbedding(nn.Module):
    """
    Generic continuous-scalar -> d_model embedding, same log-spaced Fourier
    idea as `SinusoidalMZEmbedding` but for scalars with a different natural
    range/unit (e.g. collision energy in eV, typically 0-200 rather than
    0-2000+ Da). Kept separate from the m/z embedding rather than reused
    directly so the frequency band can be tuned per unit without perturbing
    the m/z resolution behavior documented in tokenizer.py.
    """

    def __init__(
        self,
        d_model: int = config.D_MODEL,
        min_freq: float = 1e-2,
        max_freq: float = 1e2,
    ):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for sin/cos pairing, got {d_model}")
        num_freqs = d_model // 2
        freqs = torch.logspace(
            math.log10(min_freq), math.log10(max_freq), steps=num_freqs, base=10.0
        )
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (...,) raw scalar values -> (..., d_model)."""
        angles = x.unsqueeze(-1) * self.freqs
        sinusoid = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.proj(sinusoid)


class PositionalEncoding(nn.Module):
    """
    Classic fixed (non-learned) index-based sinusoidal positional encoding,
    precomputed up to `max_len` and added elementwise to the input sequence.
    Encodes *sequence position*, independent of and complementary to the
    continuous m/z value each peak token also carries.
    """

    def __init__(self, d_model: int = config.D_MODEL, max_len: int = 4096):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model) -> same shape, with PE added."""
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"sequence length {seq_len} exceeds PositionalEncoding max_len={self.pe.size(1)}"
            )
        return x + self.pe[:, :seq_len, :]


class SpectrumEncoder(nn.Module):
    """
    Fuses per-peak continuous embeddings with precursor/adduct/collision-
    energy metadata tokens, applies positional encoding, and runs the result
    through a stack of Transformer encoder layers.
    """

    def __init__(
        self,
        d_model: int = config.D_MODEL,
        n_heads: int = config.N_HEADS,
        num_encoder_layers: int = config.NUM_ENCODER_LAYERS,
        dim_feedforward: int = config.DIM_FEEDFORWARD,
        dropout: float = config.DROPOUT,
        max_peaks: int = config.MAX_PEAKS,
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Continuous peak embedding (m/z + intensity per peak).
        self.peak_embedding = SinusoidalMZEmbedding(d_model=d_model, use_intensity=True)

        # 2. Metadata embeddings.
        self.precursor_embedding = SinusoidalMZEmbedding(d_model=d_model, use_intensity=False)
        self.adduct_embedding = nn.Embedding(len(ADDUCT_VOCAB), d_model)
        self.collision_energy_embedding = SinusoidalScalarEmbedding(d_model=d_model)

        # 3. Positional encoding over the full [metadata; peaks] sequence.
        self.positional_encoding = PositionalEncoding(
            d_model=d_model, max_len=max_peaks + NUM_METADATA_TOKENS
        )
        self.input_dropout = nn.Dropout(dropout)

        # 4. Transformer encoder stack.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm: much more stable under fp16 than the post-norm default
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers, norm=nn.LayerNorm(d_model)
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: dict as produced by `src.dataset.collate_fn`, containing
                at least:
                    "mz":               (B, P) float
                    "intensity":        (B, P) float
                    "padding_mask":     (B, P) bool, True = padded peak
                    "precursor_mz":     (B,) float
                    "adduct_id":        (B,) long
                    "collision_energy": (B,) float

        Returns:
            memory: (B, NUM_METADATA_TOKENS + P, d_model) latent memory
                matrix for decoder cross-attention.
            memory_padding_mask: (B, NUM_METADATA_TOKENS + P) bool, True
                where a position should be ignored by attention (metadata
                tokens are never padded; peak padding comes from the batch).
        """
        mz = batch["mz"]
        intensity = batch["intensity"]
        peak_padding_mask = batch["padding_mask"]  # (B, P), True = padded
        precursor_mz = batch["precursor_mz"]  # (B,)
        adduct_id = batch["adduct_id"]  # (B,)
        collision_energy = batch["collision_energy"]  # (B,)

        batch_size = mz.size(0)

        # -- Token embeddings -------------------------------------------------
        peak_tokens = self.peak_embedding(mz, intensity)  # (B, P, D)

        precursor_token = self.precursor_embedding(precursor_mz).unsqueeze(1)  # (B, 1, D)
        adduct_token = self.adduct_embedding(adduct_id).unsqueeze(1)  # (B, 1, D)
        collision_token = self.collision_energy_embedding(collision_energy).unsqueeze(1)  # (B, 1, D)

        # -- Assemble sequence: [precursor, adduct, collision_energy, peaks...]
        tokens = torch.cat(
            [precursor_token, adduct_token, collision_token, peak_tokens], dim=1
        )  # (B, NUM_METADATA_TOKENS + P, D)

        tokens = self.positional_encoding(tokens)
        tokens = self.input_dropout(tokens)

        # -- Padding mask: metadata tokens are always real (never padded).
        metadata_mask = torch.zeros(
            batch_size, NUM_METADATA_TOKENS, dtype=torch.bool, device=peak_padding_mask.device
        )
        memory_padding_mask = torch.cat([metadata_mask, peak_padding_mask], dim=1)

        # -- Transformer encoder stack.
        memory = self.encoder(tokens, src_key_padding_mask=memory_padding_mask)

        return memory, memory_padding_mask


if __name__ == "__main__":
    from src.dataset import MassSpecDataset, collate_fn
    from src.tokenizer import SelfiesTokenizer
    from torch.utils.data import DataLoader

    tok = SelfiesTokenizer.load(config.VOCAB_PATH)
    train_ds = MassSpecDataset(config.TRAIN_PARQUET, tokenizer=tok, is_train=True)
    loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))

    encoder = SpectrumEncoder()
    memory, memory_padding_mask = encoder(batch)
    print("memory:", memory.shape)  # (4, 3 + MAX_PEAKS, D_MODEL)
    print("memory_padding_mask:", memory_padding_mask.shape)

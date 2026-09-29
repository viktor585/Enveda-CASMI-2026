"""
src/models/selfies_decoder.py

The Transformer Decoder stack ("the writer"): autoregressively predicts the
next SELFIES token, conditioned on (a) the tokens generated so far via
causal self-attention and (b) the spectrum's latent memory matrix from
`spectrum_encoder.py` via cross-attention.

Pipeline:
    1. Target SELFIES token ids -> learned embedding (scaled by sqrt(d_model),
       the standard Transformer convention that keeps embedding magnitude
       comparable to the positional encoding added next) + the same
       index-based sinusoidal `PositionalEncoding` used by the encoder, so
       the decoder knows each token's position in the output sequence.
    2. A causal mask (upper-triangular, strictly future positions = True)
       is passed as `tgt_mask` to every decoder layer's self-attention, so
       position i can only attend to positions <= i — required so training
       with teacher forcing doesn't let the model "cheat" by seeing the
       token it's supposed to predict.
    3. Each `nn.TransformerDecoderLayer` does causal self-attention, then
       cross-attention over the encoder's memory (masked by
       `memory_padding_mask` so attention ignores padded peak/metadata
       slots), then a feedforward block.
    4. A final `nn.Linear(d_model, vocab_size)` head projects each output
       position to vocabulary logits (pre-softmax; loss functions like
       `nn.CrossEntropyLoss` take logits directly).

Requires: torch
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from src import config
from src.models.spectrum_encoder import PositionalEncoding


def generate_causal_mask(size: int, device=None) -> torch.Tensor:
    """
    (size, size) bool mask where True = "not allowed to attend" (future
    positions), matching PyTorch's `tgt_mask` convention for
    `nn.TransformerDecoderLayer` when a bool mask is supplied.
    """
    return torch.triu(
        torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1
    )


class SelfiesDecoder(nn.Module):
    """
    Autoregressive Transformer decoder over the SELFIES vocabulary.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = config.D_MODEL,
        n_heads: int = config.N_HEADS,
        num_decoder_layers: int = config.NUM_DECODER_LAYERS,
        dim_feedforward: int = config.DIM_FEEDFORWARD,
        dropout: float = config.DROPOUT,
        max_selfies_len: int = config.MAX_SELFIES_LEN,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.embed_scale = math.sqrt(d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_selfies_len)
        self.input_dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm: much more stable under fp16 than the post-norm default
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers, norm=nn.LayerNorm(d_model)
        )

        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        target_ids: torch.Tensor,
        memory: torch.Tensor,
        target_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            target_ids: (B, T) long, decoder *input* token ids (i.e. the
                target sequence shifted right — see
                `SpectrumToStructureTransformer.forward` for the shift).
            memory: (B, S, d_model) latent memory matrix from
                `SpectrumEncoder`.
            target_padding_mask: (B, T) bool, True at padded target
                positions (matches `tgt_key_padding_mask`). Optional, but
                should be passed whenever `target_ids` contains padding.
            memory_padding_mask: (B, S) bool, True at padded memory
                positions, forwarded from `SpectrumEncoder` unchanged.

        Returns:
            logits: (B, T, vocab_size), pre-softmax next-token scores at
                every position.
        """
        batch_size, seq_len = target_ids.shape

        tokens = self.token_embedding(target_ids) * self.embed_scale  # (B, T, D)
        tokens = self.positional_encoding(tokens)
        tokens = self.input_dropout(tokens)

        causal_mask = generate_causal_mask(seq_len, device=target_ids.device)  # (T, T)

        decoded = self.decoder(
            tgt=tokens,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )  # (B, T, D)

        logits = self.output_head(decoded)  # (B, T, vocab_size)
        return logits


if __name__ == "__main__":
    vocab_size, pad_id = 64, 0
    decoder = SelfiesDecoder(vocab_size=vocab_size, pad_id=pad_id)

    batch_size, tgt_len, mem_len = 4, config.MAX_SELFIES_LEN - 1, config.MAX_PEAKS + 3
    target_ids = torch.randint(0, vocab_size, (batch_size, tgt_len))
    memory = torch.randn(batch_size, mem_len, config.D_MODEL)
    memory_padding_mask = torch.zeros(batch_size, mem_len, dtype=torch.bool)

    logits = decoder(target_ids, memory, memory_padding_mask=memory_padding_mask)
    print("logits:", logits.shape)  # (4, tgt_len, vocab_size)

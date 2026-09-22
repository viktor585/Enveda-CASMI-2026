"""
src/models/transformer_model.py

Top-level `nn.Module` wrapping `SpectrumEncoder` ("the reader") and
`SelfiesDecoder` ("the writer") into the single sequence-to-sequence model
that's instantiated for training and restored (from CHECKPOINT_DIR /
BEST_MODEL_PATH) for inference: `SpectrumToStructureTransformer`.

Training uses teacher forcing: the full target SELFIES id sequence (as
`src/dataset.py` produces it — `[SOS] token... [EOS] [PAD]...` fixed to
MAX_SELFIES_LEN) is shifted by one position so the decoder always predicts
the *next* real token from everything up to and including the current one:
    decoder input:  [SOS] t1 t2 ... tn [EOS] [PAD]...[PAD]   (all but last)
    decoder target:       t1 t2 ... tn [EOS] [PAD]...[PAD]   (all but first)
This is the shift `forward()` performs before calling the decoder, so
callers (train.py) just pass the batch straight from `collate_fn` and get
back logits already aligned with `target_ids[:, 1:]` for the loss.

Requires: torch
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from src import config
from src.models.selfies_decoder import SelfiesDecoder
from src.models.spectrum_encoder import SpectrumEncoder


class SpectrumToStructureTransformer(nn.Module):
    """
    Full encoder-decoder model: MS2 spectrum + precursor metadata ->
    predicted SELFIES token logits.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = config.D_MODEL,
        n_heads: int = config.N_HEADS,
        num_encoder_layers: int = config.NUM_ENCODER_LAYERS,
        num_decoder_layers: int = config.NUM_DECODER_LAYERS,
        dim_feedforward: int = config.DIM_FEEDFORWARD,
        dropout: float = config.DROPOUT,
        max_peaks: int = config.MAX_PEAKS,
        max_selfies_len: int = config.MAX_SELFIES_LEN,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id

        self.encoder = SpectrumEncoder(
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_peaks=max_peaks,
        )
        self.decoder = SelfiesDecoder(
            vocab_size=vocab_size,
            pad_id=pad_id,
            d_model=d_model,
            n_heads=n_heads,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_selfies_len=max_selfies_len,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Teacher-forced training forward pass.

        Args:
            batch: dict as produced by `src.dataset.collate_fn`, containing
                the encoder inputs ("mz", "intensity", "padding_mask",
                "precursor_mz", "adduct_id", "collision_energy") plus the
                decoder targets ("selfies_ids", "selfies_padding_mask"),
                each (B, MAX_SELFIES_LEN).

        Returns:
            logits: (B, MAX_SELFIES_LEN - 1, vocab_size). Align these with
                `batch["selfies_ids"][:, 1:]` (and mask with
                `batch["selfies_padding_mask"][:, 1:]`) when computing the
                training loss.
        """
        memory, memory_padding_mask = self.encoder(batch)

        # Shift: decoder sees tokens [:-1], predicts tokens [1:].
        decoder_input_ids = batch["selfies_ids"][:, :-1]
        decoder_input_padding_mask = batch["selfies_padding_mask"][:, :-1]

        logits = self.decoder(
            target_ids=decoder_input_ids,
            memory=memory,
            target_padding_mask=decoder_input_padding_mask,
            memory_padding_mask=memory_padding_mask,
        )
        return logits

    @torch.no_grad()
    def generate_greedy(
        self,
        batch: Dict[str, torch.Tensor],
        sos_id: int,
        eos_id: int,
        max_len: int = config.MAX_SELFIES_LEN,
    ) -> torch.Tensor:
        """
        Simple greedy autoregressive decoding for quick sanity checks during
        development. This is O(max_len) full decoder forward passes with no
        KV-caching, so it's intentionally not the inference path for
        generating competition submissions — `evaluate.py`/`submit.py`
        should implement beam search (BEAM_SIZE, from config.py) to produce
        the top-25 candidates the competition's MRR metric expects; this
        method only ever returns a single best-effort sequence.

        Args:
            batch: encoder-input fields only (no "selfies_ids" needed).
            sos_id / eos_id: from the fitted SelfiesTokenizer.
            max_len: maximum output length, including SOS/EOS.

        Returns:
            generated_ids: (B, <=max_len) long tensor of generated token
                ids, starting with sos_id. Sequences stop growing further
                once EOS is emitted but the returned tensor isn't
                individually truncated per-example (callers should stop at
                each row's first eos_id, as `SelfiesTokenizer.decode` does).
        """
        self.eval()
        memory, memory_padding_mask = self.encoder(batch)
        batch_size = memory.size(0)
        device = memory.device

        generated = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.decoder(
                target_ids=generated,
                memory=memory,
                target_padding_mask=None,
                memory_padding_mask=memory_padding_mask,
            )
            next_token = logits[:, -1, :].argmax(dim=-1)  # (B,)
            next_token = torch.where(
                finished, torch.full_like(next_token, eos_id), next_token
            )
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_id)
            if finished.all():
                break

        return generated


if __name__ == "__main__":
    from src.dataset import MassSpecDataset, collate_fn
    from src.tokenizer import SelfiesTokenizer
    from torch.utils.data import DataLoader

    tok = SelfiesTokenizer.load(config.VOCAB_PATH)
    train_ds = MassSpecDataset(config.TRAIN_PARQUET, tokenizer=tok, is_train=True)
    loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))

    model = SpectrumToStructureTransformer(vocab_size=tok.vocab_size, pad_id=tok.pad_id)

    logits = model(batch)
    print("training logits:", logits.shape)  # (4, MAX_SELFIES_LEN - 1, vocab_size)

    generated = model.generate_greedy(batch, sos_id=tok.sos_id, eos_id=tok.eos_id, max_len=32)
    print("greedy generated ids:", generated.shape)

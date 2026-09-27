"""
src/tokenizer.py

Two responsibilities live here, matching how the model consumes each modality:

1. `SinusoidalMZEmbedding` — turns continuous, unbounded m/z floats into fixed-size
   vectors the spectrum encoder can attend over. There's no fixed vocabulary for
   m/z (unlike tokens), so we use a Fourier-feature style encoding rather than a
   lookup table.

2. `SelfiesTokenizer` — builds and applies a *discrete* vocabulary over SELFIES
   tokens (the decoder's target language), with the usual special-token handling,
   padding/truncation to MAX_SELFIES_LEN, and JSON persistence to VOCAB_PATH.

Requires: torch, selfies  (pip install torch selfies)
"""

import json
import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from tqdm.auto import tqdm

try:
    import selfies as sf
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The 'selfies' package is required for SelfiesTokenizer. "
        "Install it with: pip install selfies"
    ) from e

from src import config


# ---------------------------------------------------------------------------
# 1. Continuous m/z sinusoidal embedding
# ---------------------------------------------------------------------------
class SinusoidalMZEmbedding(nn.Module):
    """
    Maps continuous m/z values to a d_model-dim vector using log-spaced
    sinusoidal (Fourier) features, in the spirit of the original Transformer's
    positional encoding but adapted for an unbounded continuous input rather
    than an integer position index.

    For a scalar m/z value x and d_model = 2k, the embedding is:
        [sin(x * f_0), cos(x * f_0), sin(x * f_1), cos(x * f_1), ..., sin(x * f_{k-1}), cos(x * f_{k-1})]
    where the frequencies f_i (units: rad/Da) are log-spaced between `min_freq`
    and `max_freq`, and x is the *raw, unnormalized* m/z value.

    x must NOT be rescaled (e.g. divided by a max m/z) before this — doing so
    shrinks every angle by that same factor and destroys the fine-grained
    resolution the high-frequency bands exist to provide. Concretely, a
    0.001 Da mass difference should already produce a ~1.0 radian phase shift
    in the highest-frequency channel (i.e. `max_freq` should be on the order
    of 1e3 rad/Da) so the model can resolve sub-Dalton differences that
    distinguish molecular formulas, isotopes, and adducts. Log-spacing (rather
    than the geometric 1/10000^(2i/d) schedule used for token positions) is
    what lets the same embedding cover both this fine end and the coarse,
    slowly-varying end needed for masses spanning ~1 to a few thousand Da.

    An optional small MLP projection is applied after the raw sinusoids so the
    model can learn to reweight/mix frequency bands, and an optional intensity
    scalar can be fused in (peak intensity is informative alongside m/z when
    embedding a full peak rather than a bare mass).
    """

    def __init__(
        self,
        d_model: int = config.D_MODEL,
        min_freq: float = 1e-3,
        max_freq: float = 1e3,
        use_intensity: bool = True,
        project: bool = True,
    ):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for sin/cos pairing, got {d_model}")

        self.d_model = d_model
        self.use_intensity = use_intensity
        num_freqs = d_model // 2

        # Log-spaced frequencies (rad/Da), registered as a buffer so they move
        # with .to(device) but are not trained. max_freq ~1e3 rad/Da is what
        # gives sub-Dalton (~0.001 Da) resolution; see class docstring.
        freqs = torch.logspace(
            math.log10(min_freq), math.log10(max_freq), steps=num_freqs, base=10.0
        )
        self.register_buffer("freqs", freqs)  # (num_freqs,)

        in_dim = d_model + (1 if use_intensity else 0)
        if project:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.proj = nn.Identity() if not use_intensity else nn.Linear(in_dim, d_model)

    def forward(
        self, mz: torch.Tensor, intensity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            mz: (..., ) float tensor of raw m/z values (any positive floats).
            intensity: optional (..., ) float tensor of normalized peak
                intensities in [0, 1]; required if `use_intensity=True`.

        Returns:
            (..., d_model) float tensor of peak embeddings.
        """
        # Use the raw m/z value directly (do NOT rescale it) — the frequency
        # bands are already chosen in rad/Da, and normalizing here would
        # compress every angle and erase sub-Dalton resolution.
        angles = mz.unsqueeze(-1) * self.freqs  # (..., num_freqs), broadcasting

        sin_part = torch.sin(angles)
        cos_part = torch.cos(angles)
        sinusoid = torch.cat([sin_part, cos_part], dim=-1)  # (..., d_model)

        if self.use_intensity:
            if intensity is None:
                raise ValueError("use_intensity=True but no intensity tensor was passed")
            sinusoid = torch.cat([sinusoid, intensity.unsqueeze(-1)], dim=-1)

        return self.proj(sinusoid)


# ---------------------------------------------------------------------------
# 2. Discrete SELFIES vocabulary / tokenizer
# ---------------------------------------------------------------------------
PAD_TOKEN = "[PAD]"
SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"
UNK_TOKEN = "[UNK]"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class SelfiesTokenizer:
    """
    Discrete tokenizer over SELFIES tokens (e.g. "[C]", "[=O]", "[Branch1]", ...).

    Wraps the `selfies` package's alphabet utilities to build a token<->id
    mapping from a corpus of SELFIES strings, then encodes/decodes SELFIES
    strings to/from fixed-length integer id sequences (padded/truncated to
    MAX_SELFIES_LEN) suitable for the decoder's embedding layer and loss.
    """

    def __init__(self, token_to_id: Optional[Dict[str, int]] = None):
        if not token_to_id:
            # No vocab supplied (or an empty dict) — seed with just the
            # special tokens so pad_id/sos_id/eos_id/unk_id are always valid,
            # even before build_vocab()/load() has been called.
            token_to_id = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.token_to_id: Dict[str, int] = token_to_id
        self.id_to_token: Dict[int, str] = {i: t for t, i in self.token_to_id.items()}

    # -- vocab construction -------------------------------------------------
    @classmethod
    def build_vocab(cls, selfies_strings: Sequence[str]) -> "SelfiesTokenizer":
        """
        Derive the full token alphabet from a corpus of SELFIES strings and
        assign integer ids, with special tokens placed first at fixed ids so
        they're stable across rebuilds.

        Args:
            selfies_strings: an iterable of SELFIES strings (e.g. the full
                train split, already converted from SMILES via
                `selfies.encoder`). Entries that are None, empty, or not
                individually parseable as SELFIES are skipped rather than
                raising, since real datasets routinely contain a few bad rows.
        """
        clean_strings: List[str] = []
        for s in tqdm(selfies_strings, desc="validating SELFIES", mininterval=1.0):
            if not s or not isinstance(s, str):
                continue
            try:
                # Cheaply validate this entry in isolation so one malformed
                # string can't take down alphabet extraction for the whole
                # corpus.
                list(sf.split_selfies(s))
            except Exception:
                continue
            clean_strings.append(s)

        if not clean_strings:
            raise ValueError("No valid SELFIES strings found to build a vocabulary from")

        alphabet = sf.get_alphabet_from_selfies(clean_strings)
        alphabet = sorted(alphabet)  # deterministic ordering

        token_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        next_id = len(token_to_id)
        for tok in alphabet:
            if tok not in token_to_id:
                token_to_id[tok] = next_id
                next_id += 1

        return cls(token_to_id)

    # -- persistence ----------------------------------------------------
    def save(self, path: str = config.VOCAB_PATH) -> None:
        with open(path, "w") as f:
            json.dump(self.token_to_id, f, indent=2)

    @classmethod
    def load(cls, path: str = config.VOCAB_PATH) -> "SelfiesTokenizer":
        with open(path, "r") as f:
            token_to_id = json.load(f)
        return cls(token_to_id)

    # -- basic properties -------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def sos_id(self) -> int:
        return self.token_to_id[SOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    # -- encode / decode -------------------------------------------------
    def encode(
        self,
        selfies_string: str,
        max_len: int = config.MAX_SELFIES_LEN,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """
        SELFIES string -> fixed-length list of token ids: [SOS] + tokens + [EOS],
        right-padded with [PAD] up to `max_len`, truncated (keeping room for
        [EOS]) if longer.
        """
        tokens = list(sf.split_selfies(selfies_string))
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in tokens]

        if add_special_tokens:
            budget = max_len - 2  # room for SOS and EOS
            ids = ids[:budget]
            ids = [self.sos_id] + ids + [self.eos_id]
        else:
            ids = ids[:max_len]

        if len(ids) < max_len:
            ids = ids + [self.pad_id] * (max_len - len(ids))

        return ids

    def decode(self, ids: Sequence[int], strip_special_tokens: bool = True) -> str:
        """
        Integer id sequence -> SELFIES string. Stops at the first [EOS] and
        drops [PAD]/[SOS]/[EOS]/[UNK] when `strip_special_tokens=True`.
        """
        tokens: List[str] = []
        for i in ids:
            tok = self.id_to_token.get(int(i), UNK_TOKEN)
            if tok == EOS_TOKEN:
                break
            if strip_special_tokens and tok in SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return "".join(tokens)

    def decode_to_smiles(self, ids: Sequence[int], strip_special_tokens: bool = True) -> str:
        """Convenience wrapper: decode ids -> SELFIES -> SMILES."""
        selfies_string = self.decode(ids, strip_special_tokens=strip_special_tokens)
        return sf.decoder(selfies_string)


if __name__ == "__main__":
    # Smoke test using a couple of small example molecules (ethanol, benzene).
    example_smiles = ["CCO", "c1ccccc1"]
    example_selfies = [sf.encoder(s) for s in example_smiles]

    tok = SelfiesTokenizer.build_vocab(example_selfies)
    print(f"vocab_size = {tok.vocab_size}")

    for s in example_selfies:
        ids = tok.encode(s)
        back = tok.decode_to_smiles(ids)
        print(s, "->", ids[:10], "... ->", back)

    mz_embed = SinusoidalMZEmbedding(d_model=config.D_MODEL)
    dummy_mz = torch.tensor([57.07, 150.5, 999.99])
    dummy_intensity = torch.tensor([1.0, 0.5, 0.1])
    out = mz_embed(dummy_mz, dummy_intensity)
    print("mz embedding shape:", out.shape)

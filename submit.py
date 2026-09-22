"""
submit.py

Runs the trained model over data/raw/test.parquet and writes submission.csv.

    1. Loads test.parquet (via MassSpecDataset, is_train=False — no SMILES
       column expected) and the best checkpoint from CHECKPOINT_DIR.
    2. Each row is already one spectrum (see src/dataset.py); a molecule_id
       can have several spectra (different adducts/collision energies), so
       predictions are generated per spectrum and then merged per
       molecule_id before writing the submission — see
       `merge_molecule_candidates` below.
    3. Runs beam search (width = BEAM_SIZE from config.py, i.e. 25) to
       generate that many candidate SELFIES sequences per spectrum.
    4. Decodes each candidate SELFIES -> SMILES, drops anything RDKit can't
       parse, canonicalizes the rest, and joins each molecule's top-25
       unique canonical SMILES (ranked by score) into a
       semicolon-separated string for submission.csv.

Requires: torch, pandas, rdkit, selfies (all via src/*)

Usage:
    python submit.py
"""

import os
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config
from src.dataset import MassSpecDataset, collate_fn
from src.models.transformer_model import SpectrumToStructureTransformer
from src.tokenizer import SelfiesTokenizer
from src.utils import canonicalize_smiles, load_checkpoint, selfies_to_smiles
from train import move_batch_to_device  # reuse the same device-move helper as train.py

SUBMIT_BATCH_SIZE = 16  # spectra per encoder forward pass (beam search itself loops per-spectrum)
LENGTH_PENALTY_ALPHA = 0.7  # >0 counteracts beam search's bias toward shorter sequences
OUTPUT_PATH = os.path.join(config.BASE_DIR, "submission.csv")
ID_PREFIX = "mol_"  # NOTE: confirm this matches the competition's sample_submission.csv id format


@torch.no_grad()
def beam_search(
    model: SpectrumToStructureTransformer,
    memory: torch.Tensor,
    memory_padding_mask: torch.Tensor,
    sos_id: int,
    eos_id: int,
    pad_id: int,
    beam_width: int,
    max_len: int,
    device: torch.device,
    length_penalty_alpha: float = LENGTH_PENALTY_ALPHA,
) -> List[Tuple[List[int], float]]:
    """
    Beam search decoding for a single spectrum's memory.

    NOTE ON PERFORMANCE: this re-runs the full decoder stack over the
    growing sequence at every step (no KV-caching), same tradeoff as
    `SpectrumToStructureTransformer.generate_greedy` — correct, but O(max_len)
    full decoder passes per spectrum. Fine for a first submission; worth
    adding KV-caching if test.parquet is large enough for this to be slow.

    Args:
        memory: (S, d_model) — one spectrum's encoder output (a single row
            sliced out of a batch, not batched over spectra).
        memory_padding_mask: (S,) bool.
        beam_width: number of candidates to keep/return (BEAM_SIZE).
        max_len: maximum generated length, including SOS/EOS.
        length_penalty_alpha: final scores are divided by
            (sequence_length ** length_penalty_alpha) so beam search doesn't
            systematically prefer shorter completions just because they
            accumulate fewer (typically negative) log-probs.

    Returns:
        Up to `beam_width` (token_ids, normalized_score) pairs, sorted best
        (highest score) first. token_ids includes the leading SOS and any
        trailing PAD — callers should decode with
        `SelfiesTokenizer.decode`, which already strips special tokens and
        stops at the first EOS.
    """
    seq_len, d_model = memory.shape
    memory_exp = memory.unsqueeze(0).repeat(beam_width, 1, 1)  # (beam_width, S, D)
    mask_exp = memory_padding_mask.unsqueeze(0).repeat(beam_width, 1)  # (beam_width, S)

    sequences = torch.full((beam_width, 1), sos_id, dtype=torch.long, device=device)
    # Only the first beam is "real" at step 0 -- every other beam starts at
    # -inf so the first expansion doesn't produce beam_width identical
    # copies of the single best first token.
    scores = torch.full((beam_width,), float("-inf"), device=device)
    scores[0] = 0.0
    finished = torch.zeros(beam_width, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        logits = model.decoder(
            target_ids=sequences,
            memory=memory_exp,
            target_padding_mask=None,
            memory_padding_mask=mask_exp,
        )  # (beam_width, cur_len, vocab_size)
        next_log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)  # (beam_width, vocab)
        vocab_size = next_log_probs.size(-1)

        if finished.any():
            # Freeze finished beams: the only zero-cost continuation is PAD,
            # so their score stops changing once they've hit EOS.
            frozen = torch.full_like(next_log_probs, float("-inf"))
            frozen[:, pad_id] = 0.0
            next_log_probs = torch.where(finished.unsqueeze(1), frozen, next_log_probs)

        candidate_scores = (scores.unsqueeze(1) + next_log_probs).view(-1)  # (beam_width * vocab)
        topk_scores, topk_flat_idx = candidate_scores.topk(beam_width)
        beam_idx = topk_flat_idx // vocab_size
        token_idx = topk_flat_idx % vocab_size

        sequences = torch.cat([sequences[beam_idx], token_idx.unsqueeze(1)], dim=1)
        scores = topk_scores
        finished = finished[beam_idx] | (token_idx == eos_id)

        if finished.all():
            break

    # Length-normalize using each beam's actual content length (up to and
    # including its first EOS, or the full generated length if it never
    # emitted one), not the shared padded tensor width.
    results: List[Tuple[List[int], float]] = []
    seq_list = sequences.tolist()
    for row, raw_score in zip(seq_list, scores.tolist()):
        content = row[1:]  # drop leading SOS
        length = content.index(eos_id) + 1 if eos_id in content else len(content)
        length = max(length, 1)
        norm_score = raw_score / (length**length_penalty_alpha)
        results.append((row, norm_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:beam_width]


def merge_molecule_candidates(
    candidates: List[Tuple[str, float]], top_k: int
) -> List[str]:
    """
    Merge (canonical_smiles, score) pairs gathered across all of one
    molecule_id's spectra into a single ranked list. The same molecule can
    surface from multiple spectra (different adducts/collision energies) or
    multiple beams within one spectrum reaching the same canonical
    structure via different SELFIES strings; when a canonical SMILES
    repeats, its best (highest) score across all occurrences is kept as
    the ranking signal -- the strongest single piece of evidence for that
    structure, rather than averaging it down with weaker duplicates.
    """
    best_score: Dict[str, float] = {}
    for smiles, score in candidates:
        if smiles not in best_score or score > best_score[smiles]:
            best_score[smiles] = score

    ranked = sorted(best_score.items(), key=lambda kv: kv[1], reverse=True)
    return [smiles for smiles, _ in ranked[:top_k]]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device = {device}")

    tokenizer = SelfiesTokenizer.load(config.VOCAB_PATH)

    model = SpectrumToStructureTransformer(
        vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id
    ).to(device)
    checkpoint = load_checkpoint(config.BEST_MODEL_PATH, model=model, device=str(device))
    model.eval()
    print(
        f"[checkpoint] loaded epoch={checkpoint.get('epoch')} "
        f"best_metric={checkpoint.get('best_metric')} from {config.BEST_MODEL_PATH}"
    )

    test_dataset = MassSpecDataset(config.TEST_PARQUET, tokenizer=None, is_train=False)
    test_loader = DataLoader(
        test_dataset, batch_size=SUBMIT_BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    print(f"[data] {len(test_dataset)} test spectra")

    # molecule_id -> list of (canonical_smiles, score) gathered from every
    # spectrum belonging to that molecule.
    molecule_candidates: Dict[object, List[Tuple[str, float]]] = defaultdict(list)
    molecule_order: List[object] = []  # preserve first-seen order for the output rows

    for batch in test_loader:
        batch = move_batch_to_device(batch, device)

        with torch.no_grad():
            memory, memory_padding_mask = model.encoder(batch)  # (B, S, D), (B, S)

        for i in range(memory.size(0)):
            molecule_id = batch["molecule_id"][i]
            if molecule_id not in molecule_candidates:
                molecule_order.append(molecule_id)

            beams = beam_search(
                model,
                memory[i],
                memory_padding_mask[i],
                sos_id=tokenizer.sos_id,
                eos_id=tokenizer.eos_id,
                pad_id=tokenizer.pad_id,
                beam_width=config.BEAM_SIZE,
                max_len=config.MAX_SELFIES_LEN,
                device=device,
            )

            for token_ids, score in beams:
                selfies_str = tokenizer.decode(token_ids)
                smiles = selfies_to_smiles(selfies_str)
                if smiles is None:
                    continue
                canonical = canonicalize_smiles(smiles)
                if canonical is None:
                    continue
                molecule_candidates[molecule_id].append((canonical, score))

        print(f"[beam search] processed {len(molecule_order)} molecules so far")

    rows = []
    num_empty = 0
    for molecule_id in molecule_order:
        top_smiles = merge_molecule_candidates(
            molecule_candidates[molecule_id], top_k=config.BEAM_SIZE
        )
        if not top_smiles:
            num_empty += 1
        rows.append(
            {
                "id": f"{ID_PREFIX}{molecule_id}",
                "smiles": ";".join(top_smiles),
            }
        )

    if num_empty:
        print(
            f"[warning] {num_empty}/{len(rows)} molecules produced no valid "
            f"canonical SMILES candidate at all (empty submission cell)"
        )

    submission_df = pd.DataFrame(rows, columns=["id", "smiles"])
    submission_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[done] wrote {len(submission_df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

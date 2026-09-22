"""
src/metrics.py

Competition scoring: Mean Reciprocal Rank (MRR) over each molecule's top-K
predicted SMILES (K = BEAM_SIZE from config.py, i.e. top-25 candidates from
beam search).

    MRR = (1/N) * sum_i (1 / rank_i)

where rank_i is the 1-indexed position of the first predicted candidate for
molecule i that matches its true structure, or the reciprocal rank
contributes 0 if no candidate in the top-K matches at all.

Matching is done on **canonical SMILES**, not raw string equality: a
generated SMILES and the ground-truth SMILES can describe the identical
molecule while differing in atom order (e.g. "CCO" vs "OCC"), so both sides
are run through `utils.canonicalize_smiles` before comparing. An
unparseable prediction simply can't match anything and is skipped, rather
than raising and aborting evaluation of the whole batch.

Requires: rdkit (via src.utils)
"""

from typing import Dict, List, Optional, Sequence

from src import config
from src.utils import canonicalize_smiles


def reciprocal_rank(
    predicted_smiles: Sequence[str],
    true_smiles: str,
    k: int = config.BEAM_SIZE,
) -> float:
    """
    Reciprocal rank for a single molecule.

    Args:
        predicted_smiles: ranked list of candidate SMILES (best first), as
            produced by beam search — e.g. from
            `SpectrumToStructureTransformer` beam decoding in evaluate.py.
            Only the first `k` entries are considered, matching the
            competition's top-25 format.
        true_smiles: the ground-truth SMILES for this molecule.
        k: how many of the top predictions to search (default: BEAM_SIZE).

    Returns:
        1/rank of the first canonical match within the top-k predictions,
        or 0.0 if none match (including if `true_smiles` itself is invalid,
        which should not happen for real ground truth but is handled
        defensively rather than raising mid-evaluation).
    """
    canon_true = canonicalize_smiles(true_smiles)
    if canon_true is None:
        return 0.0

    for rank, candidate in enumerate(predicted_smiles[:k], start=1):
        canon_pred = canonicalize_smiles(candidate)
        if canon_pred is not None and canon_pred == canon_true:
            return 1.0 / rank

    return 0.0


def mrr_score(
    all_predicted_smiles: Sequence[Sequence[str]],
    all_true_smiles: Sequence[str],
    k: int = config.BEAM_SIZE,
) -> float:
    """
    Mean Reciprocal Rank across a full validation/test set.

    Args:
        all_predicted_smiles: one ranked candidate list per molecule, i.e.
            `all_predicted_smiles[i]` is molecule i's top-k SMILES
            predictions (best first).
        all_true_smiles: `all_true_smiles[i]` is molecule i's ground-truth
            SMILES. Must be the same length as `all_predicted_smiles`.
        k: how many of each molecule's top predictions to consider.

    Returns:
        The mean reciprocal rank over all molecules, in [0, 1].
    """
    if len(all_predicted_smiles) != len(all_true_smiles):
        raise ValueError(
            f"got {len(all_predicted_smiles)} prediction lists but "
            f"{len(all_true_smiles)} ground-truth SMILES; these must align 1:1"
        )
    if len(all_true_smiles) == 0:
        raise ValueError("cannot compute MRR over an empty set of molecules")

    scores = [
        reciprocal_rank(preds, true, k=k)
        for preds, true in zip(all_predicted_smiles, all_true_smiles)
    ]
    return sum(scores) / len(scores)


def mrr_score_by_id(
    predictions_by_id: Dict[object, Sequence[str]],
    targets_by_id: Dict[object, str],
    k: int = config.BEAM_SIZE,
) -> float:
    """
    Convenience variant keyed by molecule_id/spectrum_id, for when
    predictions and targets are naturally dict-shaped (e.g. assembled from
    a DataLoader that isn't guaranteed to preserve row order). Only ids
    present in both dicts are scored; any id missing predictions is treated
    as a full miss (reciprocal rank 0) rather than silently excluded, since
    a missing prediction is still a miss under the competition's scoring.

    Args:
        predictions_by_id: {id: ranked candidate SMILES list}.
        targets_by_id: {id: ground-truth SMILES}.
        k: how many of each molecule's top predictions to consider.

    Returns:
        The mean reciprocal rank over all ids in `targets_by_id`.
    """
    if len(targets_by_id) == 0:
        raise ValueError("cannot compute MRR over an empty set of molecules")

    scores = []
    for mol_id, true_smiles in targets_by_id.items():
        preds = predictions_by_id.get(mol_id, [])
        scores.append(reciprocal_rank(preds, true_smiles, k=k))

    return sum(scores) / len(scores)


if __name__ == "__main__":
    # Small smoke test: molecule 0 matches at rank 2 (different atom order,
    # same molecule via canonicalization), molecule 1 has no match.
    predictions = [
        ["c1ccccc1", "OCC", "CCN"],  # true is "CCO" -> canonical match at rank 2
        ["CCC", "CCCC"],  # no match
    ]
    targets = ["CCO", "CCCCC"]

    for i, (preds, true) in enumerate(zip(predictions, targets)):
        print(f"molecule {i} reciprocal rank:", reciprocal_rank(preds, true))

    print("MRR:", mrr_score(predictions, targets))

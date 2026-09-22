"""
src/utils.py

Chemistry helpers (SELFIES -> SMILES decoding, SMILES validation/
canonicalization) and model checkpoint I/O.

Requires: torch, selfies, rdkit (pip install rdkit)
"""

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import selfies as sf
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The 'selfies' package is required. Install it with: pip install selfies"
    ) from e

try:
    from rdkit import Chem
    from rdkit import RDLogger

    # RDKit logs a warning/error to stderr for every malformed SMILES it's
    # asked to parse. During evaluation we *expect* a steady stream of
    # invalid model outputs and handle them programmatically (see
    # canonicalize_smiles below), so the per-call console spam is just
    # noise — silence RDKit's own logger rather than RDKit's caller trying
    # to filter stderr.
    RDLogger.DisableLog("rdApp.*")
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The 'rdkit' package is required. Install it with: pip install rdkit"
    ) from e

from src import config


# ---------------------------------------------------------------------------
# Chemistry helpers
# ---------------------------------------------------------------------------
def selfies_to_smiles(selfies_string: str) -> Optional[str]:
    """
    Decode a generated SELFIES string back to SMILES via the `selfies`
    library. SELFIES is constructed by design so every string in its
    alphabet decodes to *some* syntactically valid molecule, but a model can
    still emit a malformed string (bad special tokens leaking in, truncation
    mid-token, etc.), so this still guards with try/except.

    Returns:
        The SMILES string, or None if `selfies_string` could not be decoded.
    """
    if not selfies_string:
        return None
    try:
        return sf.decoder(selfies_string)
    except Exception:
        return None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Validate a SMILES string with RDKit and standardize it to RDKit's
    canonical atom ordering, so two SMILES strings describing the same
    molecule (but written with atoms in a different order) compare equal as
    strings.

    Returns:
        The canonical SMILES string, or None if `smiles` is None, empty, or
        not parseable as a valid molecule (RDKit returns None from
        `MolFromSmiles` for anything chemically invalid).
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def is_valid_smiles(smiles: str) -> bool:
    """Convenience wrapper: True iff `smiles` parses as a valid molecule."""
    return canonicalize_smiles(smiles) is not None


def selfies_to_canonical_smiles(selfies_string: str) -> Optional[str]:
    """Convenience wrapper: SELFIES -> SMILES -> canonical SMILES in one call."""
    smiles = selfies_to_smiles(selfies_string)
    if smiles is None:
        return None
    return canonicalize_smiles(smiles)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    epoch: int = 0,
    best_metric: Optional[float] = None,
    path: str = config.BEST_MODEL_PATH,
    **extra: Any,
) -> None:
    """
    Save model weights, optimizer state, and training progress to `path`.

    Args:
        model: the model to checkpoint (`model.state_dict()` is saved).
        optimizer: optional optimizer to checkpoint alongside the model, so
            training can resume with matching momentum/Adam moment state.
        epoch: the epoch index just completed (0-indexed by convention;
            callers decide).
        best_metric: the validation metric (e.g. MRR) associated with this
            checkpoint, so `load_checkpoint` can report/compare it later.
        path: destination file; parent directories are created if needed
            (defaults to CHECKPOINT_DIR/best_model.pt from config.py).
        **extra: any additional JSON/tensor-serializable state to persist
            (e.g. the SelfiesTokenizer's vocab_size, config snapshot).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        **extra,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str = config.BEST_MODEL_PATH,
    model: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load a checkpoint saved by `save_checkpoint`. If `model` (and/or
    `optimizer`) is provided, their state dicts are loaded in place;
    otherwise the raw checkpoint dict is returned for the caller to handle
    (e.g. inspecting `best_metric` without allocating a model).

    Args:
        path: checkpoint file to load (defaults to BEST_MODEL_PATH).
        model: if given, `model.load_state_dict(...)` is called on it.
        optimizer: if given, `optimizer.load_state_dict(...)` is called on
            it (only if the checkpoint actually has optimizer state).
        device: device to map tensors to on load.

    Returns:
        The checkpoint dict (epoch, best_metric, and any extra fields
        passed to `save_checkpoint`; state dicts are still present too).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"no checkpoint found at {path}")

    checkpoint = torch.load(path, map_location=device)

    if model is not None and checkpoint.get("model_state_dict") is not None:
        model.load_state_dict(checkpoint["model_state_dict"])

    if (
        optimizer is not None
        and checkpoint.get("optimizer_state_dict") is not None
    ):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


if __name__ == "__main__":
    # Smoke test for the chemistry helpers (ethanol).
    example_smiles = "CCO"
    example_selfies = sf.encoder(example_smiles)
    print("selfies:", example_selfies)

    decoded = selfies_to_smiles(example_selfies)
    print("decoded smiles:", decoded)

    canon = canonicalize_smiles(decoded)
    print("canonical smiles:", canon)

    print("invalid smiles check:", is_valid_smiles("not_a_smiles((("))

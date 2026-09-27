"""
src/dataset.py

Loads a train/test parquet matching the CASMI 2026 schema, where **each row
is already a full spectrum** (not one row per peak):

    spectrum_id | molecule_id | ms2_mzs        | ms2_normalized_intensities | precursor_mz | adduct  | collision_energy_ev | normalized_smiles
    ----------- | ----------- | -------------- | --------------------------- | ------------ | ------- | -------------------- | ------------------
    0           | 0           | [57.07, 91.05] | [0.81, 1.0]                  | 106.08       | [M+H]+  | 20.0                  | CCO
    1           | 0           | [45.03, 78.9]  | [0.24, 1.0]                  | 106.08       | [M+H]+  | 40.0                  | CCO
    ...

`ms2_mzs` / `ms2_normalized_intensities` are pre-aligned list/array columns
— **no groupby is needed (or correct) to assemble a spectrum's peaks**;
doing a `groupby(...).agg(list)` over columns that already contain lists
produces nested lists-of-lists and breaks tensor conversion. Each row is
one training/inference example.

`molecule_id` groups multiple spectra (different adducts / collision
energies) that share the same target structure. It's exposed here only for
leakage-safe splitting downstream (`get_molecule_id_groups`) — e.g. so
GroupKFold keeps every spectrum of a molecule on the same side of a
train/val split — not for merging peaks across spectra, since precursor_mz/
adduct/collision_energy differ per spectrum and would be meaningless if
pooled.

Requires: torch, pandas, pyarrow, selfies, numpy
"""

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

try:
    import selfies as sf
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The 'selfies' package is required for MassSpecDataset. "
        "Install it with: pip install selfies"
    ) from e

from src import config
from src.tokenizer import SelfiesTokenizer


# ---------------------------------------------------------------------------
# Adduct vocabulary
# ---------------------------------------------------------------------------
# Fixed, hand-specified vocabulary (rather than one built per-split) so
# adduct ids stay consistent between train, val, and test/submission — an
# adduct seen only at test time must not silently shift every other id.
ADDUCT_VOCAB: Dict[str, int] = {
    "[UNK_ADDUCT]": 0,
    "[M+H]+": 1,
    "[M-H]-": 2,
    "[M+Na]+": 3,
    "[M+NH4]+": 4,
    "[M+K]+": 5,
    "[M+H-H2O]+": 6,
    "[M+2H]2+": 7,
    "[M-H2O-H]-": 8,
    "[M+Cl]-": 9,
    "[M+FA-H]-": 10,
}


def adduct_to_id(adduct: str) -> int:
    return ADDUCT_VOCAB.get(adduct, ADDUCT_VOCAB["[UNK_ADDUCT]"])


# ---------------------------------------------------------------------------
# Peak filtering
# ---------------------------------------------------------------------------
def _to_float_list(x) -> List[float]:
    """
    Normalize a cell that may be a python list, tuple, numpy array, or a
    pyarrow-backed list scalar (what `ms2_mzs`/`ms2_normalized_intensities`
    typically deserialize to) into a flat python list of floats.
    """
    if x is None:
        return []
    return np.asarray(x, dtype=np.float64).ravel().tolist()


def filter_and_pad_peaks(
    mz_values,
    intensity_values,
    max_peaks: int = config.MAX_PEAKS,
    min_relative_intensity: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Reduce one spectrum's (already 1-D) peak arrays to a fixed-length,
    padded representation.

    Steps:
        1. Normalize intensities to relative units (max peak = 1.0). The
           CASMI columns are already named "normalized_intensities", but we
           re-normalize defensively in case a spectrum's max isn't 1.0.
        2. Drop peaks below `min_relative_intensity` (noise floor).
        3. If more than `max_peaks` peaks remain, keep the `max_peaks`
           highest-intensity ones.
        4. Sort survivors by m/z ascending — a canonical, deterministic
           peak order.
        5. Right-pad with zeros to `max_peaks` and build a boolean padding
           mask (True = padding, matching `src_key_padding_mask`).

    Args:
        mz_values / intensity_values: 1-D sequences (list, tuple, or
            ndarray) of equal length — one spectrum's peaks. NOT list-of-
            lists; each element is a single float.
        max_peaks: fixed output length.
        min_relative_intensity: peaks below this relative intensity
            (post-normalization) are discarded before the top-`max_peaks`
            selection. 0.0 keeps everything.

    Returns:
        dict with "mz", "intensity" (both (max_peaks,) float32) and
        "padding_mask" ((max_peaks,) bool, True where padded).
    """
    mz_values = _to_float_list(mz_values)
    intensity_values = _to_float_list(intensity_values)

    if len(mz_values) != len(intensity_values):
        raise ValueError(
            f"mz/intensity length mismatch: {len(mz_values)} vs {len(intensity_values)}"
        )

    if len(mz_values) == 0:
        return {
            "mz": torch.zeros(max_peaks, dtype=torch.float32),
            "intensity": torch.zeros(max_peaks, dtype=torch.float32),
            "padding_mask": torch.ones(max_peaks, dtype=torch.bool),
        }

    mz_arr = torch.tensor(mz_values, dtype=torch.float32)
    inten_arr = torch.tensor(intensity_values, dtype=torch.float32)

    # 1. Defensive relative-intensity re-normalization.
    base_peak = inten_arr.max()
    if base_peak > 0:
        inten_arr = inten_arr / base_peak

    # 2. Noise-floor filtering.
    if min_relative_intensity > 0.0:
        keep = inten_arr >= min_relative_intensity
        mz_arr, inten_arr = mz_arr[keep], inten_arr[keep]

    # 3. Cap to the max_peaks strongest peaks.
    if mz_arr.numel() > max_peaks:
        top_idx = torch.topk(inten_arr, k=max_peaks).indices
        mz_arr, inten_arr = mz_arr[top_idx], inten_arr[top_idx]

    # 4. Canonical m/z-ascending order.
    order = torch.argsort(mz_arr)
    mz_arr, inten_arr = mz_arr[order], inten_arr[order]

    # 5. Right-pad + padding mask.
    n = mz_arr.numel()
    mz_t = torch.zeros(max_peaks, dtype=torch.float32)
    inten_t = torch.zeros(max_peaks, dtype=torch.float32)
    mask_t = torch.ones(max_peaks, dtype=torch.bool)
    mz_t[:n] = mz_arr
    inten_t[:n] = inten_arr
    mask_t[:n] = False

    return {"mz": mz_t, "intensity": inten_t, "padding_mask": mask_t}


# ---------------------------------------------------------------------------
# Molecule-level grouping (for leakage-safe splitting, NOT peak pooling)
# ---------------------------------------------------------------------------
def get_molecule_id_groups(df: pd.DataFrame, id_col: str = "molecule_id") -> Dict:
    """
    Map molecule_id -> list of row indices (positions) that share it.

    Use this with e.g. `sklearn.model_selection.GroupKFold(groups=...)` when
    splitting train/val, so every spectrum of a given molecule (its several
    adducts/collision energies) stays on one side of the split. This is the
    correct place "group by molecule_id" belongs for this schema — spectra
    themselves must stay one-row-per-example, since precursor_mz/adduct/
    collision_energy_ev are only meaningful per individual spectrum.
    """
    groups: Dict = {}
    for pos, mol_id in enumerate(df[id_col].tolist()):
        groups.setdefault(mol_id, []).append(pos)
    return groups


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MassSpecDataset(Dataset):
    """
    One example = one spectrum row: its (filtered, padded) peak arrays,
    precursor metadata, and — for training — a tokenized SELFIES target
    derived on the fly from the row's SMILES.
    """

    def __init__(
        self,
        parquet_path: str,
        tokenizer: Optional[SelfiesTokenizer] = None,
        max_peaks: int = config.MAX_PEAKS,
        max_selfies_len: int = config.MAX_SELFIES_LEN,
        min_relative_intensity: float = 0.0,
        id_col: str = "molecule_id",
        spectrum_id_col: str = "spectrum_id",
        mz_col: str = "ms2_mzs",
        intensity_col: str = "ms2_normalized_intensities",
        precursor_mz_col: str = "precursor_mz",
        adduct_col: str = "adduct",
        collision_energy_col: str = "collision_energy_ev",
        smiles_col: Optional[str] = None,
        is_train: bool = True,
    ):
        """
        Args:
            parquet_path: path to train.parquet or test.parquet.
            tokenizer: fitted SelfiesTokenizer; required when `is_train`.
            max_peaks / max_selfies_len: fixed output lengths (default from
                config.py).
            min_relative_intensity: passed through to filter_and_pad_peaks.
            id_col/spectrum_id_col/mz_col/intensity_col/precursor_mz_col/
            adduct_col/collision_energy_col: column names, overridable if
                the real schema differs from what's documented above.
            smiles_col: name of the SMILES target column. If None
                (default), auto-detects by trying "normalized_smiles" then
                "smiles". Ignored when `is_train=False`.
            is_train: whether to expect/encode a SMILES -> SELFIES target.
        """
        if is_train and tokenizer is None:
            raise ValueError("tokenizer is required when is_train=True")

        self.tokenizer = tokenizer
        self.max_peaks = max_peaks
        self.max_selfies_len = max_selfies_len
        self.min_relative_intensity = min_relative_intensity
        self.is_train = is_train

        print(f"[MassSpecDataset] reading {parquet_path} ...")
        t0 = time.time()
        df = pd.read_parquet(parquet_path)
        print(f"[MassSpecDataset] read {len(df):,} rows in {time.time() - t0:.1f}s")

        required_cols = {
            id_col,
            mz_col,
            intensity_col,
            precursor_mz_col,
            adduct_col,
            collision_energy_col,
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise KeyError(
                f"parquet at {parquet_path} is missing expected column(s) {missing}; "
                f"pass the matching *_col argument(s) if the schema differs"
            )

        if is_train:
            if smiles_col is None:
                for candidate in ("normalized_smiles", "smiles"):
                    if candidate in df.columns:
                        smiles_col = candidate
                        break
                if smiles_col is None:
                    raise KeyError(
                        "is_train=True but neither 'normalized_smiles' nor 'smiles' "
                        "was found; pass smiles_col explicitly"
                    )
            elif smiles_col not in df.columns:
                raise KeyError(f"smiles_col={smiles_col!r} not found in parquet")

        # Each row is already one full spectrum -- no groupby here.
        self.spectrum_ids = (
            df[spectrum_id_col].tolist() if spectrum_id_col in df.columns else df.index.tolist()
        )
        self.molecule_ids = df[id_col].tolist()
        self.mz_arrays = df[mz_col].tolist()
        self.intensity_arrays = df[intensity_col].tolist()
        self.precursor_mz = df[precursor_mz_col].astype(float).tolist()
        self.adduct_ids = [adduct_to_id(a) for a in df[adduct_col].tolist()]
        def _parse_float_scalar(val):
            if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
                return float(val[0]) if len(val) > 0 else 0.0
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        self.collision_energy = [_parse_float_scalar(x) for x in df[collision_energy_col]]

        self.selfies_list: Optional[List[str]] = None
        if is_train:
            smiles_list = df[smiles_col].tolist()
            selfies_list: List[str] = []
            dropped = 0
            keep_mask = []
            for smiles in tqdm(smiles_list, desc="[MassSpecDataset] encoding SELFIES"):
                try:
                    selfies_list.append(sf.encoder(smiles))
                    keep_mask.append(True)
                except Exception:
                    dropped += 1
                    keep_mask.append(False)

            if dropped:
                print(
                    f"[MassSpecDataset] dropped {dropped}/{len(smiles_list)} rows "
                    f"whose SMILES could not be encoded to SELFIES"
                )
                keep_idx = [i for i, k in enumerate(keep_mask) if k]
                self.spectrum_ids = [self.spectrum_ids[i] for i in keep_idx]
                self.molecule_ids = [self.molecule_ids[i] for i in keep_idx]
                self.mz_arrays = [self.mz_arrays[i] for i in keep_idx]
                self.intensity_arrays = [self.intensity_arrays[i] for i in keep_idx]
                self.precursor_mz = [self.precursor_mz[i] for i in keep_idx]
                self.adduct_ids = [self.adduct_ids[i] for i in keep_idx]
                self.collision_energy = [self.collision_energy[i] for i in keep_idx]

            self.selfies_list = selfies_list

    def __len__(self) -> int:
        return len(self.spectrum_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        peaks = filter_and_pad_peaks(
            self.mz_arrays[idx],
            self.intensity_arrays[idx],
            max_peaks=self.max_peaks,
            min_relative_intensity=self.min_relative_intensity,
        )

        item: Dict[str, torch.Tensor] = {
            "spectrum_id": self.spectrum_ids[idx],
            "molecule_id": self.molecule_ids[idx],
            "mz": peaks["mz"],
            "intensity": peaks["intensity"],
            "padding_mask": peaks["padding_mask"],
            "precursor_mz": torch.tensor(self.precursor_mz[idx], dtype=torch.float32),
            "adduct_id": torch.tensor(self.adduct_ids[idx], dtype=torch.long),
            "collision_energy": torch.tensor(
                self.collision_energy[idx], dtype=torch.float32
            ),
        }

        if self.is_train:
            ids = self.tokenizer.encode(
                self.selfies_list[idx], max_len=self.max_selfies_len
            )
            target = torch.tensor(ids, dtype=torch.long)
            item["selfies_ids"] = target
            item["selfies_padding_mask"] = target == self.tokenizer.pad_id

        return item


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Fixed-length tensor fields are stacked; id-like fields stay as lists."""
    out: Dict[str, torch.Tensor] = {
        "spectrum_id": [b["spectrum_id"] for b in batch],
        "molecule_id": [b["molecule_id"] for b in batch],
        "mz": torch.stack([b["mz"] for b in batch]),
        "intensity": torch.stack([b["intensity"] for b in batch]),
        "padding_mask": torch.stack([b["padding_mask"] for b in batch]),
        "precursor_mz": torch.stack([b["precursor_mz"] for b in batch]),
        "adduct_id": torch.stack([b["adduct_id"] for b in batch]),
        "collision_energy": torch.stack([b["collision_energy"] for b in batch]),
    }
    if "selfies_ids" in batch[0]:
        out["selfies_ids"] = torch.stack([b["selfies_ids"] for b in batch])
        out["selfies_padding_mask"] = torch.stack(
            [b["selfies_padding_mask"] for b in batch]
        )
    return out


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    tok = SelfiesTokenizer.load(config.VOCAB_PATH)
    train_ds = MassSpecDataset(config.TRAIN_PARQUET, tokenizer=tok, is_train=True)
    print(f"train examples (one per spectrum row): {len(train_ds)}")

    loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    batch = next(iter(loader))
    print("mz:", batch["mz"].shape)
    print("precursor_mz:", batch["precursor_mz"].shape)
    print("adduct_id:", batch["adduct_id"].shape)
    print("selfies_ids:", batch["selfies_ids"].shape)

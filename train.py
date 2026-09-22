"""
train.py

Main training entrypoint for the spectrum -> SELFIES/SMILES transformer.

    1. Hyperparameters come from src/config.py.
    2. Loads train.parquet via src/dataset.py, splits it into train/val by
       molecule_id (GroupKFold, so no molecule's spectra leak across the
       split), and wraps both in DataLoaders.
    3. Instantiates SpectrumToStructureTransformer, an AdamW optimizer, and
       a CosineAnnealingLR schedule.
    4. Trains with teacher forcing: SpectrumToStructureTransformer.forward()
       already shifts the target internally (decoder sees
       selfies_ids[:, :-1], predicts selfies_ids[:, 1:]), so this loop just
       computes label-smoothed cross-entropy on the returned logits against
       that shifted target.
    5. At the end of every epoch, computes teacher-forced validation loss
       and saves a checkpoint via src/utils.py whenever it's the best seen
       so far.

Note on the validation metric: this loop tracks teacher-forced
cross-entropy loss (cheap, one forward pass per val batch) as the
"best model" criterion, not the competition's MRR — computing MRR requires
autoregressive beam-search generation (BEAM_SIZE candidates per molecule),
which is orders of magnitude slower per batch and belongs in evaluate.py as
a separate, less-frequent check. Lower validation loss is a reasonable
cheap proxy to select checkpoints during training; evaluate.py should be
run against saved checkpoints to get the actual MRR before submitting.

Usage:
    python train.py
"""

import os
import time

import pandas as pd
import selfies as sf
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset

from src import config
from src.dataset import MassSpecDataset, collate_fn
from src.models.transformer_model import SpectrumToStructureTransformer
from src.tokenizer import SelfiesTokenizer
from src.utils import save_checkpoint

SEED = 42
N_SPLITS = 5  # 1/N_SPLITS of molecules held out for validation
LOG_EVERY = 50  # print training loss every this many steps
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 0  # Set to 0 on Windows to avoid pickling 2.5M spectra across spawned processes


def get_or_build_tokenizer(parquet_path: str, vocab_path: str) -> SelfiesTokenizer:
    """
    Load the SELFIES vocabulary if it's already been built (e.g. by a
    previous run), otherwise derive it from train.parquet's SMILES column
    and persist it, so every subsequent run (and evaluate.py/submit.py)
    sees the exact same token<->id mapping.
    """
    if os.path.exists(vocab_path):
        print(f"[tokenizer] loading existing vocab from {vocab_path}")
        return SelfiesTokenizer.load(vocab_path)

    print(f"[tokenizer] no vocab found at {vocab_path}, building from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    smiles_col = "normalized_smiles" if "normalized_smiles" in df.columns else "smiles"
    if smiles_col not in df.columns:
        raise KeyError(
            f"{parquet_path} has neither 'normalized_smiles' nor 'smiles' column"
        )

    selfies_strings = []
    for smiles in df[smiles_col].tolist():
        try:
            selfies_strings.append(sf.encoder(smiles))
        except Exception:
            continue  # dropped rows are re-filtered identically inside MassSpecDataset

    tokenizer = SelfiesTokenizer.build_vocab(selfies_strings)
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    tokenizer.save(vocab_path)
    print(f"[tokenizer] built vocab of size {tokenizer.vocab_size}, saved to {vocab_path}")
    return tokenizer


def split_train_val(dataset: MassSpecDataset, n_splits: int = N_SPLITS, seed: int = SEED):
    """
    Group-aware split: every spectrum belonging to the same molecule_id
    stays entirely in train or entirely in val, so validation loss isn't
    inflated by the model having seen a near-duplicate spectrum (same
    molecule, different adduct/collision energy) during training.
    """
    gkf = GroupKFold(n_splits=n_splits)
    # GroupKFold doesn't take a random seed directly; molecule_id order is
    # already arbitrary (parquet row order), so the first fold is used as-is.
    # X is only used by sklearn to infer n_samples, so pass a cheap dummy
    # index array rather than dataset.mz_arrays -- that's a list of
    # variable-length (ragged) peak arrays, and sklearn's internal
    # np.asarray(X) either raises or silently produces a broken object
    # array when the per-row lengths differ.
    dummy_X = range(len(dataset))
    train_idx, val_idx = next(gkf.split(dummy_X, groups=dataset.molecule_ids))
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move only tensor fields; molecule_id/spectrum_id stay as python lists."""
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
    }


def compute_loss(
    logits: torch.Tensor, batch: dict, loss_fn: nn.Module
) -> torch.Tensor:
    """
    logits: (B, MAX_SELFIES_LEN - 1, vocab_size), from
    SpectrumToStructureTransformer.forward(), already aligned with
    selfies_ids[:, 1:] (see that module's docstring for the shift).
    """
    target = batch["selfies_ids"][:, 1:]
    return loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    epoch: int = 0,
) -> float:
    """
    One pass over `loader`. If `optimizer` is given, runs in training mode
    (backward + step + grad clipping); otherwise runs in eval mode under
    no_grad for validation. Returns the mean per-batch loss.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for step, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)

            if is_train:
                optimizer.zero_grad()

            logits = model(batch)
            loss = compute_loss(logits, batch, loss_fn)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if is_train and (step + 1) % LOG_EVERY == 0:
                elapsed = time.time() - start_time
                print(
                    f"  epoch {epoch} step {step + 1}/{len(loader)} "
                    f"loss {total_loss / num_batches:.4f} ({elapsed:.1f}s elapsed)"
                )

    return total_loss / max(num_batches, 1)


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device = {device}")

    tokenizer = get_or_build_tokenizer(config.TRAIN_PARQUET, config.VOCAB_PATH)

    full_dataset = MassSpecDataset(
        config.TRAIN_PARQUET,
        tokenizer=tokenizer,
        is_train=True,
        id_col="inchikey14",
        smiles_col="normalized_smiles",
        mz_col="ms2_mzs",
        intensity_col="ms2_normalized_intensities",
    )
    train_dataset, val_dataset = split_train_val(full_dataset)
    print(
        f"[data] {len(full_dataset)} spectra total -> "
        f"{len(train_dataset)} train / {len(val_dataset)} val"
    )

    # pin_memory only helps when batches are subsequently moved to a CUDA
    # device (it lets that H2D copy happen asynchronously); it's a no-op
    # cost on CPU-only runs, so gate it on availability rather than always
    # enabling it. persistent_workers avoids respawning the worker pool
    # every epoch, which matters once num_workers > 0.
    loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    model = SpectrumToStructureTransformer(
        vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id
    ).to(device)
    print(
        f"[model] {sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
        f"trainable parameters"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS
    )
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id, label_smoothing=config.LABEL_SMOOTHING
    )

    best_val_loss = float("inf")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\n[epoch {epoch}/{config.NUM_EPOCHS}] lr = {scheduler.get_last_lr()[0]:.2e}")

        train_loss = run_epoch(
            model, train_loader, loss_fn, device, optimizer=optimizer, epoch=epoch
        )
        val_loss = run_epoch(model, val_loader, loss_fn, device, optimizer=None)
        scheduler.step()

        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch=epoch,
                best_metric=best_val_loss,
                path=config.BEST_MODEL_PATH,
                vocab_size=tokenizer.vocab_size,
                pad_id=tokenizer.pad_id,
                scheduler_state_dict=scheduler.state_dict(),
            )
            print(
                f"[checkpoint] new best val_loss={best_val_loss:.4f}, "
                f"saved to {config.BEST_MODEL_PATH}"
            )

    print(f"\n[done] best val_loss = {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
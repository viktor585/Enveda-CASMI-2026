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

Usage:
    python train.py
"""

import gc
import os
import time

import pandas as pd
import pyarrow.parquet as pq
import selfies as sf
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from src import config
from src.dataset import MassSpecDataset, collate_fn
from src.models.transformer_model import SpectrumToStructureTransformer
from src.tokenizer import SelfiesTokenizer
from src.utils import save_checkpoint

SEED = 42
N_SPLITS = 5  # 1/N_SPLITS of molecules held out for validation
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = getattr(config, "NUM_WORKERS", 0)  # Safe fallback to config setting
CHECKPOINT_EVERY_STEPS = 2000  # Save intermediate safety checkpoints


def get_or_build_tokenizer(parquet_path: str, vocab_path: str) -> SelfiesTokenizer:
    """
    Load the SELFIES vocabulary if it's already been built (e.g. by a
    previous run), otherwise derive it from train.parquet's SMILES column
    and persist it.
    """
    if os.path.exists(vocab_path):
        print(f"[tokenizer] loading existing vocab from {vocab_path}")
        return SelfiesTokenizer.load(vocab_path)

    print(f"[tokenizer] no vocab found at {vocab_path}, building from {parquet_path}")

    # Only read the schema first (cheap) to pick the right column name,
    # then load JUST that column. Reading the full parquet here would also
    # pull in ms2_mzs / ms2_normalized_intensities -- per-spectrum arrays
    # up to MAX_PEAKS long -- which we don't need for vocab building and
    # which can blow up memory badly at millions of rows (pandas stores
    # array-valued columns as Python list objects per cell, with overhead
    # far beyond the raw float size).
    available_cols = pq.ParquetFile(parquet_path).schema.names
    smiles_col = "normalized_smiles" if "normalized_smiles" in available_cols else "smiles"
    if smiles_col not in available_cols:
        raise KeyError(
            f"{parquet_path} has neither 'normalized_smiles' nor 'smiles' column"
        )

    df = pd.read_parquet(parquet_path, columns=[smiles_col])
    smiles_list = df[smiles_col].tolist()
    del df
    gc.collect()

    selfies_strings = []
    for smiles in tqdm(smiles_list, desc="encoding SELFIES"):
        try:
            selfies_strings.append(sf.encoder(smiles))
        except Exception:
            continue
    del smiles_list
    gc.collect()

    tokenizer = SelfiesTokenizer.build_vocab(selfies_strings)
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    tokenizer.save(vocab_path)
    print(f"[tokenizer] built vocab of size {tokenizer.vocab_size}, saved to {vocab_path}")
    return tokenizer


def split_train_val(dataset: MassSpecDataset, n_splits: int = N_SPLITS, seed: int = SEED):
    """Group-aware train/val split ensuring no molecule leakage across splits."""
    gkf = GroupKFold(n_splits=n_splits)
    dummy_X = range(len(dataset))
    train_idx, val_idx = next(gkf.split(dummy_X, groups=dataset.molecule_ids))
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move tensor fields to target device."""
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
    }


def compute_loss(
    logits: torch.Tensor, batch: dict, loss_fn: nn.Module
) -> torch.Tensor:
    """Compute cross-entropy loss against right-shifted SELFIES target ids."""
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
    One pass over `loader` with a live tqdm progress bar.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    num_batches = 0

    desc = f"Epoch {epoch} [Train]" if is_train else f"Epoch {epoch} [Val]"
    pbar = tqdm(loader, desc=desc, leave=True, dynamic_ncols=True)

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for step, batch in enumerate(pbar, start=1):
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
            running_loss = total_loss / num_batches

            # Live update progress bar metrics
            pbar.set_postfix({"loss": f"{running_loss:.4f}"})

            # Intermediate safety checkpointing during long training steps
            if is_train and step % CHECKPOINT_EVERY_STEPS == 0:
                chkpt_path = f"/kaggle/working/checkpoint_epoch{epoch}_step{step}.pt"
                torch.save(model.state_dict(), chkpt_path)

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

    epochs = getattr(config, "EPOCHS", getattr(config, "NUM_EPOCHS", 20))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id, label_smoothing=config.LABEL_SMOOTHING
    )

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} (lr = {scheduler.get_last_lr()[0]:.2e}) ---")

        train_loss = run_epoch(
            model, train_loader, loss_fn, device, optimizer=optimizer, epoch=epoch
        )
        val_loss = run_epoch(model, val_loader, loss_fn, device, optimizer=None, epoch=epoch)
        scheduler.step()

        print(f"Summary: train_loss = {train_loss:.4f} | val_loss = {val_loss:.4f}")

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
                f"[checkpoint] new best val_loss = {best_val_loss:.4f}, "
                f"saved to {config.BEST_MODEL_PATH}"
            )

    print(f"\n[done] best val_loss = {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
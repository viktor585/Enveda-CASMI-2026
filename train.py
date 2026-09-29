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
import math
import os
import time

import pandas as pd
import pyarrow as pa
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
CHECKPOINT_EVERY_STEPS = 2000  # Save intermediate safety checkpoints (one rolling file)


def _setting(name: str, default, cast):
    """
    Resolve a tunable as: environment variable > config.py attribute > default.
    The env-var route lets you flip a knob from a notebook cell (`%env USE_AMP=0`)
    without editing any file.
    """
    raw = os.environ.get(name)
    if raw is None:
        raw = getattr(config, name, default)
    if cast is bool:
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    return cast(raw)


NUM_WORKERS = _setting("NUM_WORKERS", 2, int)  # CPU-side batch prep overlapped with the GPU
USE_AMP = _setting("USE_AMP", True, bool)  # fp16 mixed precision (big win on T4 tensor cores)
USE_DATAPARALLEL = _setting("USE_DATAPARALLEL", False, bool)  # split each batch across all GPUs
VAL_MAX_SAMPLES = _setting("VAL_MAX_SAMPLES", 50_000, int)  # 0 = use the full validation split
MAX_STEPS_PER_EPOCH = _setting("MAX_STEPS_PER_EPOCH", 0, int)  # 0 = full epoch
WARMUP_STEPS = _setting("WARMUP_STEPS", 1000, int)  # linear LR ramp before the cosine decay


class _AutocastForward(nn.Module):
    """
    Runs the wrapped model's forward under fp16 autocast.

    Autocast is entered *inside* forward (rather than around the call site) so
    that it also takes effect in nn.DataParallel's per-GPU replica threads --
    autocast state is thread-local and would otherwise be silently skipped
    there. The wrapped model is kept as `.model` so its parameters/state_dict
    can still be saved and loaded unwrapped.
    """

    def __init__(self, model: nn.Module, enabled: bool):
        super().__init__()
        self.model = model
        self.enabled = enabled

    def forward(self, batch):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.enabled):
            return self.model(batch)


def _make_grad_scaler(enabled: bool):
    try:  # torch >= 2.3
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


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
    # Spectra vastly outnumber molecules and share SMILES, so encode each
    # distinct SMILES once -- the alphabet (a set) comes out identical.
    smiles_list = list(dict.fromkeys(df[smiles_col].tolist()))
    del df
    gc.collect()
    pa.default_memory_pool().release_unused()
    print(f"[tokenizer] encoding {len(smiles_list):,} unique SMILES")

    selfies_strings = []
    for smiles in tqdm(smiles_list, desc="encoding SELFIES", mininterval=1.0):
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
    logits = logits.float()  # logits may be fp16 under autocast; keep the loss in fp32
    return loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    epoch: int = 0,
    scaler=None,
    max_steps: int = 0,
    ckpt_model: nn.Module = None,
    step_scheduler=None,
) -> float:
    """
    One pass over `loader` with a live tqdm progress bar.

    `model` is whatever should be called for the forward pass (possibly wrapped
    for autocast / DataParallel); `ckpt_model` is the plain, unwrapped model
    whose state_dict gets written to the rolling safety checkpoint.
    Batches whose loss is not finite are skipped (no backward/step) and excluded
    from the reported average rather than poisoning it.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    ckpt_model = ckpt_model if ckpt_model is not None else model

    total_loss = 0.0
    num_batches = 0
    skipped = 0

    desc = f"Epoch {epoch} [Train]" if is_train else f"Epoch {epoch} [Val]"
    total_steps = min(len(loader), max_steps) if max_steps else len(loader)
    pbar = tqdm(
        loader, desc=desc, total=total_steps, leave=True, dynamic_ncols=True, mininterval=1.0
    )

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for step, batch in enumerate(pbar, start=1):
            if max_steps and step > max_steps:
                break

            batch = move_batch_to_device(batch, device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(batch)
            loss = compute_loss(logits, batch, loss_fn)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                skipped += 1
                continue

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)  # so clipping sees true gradient magnitudes
                    torch.nn.utils.clip_grad_norm_(ckpt_model.parameters(), GRAD_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ckpt_model.parameters(), GRAD_CLIP_NORM)
                    optimizer.step()

                if step_scheduler is not None:
                    step_scheduler.step()

            total_loss += loss_value
            num_batches += 1
            running_loss = total_loss / num_batches

            # Live update progress bar metrics
            pbar.set_postfix({"loss": f"{running_loss:.4f}"})

            # Rolling safety checkpoint (one file, overwritten) so a killed session
            # doesn't lose everything, without filling the disk with copies.
            if is_train and step % CHECKPOINT_EVERY_STEPS == 0:
                os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
                chkpt_path = os.path.join(config.CHECKPOINT_DIR, "last_state_dict.pt")
                torch.save(ckpt_model.state_dict(), chkpt_path)

    if skipped:
        print(f"[warn] skipped {skipped} batch(es) with non-finite loss in {desc}")
    if num_batches == 0:
        # Every batch this epoch was non-finite. Returning 0.0 here (the old
        # `total_loss / max(num_batches, 1)` behavior) would look like a
        # perfect loss and trick main()'s `val_loss < best_val_loss` check
        # into overwriting the best checkpoint with a fully-diverged model.
        # inf is never "better", so a wrecked epoch can never win.
        return float("inf")
    return total_loss / num_batches


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
    if VAL_MAX_SAMPLES and len(val_dataset) > VAL_MAX_SAMPLES:
        # A fixed random subset of the (already molecule-disjoint) validation
        # split. Plenty for tracking val loss / picking the best checkpoint,
        # at a small fraction of the cost of scoring every held-out spectrum.
        gen = torch.Generator().manual_seed(SEED)
        keep = torch.randperm(len(val_dataset), generator=gen)[:VAL_MAX_SAMPLES].tolist()
        val_dataset = Subset(val_dataset, keep)
        print(f"[data] validating on a fixed {len(val_dataset)}-spectrum subset")

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

    use_amp = USE_AMP and device.type == "cuda"
    # `model` stays the plain module (used for the optimizer and checkpoints);
    # `fwd_model` is what actually gets called for forward passes.
    fwd_model = _AutocastForward(model, enabled=use_amp)
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    if USE_DATAPARALLEL and n_gpus > 1:
        fwd_model = nn.DataParallel(fwd_model)
        print(f"[setup] DataParallel across {n_gpus} GPUs")
    elif USE_DATAPARALLEL:
        print("[setup] USE_DATAPARALLEL requested but only one device found; ignoring")
    scaler = _make_grad_scaler(enabled=use_amp)
    print(
        f"[setup] mixed precision (fp16) = {use_amp} | workers = {NUM_WORKERS} | "
        f"max steps/epoch = {MAX_STEPS_PER_EPOCH or 'full'}"
    )

    epochs = int(
        os.environ.get("NUM_EPOCHS", getattr(config, "EPOCHS", getattr(config, "NUM_EPOCHS", 20)))
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # Linear warmup for WARMUP_STEPS steps, then cosine decay to ~0 over the
    # remaining steps. Stepped once per optimizer update (inside run_epoch),
    # not once per epoch -- post-norm Transformer layers are prone to fp16
    # overflow if the LR jumps straight to its full value, so the ramp needs
    # to happen gradually within epoch 1, not only change epoch-to-epoch.
    steps_per_epoch = len(train_loader) if not MAX_STEPS_PER_EPOCH else min(
        len(train_loader), MAX_STEPS_PER_EPOCH
    )
    total_steps = max(steps_per_epoch * epochs, 1)
    warmup_steps = min(WARMUP_STEPS, max(total_steps // 10, 1))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    print(f"[setup] LR warmup = {warmup_steps} steps, then cosine decay over {total_steps} total steps")

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id, label_smoothing=config.LABEL_SMOOTHING
    )

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} (lr = {optimizer.param_groups[0]['lr']:.2e}) ---")

        train_loss = run_epoch(
            fwd_model,
            train_loader,
            loss_fn,
            device,
            optimizer=optimizer,
            epoch=epoch,
            scaler=scaler,
            max_steps=MAX_STEPS_PER_EPOCH,
            ckpt_model=model,
            step_scheduler=scheduler,
        )
        val_loss = run_epoch(
            fwd_model,
            val_loader,
            loss_fn,
            device,
            optimizer=None,
            epoch=epoch,
            ckpt_model=model,
        )

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
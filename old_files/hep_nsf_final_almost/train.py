"""
train.py
========
Self-contained training loop for all flow models.

Uses model.log_prob() directly so each model's own base distribution
is respected:
  s2  uniform on S2      : log p(z) = -log(4*pi)
  r2  Gaussian in R2     : log p(z) = standard normal
  r3  Gaussian in R3     : log p(z) = standard normal
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .utils import save_checkpoint, save_losses, log_std_correction


def train_model(model: nn.Module,
                train_loader,
                val_loader,
                std_tensor:       torch.Tensor,
                device:           torch.device,
                model_type:       str   = "",
                num_layers:       int   = 2,
                arch:             str   = "mlp",
                activation:       str   = "relu",
                lr:               float = 1e-3,
                weight_decay:     float = 0.0,
                max_epochs:       int   = 10_000,
                patience:         int   = 20,
                ema_alpha:        float = 0.3,
                clip_grad:        float = 5.0,
                use_plateau:      bool  = True,
                plateau_factor:   float = 0.5,
                plateau_patience: int   = 10,
                use_cosine:       bool  = False,
                cosine_t_max:     int   = 500,
                log_every:        int   = 10,
                save_dir:         str   = "checkpoints",
                run_name:         str   = "model",
                resume_from:      Optional[str] = None,
                ) -> tuple[nn.Module, dict]:
    """Train a normalising-flow model.

    Parameters
    ----------
    model_type : str
        Registry key of the model ('s2', 'r2', 'r3').
        Saved into the checkpoint so the architecture can be recovered
        without needing to specify flags at sample/evaluate time.
    run_name : str
        Prefix for checkpoint and loss filenames.
        Does NOT have to match model_type — run_name is your experiment
        label (e.g. 'mup_s2_k2'), model_type is the architecture key.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    best_path = Path(save_dir) / f"{run_name}_best.pt"

    model = model.to(device)
    std_correction = log_std_correction(std_tensor).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=weight_decay)

    schedulers = []
    if use_plateau:
        schedulers.append(
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=plateau_factor,
                patience=plateau_patience))
    if use_cosine:
        schedulers.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cosine_t_max))

    start_epoch      = 0
    best_val_loss    = float("inf")
    patience_counter = 0
    smoothed_val:    Optional[float] = None
    epoch_train:     list[float] = []
    epoch_val:       list[float] = []

    if resume_from is not None:
        from .utils import load_checkpoint
        meta          = load_checkpoint(resume_from, model, optimizer, device=device)
        start_epoch   = meta.get("epoch", 0) + 1
        best_val_loss = meta.get("best_val_loss", float("inf"))
        epoch_train   = meta.get("train_losses", [])
        epoch_val     = meta.get("val_losses",   [])
        print(f"Resuming from epoch {start_epoch}, "
              f"best val = {best_val_loss:.6f}")

    t0 = time.time()

    for epoch in range(start_epoch, max_epochs):

        # ── Training ────────────────────────────────────────────────────── #
        model.train()
        total_loss, n_batches = 0.0, 0

        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            log_prob = model.log_prob(xb)
            loss     = -log_prob.mean() + std_correction

            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                total_loss += loss.item()
                n_batches  += 1

        avg_train = total_loss / max(n_batches, 1)
        epoch_train.append(avg_train)

        # ── Validation ──────────────────────────────────────────────────── #
        model.eval()
        val_total, val_batches = 0.0, 0

        with torch.no_grad():
            for (xb,) in val_loader:
                xb       = xb.to(device)
                log_prob = model.log_prob(xb)
                val_l    = -log_prob.mean() + std_correction
                if torch.isfinite(val_l):
                    val_total  += val_l.item()
                    val_batches += 1

        avg_val = val_total / max(val_batches, 1)
        epoch_val.append(avg_val)

        smoothed_val = (avg_val if smoothed_val is None
                        else ema_alpha * avg_val + (1 - ema_alpha) * smoothed_val)

        # ── Schedulers ──────────────────────────────────────────────────── #
        for sched in schedulers:
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(smoothed_val)
            else:
                sched.step()

        # ── Checkpoint ──────────────────────────────────────────────────── #
        if smoothed_val < best_val_loss - 1e-6:
            best_val_loss    = smoothed_val
            patience_counter = 0

            # Derive hidden_dim from whichever attribute exists.
            # Must handle both MLP (.net[0]) and ResNet (.proj) architectures.
            try:
                if hasattr(model, "phi_nets"):
                    net = model.phi_nets[0]
                elif hasattr(model, "nets_phi"):
                    net = model.nets_phi[0]
                elif hasattr(model, "nets1"):
                    net = model.nets1[0]
                else:
                    net = None

                if net is None:
                    hd = 64
                elif hasattr(net, "proj"):   # ResNet
                    hd = net.proj.out_features
                elif hasattr(net, "net"):    # MLP
                    hd = net.net[0].out_features
                else:
                    hd = 64
            except Exception:
                hd = 64

            save_checkpoint(
                best_path, model, optimizer, epoch, best_val_loss,
                extra={
                    "train_losses": epoch_train,
                    "val_losses":   epoch_val,
                    "model_type":   model_type,
                    "num_bins":     model.num_bins,
                    "num_splines":  model.num_splines,
                    "bound":        model.bound if hasattr(model, "bound") else None,
                    "hidden_dim":   hd,
                    "num_layers":   num_layers,
                    "arch":         arch,
                    "activation":   activation,
                })
        else:
            patience_counter += 1

        # ── Logging ─────────────────────────────────────────────────────── #
        if epoch % log_every == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:5d}/{max_epochs} "
                  f"| Train: {avg_train:9.4f} "
                  f"| Val: {avg_val:9.4f} "
                  f"| Smooth: {smoothed_val:9.4f} "
                  f"| LR: {lr_now:.2e} "
                  f"| Patience: {patience_counter}/{patience} "
                  f"| Elapsed: {time.time()-t0:.0f}s")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}.")
            break

    print(f"\nBest val loss = {best_val_loss:.6f}")
    print(f"Checkpoint    : {best_path}")

    from .utils import load_checkpoint
    load_checkpoint(best_path, model, device=device)

    losses = {"train": epoch_train, "val": epoch_val}
    save_losses(losses, Path(save_dir) / f"{run_name}_losses.json")
    return model, losses

#!/usr/bin/env python
"""
main.py
=======
Command-line entry point for HEP Normalising-Spline Flows.

Supported modes (``--mode``):
  train    — train a flow model from scratch or resume.
  sample   — load a saved checkpoint and draw new samples.
  evaluate — compute KL, ESS, Wasserstein on saved samples.
  plot     — generate all diagnostic plots from a checkpoint.

Quick-start examples
--------------------
Train the S² (angular) flow with 2 stacked splines::

    python main.py --mode train \\
        --model s2 --data eemumu_mup.json \\
        --num_bins 32 --num_splines 2 \\
        --hidden_dim 64 --num_layers 2 \\
        --lr 1e-3 --batch_size 8192 \\
        --epochs 5000 --patience 20 \\
        --run_name mup_s2_k2 --save_dir checkpoints

Train the R³ (Cartesian) flow with 3 stacked splines::

    python main.py --mode train \\
        --model r3 --data eemumu_mup.json \\
        --num_bins 32 --num_splines 3 \\
        --run_name mup_r3_k3

Generate samples from a saved S² checkpoint::

    python main.py --mode sample \\
        --model s2 --checkpoint checkpoints/mup_s2_k2_best.pt \\
        --data eemumu_mup.json \\
        --num_samples 50000 --output_dir outputs

Evaluate a saved run::

    python main.py --mode evaluate \\
        --model s2 --checkpoint checkpoints/mup_s2_k2_best.pt \\
        --data eemumu_mup.json

Cluster usage
-------------
All arguments can be set via a YAML config file with ``--config``::

    python main.py --config configs/s2_default.yaml

Command-line arguments override values from the YAML config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hep_nsf",
        description="HEP Neural Spline Flows — train, sample, evaluate, plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Config file ──────────────────────────────────────────────────────── #
    p.add_argument("--config", type=str, default=None,
                   help="Path to a YAML config file.  CLI args override YAML.")

    # ── Mode ─────────────────────────────────────────────────────────────── #
    p.add_argument("--mode", choices=["train", "sample", "evaluate", "plot"],
                   default="train",
                   help="Execution mode.")

    # ── Data ─────────────────────────────────────────────────────────────── #
    data = p.add_argument_group("Data")
    data.add_argument("--data", type=str, required=True,
                      help="Path to the input JSON data file "
                           "(list of [px,py,pz] vectors).")
    data.add_argument("--val_frac", type=float, default=0.2,
                      help="Fraction of data held out for validation.")
    data.add_argument("--seed", type=int, default=42,
                      help="Global random seed.")

    # ── Model ────────────────────────────────────────────────────────────── #
    mdl = p.add_argument_group("Model")
    mdl.add_argument("--model", choices=["s2", "r3", "s2_recursive"],
                     default="s2",
                     help="Flow architecture.  "
                          "s2=angular S², r3=Cartesian R³, "
                          "s2_recursive=parameter-only sphere.")
    mdl.add_argument("--num_bins", type=int, default=32,
                     help="Number of RQS bins per spline layer.")
    mdl.add_argument("--num_splines", type=int, default=1,
                     help="Number of stacked spline coupling blocks.  "
                          "1 reproduces the original single-spline network; "
                          "higher values increase expressiveness.")
    mdl.add_argument("--bound", type=float, default=5.0,
                     help="Spline domain half-width in standardised space.")
    mdl.add_argument("--hidden_dim", type=int, default=64,
                     help="Hidden dimension of conditioner MLPs.")
    mdl.add_argument("--num_layers", type=int, default=2,
                     help="Number of hidden layers (MLP) or residual "
                          "blocks (ResNet) in each conditioner.")
    mdl.add_argument("--arch", choices=["mlp", "resnet"], default="mlp",
                     help="Conditioner network architecture.")
    mdl.add_argument("--activation",
                     choices=["relu", "tanh", "elu", "leaky_relu",
                               "silu", "gelu"],
                     default="relu",
                     help="Activation function for conditioner networks.")
    mdl.add_argument("--dropout", type=float, default=0.0,
                     help="Dropout probability in conditioner networks.")

    # ── Training ─────────────────────────────────────────────────────────── #
    trn = p.add_argument_group("Training")
    trn.add_argument("--lr", type=float, default=1e-3,
                     help="Initial learning rate.")
    trn.add_argument("--weight_decay", type=float, default=0.0,
                     help="L2 weight decay for Adam.")
    trn.add_argument("--batch_size", type=int, default=8192,
                     help="Training batch size.")
    trn.add_argument("--epochs", type=int, default=10_000,
                     help="Maximum training epochs.")
    trn.add_argument("--patience", type=int, default=20,
                     help="Early-stopping patience (epochs).")
    trn.add_argument("--ema_alpha", type=float, default=0.3,
                     help="EMA smoothing factor for validation loss.")
    trn.add_argument("--clip_grad", type=float, default=5.0,
                     help="Gradient clipping max-norm.")
    trn.add_argument("--use_plateau", action="store_true", default=True,
                     help="Use ReduceLROnPlateau scheduler.")
    trn.add_argument("--no_plateau", dest="use_plateau", action="store_false",
                     help="Disable ReduceLROnPlateau.")
    trn.add_argument("--plateau_factor", type=float, default=0.5,
                     help="LR reduction factor for plateau scheduler.")
    trn.add_argument("--plateau_patience", type=int, default=10,
                     help="Patience for plateau LR reduction.")
    trn.add_argument("--use_cosine", action="store_true", default=False,
                     help="Add cosine-annealing LR scheduler.")
    trn.add_argument("--cosine_t_max", type=int, default=500,
                     help="Period for cosine annealing.")
    trn.add_argument("--log_every", type=int, default=10,
                     help="Print training status every N epochs.")
    trn.add_argument("--resume_from", type=str, default=None,
                     help="Path to a checkpoint to resume training from.")

    # ── I/O ──────────────────────────────────────────────────────────────── #
    io = p.add_argument_group("I/O")
    io.add_argument("--run_name", type=str, default="model",
                    help="Identifier prefix for checkpoint and loss files.")
    io.add_argument("--save_dir", type=str, default="checkpoints",
                    help="Directory for checkpoints and loss JSON files.")
    io.add_argument("--output_dir", type=str, default="outputs",
                    help="Directory for plots and generated samples.")
    io.add_argument("--checkpoint", type=str, default=None,
                    help="Checkpoint path (required for sample/evaluate/plot).")

    # ── Sampling ─────────────────────────────────────────────────────────── #
    spl = p.add_argument_group("Sampling")
    spl.add_argument("--num_samples", type=int, default=10_000,
                     help="Number of samples to generate.")
    spl.add_argument("--target_r", type=float, default=500.0,
                     help="Momentum magnitude (GeV) assumed when back-"
                          "converting angular samples to Cartesian.")

    # ── Device ───────────────────────────────────────────────────────────── #
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                   default="auto",
                   help="Compute device.")

    # ── Plotting ─────────────────────────────────────────────────────────── #
    plt_g = p.add_argument_group("Plotting")
    plt_g.add_argument("--no_plots", action="store_true", default=False,
                       help="Suppress all plot generation.")
    plt_g.add_argument("--dpi", type=int, default=400,
                       help="DPI for saved figures.")

    return p


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_config(args: argparse.Namespace,
                 yaml_cfg: dict) -> argparse.Namespace:
    """Override argparse defaults with YAML values, unless CLI was explicit."""
    # argparse does not distinguish 'set by user' vs 'default'.
    # We merge conservatively: YAML only fills values that equal the parser
    # defaults (i.e. were not explicitly provided on the command line).
    parser   = build_parser()
    defaults = vars(parser.parse_args([
        "--data", args.data or "dummy",  # need a required arg
    ]))
    for key, yaml_val in yaml_cfg.items():
        if key in defaults and vars(args)[key] == defaults[key]:
            setattr(args, key, yaml_val)
    return args


# ---------------------------------------------------------------------------
# Preprocessing helper  (shared across modes)
# ---------------------------------------------------------------------------

def _prepare_data(args: argparse.Namespace):
    """Load data and return normalised tensor + stats."""
    from hep_nsf.utils import (load_json_data, normalise,
                                cartesian_to_spherical, set_seed)
    set_seed(args.seed)

    raw = load_json_data(args.data)  # (N, 3) Cartesian

    if args.model in ("s2", "s2_recursive"):
        # Project to unit sphere → (cos θ, φ)
        data = cartesian_to_spherical(raw, phi_range="0_2pi")
    else:
        # R³: work directly in Cartesian
        data = raw

    data_norm, mean, std = normalise(data)
    return data_norm, mean, std, data


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def run_train(args: argparse.Namespace) -> None:
    from hep_nsf.networks import build_model
    from hep_nsf.utils    import make_dataloaders
    from hep_nsf.train    import train_model
    from hep_nsf.utils    import get_device
    from hep_nsf.analysis import model_summary
    import hep_nsf.plotting as plt_mod

    device = get_device(args.device)
    print(f"Device: {device}")

    data_norm, mean, std, _ = _prepare_data(args)

    train_loader, val_loader = make_dataloaders(
        data_norm, batch_size=args.batch_size,
        val_frac=args.val_frac, seed=args.seed)

    # Build model
    model_kwargs = dict(
        num_bins=args.num_bins,
        num_splines=args.num_splines,
        bound=args.bound,
    )
    if args.model != "s2_recursive":
        model_kwargs.update(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            arch=args.arch,
            activation=args.activation,
            dropout=args.dropout,
        )

    model = build_model(args.model, **model_kwargs)
    model_summary(model, model_type=args.model)

    model, losses = train_model(
        model, train_loader, val_loader,
        std_tensor=std,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        patience=args.patience,
        ema_alpha=args.ema_alpha,
        clip_grad=args.clip_grad,
        use_plateau=args.use_plateau,
        plateau_factor=args.plateau_factor,
        plateau_patience=args.plateau_patience,
        use_cosine=args.use_cosine,
        cosine_t_max=args.cosine_t_max,
        log_every=args.log_every,
        save_dir=args.save_dir,
        run_name=args.run_name,
        resume_from=args.resume_from,
    )

    if not args.no_plots:
        plt_mod.plot_loss_curves(losses["train"], losses["val"],
                                 title=f"Training Curves — {args.run_name}",
                                 output_dir=args.output_dir)
    print("Training complete.")


def run_sample(args: argparse.Namespace) -> None:
    from hep_nsf.networks  import build_model
    from hep_nsf.utils     import (load_checkpoint, get_device,
                                    denormalise, spherical_to_cartesian)
    import hep_nsf.plotting as plt_mod

    if args.checkpoint is None:
        sys.exit("--checkpoint is required for mode=sample")

    device = get_device(args.device)
    data_norm, mean, std, data_phys = _prepare_data(args)

    model_kwargs = dict(
        num_bins=args.num_bins,
        num_splines=args.num_splines,
        bound=args.bound,
    )
    if args.model != "s2_recursive":
        model_kwargs.update(
            hidden_dim=args.hidden_dim, num_layers=args.num_layers,
            arch=args.arch, activation=args.activation, dropout=args.dropout)

    model = build_model(args.model, **model_kwargs).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    # Generate samples
    with torch.no_grad():
        z_samples = model.sample(args.num_samples, device=device)
        x_phys    = denormalise(z_samples, mean, std)

    x_np = x_phys.cpu().numpy()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_dir) / f"{args.run_name}_samples.npy"
    np.save(out_path, x_np)
    print(f"Saved {args.num_samples} samples → {out_path}")

    if not args.no_plots:
        labels = ([r"$\cos\theta$", r"$\phi$"]
                  if args.model in ("s2", "s2_recursive")
                  else [r"$p_x$", r"$p_y$", r"$p_z$"])
        data_np = data_phys.numpy()[:args.num_samples]

        plt_mod.plot_marginal_1d(
            data_np, x_np, labels=labels,
            title=f"1-D Marginals — {args.run_name}",
            output_dir=args.output_dir)
        plt_mod.plot_marginal_2d(
            data_np, x_np, labels=labels,
            title=f"2-D Marginals — {args.run_name}",
            output_dir=args.output_dir)

        if args.model in ("s2", "s2_recursive"):
            from hep_nsf.utils import load_json_data, cartesian_to_spherical
            raw = load_json_data(args.data)
            cyl_data = cartesian_to_spherical(raw)
            plt_mod.plot_mollweide_kde(cyl_data, "Data KDE",
                                       output_dir=args.output_dir)
            plt_mod.plot_mollweide_kde(
                torch.tensor(x_np), "Flow KDE",
                output_dir=args.output_dir)
        else:
            from hep_nsf.utils import load_json_data
            raw = load_json_data(args.data)
            plt_mod.plot_physics_comparison(
                raw, torch.tensor(x_np),
                title=f"Physics Comparison — {args.run_name}",
                output_dir=args.output_dir)
            plt_mod.plot_radius_distribution(
                x_np, output_dir=args.output_dir)


def run_evaluate(args: argparse.Namespace) -> None:
    from hep_nsf.networks  import build_model
    from hep_nsf.utils     import (load_checkpoint, get_device,
                                    denormalise, cartesian_to_spherical,
                                    load_json_data)
    from hep_nsf.analysis  import evaluate, consistency_report

    if args.checkpoint is None:
        sys.exit("--checkpoint is required for mode=evaluate")

    device = get_device(args.device)
    data_norm, mean, std, data_phys = _prepare_data(args)

    model_kwargs = dict(
        num_bins=args.num_bins, num_splines=args.num_splines,
        bound=args.bound)
    if args.model != "s2_recursive":
        model_kwargs.update(
            hidden_dim=args.hidden_dim, num_layers=args.num_layers,
            arch=args.arch, activation=args.activation, dropout=args.dropout)

    model = build_model(args.model, **model_kwargs).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    with torch.no_grad():
        z_samples = model.sample(args.num_samples, device=device)
        x_phys    = denormalise(z_samples, mean, std)

    raw = load_json_data(args.data)

    if args.model in ("s2", "s2_recursive"):
        cyl_d = cartesian_to_spherical(raw).numpy()
        cyl_s = x_phys.cpu().numpy()
        evaluate(cyl_d[:, 0], cyl_d[:, 1],
                 cyl_s[:, 0], cyl_s[:, 1])
        consistency_report(torch.tensor(cyl_s), target_r=args.target_r)
    else:
        from hep_nsf.utils import cartesian_to_spherical
        cyl_d = cartesian_to_spherical(raw).numpy()
        cyl_s = cartesian_to_spherical(x_phys.cpu()).numpy()
        evaluate(cyl_d[:, 0], cyl_d[:, 1],
                 cyl_s[:, 0], cyl_s[:, 1])


def run_plot(args: argparse.Namespace) -> None:
    from hep_nsf.utils import load_losses
    if args.checkpoint is not None:
        loss_path = Path(args.checkpoint).with_name(
            Path(args.checkpoint).stem.replace("_best", "_losses") + ".json")
        if loss_path.exists():
            losses = load_losses(loss_path)
            import hep_nsf.plotting as plt_mod
            plt_mod.plot_loss_curves(
                losses["train"], losses["val"],
                title="Training Curves",
                output_dir=args.output_dir)
            print(f"Saved loss curve → {args.output_dir}")
    # Delegate to sample which also saves plots
    run_sample(args)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = build_parser()
    args   = parser.parse_args(argv)

    # Load YAML config if provided
    if args.config is not None:
        yaml_cfg = load_yaml_config(args.config)
        args = merge_config(args, yaml_cfg)

    print("=" * 60)
    print(" HEP Neural Spline Flow")
    print(f" Mode      : {args.mode}")
    print(f" Model     : {args.model}")
    print(f" Data      : {args.data}")
    print(f" Bins      : {args.num_bins}   Splines: {args.num_splines}")
    print("=" * 60 + "\n")

    if args.mode == "train":
        run_train(args)
    elif args.mode == "sample":
        run_sample(args)
    elif args.mode == "evaluate":
        run_evaluate(args)
    elif args.mode == "plot":
        run_plot(args)


if __name__ == "__main__":
    main()

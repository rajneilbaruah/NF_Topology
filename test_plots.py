"""
test_plots.py
=============
Standalone script to test every plotting function in hep_nsf/plotting.py.

Run from the NF_Sphere/ directory (parent of hep_nsf/):

    python test_plots.py --data ../Data/NFSpheres/eemumu_mup.json

If you already have a trained checkpoint, pass it too:

    python test_plots.py \
        --data ../Data/NFSpheres/eemumu_mup.json \
        --checkpoint checkpoints/smoke_test_best.pt \
        --num_bins 16 --num_splines 2 --hidden_dim 32

Each plot is saved to test_plot_outputs/ and a PASS/FAIL summary is printed.
"""

import argparse
import traceback
import numpy as np
import torch

# ── argument parsing ─────────────────────────────────────────────────────── #
parser = argparse.ArgumentParser()
parser.add_argument("--data",        required=True)
parser.add_argument("--checkpoint",  default=None)
parser.add_argument("--model",       default="s2")
parser.add_argument("--num_bins",    type=int, default=16)
parser.add_argument("--num_splines", type=int, default=1)
parser.add_argument("--hidden_dim",  type=int, default=32)
parser.add_argument("--num_layers",  type=int, default=2)
parser.add_argument("--bound",       type=float, default=5.0)
parser.add_argument("--num_samples", type=int, default=3000)
parser.add_argument("--output_dir",  default="test_plot_outputs")
args = parser.parse_args()

# ── imports ──────────────────────────────────────────────────────────────── #
from hep_nsf.utils import (
    load_json_data, cartesian_to_spherical, spherical_to_cartesian,
    normalise, denormalise, get_device, load_checkpoint
)
from hep_nsf.networks  import build_model
import hep_nsf.plotting as P

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
results = {}

def run(name, fn):
    try:
        fn()
        print(f"{PASS}  {name}")
        results[name] = "PASS"
    except Exception:
        print(f"{FAIL}  {name}")
        traceback.print_exc()
        results[name] = "FAIL"

# ── prepare data ─────────────────────────────────────────────────────────── #
print("\nLoading data...")
raw       = load_json_data(args.data)                     # (N, 3)
cyl       = cartesian_to_spherical(raw)                   # (N, 2)  cos_theta, phi
norm, mean, std = normalise(cyl)

# Fake "flow samples" = slightly noisy data (good enough for plot testing)
rng     = np.random.default_rng(42)
noise   = torch.tensor(rng.normal(0, 0.05, cyl.shape), dtype=torch.float32)
cyl_gen = (cyl + noise).clamp(
    torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 2*np.pi]))

data_np = cyl.numpy()
gen_np  = cyl_gen.numpy()

# For R³ plots — use the raw Cartesian data with small noise so radius varies
cart_data = raw
cart_noise = torch.tensor(rng.normal(0, 5.0, raw.shape), dtype=torch.float32)
cart_gen  = raw + cart_noise

# Fake loss curves
train_losses = list(np.linspace(3.0, 1.5, 40) + rng.normal(0, 0.05, 40))
val_losses   = list(np.linspace(3.2, 1.6, 40) + rng.normal(0, 0.05, 40))

OUT = args.output_dir
print(f"Saving all plots to: {OUT}/\n")

# ── load model if checkpoint given ───────────────────────────────────────── #
model = None
device = get_device("cpu")

if args.checkpoint:
    try:
        model = build_model(
            args.model,
            num_bins=args.num_bins,
            num_splines=args.num_splines,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            bound=args.bound,
        ).to(device)
        load_checkpoint(args.checkpoint, model, device=device)
        model.eval()
        print("Checkpoint loaded successfully.\n")
    except Exception:
        print("Warning: checkpoint load failed. Skipping model-dependent plots.\n")
        model = None

# ── run each plot test ───────────────────────────────────────────────────── #
print("=" * 55)
print("  Testing all plotting functions")
print("=" * 55)

run("plot_loss_curves",
    lambda: P.plot_loss_curves(
        train_losses, val_losses,
        title="Test Loss Curves",
        save="test_loss_curves.png",
        output_dir=OUT))

run("plot_marginal_1d  (2D angular data)",
    lambda: P.plot_marginal_1d(
        data_np, gen_np,
        labels=[r"$\cos\theta$", r"$\phi$"],
        title="Test 1-D Marginals (S2)",
        save="test_marginals_1d_s2.png",
        output_dir=OUT))

run("plot_marginal_1d  (3D Cartesian data)",
    lambda: P.plot_marginal_1d(
        cart_data.numpy(), cart_gen.numpy(),
        labels=[r"$p_x$", r"$p_y$", r"$p_z$"],
        title="Test 1-D Marginals (R3)",
        save="test_marginals_1d_r3.png",
        output_dir=OUT))

run("plot_marginal_2d  (2D angular data)",
    lambda: P.plot_marginal_2d(
        data_np, gen_np,
        labels=[r"$\cos\theta$", r"$\phi$"],
        title="Test 2-D Marginals (S2)",
        save="test_marginals_2d_s2.png",
        output_dir=OUT))

run("plot_marginal_2d  (3D Cartesian data)",
    lambda: P.plot_marginal_2d(
        cart_data.numpy(), cart_gen.numpy(),
        labels=[r"$p_x$", r"$p_y$", r"$p_z$"],
        title="Test 2-D Marginals (R3)",
        save="test_marginals_2d_r3.png",
        output_dir=OUT))

run("plot_mollweide_kde  (data)",
    lambda: P.plot_mollweide_kde(
        cyl,
        title="Test Data KDE",
        save="test_mollweide_data.png",
        output_dir=OUT))

run("plot_mollweide_kde  (generated)",
    lambda: P.plot_mollweide_kde(
        cyl_gen,
        title="Test Flow KDE",
        save="test_mollweide_flow.png",
        output_dir=OUT))

run("plot_physics_comparison",
    lambda: P.plot_physics_comparison(
        cart_data, cart_gen,
        title="Test Physics Comparison",
        save="test_physics_comparison.png",
        output_dir=OUT))

run("plot_radius_distribution",
    lambda: P.plot_radius_distribution(
        cart_gen.numpy(),
        title="Test |p| Distribution",
        save="test_radius_dist.png",
        output_dir=OUT))

# Model-dependent plots — only run if checkpoint was loaded
if model is not None:
    run("plot_base_mapping",
        lambda: P.plot_base_mapping(
            model, device, mean, std,
            dim=2,
            title="Test Base Mapping",
            save="test_base_mapping.png",
            output_dir=OUT))

    run("plot_jacobian_map",
        lambda: P.plot_jacobian_map(
            model, device, mean, std,
            n_samples=1000,
            title="Test Jacobian Map",
            save="test_jacobian_map.png",
            output_dir=OUT))
else:
    print("  SKIP  plot_base_mapping   (no checkpoint provided)")
    print("  SKIP  plot_jacobian_map   (no checkpoint provided)")

# ── summary ──────────────────────────────────────────────────────────────── #
print("\n" + "=" * 55)
print("  Summary")
print("=" * 55)
passed = sum(1 for v in results.values() if v == "PASS")
failed = sum(1 for v in results.values() if v == "FAIL")
print(f"  Passed : {passed}")
print(f"  Failed : {failed}")
if failed == 0:
    print("\n  \033[92mAll plotting functions OK.\033[0m")
else:
    print("\n  \033[91mSome plots failed — see tracebacks above.\033[0m")
print("=" * 55)
print(f"\nAll output files saved to: {OUT}/\n")

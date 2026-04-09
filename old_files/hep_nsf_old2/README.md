# HEP Neural Spline Flows (`hep_nsf`)

A clean, cluster-ready Python package for learning probability densities of
high-energy physics (HEP) 3-momentum data using **Rational-Quadratic Spline
(RQS) normalising flows**.

Originally developed from a set of exploratory Jupyter notebooks covering
angular flows on S² and Cartesian flows in R³, this package restructures
everything into a production-quality codebase with full CLI support and
configurable multi-spline stacking.

---

## Contents

- [Features](#features)
- [Package Structure](#package-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Multi-Spline Stacking](#multi-spline-stacking)
- [CLI Reference](#cli-reference)
- [Config Files (YAML)](#config-files-yaml)
- [Cluster Usage (SLURM)](#cluster-usage-slurm)
- [Python API](#python-api)
- [Module Reference](#module-reference)
- [Output Files](#output-files)
- [Evaluation Metrics](#evaluation-metrics)
- [Architecture Details](#architecture-details)

---

## Features

| Feature | Details |
|---|---|
| **Three flow architectures** | S² angular, R³ Cartesian, fast recursive sphere |
| **Configurable spline depth** | Stack N coupling-layer blocks via `--num_splines` |
| **Conditioner networks** | Plain MLP or ResNet with layer-norm |
| **Robust training loop** | EMA val loss, early stopping, grad clipping, NaN filtering |
| **Dual LR schedulers** | ReduceLROnPlateau + optional CosineAnnealing |
| **Full checkpoint system** | Save/resume with epoch + optimiser state |
| **Rich analysis suite** | KL divergence, ESS, Wasserstein-1, Jensen–Shannon, consistency |
| **Diagnostic plots** | 1-D/2-D marginals, Mollweide KDE, physics variables, Jacobian map |
| **CLI + YAML configs** | Every hyperparameter is a CLI flag; override groups via YAML |
| **Cluster-ready** | No interactive display needed; `Agg` backend; SLURM examples included |

---

## Package Structure

```
hep_nsf/
├── __init__.py          Public API re-exports
├── splines.py           Core RQS bijection (forward + inverse)
├── mlps.py              Conditioner networks (MLP, ResNet, factory)
├── networks.py          Flow models (AngularSphereFlow, CartesianNSF,
│                                     RecursiveSphereFlow, build_model)
├── utils.py             Coordinate transforms, data I/O, normalisation,
│                        DataLoader factory, checkpoint helpers
├── train.py             Training loop (NLL loss, early stopping, schedulers)
├── analysis.py          KL divergence, ESS, Wasserstein, consistency report
├── plotting.py          All matplotlib visualisation functions
├── main.py              CLI entry point (train / sample / evaluate / plot)
├── configs/
│   ├── s2_default.yaml  Defaults for angular S² flow
│   └── r3_default.yaml  Defaults for Cartesian R³ flow
├── requirements.txt
├── setup.py
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/hep_nsf.git
cd hep_nsf

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install
pip install -e .
# or, without the editable install:
pip install -r requirements.txt
```

**Dependencies:** `torch >= 2.0`, `numpy`, `scipy`, `matplotlib`, `pyyaml`.

No `normflows` package is required — the RQS is implemented from scratch.

---

## Quick Start

### Train the S² angular flow

```bash
python -m hep_nsf.main --mode train \
    --model s2 \
    --data eemumu_mup.json \
    --num_bins 32 \
    --num_splines 1 \
    --run_name mup_s2
```

### Train the R³ Cartesian flow

```bash
python -m hep_nsf.main --mode train \
    --model r3 \
    --data eemumu_mup.json \
    --num_bins 32 \
    --num_splines 1 \
    --run_name mup_r3
```

### Generate samples from a saved checkpoint

```bash
python -m hep_nsf.main --mode sample \
    --model s2 \
    --data eemumu_mup.json \
    --checkpoint checkpoints/mup_s2_best.pt \
    --num_samples 50000 \
    --output_dir outputs
```

### Evaluate a trained model

```bash
python -m hep_nsf.main --mode evaluate \
    --model s2 \
    --data eemumu_mup.json \
    --checkpoint checkpoints/mup_s2_best.pt
```

---

## Multi-Spline Stacking

The key new feature over the original notebooks is **stackable spline blocks**.
Each block is a complete alternating-coupling pass through all dimensions.

| `--num_splines` | S² layers | R³ layers | Notes |
|---|---|---|---|
| 1 | 2 RQS layers | 2 RQS layers | Original notebook behaviour |
| 2 | 4 RQS layers | 4 RQS layers | Double expressiveness |
| 3 | 6 RQS layers | 6 RQS layers | Recommended for hard distributions |
| 4+ | … | … | Diminishing returns; check compute budget |

**How the stacking works (S² example with `num_splines=2`):**

```
Block 1:
  φ   ← RQS(φ   | cos θ)    [net_phi[0]]
  cos θ ← RQS(cos θ | φ)    [net_costheta[0]]

Block 2:
  φ   ← RQS(φ   | cos θ)    [net_phi[1]]
  cos θ ← RQS(cos θ | φ)    [net_costheta[1]]
```

Each block has its own independent conditioner network, so the total
parameter count scales linearly with `num_splines`.

**Example — sweep over spline depth:**

```bash
for K in 1 2 3 4; do
  python -m hep_nsf.main --mode train \
      --model s2 --data eemumu_mup.json \
      --num_splines $K \
      --run_name mup_s2_k${K}
done
```

---

## CLI Reference

```
usage: hep_nsf [-h] [--config CONFIG]
               [--mode {train,sample,evaluate,plot}]
               --data DATA [--val_frac VAL_FRAC] [--seed SEED]
               [--model {s2,r3,s2_recursive}]
               [--num_bins NUM_BINS] [--num_splines NUM_SPLINES]
               [--bound BOUND] [--hidden_dim HIDDEN_DIM]
               [--num_layers NUM_LAYERS] [--arch {mlp,resnet}]
               [--activation ACTIVATION] [--dropout DROPOUT]
               [--lr LR] [--weight_decay WEIGHT_DECAY]
               [--batch_size BATCH_SIZE] [--epochs EPOCHS]
               [--patience PATIENCE] [--ema_alpha EMA_ALPHA]
               [--clip_grad CLIP_GRAD]
               [--use_plateau | --no_plateau]
               [--plateau_factor PLATEAU_FACTOR]
               [--plateau_patience PLATEAU_PATIENCE]
               [--use_cosine] [--cosine_t_max COSINE_T_MAX]
               [--log_every LOG_EVERY] [--resume_from RESUME_FROM]
               [--run_name RUN_NAME] [--save_dir SAVE_DIR]
               [--output_dir OUTPUT_DIR] [--checkpoint CHECKPOINT]
               [--num_samples NUM_SAMPLES] [--target_r TARGET_R]
               [--device {auto,cpu,cuda,mps}]
               [--no_plots] [--dpi DPI]
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `s2` | `s2` = angular S², `r3` = Cartesian R³, `s2_recursive` = no-MLP |
| `--num_bins` | 32 | RQS bins per layer |
| `--num_splines` | 1 | **Number of stacked coupling blocks** |
| `--bound` | 5.0 | Spline domain half-width (standardised space) |
| `--hidden_dim` | 64 | Conditioner MLP hidden width |
| `--num_layers` | 2 | Conditioner depth (hidden layers or residual blocks) |
| `--arch` | `mlp` | `mlp` or `resnet` |
| `--activation` | `relu` | relu/tanh/elu/leaky_relu/silu/gelu |
| `--lr` | 1e-3 | Adam learning rate |
| `--patience` | 20 | Early-stopping patience (epochs) |
| `--ema_alpha` | 0.3 | Exponential moving average factor for val loss |
| `--resume_from` | — | Resume training from a checkpoint |
| `--checkpoint` | — | Load model for sample/evaluate/plot |
| `--num_samples` | 10000 | Samples to generate in sample/evaluate mode |
| `--device` | `auto` | `auto` / `cpu` / `cuda` / `mps` |

---

## Config Files (YAML)

Instead of typing long CLI commands, put settings in a YAML file:

```yaml
# my_run.yaml
mode: train
model: s2
num_bins: 64
num_splines: 3
hidden_dim: 128
num_layers: 3
arch: resnet
lr: 0.0005
batch_size: 4096
patience: 30
run_name: mup_s2_deep
```

```bash
python -m hep_nsf.main --config my_run.yaml --data eemumu_mup.json
```

CLI arguments always override YAML values.

---

## Cluster Usage (SLURM)

### Example SLURM script

```bash
#!/bin/bash
#SBATCH --job-name=hep_nsf_train
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out

module load python/3.11 cuda/12.1

source /path/to/.venv/bin/activate

python -m hep_nsf.main \
    --config configs/s2_default.yaml \
    --data /scratch/$USER/eemumu_mup.json \
    --num_splines 3 \
    --run_name mup_s2_k3_${SLURM_JOB_ID} \
    --save_dir /scratch/$USER/checkpoints \
    --output_dir /scratch/$USER/outputs \
    --device cuda
```

### Parameter sweep over spline depth

```bash
#!/bin/bash
for K in 1 2 3 4; do
    sbatch --export=ALL,NUM_SPLINES=$K run_job.sh
done
```

Inside `run_job.sh`:
```bash
python -m hep_nsf.main --num_splines $NUM_SPLINES \
    --run_name mup_s2_k${NUM_SPLINES} ...
```

### Tips for cluster use

- Set `--device cuda` explicitly — `auto` works but `cuda` is clearer in logs.
- Use `--no_plots` on headless nodes if matplotlib causes issues (outputs
  are still saved as `.npy`/`.json`; generate plots locally after).
- Use `--resume_from` to continue interrupted jobs without losing progress.
- Loss curves are saved as `<run_name>_losses.json` alongside checkpoints;
  you can plot them offline with `plot_loss_curves()`.

---

## Python API

### Minimal training example

```python
import torch
from hep_nsf import (build_model, load_json_data, normalise,
                      cartesian_to_spherical, make_dataloaders,
                      train_model, get_device)

device = get_device("auto")

# Load and preprocess data
raw      = load_json_data("eemumu_mup.json")          # (N, 3) GeV
cyl      = cartesian_to_spherical(raw)                 # (N, 2) [cos θ, φ]
data_norm, mean, std = normalise(cyl)

train_loader, val_loader = make_dataloaders(data_norm, batch_size=8192)

# Build model with 2 stacked spline blocks
model = build_model("s2", num_bins=32, num_splines=2,
                    hidden_dim=64, num_layers=2)

# Train
model, losses = train_model(
    model, train_loader, val_loader,
    std_tensor=std, device=device,
    lr=1e-3, patience=20, run_name="my_flow")
```

### Sampling

```python
import torch
from hep_nsf import denormalise, spherical_to_cartesian

model.eval()
with torch.no_grad():
    z_norm = model.sample(50_000, device=device)
    z_phys = denormalise(z_norm, mean, std)     # physical (cos θ, φ)
    cart   = spherical_to_cartesian(z_phys, r=500.0)   # (px, py, pz) GeV
```

### Evaluation

```python
from hep_nsf import evaluate, consistency_report
import numpy as np

cyl_data = cyl.numpy()
cyl_flow = z_phys.cpu().numpy()

metrics = evaluate(cyl_data[:, 0], cyl_data[:, 1],
                   cyl_flow[:, 0], cyl_flow[:, 1])
# {'kl': ..., 'ess': ..., 'ess_frac': ..., 'W1_costheta': ..., 'W1_phi': ...}

consistency_report(torch.tensor(cyl_flow), target_r=500.0)
```

### Plotting

```python
from hep_nsf import plotting as plt_mod
import numpy as np

plt_mod.plot_mollweide_kde(z_phys, "Flow Density", output_dir="outputs")
plt_mod.plot_physics_comparison(raw, cart, output_dir="outputs")
plt_mod.plot_loss_curves(losses["train"], losses["val"], output_dir="outputs")
```

---

## Module Reference

### `splines.py`

| Function | Description |
|---|---|
| `rqs(inputs, w, h, d, inverse, bound)` | Unified RQS entry point |
| `rqs_forward(...)` | Data → base transform + log-det |
| `rqs_inverse(...)` | Base → data transform (quadratic solve) |

### `mlps.py`

| Class / Function | Description |
|---|---|
| `MLP` | Plain feed-forward conditioner |
| `ResNet` | Residual conditioner with LayerNorm |
| `build_conditioner(arch, ...)` | Factory: `'mlp'` or `'resnet'` |

### `networks.py`

| Class / Function | Description |
|---|---|
| `AngularSphereFlow` | 2-D S² flow; input `(cos θ, φ)` normalised |
| `CartesianNSF` | 3-D R³ flow; input `(px, py, pz)` standardised |
| `RecursiveSphereFlow` | Parameter-only sphere flow (no MLP) |
| `build_model(model_type, **kwargs)` | Registry-based factory |

All models expose:
- `forward(x, inverse=False)` → `(z, ldj)` or `x_sample`
- `log_prob(x)` → log-probability per sample
- `sample(n, device)` → `n` new samples

### `utils.py`

| Function | Description |
|---|---|
| `set_seed(seed)` | Fix all random seeds |
| `get_device(prefer)` | Best available device |
| `cartesian_to_spherical(p)` | `(px,py,pz)` → `(cos θ, φ)` |
| `spherical_to_cartesian(sph, r)` | `(cos θ, φ)` → `(px,py,pz)` |
| `cartesian_to_physics(p)` | Dict of HEP variables |
| `load_json_data(path)` | JSON → float32 tensor |
| `normalise(data)` | Standardise; return `(norm, mean, std)` |
| `denormalise(norm, mean, std)` | Reverse standardisation |
| `make_dataloaders(data, ...)` | Train/val split + DataLoader |
| `save_checkpoint / load_checkpoint` | Full checkpoint I/O |
| `count_parameters(model)` | Total trainable parameters |

### `train.py`

| Function | Description |
|---|---|
| `train_model(model, ...)` | Full training loop with early stopping |
| `nll_loss(z, ldj, std_correction)` | Batch-mean NLL |

### `analysis.py`

| Function | Description |
|---|---|
| `kl_divergence_kde(...)` | KDE-based KL divergence on S² |
| `effective_sample_size(...)` | Importance-weighted ESS |
| `wasserstein_1d(data, samples)` | Per-feature W₁ distance |
| `js_divergence_1d(data, samples)` | Per-feature JSD |
| `consistency_report(cyl_samples)` | Unphysical fraction + radius stats |
| `evaluate(...)` | Full metric suite with printout |
| `model_summary(model)` | Parameter count + config |

### `plotting.py`

| Function | Description |
|---|---|
| `plot_loss_curves(train, val)` | Training / validation NLL |
| `plot_marginal_1d(data, samples)` | Overlaid 1-D histograms |
| `plot_marginal_2d(data, samples)` | 2-D scatter pairs |
| `plot_mollweide_kde(cyl_data)` | Sky-map KDE on S² |
| `plot_physics_comparison(data, samples)` | 6-panel `px,py,pz,pT,η,φ` |
| `plot_base_mapping(model, ...)` | Base → data mapping (2-D) |
| `plot_jacobian_map(model, ...)` | log-det-Jacobian in data space |
| `plot_radius_distribution(samples)` | `|p|` histogram |

---

## Output Files

After training, the following files are created:

```
checkpoints/
├── <run_name>_best.pt          Best model checkpoint
└── <run_name>_losses.json      Train/val loss curves

outputs/
├── loss_curves.png
├── marginals_1d.png
├── marginals_2d.png
├── mollweide_*.png             (S² models)
├── physics_comparison.png      (R³ models)
├── radius_distribution.png     (R³ models)
└── <run_name>_samples.npy      Generated samples array (N, D)
```

---

## Evaluation Metrics

| Metric | Interpretation |
|---|---|
| **KL(data ‖ flow)** | 0 = perfect; larger = flow misses data structure |
| **ESS / ESS%** | ESS% close to 100% = flow closely matches data density |
| **W₁** | Earth-mover distance per feature; smaller is better |
| **JSD** | 0 = identical distributions; ln 2 ≈ 0.693 = maximally different |

---

## Architecture Details

### RQS Bijection

The Rational-Quadratic Spline maps an input `x ∈ [−K, K]` to an output
`y ∈ [−K, K]` via a piecewise rational quadratic function with `B` bins.

Parameters per dimension: `B` widths + `B` heights + `(B+1)` derivatives
= `3B+1` scalars.

The forward pass is analytic; the inverse uses a closed-form quadratic root.
Both are numerically stable with ε-flooring of derivatives.

### Coupling Layers

A single **spline block** for the 2-D S² model performs:

```
φ_new, ldj₁   = RQS(φ     | conditioner₁(cos θ))
θ_new, ldj₂   = RQS(cos θ | conditioner₂(φ_new))
log_det_total += ldj₁ + ldj₂
```

With `num_splines=N`, this block is repeated N times with independent
conditioner parameters, giving `2N` total RQS applications and a
log-determinant that sums contributions from all layers.

### Loss Function

Training minimises the batch-mean negative log-likelihood (NLL):

```
L = −E_x[log q_θ(x)]
  = −E_x[log p_z(f(x)) + log |det J_f(x)|] + Σ log σ_i
```

The last term `Σ log σ_i` is the normalisation correction for standardised
data; it is a constant w.r.t. model parameters but included for correct
absolute NLL values.

---

## Citation

If you use this code in your research, please cite the relevant papers:

```bibtex
@article{durkan2019neural,
  title={Neural Spline Flows},
  author={Durkan, Conor and Bekasov, Artur and Murray, Iain and Papamakarios, George},
  journal={Advances in Neural Information Processing Systems},
  year={2019}
}
```

---

## License

MIT License — see `LICENSE` file for details.

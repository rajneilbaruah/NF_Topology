# HEP Neural Spline Flows (`hep_nsf`)

A clean, cluster-ready Python package for learning probability densities of
high-energy physics (HEP) 3-momentum data using **Rational-Quadratic Spline
(RQS) normalising flows**.

Originally developed from three Jupyter notebooks:

| Notebook | Model | Key |
|---|---|---|
| `Circular_Splines.ipynb` | `RecursiveSphereFlow` | `s2` |
| `Cartesian_RQS_3MomS2.ipynb` | `AngularSphereFlow` | `r2` |
| `Cartesian_RQS_3MomR3.ipynb` | `CartesianNSF` | `r3` |

---

## Model Naming Convention

This is the most important thing to understand before using the package.

| Key | Class | Input space | Base distribution | Notes |
|---|---|---|---|---|
| `s2` | `RecursiveSphereFlow` | Physical `(cos θ, φ)` | **Uniform on S²** | cos θ ∈ (−1,1), φ ∈ (0,2π). No normalisation applied. Free-parameter Cartesian spline for cos θ, MLP-conditioned **circular** spline for φ (enforces d[0]=d[K] for periodicity). |
| `r2` | `AngularSphereFlow` | **Standardised** `(cos θ, φ)` | **Gaussian in R²** | Standardises angular coordinates to zero mean / unit variance before training. Both dimensions use MLP-conditioned standard RQS. |
| `r3` | `CartesianNSF` | **Standardised** `(px, py, pz)` | **Gaussian in R³** | Works directly in 3D Cartesian momentum space after standardisation. |

The key distinction between `s2` and `r2` is:

- `s2` works in **physical coordinates** — the spline boundaries match the physical domain. The base distribution is the uniform measure on the sphere (the physically natural choice).
- `r2` works in **normalised coordinates** — data is standardised first, so the Gaussian base is appropriate.

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
| **Three flow architectures** | `s2` (uniform S²), `r2` (Gaussian R²), `r3` (Gaussian R³) |
| **Physically correct bases** | `s2` uses uniform-on-sphere base; `r2`/`r3` use standard Gaussian |
| **Circular spline for φ** | `s2` enforces d[0]=d[K] — smooth across the φ=0/2π boundary |
| **Configurable spline depth** | Stack N coupling blocks via `--num_splines` |
| **Conditioner networks** | Plain MLP or ResNet with layer-norm |
| **Self-describing checkpoints** | Architecture saved in checkpoint — no flags needed at inference |
| **Robust training loop** | EMA val loss, early stopping, grad clipping, NaN filtering |
| **Dual LR schedulers** | ReduceLROnPlateau + optional CosineAnnealing |
| **Rich analysis suite** | KL, ESS, Wasserstein-1, Jensen–Shannon, unphysical fraction |
| **Comprehensive plots** | Angular + Cartesian marginals, Mollweide KDE, physics variables, base mapping, Jacobian map |
| **CLI + YAML configs** | Every hyperparameter is a CLI flag; YAML override supported |
| **Cluster-ready** | Headless `Agg` backend; SLURM examples included |

---

## Package Structure

```
hep_nsf/
├── __init__.py          Public API re-exports
├── splines.py           RQS bijection: rqs, rqs_with_bounds, rqs_circular
├── mlps.py              Conditioner networks: MLP, ResNet, build_conditioner
├── networks.py          Flow models: RecursiveSphereFlow, AngularSphereFlow,
│                                     CartesianNSF, build_model
├── utils.py             Coordinate transforms, data I/O, normalisation,
│                        DataLoader factory, checkpoint helpers
├── train.py             Training loop (model.log_prob, early stopping, schedulers)
├── analysis.py          KL, ESS, Wasserstein, JSD, consistency report
├── plotting.py          All matplotlib visualisation functions
├── main.py              CLI entry point (train / sample / evaluate / plot)
├── configs/
│   ├── s2_default.yaml
│   ├── r2_default.yaml
│   └── r3_default.yaml
├── requirements.txt
└── setup.py

# Scripts in the parent directory (NF_Sphere/)
train_all.py             Train all three models on the same dataset
plot_all.py              Generate all plots for all three models
analyse_all.py           Full metric suite comparison across models
run_tests.py             Exhaustive test suite (251 tests)
test_plots.py            Targeted plotting function test
```

---

## Installation

```bash
git clone https://github.com/<your-username>/hep_nsf.git
cd hep_nsf
python -m venv .venv
source .venv/bin/activate   # Linux / macOS

pip install -e .
# or
pip install -r requirements.txt
```

Dependencies: `torch>=2.0`, `numpy`, `scipy`, `matplotlib`, `pyyaml`.

---

## Quick Start

### Train `s2` — physical angular flow (uniform S² base)

```bash
python -m hep_nsf.main --mode train \
    --model s2 \
    --data eemumu_mup.json \
    --run_name mup_s2
```

### Train `r2` — normalised angular flow (Gaussian R² base)

```bash
python -m hep_nsf.main --mode train \
    --model r2 \
    --data eemumu_mup.json \
    --run_name mup_r2
```

### Train `r3` — Cartesian 3-momentum flow

```bash
python -m hep_nsf.main --mode train \
    --model r3 \
    --data eemumu_mup.json \
    --run_name mup_r3
```

### Sample from a checkpoint (no architecture flags needed)

```bash
python -m hep_nsf.main --mode sample \
    --checkpoint checkpoints/mup_s2_best.pt \
    --data eemumu_mup.json \
    --num_samples 50000
```

The checkpoint is **self-describing** — it stores `model_type`, `num_bins`,
`num_splines`, `hidden_dim`, `num_layers`, `arch`, and `bound`. You never
need to re-specify the architecture at inference time.

### Train all three models at once

```bash
python train_all.py --data eemumu_mup.json
```

### Generate all plots for all models

```bash
python plot_all.py --data eemumu_mup.json
```

### Full statistical analysis

```bash
python analyse_all.py --data eemumu_mup.json
```

---

## Multi-Spline Stacking

Pass `--num_splines N` to stack N coupling blocks.

`num_splines=1` reproduces the original single-block notebook behaviour.

**How stacking works for `s2` (example with `num_splines=2`):**

```
Block 1:
  cos θ → rqs(cos θ,  free_params[0])          # Cartesian spline, no MLP
  φ     → rqs_circular(φ, MLP_0(cos θ'))       # circular spline, conditioned

Block 2:
  cos θ → rqs(cos θ', free_params[1])
  φ     → rqs_circular(φ', MLP_1(cos θ''))
```

Note: the inverse correctly undoes each block in reverse order, using the
**transformed** cos θ to condition the φ inversion before then inverting cos θ.

**Practical guidance:**

| `--num_splines` | Use case |
|---|---|
| 1 | Smooth unimodal distributions — original notebook behaviour |
| 2 | Good default upgrade, captures more structure |
| 3–4 | Multimodal or sharply peaked distributions |
| 5+ | Rarely needed; check validation loss still improves |

**Sweep example:**

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
python -m hep_nsf.main [--config CONFIG]
    --mode  {train, sample, evaluate, plot}
    --data  DATA_PATH

Model:
    --model         {s2, r2, r3}       default: s2
    --num_bins      int                default: 32
    --num_splines   int                default: 1
    --bound         float              default: 5.0   (r2/r3 only)
    --hidden_dim    int                default: 64
    --num_layers    int                default: 2
    --arch          {mlp, resnet}      default: mlp
    --activation    {relu,tanh,elu,leaky_relu,silu,gelu}
    --dropout       float              default: 0.0

Training:
    --lr            float              default: 1e-3
    --weight_decay  float              default: 0.0
    --batch_size    int                default: 8192
    --epochs        int                default: 10000
    --patience      int                default: 20
    --ema_alpha     float              default: 0.3
    --clip_grad     float              default: 5.0
    --use_plateau / --no_plateau       default: plateau enabled
    --plateau_factor    float          default: 0.5
    --plateau_patience  int            default: 10
    --use_cosine                       default: off
    --cosine_t_max  int                default: 500
    --log_every     int                default: 10
    --resume_from   PATH

I/O:
    --run_name      str                default: model
    --save_dir      str                default: checkpoints
    --output_dir    str                default: outputs
    --checkpoint    PATH               (required for sample/evaluate/plot)

Sampling:
    --num_samples   int                default: 10000
    --target_r      float              default: 500.0 (GeV)

Device:
    --device        {auto, cpu, cuda, mps}
    --no_plots
```

---

## Config Files (YAML)

```yaml
# configs/s2_default.yaml
mode: train
model: s2
num_bins: 32
num_splines: 1       # increase for more expressiveness
hidden_dim: 64
num_layers: 2
arch: mlp
activation: relu
lr: 0.001
batch_size: 8192
epochs: 10000
patience: 20
run_name: s2_flow
save_dir: checkpoints
output_dir: outputs
device: auto
```

```bash
python -m hep_nsf.main --config configs/s2_default.yaml --data eemumu_mup.json
# Override one value on CLI:
python -m hep_nsf.main --config configs/s2_default.yaml \
    --data eemumu_mup.json --num_splines 3 --run_name mup_s2_k3
```

---

## Cluster Usage (SLURM)

```bash
#!/bin/bash
#SBATCH --job-name=hep_nsf
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out

source /path/to/.venv/bin/activate

python -m hep_nsf.main \
    --config configs/s2_default.yaml \
    --data /scratch/$USER/eemumu_mup.json \
    --num_splines $NUM_SPLINES \
    --run_name mup_s2_k${NUM_SPLINES}_${SLURM_JOB_ID} \
    --save_dir /scratch/$USER/checkpoints \
    --output_dir /scratch/$USER/outputs \
    --device cuda \
    --no_plots
```

**Sweep over spline depth:**

```bash
for K in 1 2 3 4; do
    sbatch --export=ALL,NUM_SPLINES=$K submit.sh
done
```

**Tips:**

- Use `--no_plots` on headless nodes. Download outputs and plot locally.
- Use `--resume_from` to continue interrupted jobs.
- Loss curves are saved as `<run_name>_losses.json` alongside checkpoints.

---

## Python API

```python
import torch
from hep_nsf import (
    build_model, load_json_data, cartesian_to_spherical,
    normalise, denormalise, make_dataloaders,
    train_model, get_device, set_seed
)

set_seed(42)
device = get_device("auto")

# Load data
raw = load_json_data("eemumu_mup.json")          # (N, 3) GeV

# --- s2: physical (cos_theta, phi), uniform S2 base ---
cyl = cartesian_to_spherical(raw, phi_range="0_2pi")   # (N, 2)
cyl_norm, mean, std = normalise(cyl)
train_loader, val_loader = make_dataloaders(cyl_norm, batch_size=8192)

model = build_model("s2", num_bins=32, num_splines=2,
                    hidden_dim=64, num_layers=2)

model, losses = train_model(
    model, train_loader, val_loader,
    std_tensor=std, device=device,
    model_type="s2",          # saved into checkpoint
    num_layers=2, arch="mlp",
    lr=1e-3, patience=20, run_name="mup_s2")

# Sampling — s2.sample() returns physical (cos_theta, phi) directly
with torch.no_grad():
    samples = model.sample(50_000, device=device)  # (50000, 2) physical

# --- r2: standardised (cos_theta, phi), Gaussian R2 base ---
model_r2 = build_model("r2", num_bins=32, num_splines=2,
                        bound=5.0, hidden_dim=64, num_layers=2)

model_r2, _ = train_model(model_r2, train_loader, val_loader,
                           std_tensor=std, device=device,
                           model_type="r2", num_layers=2, arch="mlp",
                           run_name="mup_r2")

# r2.sample() returns normalised coords — must denormalise
with torch.no_grad():
    z_norm = model_r2.sample(50_000, device=device)
    z_phys = denormalise(z_norm, mean, std)        # (50000, 2) physical

# --- r3: standardised (px, py, pz), Gaussian R3 base ---
raw_norm, mean3, std3 = normalise(raw)
tl3, vl3 = make_dataloaders(raw_norm, batch_size=8192)

model_r3 = build_model("r3", num_bins=32, num_splines=1,
                        bound=5.0, hidden_dim=64, num_layers=2)

model_r3, _ = train_model(model_r3, tl3, vl3,
                           std_tensor=std3, device=device,
                           model_type="r3", num_layers=2, arch="mlp",
                           run_name="mup_r3")
```

**Important**: `s2.sample()` returns **physical** coordinates — no
`denormalise()` needed. `r2.sample()` and `r3.sample()` return
**normalised** coordinates — always call `denormalise()` before use.

---

## Module Reference

### `splines.py`

| Function | Description |
|---|---|
| `rqs(inputs, w, h, d, inverse, bound)` | Standard symmetric RQS on `[-bound, bound]`. Used by `r2` and `r3`. |
| `rqs_with_bounds(inputs, w, h, d, inverse, b_x, b_y)` | Asymmetric RQS with explicit domain. Used for `s2` cos θ. |
| `rqs_circular(inputs, w, h, d, inverse, b_x, b_y)` | Periodic RQS — enforces d[0]=d[K]. Used for `s2` φ. |

### `mlps.py`

| Class / Function | Description |
|---|---|
| `MLP` | Plain feed-forward conditioner |
| `ResNet` | Residual conditioner with LayerNorm |
| `build_conditioner(arch, in_dim, out_dim, ...)` | Factory: `'mlp'` or `'resnet'` |

### `networks.py`

| Class / Function | Description |
|---|---|
| `RecursiveSphereFlow` | `s2` — physical coords, uniform S² base, circular φ spline |
| `AngularSphereFlow` | `r2` — standardised angular, Gaussian R² base |
| `CartesianNSF` | `r3` — standardised Cartesian, Gaussian R³ base |
| `build_model(model_type, **kwargs)` | Registry factory: `'s2'`, `'r2'`, `'r3'` |

All models expose:
- `forward(x, inverse=False)` → `(z, ldj)` or `x_sample`
- `log_prob(x)` → log-probability per sample (using each model's correct base)
- `sample(n, device)` → `n` new samples

### `utils.py`

| Function | Description |
|---|---|
| `set_seed(seed)` | Fix all random seeds |
| `get_device(prefer)` | Best available device |
| `cartesian_to_spherical(p, phi_range)` | `(px,py,pz)` → `(cos θ, φ)` |
| `spherical_to_cartesian(sph, r)` | `(cos θ, φ)` → `(px,py,pz)` |
| `cartesian_to_physics(p)` | Dict: px, py, pz, pT, η, φ, \|p\| |
| `load_json_data(path)` | JSON → float32 Tensor |
| `normalise(data)` | Standardise; return `(norm, mean, std)` |
| `denormalise(norm, mean, std)` | Reverse standardisation |
| `make_dataloaders(data, ...)` | Train/val split + DataLoader |
| `save_checkpoint / load_checkpoint` | Full checkpoint I/O |

### `train.py`

| Function | Description |
|---|---|
| `train_model(model, ..., model_type, num_layers, arch)` | Full training loop. Uses `model.log_prob()` directly — each model's own base is respected automatically. Saves `model_type`, `num_bins`, `num_splines`, `hidden_dim`, `num_layers`, `arch`, `bound` into checkpoint. |

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
| `plot_loss_curves` | Training / validation NLL |
| `plot_marginal_1d` | Overlaid 1-D histograms |
| `plot_marginal_2d` | 2-D scatter pairs |
| `plot_mollweide_kde` | Sky-map KDE on S² |
| `plot_physics_comparison` | 6-panel px, py, pz, pT, η, φ |
| `plot_base_mapping` | Gaussian polar grid → data space (`r2`) |
| `plot_base_mapping_s2` | Uniform lat-lon grid → data space (`s2`) |
| `plot_jacobian_map` | log-det-Jacobian scatter (`s2`, `r2`) |
| `plot_jacobian_map_r3` | 3 projections + Mollweide + histogram (`r3`) |
| `plot_radius_distribution` | \|p\| histogram (`r3` only) |

---

## Output Files

```
checkpoints/
├── <run_name>_best.pt          Best model (self-describing checkpoint)
└── <run_name>_losses.json      Train/val loss curves

outputs/
├── <model>/
│   ├── <model>_loss_curves.png
│   ├── <model>_marginals_angular_1d.png
│   ├── <model>_marginals_angular_2d.png
│   ├── <model>_marginals_cartesian_1d.png
│   ├── <model>_marginals_cartesian_2d.png
│   ├── <model>_mollweide_data.png
│   ├── <model>_mollweide_flow.png
│   ├── <model>_physics_comparison.png
│   ├── <model>_base_mapping.png
│   ├── <model>_jacobian_map*.png
│   └── <model>_radius_dist.png     (r3 only)
└── combined_physics_comparison.png
```

---

## Evaluation Metrics

| Metric | Interpretation |
|---|---|
| **KL(data ‖ flow)** | 0 = perfect. Larger = flow misses data structure. |
| **ESS%** | 100% = perfect density match. Below ~50% suggests missing features. |
| **W₁** | Earth-mover distance per feature. Smaller is better. |
| **JSD** | 0 = identical. ln 2 ≈ 0.693 = maximally different. |
| **Unphysical %** | Fraction of `s2`/`r2` samples where cos θ ∉ [−1,1] or φ ∉ [0,2π]. Should be near zero. |

---

## Architecture Details

### Three bases

**`s2` — Uniform on S²:**
The natural base for angular data. The density is `1/(4π)` everywhere on
the sphere, so `log p(z) = −log(4π)` — a constant that appears only in the
NLL correction term during training. Sampling draws `cos θ ~ U[−1,1]`
and `φ ~ U[0, 2π]`.

**`r2` and `r3` — Standard Gaussian:**
After standardising to zero mean / unit variance, the Gaussian is the
natural base. `log p(z) = −½ ∑(z² + log 2π)`.

### Circular spline for φ in `s2`

φ lives on a circle: φ = 0 and φ = 2π are the same point. A standard RQS
applied to φ can create a discontinuity at this boundary. The circular
spline enforces:

```
d[K] = d[0]    (derivative at right edge = derivative at left edge)
```

Implemented as:
```python
d_circular = torch.cat([d, d[:, 0:1]], dim=-1)   # (B, K) → (B, K+1)
```

This guarantees the transform is smooth across the wrap-around point.

### RQS bijection

The Rational-Quadratic Spline maps `x ∈ [left, right]` to
`y ∈ [bottom, top]` via a piecewise rational quadratic with `K` bins.

Parameters per dimension: `K` widths + `K` heights + `(K+1)` derivatives
(or `K` for the circular variant) = `3K+1` scalars total.

Forward pass: analytic.
Inverse pass: closed-form quadratic root.

### NLL loss

Training minimises the batch-mean negative log-likelihood:

```
L = −E_x[log p_θ(x)]
  = −E_x[log p_z(f(x)) + log |det J_f(x)|] + Σ log σ_i
```

The `Σ log σ_i` term is the normalisation correction when training on
standardised data (constant w.r.t. model parameters but needed for correct
absolute NLL values). For `s2`, `p_z(z) = 1/(4π)` replaces the Gaussian.

---

## Running the Test Suite

```bash
python run_tests.py --data eemumu_mup.json
```

Runs 251 tests across all functions, all argument combinations, all
model × arch × activation × num_splines combinations, checkpoint
save/load/resume, CLI modes, YAML configs, and helper scripts.

Results in `test_run.log` and `test_summary.md`.

Skip specific sections:

```bash
python run_tests.py --data eemumu_mup.json --skip_sections 6 12
```

---

## Citation

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

MIT License.

# hep_nsf Test Suite — Summary

Run at: 2026-04-09 07:48:08

| Result | Count |
|--------|-------|
| ✓ PASS | 254 |
| ✗ FAIL | 0 |
| ○ SKIP | 2 |
| **Total** | **256** |


## ✓ Analysis  (9 pass / 0 fail / 0 skip)

| Status | Test | Time (s) |
|--------|------|----------|
| ✓ PASS | `kl_divergence_kde` | 0.04 |
| ✓ PASS | `effective_sample_size` | 0.02 |
| ✓ PASS | `wasserstein_1d` | 0.01 |
| ✓ PASS | `js_divergence_1d` | 0.01 |
| ✓ PASS | `consistency_report` | 0.02 |
| ✓ PASS | `evaluate  (full suite)` | 0.03 |
| ✓ PASS | `model_summary  s2` | 0.02 |
| ✓ PASS | `model_summary  r2` | 0.0 |
| ✓ PASS | `model_summary  r3` | 0.0 |

## ✓ CLIInference  (18 pass / 0 fail / 0 skip)

| Status | Test | Time (s) |
|--------|------|----------|
| ✓ PASS | `sample  s2  with plots` | 15.06 |
| ✓ PASS | `sample  s2  --no_plots` | 3.72 |
| ✓ PASS | `evaluate  s2` | 4.48 |
| ✓ PASS | `plot  s2` | 15.38 |
| ✓ PASS | `sample  r2  with plots` | 14.53 |
| ✓ PASS | `sample  r2  --no_plots` | 3.03 |
| ✓ PASS | `evaluate  r2` | 4.29 |
| ✓ PASS | `plot  r2` | 15.02 |
| ✓ PASS | `sample  r3  with plots` | 16.42 |
| ✓ PASS | `sample  r3  --no_plots` | 2.99 |
| ✓ PASS | `evaluate  r3` | 4.2 |
| ✓ PASS | `plot  r3` | 16.83 |
| ✓ PASS | `sample  s2  self-describing (no arch flags)` | 2.99 |
| ✓ PASS | `sample  r2  self-describing (no arch flags)` | 3.17 |
| ✓ PASS | `sample  r3  self-describing (no arch flags)` | 3.04 |
| ✓ PASS | `sample  r2  ns=2  self-describing` | 3.0 |
| ✓ PASS | `sample  r2  resnet  self-describing` | 2.98 |
| ✓ PASS | `sample  missing checkpoint raises error` | 2.92 |

## ✓ CLITrain  (41 pass / 0 fail / 0 skip)

| Status | Test | Time (s) |
|--------|------|----------|
| ✓ PASS | `train  s2  defaults` | 4.41 |
| ✓ PASS | `train  r2  defaults` | 4.56 |
| ✓ PASS | `train  r3  defaults` | 4.53 |
| ✓ PASS | `train  s2  mlp` | 4.37 |
| ✓ PASS | `train  s2  resnet` | 4.77 |
| ✓ PASS | `train  r2  mlp` | 4.43 |
| ✓ PASS | `train  r2  resnet` | 4.44 |
| ✓ PASS | `train  r3  mlp` | 4.49 |
| ✓ PASS | `train  r3  resnet` | 4.52 |
| ✓ PASS | `train  r2  act=relu` | 4.4 |
| ✓ PASS | `train  r2  act=tanh` | 4.84 |
| ✓ PASS | `train  r2  act=elu` | 5.06 |
| ✓ PASS | `train  r2  act=leaky_relu` | 5.27 |
| ✓ PASS | `train  r2  act=silu` | 4.67 |
| ✓ PASS | `train  r2  act=gelu` | 4.94 |
| ✓ PASS | `train  s2  ns=1` | 5.16 |
| ✓ PASS | `train  s2  ns=2` | 4.78 |
| ✓ PASS | `train  s2  ns=3` | 5.24 |
| ✓ PASS | `train  r2  ns=1` | 4.92 |
| ✓ PASS | `train  r2  ns=2` | 4.86 |
| ✓ PASS | `train  r2  ns=3` | 5.21 |
| ✓ PASS | `train  r3  ns=1` | 5.17 |
| ✓ PASS | `train  r3  ns=2` | 5.39 |
| ✓ PASS | `train  r3  ns=3` | 6.01 |
| ✓ PASS | `train  r2  dropout=0.2` | 8.62 |
| ✓ PASS | `train  r2  no_plateau` | 6.04 |
| ✓ PASS | `train  r2  use_cosine` | 5.0 |
| ✓ PASS | `train  r2  plateau+cosine` | 5.32 |
| ✓ PASS | `train  r2  no_plateau+cosine` | 6.58 |
| ✓ PASS | `train  r2  bound=3.0` | 6.41 |
| ✓ PASS | `train  r2  bound=5.0` | 4.99 |
| ✓ PASS | `train  r2  bound=8.0` | 4.56 |
| ✓ PASS | `train  r2  hidden_dim=16` | 6.03 |
| ✓ PASS | `train  r2  hidden_dim=32` | 6.87 |
| ✓ PASS | `train  r2  hidden_dim=64` | 7.3 |
| ✓ PASS | `train  r2  num_layers=1` | 6.22 |
| ✓ PASS | `train  r2  num_layers=2` | 5.02 |
| ✓ PASS | `train  r2  num_layers=3` | 4.94 |
| ✓ PASS | `train  r2  resnet  act=relu` | 4.63 |
| ✓ PASS | `train  r2  resnet  act=tanh` | 4.81 |
| ✓ PASS | `train  r2  resnet  act=elu` | 4.92 |

## ✓ Devices  (1 pass / 0 fail / 2 skip)

| Status | Test | Time (s) |
|--------|------|----------|
| ✓ PASS | `device=cpu  forward+sample  all models` | 0.05 |
| ○ SKIP | `device=cuda  forward+sample` | 0.0 |
| ○ SKIP | `device=mps  forward+sample` | 0.0 |

## ✓ HelperScripts  (7 pass / 0 fail / 0 skip)

| Status | Test | Time (s) |
|--------|------|----------|
| ✓ PASS | `train_all  s2+r2+r3` | 7.47 |
| ✓ PASS | `train_all  --skip r3` | 5.68 |
| ✓ PASS | `train_all  --skip s2 r2` | 5.08 |
| ✓ PASS | `plot_all  s2+r2+r3` | 45.38 |
| ✓ PASS | `plot_all  --skip r3` | 29.56 |
| ✓ PASS | `analyse_all  s2+r2+r3` | 9.95 |
| ✓ PASS | `analyse_all  --skip s2` | 7.63 |

## ✓ MLPs  (31 pass / 0 fail / 0 skip)

| Status | Test | Time (s) |
|--------|------|----------|
| ✓ PASS | `MLP  activation=relu` | 0.0 |
| ✓ PASS | `MLP  activation=tanh` | 0.0 |
| ✓ PASS | `MLP  activation=elu` | 0.0 |
| ✓ PASS | `MLP  activation=leaky_relu` | 0.0 |
| ✓ PASS | `MLP  activation=silu` | 0.0 |
| ✓ PASS | `MLP  activation=gelu` | 0.0 |
| ✓ PASS | `MLP  num_layers=1` | 0.0 |
| ✓ PASS | `MLP  num_layers=2` | 0.0 |
| ✓ PASS | `MLP  num_layers=3` | 0.0 |
| ✓ PASS | `MLP  num_layers=4` | 0.0 |
| ✓ PASS | `MLP  hidden_dim=16` | 0.0 |
| ✓ PASS | `MLP  hidden_dim=32` | 0.0 |
| ✓ PASS | `MLP  hidden_dim=64` | 0.0 |
| ✓ PASS | `MLP  hidden_dim=128` | 0.0 |
| ✓ PASS | `MLP  dropout=0.0` | 0.0 |
| ✓ PASS | `MLP  dropout=0.1` | 0.0 |
| ✓ PASS | `MLP  dropout=0.3` | 0.0 |
| ✓ PASS | `ResNet  activation=relu` | 0.02 |
| ✓ PASS | `ResNet  activation=tanh` | 0.02 |
| ✓ PASS | `ResNet  activation=elu` | 0.0 |
| ✓ PASS | `ResNet  activation=leaky_relu` | 0.0 |
| ✓ PASS | `ResNet  activation=silu` | 0.0 |
| ✓ PASS | `ResNet  activation=gelu` | 0.0 |
| ✓ PASS | `ResNet  num_layers(blocks)=1` | 0.0 |
| ✓ PASS | `ResNet  num_layers(blocks)=2` | 0.0 |
| ✓ PASS | `ResNet  num_layers(blocks)=3` | 0.0 |
| ✓ PASS | `ResNet  hidden_dim=16` | 0.0 |
| ✓ PASS | `ResNet  hidden_dim=32` | 0.0 |
| ✓ PASS | `ResNet  hidden_dim=64` | 0.0 |
| ✓ PASS | `ResNet  dropout=0.0  layer_norm=True` | 0.0 |
| ✓ PASS | `ResNet  dropout=0.2  layer_norm=True` | 0.0 |

## ✓ ModelAPI  (73 pass / 0 fail / 0 skip)

| Status | Test | Time (s) |
|--------|------|----------|
| ✓ PASS | `forward  s2  mlp  ns=1` | 0.0 |
| ✓ PASS | `round-trip  s2  mlp  ns=1` | 0.0 |
| ✓ PASS | `log_prob  s2  mlp  ns=1` | 0.0 |
| ✓ PASS | `sample  s2  mlp  ns=1` | 0.0 |
| ✓ PASS | `forward  s2  mlp  ns=2` | 0.0 |
| ✓ PASS | `round-trip  s2  mlp  ns=2` | 0.01 |
| ✓ PASS | `log_prob  s2  mlp  ns=2` | 0.0 |
| ✓ PASS | `sample  s2  mlp  ns=2` | 0.0 |
| ✓ PASS | `forward  s2  mlp  ns=3` | 0.01 |
| ✓ PASS | `round-trip  s2  mlp  ns=3` | 0.01 |
| ✓ PASS | `log_prob  s2  mlp  ns=3` | 0.01 |
| ✓ PASS | `sample  s2  mlp  ns=3` | 0.01 |
| ✓ PASS | `forward  s2  resnet  ns=1` | 0.0 |
| ✓ PASS | `round-trip  s2  resnet  ns=1` | 0.0 |
| ✓ PASS | `log_prob  s2  resnet  ns=1` | 0.0 |
| ✓ PASS | `sample  s2  resnet  ns=1` | 0.0 |
| ✓ PASS | `forward  s2  resnet  ns=2` | 0.0 |
| ✓ PASS | `round-trip  s2  resnet  ns=2` | 0.01 |
| ✓ PASS | `log_prob  s2  resnet  ns=2` | 0.0 |
| ✓ PASS | `sample  s2  resnet  ns=2` | 0.0 |
| ✓ PASS | `forward  s2  resnet  ns=3` | 0.01 |
| ✓ PASS | `round-trip  s2  resnet  ns=3` | 0.01 |
| ✓ PASS | `log_prob  s2  resnet  ns=3` | 0.01 |
| ✓ PASS | `sample  s2  resnet  ns=3` | 0.0 |
| ✓ PASS | `forward  r2  mlp  ns=1` | 0.0 |
| ✓ PASS | `round-trip  r2  mlp  ns=1` | 0.0 |
| ✓ PASS | `log_prob  r2  mlp  ns=1` | 0.0 |
| ✓ PASS | `sample  r2  mlp  ns=1` | 0.0 |
| ✓ PASS | `forward  r2  mlp  ns=2` | 0.0 |
| ✓ PASS | `round-trip  r2  mlp  ns=2` | 0.01 |
| ✓ PASS | `log_prob  r2  mlp  ns=2` | 0.0 |
| ✓ PASS | `sample  r2  mlp  ns=2` | 0.0 |
| ✓ PASS | `forward  r2  mlp  ns=3` | 0.0 |
| ✓ PASS | `round-trip  r2  mlp  ns=3` | 0.01 |
| ✓ PASS | `log_prob  r2  mlp  ns=3` | 0.0 |
| ✓ PASS | `sample  r2  mlp  ns=3` | 0.0 |
| ✓ PASS | `forward  r2  resnet  ns=1` | 0.0 |
| ✓ PASS | `round-trip  r2  resnet  ns=1` | 0.0 |
| ✓ PASS | `log_prob  r2  resnet  ns=1` | 0.0 |
| ✓ PASS | `sample  r2  resnet  ns=1` | 0.0 |
| ✓ PASS | `forward  r2  resnet  ns=2` | 0.01 |
| ✓ PASS | `round-trip  r2  resnet  ns=2` | 0.01 |
| ✓ PASS | `log_prob  r2  resnet  ns=2` | 0.0 |
| ✓ PASS | `sample  r2  resnet  ns=2` | 0.0 |
| ✓ PASS | `forward  r2  resnet  ns=3` | 0.01 |
| ✓ PASS | `round-trip  r2  resnet  ns=3` | 0.01 |
| ✓ PASS | `log_prob  r2  resnet  ns=3` | 0.01 |
| ✓ PASS | `sample  r2  resnet  ns=3` | 0.01 |
| ✓ PASS | `forward  r3  mlp  ns=1` | 0.0 |
| ✓ PASS | `round-trip  r3  mlp  ns=1` | 0.0 |
| ✓ PASS | `log_prob  r3  mlp  ns=1` | 0.0 |
| ✓ PASS | `sample  r3  mlp  ns=1` | 0.0 |
| ✓ PASS | `forward  r3  mlp  ns=2` | 0.0 |
| ✓ PASS | `round-trip  r3  mlp  ns=2` | 0.01 |
| ✓ PASS | `log_prob  r3  mlp  ns=2` | 0.0 |
| ✓ PASS | `sample  r3  mlp  ns=2` | 0.0 |
| ✓ PASS | `forward  r3  mlp  ns=3` | 0.0 |
| ✓ PASS | `round-trip  r3  mlp  ns=3` | 0.01 |
| ✓ PASS | `log_prob  r3  mlp  ns=3` | 0.0 |
| ✓ PASS | `sample  r3  mlp  ns=3` | 0.0 |
| ✓ PASS | `forward  r3  resnet  ns=1` | 0.0 |
| ✓ PASS | `round-trip  r3  resnet  ns=1` | 0.0 |
| ✓ PASS | `log_prob  r3  resnet  ns=1` | 0.0 |
| ✓ PASS | `sample  r3  resnet  ns=1` | 0.0 |
| ✓ PASS | `forward  r3  resnet  ns=2` | 0.01 |
| ✓ PASS | `round-trip  r3  resnet  ns=2` | 0.01 |
| ✓ PASS | `log_prob  r3  resnet  ns=2` | 0.0 |
| ✓ PASS | `sample  r3  resnet  ns=2` | 0.0 |
| ✓ PASS | `forward  r3  resnet  ns=3` | 0.01 |
| ✓ PASS | `round-trip  r3  resnet  ns=3` | 0.01 |
| ✓ PASS | `log_prob  r3  resnet  ns=3` | 0.01 |
| ✓ PASS | `sample  r3  resnet  ns=3` | 0.01 |
| ✓ PASS | `build_model unknown key raises ValueError` | 0.0 |

## ✓ Plotting  (15 pass / 0 fail / 0 skip)

| Status | Test | Time (s) |
|--------|------|----------|
| ✓ PASS | `plot_loss_curves` | 0.67 |
| ✓ PASS | `plot_marginal_1d  [angular]` | 2.06 |
| ✓ PASS | `plot_marginal_1d  [cartesian]` | 1.02 |
| ✓ PASS | `plot_marginal_2d  [angular]` | 0.46 |
| ✓ PASS | `plot_marginal_2d  [cartesian]` | 1.21 |
| ✓ PASS | `plot_mollweide_kde  [tensor input]` | 0.83 |
| ✓ PASS | `plot_mollweide_kde  [ndarray input]` | 0.84 |
| ✓ PASS | `plot_physics_comparison` | 1.93 |
| ✓ PASS | `plot_radius_distribution` | 0.6 |
| ✓ PASS | `plot_radius_distribution  [near-zero range  auto bins]` | 0.36 |
| ✓ PASS | `plot_base_mapping_s2  [s2]` | 1.3 |
| ✓ PASS | `plot_jacobian_map     [s2]` | 0.62 |
| ✓ PASS | `plot_base_mapping     [r2]` | 1.71 |
| ✓ PASS | `plot_jacobian_map     [r2]` | 0.6 |
| ✓ PASS | `plot_jacobian_map_r3  [r3]` | 2.6 |

## ✓ PyAPITrain  (26 pass / 0 fail / 0 skip)

| Status | Test | Time (s) |
|--------|------|----------|
| ✓ PASS | `train  s2  mlp` | 0.28 |
| ✓ PASS | `train  s2  resnet` | 0.32 |
| ✓ PASS | `train  r2  mlp` | 0.31 |
| ✓ PASS | `train  r2  resnet` | 0.48 |
| ✓ PASS | `train  r3  mlp` | 0.33 |
| ✓ PASS | `train  r3  resnet` | 0.42 |
| ✓ PASS | `train  r2  mlp  act=relu` | 0.31 |
| ✓ PASS | `train  r2  mlp  act=tanh` | 0.32 |
| ✓ PASS | `train  r2  mlp  act=elu` | 0.31 |
| ✓ PASS | `train  r2  mlp  act=leaky_relu` | 0.33 |
| ✓ PASS | `train  r2  mlp  act=silu` | 0.41 |
| ✓ PASS | `train  r2  mlp  act=gelu` | 0.33 |
| ✓ PASS | `train  s2  ns=1` | 0.28 |
| ✓ PASS | `train  s2  ns=2` | 0.44 |
| ✓ PASS | `train  s2  ns=3` | 0.59 |
| ✓ PASS | `train  r2  ns=1` | 0.31 |
| ✓ PASS | `train  r2  ns=2` | 0.49 |
| ✓ PASS | `train  r2  ns=3` | 0.74 |
| ✓ PASS | `train  r3  ns=1` | 0.33 |
| ✓ PASS | `train  r3  ns=2` | 0.53 |
| ✓ PASS | `train  r3  ns=3` | 0.72 |
| ✓ PASS | `train  r2  mlp  dropout=0.2` | 0.3 |
| ✓ PASS | `train  r2  plateau=True  cosine=False` | 0.3 |
| ✓ PASS | `train  r2  plateau=False  cosine=True` | 0.3 |
| ✓ PASS | `train  r2  plateau=True  cosine=True` | 0.4 |
| ✓ PASS | `train  r2  plateau=False  cosine=False` | 0.31 |

## ✓ Resume  (5 pass / 0 fail / 0 skip)

| Status | Test | Time (s) |
|--------|------|----------|
| ✓ PASS | `resume  s2` | 5.18 |
| ✓ PASS | `resume  r2` | 5.83 |
| ✓ PASS | `resume  r3` | 6.18 |
| ✓ PASS | `resume  r2  mlp  checkpoint` | 5.49 |
| ✓ PASS | `resume  r2  resnet  checkpoint` | 5.15 |

## ✓ Splines  (9 pass / 0 fail / 0 skip)

| Status | Test | Time (s) |
|--------|------|----------|
| ✓ PASS | `rqs forward+inverse round-trip` | 0.0 |
| ✓ PASS | `rqs log-det finite` | 0.0 |
| ✓ PASS | `rqs_with_bounds round-trip  (cos_theta domain)` | 0.0 |
| ✓ PASS | `rqs_with_bounds round-trip  (phi domain)` | 0.0 |
| ✓ PASS | `rqs_circular round-trip  (periodic phi)` | 0.0 |
| ✓ PASS | `rqs_circular LDJ finite (periodicity check)` | 0.0 |
| ✓ PASS | `rqs_with_bounds  num_bins=4` | 0.0 |
| ✓ PASS | `rqs_with_bounds  num_bins=16` | 0.0 |
| ✓ PASS | `rqs_with_bounds  num_bins=32` | 0.0 |

## ✓ Utils  (14 pass / 0 fail / 0 skip)

| Status | Test | Time (s) |
|--------|------|----------|
| ✓ PASS | `set_seed` | 0.0 |
| ✓ PASS | `get_device(auto)` | 0.0 |
| ✓ PASS | `get_device(cpu)` | 0.0 |
| ✓ PASS | `load_json_data` | 0.01 |
| ✓ PASS | `cartesian_to_spherical` | 0.01 |
| ✓ PASS | `spherical_to_cartesian` | 0.01 |
| ✓ PASS | `normalise` | 0.08 |
| ✓ PASS | `denormalise (round-trip)` | 0.01 |
| ✓ PASS | `make_dataloaders` | 0.01 |
| ✓ PASS | `cartesian_to_physics` | 0.01 |
| ✓ PASS | `save/load_checkpoint` | 1.19 |
| ✓ PASS | `save/load_losses` | 0.0 |
| ✓ PASS | `count_parameters` | 0.0 |
| ✓ PASS | `log_std_correction` | 0.0 |

## ✓ YAML  (5 pass / 0 fail / 0 skip)

| Status | Test | Time (s) |
|--------|------|----------|
| ✓ PASS | `train via YAML config` | 5.14 |
| ✓ PASS | `YAML + CLI override model` | 4.68 |
| ✓ PASS | `train via shipped config s2_default.yaml` | 4.63 |
| ✓ PASS | `train via shipped config r2_default.yaml` | 5.1 |
| ✓ PASS | `train via shipped config r3_default.yaml` | 5.38 |
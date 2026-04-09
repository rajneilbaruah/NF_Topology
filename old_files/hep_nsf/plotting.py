"""
plotting.py
===========
All visualisation routines.

Key design notes
----------------
- s2 works in physical (cos_theta, phi): model.sample() returns physical.
  No denormalisation needed. mean/std are only for NLL correction display.
- r2 works in normalised space: model.sample() returns normalised.
  Denormalise before display.
- r3 same as r2 but 3D.
- plot_jacobian_map: for BOTH s2 and r2, model.sample() output can be
  passed directly to model.forward() for LDJ — no mean/std adjustment.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import gaussian_kde

matplotlib.use("Agg")

TRUE_COLOR  = "#00234e"
MODEL_COLOR = "#A61200"
DPI         = 400


def _setup_style():
    rcParams["font.family"]      = "serif"
    rcParams["mathtext.fontset"] = "dejavuserif"
    rcParams["axes.grid"]        = True
    rcParams["grid.linestyle"]   = "--"
    rcParams["grid.alpha"]       = 0.5

_setup_style()


def _savefig(fig, save, default_name, output_dir="outputs"):
    if save is False:
        return
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = save if isinstance(save, str) else default_name
    fig.savefig(Path(output_dir) / fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(train_losses, val_losses,
                     title="Training Curves",
                     save=True, output_dir="outputs"):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs  = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color=TRUE_COLOR,  lw=2, label="Train")
    ax.plot(epochs, val_losses,   color=MODEL_COLOR, lw=2, label="Validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("NLL Loss"); ax.set_title(title)
    ax.legend(); fig.tight_layout()
    _savefig(fig, save, "loss_curves.png", output_dir)
    return fig


# ---------------------------------------------------------------------------
# 1-D marginals
# ---------------------------------------------------------------------------

def plot_marginal_1d(data, samples, labels=None,
                     title="1-D Marginals", density=True,
                     n_bins=50, save=True, output_dir="outputs"):
    D = data.shape[1]
    if labels is None:
        labels = [f"$x_{{{i}}}$" for i in range(D)]

    ncols = min(D, 3)
    nrows = (D + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    axes = np.array(axes).flatten()

    for i in range(D):
        ax  = axes[i]
        d   = data[:, i][np.isfinite(data[:, i])]
        s   = samples[:, i][np.isfinite(samples[:, i])]
        lo  = np.percentile(np.concatenate([d, s]), 0.1)
        hi  = np.percentile(np.concatenate([d, s]), 99.9)
        if lo == hi: lo -= 1; hi += 1
        bins = np.linspace(lo, hi, n_bins + 1)
        ax.hist(d, bins=bins, density=density, histtype="step",
                lw=2.5, color=TRUE_COLOR,  label="Data")
        ax.hist(s, bins=bins, density=density, histtype="step",
                lw=2.5, color=MODEL_COLOR, label="Flow")
        ax.set_xlabel(labels[i]); ax.legend(fontsize=9)

    for j in range(D, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title, fontsize=14); fig.tight_layout()
    _savefig(fig, save, "marginals_1d.png", output_dir)
    return fig


# ---------------------------------------------------------------------------
# 2-D scatter pairs
# ---------------------------------------------------------------------------

def plot_marginal_2d(data, samples, labels=None,
                     title="2-D Marginals", max_points=5000,
                     save=True, output_dir="outputs"):
    D = data.shape[1]
    if labels is None:
        labels = [f"$x_{{{i}}}$" for i in range(D)]

    pairs = [(i, j) for i in range(D) for j in range(i+1, D)]
    ncols = min(len(pairs), 3)
    nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    axes = np.array(axes).flatten()

    rng   = np.random.default_rng(42)
    idx_d = rng.choice(len(data),    min(max_points, len(data)),    replace=False)
    idx_s = rng.choice(len(samples), min(max_points, len(samples)), replace=False)

    for k, (i, j) in enumerate(pairs):
        ax = axes[k]
        ax.scatter(data[idx_d, i],    data[idx_d, j],
                   s=4, alpha=0.4, color=TRUE_COLOR,  label="Data", rasterized=True)
        ax.scatter(samples[idx_s, i], samples[idx_s, j],
                   s=4, alpha=0.4, color=MODEL_COLOR, label="Flow", rasterized=True)
        ax.set_xlabel(labels[i]); ax.set_ylabel(labels[j])
        ax.legend(markerscale=2, fontsize=9)

    for j in range(len(pairs), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title, fontsize=14); fig.tight_layout()
    _savefig(fig, save, "marginals_2d.png", output_dir)
    return fig


# ---------------------------------------------------------------------------
# Mollweide KDE
# ---------------------------------------------------------------------------

def plot_mollweide_kde(cyl_data, title="KDE Density",
                       bw=0.2, grid_lon=150, grid_lat=75,
                       save=True, output_dir="outputs"):
    if isinstance(cyl_data, tuple):
        cyl_data = cyl_data[0]
    if isinstance(cyl_data, torch.Tensor):
        cyl_data = cyl_data.detach().cpu()

    lat = np.arcsin(np.clip(np.array(cyl_data[:, 0]), -1.0, 1.0))
    lon = np.array(cyl_data[:, 1]) - np.pi

    lon_g, lat_g = np.meshgrid(
        np.linspace(-np.pi,   np.pi,   grid_lon),
        np.linspace(-np.pi/2, np.pi/2, grid_lat))
    kde = gaussian_kde(np.vstack([lon, lat]), bw_method=bw)
    zi  = kde(np.vstack([lon_g.ravel(), lat_g.ravel()])).reshape(lon_g.shape)

    fig = plt.figure(figsize=(10, 5))
    ax  = fig.add_subplot(111, projection="mollweide")
    im  = ax.pcolormesh(lon_g, lat_g, zi, cmap="viridis", shading="auto")
    ax.set_title(title, fontsize=13)
    plt.colorbar(im, orientation="horizontal", pad=0.1, label="Density")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    _savefig(fig, save, f"{title.replace(' ','_')}.png", output_dir)
    return fig


# ---------------------------------------------------------------------------
# Physics comparison
# ---------------------------------------------------------------------------

def plot_physics_comparison(data_cart, sample_cart,
                             title="Physics Comparison",
                             n_bins=50, density=True,
                             save=True, output_dir="outputs"):
    from .utils import cartesian_to_physics

    def _get(p):
        d = cartesian_to_physics(p)
        return [d["px"].numpy(), d["py"].numpy(), d["pz"].numpy(),
                d["pT"].numpy(), d["eta"].numpy(), d["phi"].numpy()]

    d_v = _get(data_cart);  s_v = _get(sample_cart)
    lbls = [r"$p_x$ [GeV]", r"$p_y$ [GeV]", r"$p_z$ [GeV]",
            r"$p_T$ [GeV]", r"$\eta$",       r"$\phi$ [rad]"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for i, ax in enumerate(axes.flatten()):
        d  = d_v[i][np.isfinite(d_v[i])]
        s  = s_v[i][np.isfinite(s_v[i])]
        lo = np.percentile(d, 0.1); hi = np.percentile(d, 99.9)
        if lo == hi: lo -= 1; hi += 1
        bins = np.linspace(lo, hi, n_bins + 1)
        ax.hist(d, bins=bins, density=density, histtype="step",
                lw=2.5, color=TRUE_COLOR,  label="Data")
        ax.hist(s, bins=bins, density=density, histtype="step",
                lw=2.5, color=MODEL_COLOR, label="Flow")
        ax.set_xlabel(lbls[i], fontsize=12); ax.legend(fontsize=9)

    fig.suptitle(title, fontsize=14); fig.tight_layout()
    _savefig(fig, save, "physics_comparison.png", output_dir)
    return fig


# ---------------------------------------------------------------------------
# Base mapping — r2 (Gaussian polar grid)
# ---------------------------------------------------------------------------

def plot_base_mapping(model, device, mean, std,
                      dim=2, n_r=50, n_phi=100, r_max=4.0,
                      title="Base Mapping (Gaussian Grid)",
                      save=True, output_dir="outputs"):
    """Gaussian polar grid mapped through r2.

    r2 works in normalised space. The base is Gaussian.
    We create a polar grid in base space and pass it through
    the inverse flow, then denormalise to physical (cos_theta, phi).
    """
    if dim != 2:
        raise ValueError("plot_base_mapping supports 2-D models only.")

    r_vals   = np.linspace(0.01, r_max, n_r)
    phi_vals = np.linspace(0, 2*np.pi, n_phi)
    R, PHI   = np.meshgrid(r_vals, phi_vals)
    z1 = (R * np.cos(PHI)).ravel()
    z2 = (R * np.sin(PHI)).ravel()
    z_t = torch.tensor(np.stack([z1, z2], axis=1),
                        dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        x_norm = model.forward(z_t, inverse=True)
        x_phys = x_norm * std.to(device) + mean.to(device)
        x_np   = x_phys.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc1 = axes[0].scatter(x_np[:, 0], x_np[:, 1],
                          c=R.ravel(), cmap="viridis", s=4, rasterized=True)
    axes[0].set_title(r"Coloured by base radius $r$")
    axes[0].set_xlabel(r"$\cos\theta$"); axes[0].set_ylabel(r"$\phi$")
    plt.colorbar(sc1, ax=axes[0], label="Gaussian sigmas")

    sc2 = axes[1].scatter(x_np[:, 0], x_np[:, 1],
                          c=PHI.ravel(), cmap="hsv", s=4, rasterized=True)
    axes[1].set_title(r"Coloured by base angle $\phi_{\rm base}$")
    axes[1].set_xlabel(r"$\cos\theta$")
    plt.colorbar(sc2, ax=axes[1], label="Base angle [rad]")

    fig.suptitle(title, fontsize=13); fig.tight_layout()
    _savefig(fig, save, "base_mapping.png", output_dir)
    return fig


# ---------------------------------------------------------------------------
# Base mapping — s2 (Uniform lat-lon grid)
# ---------------------------------------------------------------------------

def plot_base_mapping_s2(model, device, mean, std,
                          n_lat=30, n_lon=60,
                          title="Base Mapping (Uniform S2 Grid)",
                          save=True, output_dir="outputs"):
    """Uniform lat-lon grid on S2 mapped through s2.

    s2 works in physical (cos_theta, phi). Base is uniform on S2.
    We create a regular grid in (cos_theta, phi) base space and
    pass it through the inverse flow.

    Left panel  : coloured by cos_theta_base (latitude)
    Right panel : coloured by phi_base (longitude)

    Grid lines in base space → curved iso-density contours in data space,
    showing where the flow stretches/compresses probability mass.
    """
    TWO_PI = 2.0 * np.pi

    ct_vals  = np.linspace(-0.95, 0.95, n_lat)
    phi_vals = np.linspace(0.1, TWO_PI - 0.1, n_lon)
    CT, PHI  = np.meshgrid(ct_vals, phi_vals)

    z_t = torch.tensor(
        np.stack([CT.ravel(), PHI.ravel()], axis=1),
        dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        # s2 inverse: base physical (cos_theta, phi) -> data physical
        x_np = model.forward(z_t, inverse=True).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc1 = axes[0].scatter(x_np[:, 0], x_np[:, 1],
                          c=CT.ravel(), cmap="coolwarm", s=4,
                          vmin=-1, vmax=1, rasterized=True)
    axes[0].set_title(r"Coloured by base $\cos\theta$ (latitude)")
    axes[0].set_xlabel(r"$\cos\theta$ (data)")
    axes[0].set_ylabel(r"$\phi$ (data) [rad]")
    axes[0].set_xlim(-1.1, 1.1)
    axes[0].set_ylim(-0.2, TWO_PI + 0.2)
    plt.colorbar(sc1, ax=axes[0], label=r"$\cos\theta_{\rm base}$")

    sc2 = axes[1].scatter(x_np[:, 0], x_np[:, 1],
                          c=PHI.ravel(), cmap="hsv", s=4,
                          vmin=0, vmax=TWO_PI, rasterized=True)
    axes[1].set_title(r"Coloured by base $\phi$ (longitude)")
    axes[1].set_xlabel(r"$\cos\theta$ (data)")
    axes[1].set_xlim(-1.1, 1.1)
    axes[1].set_ylim(-0.2, TWO_PI + 0.2)
    plt.colorbar(sc2, ax=axes[1], label=r"$\phi_{\rm base}$ [rad]")

    fig.suptitle(title, fontsize=13); fig.tight_layout()
    _savefig(fig, save, "s2_base_mapping.png", output_dir)
    return fig


# ---------------------------------------------------------------------------
# Jacobian map — s2 and r2
# ---------------------------------------------------------------------------

def plot_jacobian_map(model, device, mean, std,
                      n_samples=5000,
                      title="Log-Det-Jacobian Map",
                      save=True, output_dir="outputs"):
    """Scatter of samples coloured by log |det J|.

    For s2: model.sample() returns physical (cos_theta, phi).
            Pass directly to model.forward() for LDJ.
            Display as-is (already physical).

    For r2: model.sample() returns normalised coords.
            Pass directly to model.forward() for LDJ.
            Denormalise for display only.

    In both cases x_in = z (the sample output) — no mean/std adjustment.
    """
    model.eval()
    with torch.no_grad():
        z = model.sample(n_samples, device=device)

        # x_in for forward pass = z directly in both cases
        # s2: z is physical, forward() expects physical
        # r2: z is normalised, forward() expects normalised
        _, ldj = model.forward(z, inverse=False)

        # For display: physical coordinates
        if hasattr(model, "bound"):  # r2 — normalised, denormalise for display
            x_disp = (z * std.to(device) + mean.to(device)).cpu().numpy()
        else:                        # s2 — already physical
            x_disp = z.cpu().numpy()

    ldj_np = ldj.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(x_disp[:, 0], x_disp[:, 1],
                    c=ldj_np, cmap="magma", s=4, alpha=0.7, rasterized=True)
    plt.colorbar(sc, ax=ax, label=r"$\ln|\det J|$")
    ax.set_xlabel(r"$\cos\theta$"); ax.set_ylabel(r"$\phi$ [rad]")
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    _savefig(fig, save, "jacobian_map.png", output_dir)
    return fig


# ---------------------------------------------------------------------------
# Jacobian map — r3
# ---------------------------------------------------------------------------

def plot_jacobian_map_r3(model, device, mean, std,
                          n_samples=5000,
                          title="R3 Jacobian Map",
                          save=True, output_dir="outputs"):
    """Three diagnostics for r3:
      A — three 2-D Cartesian projections coloured by LDJ
      B — Mollweide coloured by LDJ
      C — 1-D histogram of LDJ values
    """
    from .utils import cartesian_to_spherical, denormalise

    model.eval()
    with torch.no_grad():
        z_norm = model.sample(n_samples, device=device)
        x_phys = denormalise(z_norm, mean, std)
        _, ldj = model.forward(z_norm, inverse=False)

    x_np   = x_phys.cpu().numpy()
    ldj_np = ldj.cpu().numpy()

    # ── A: Cartesian projections ──────────────────────────────────────────── #
    labels = [r"$p_x$ [GeV]", r"$p_y$ [GeV]", r"$p_z$ [GeV]"]
    pairs  = [(0,1),(0,2),(1,2)]

    fig_a, axes_a = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (i, j) in zip(axes_a, pairs):
        sc = ax.scatter(x_np[:,i], x_np[:,j],
                        c=ldj_np, cmap="magma", s=3, alpha=0.7, rasterized=True)
        ax.set_xlabel(labels[i]); ax.set_ylabel(labels[j])
        ax.set_title(f"{labels[i]} vs {labels[j]}", fontsize=11)
    plt.colorbar(sc, ax=axes_a[-1], label=r"$\ln|\det J|$")
    fig_a.suptitle(f"{title} — Cartesian Projections", fontsize=13)
    fig_a.tight_layout()
    suffix_cart = save.replace(".png","_cartesian.png") if isinstance(save,str) \
                  else "r3_jacobian_cartesian.png"
    _savefig(fig_a, suffix_cart, "r3_jacobian_cartesian.png", output_dir)

    # ── B: Mollweide coloured by LDJ ──────────────────────────────────────── #
    cyl = cartesian_to_spherical(x_phys.cpu()).numpy()
    lat = np.arcsin(np.clip(cyl[:, 0], -1.0, 1.0))
    lon = cyl[:, 1] - np.pi

    fig_b = plt.figure(figsize=(10, 5))
    ax_b  = fig_b.add_subplot(111, projection="mollweide")
    sc_b  = ax_b.scatter(lon, lat, c=ldj_np, cmap="magma",
                         s=3, alpha=0.7, rasterized=True)
    ax_b.set_title(f"{title} — Mollweide", fontsize=13)
    plt.colorbar(sc_b, orientation="horizontal", pad=0.1,
                 label=r"$\ln|\det J|$")
    ax_b.grid(True, linestyle=":", alpha=0.6)
    fig_b.tight_layout()
    suffix_moll = save.replace(".png","_mollweide.png") if isinstance(save,str) \
                  else "r3_jacobian_mollweide.png"
    _savefig(fig_b, suffix_moll, "r3_jacobian_mollweide.png", output_dir)

    # ── C: 1-D histogram of LDJ ───────────────────────────────────────────── #
    fig_c, ax_c = plt.subplots(figsize=(7, 5))
    ax_c.hist(ldj_np, bins=50, color=MODEL_COLOR, alpha=0.8,
              histtype="stepfilled", edgecolor=MODEL_COLOR, lw=1.5)
    ax_c.axvline(np.mean(ldj_np), color="black", lw=2, linestyle="--",
                 label=f"mean = {np.mean(ldj_np):.2f}")
    ax_c.set_xlabel(r"$\ln|\det J|$", fontsize=12)
    ax_c.set_ylabel("Counts", fontsize=12)
    ax_c.set_title(f"{title} — Distribution", fontsize=13)
    ax_c.legend(fontsize=10); fig_c.tight_layout()
    suffix_hist = save.replace(".png","_hist.png") if isinstance(save,str) \
                  else "r3_jacobian_hist.png"
    _savefig(fig_c, suffix_hist, "r3_jacobian_hist.png", output_dir)

    return fig_a, fig_b, fig_c


# ---------------------------------------------------------------------------
# Radius distribution — r3 only
# ---------------------------------------------------------------------------

def plot_radius_distribution(samples_cart,
                              title="Generated |p| Distribution",
                              n_bins=50, save=True, output_dir="outputs"):
    """Histogram of |p| magnitude. Only meaningful for r3 (learned magnitude).

    Angular models (s2, r2) fix r by back-conversion so this is not called
    for them — the plot would just be a spike at target_r.
    """
    if isinstance(samples_cart, torch.Tensor):
        samples_cart = samples_cart.detach().cpu().numpy()
    radii = np.linalg.norm(samples_cart, axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    r_min, r_max = radii.min(), radii.max()
    if r_max - r_min < 1e-6:
        # All values identical (e.g. fixed-radius back-conversion)
        ax.axvline(r_min, color=MODEL_COLOR, lw=2,
                   label=f"all = {r_min:.2f} GeV")
        ax.set_xlim(r_min - 1, r_min + 1)
        ax.legend(fontsize=10)
    else:
        bins = n_bins if (r_max - r_min) > 1e-6 * n_bins else max(int(n_bins/5), 2)
        ax.hist(radii, bins=bins, color=MODEL_COLOR, alpha=0.75,
                histtype="stepfilled", edgecolor=MODEL_COLOR, lw=1.5)
    ax.set_xlabel(r"$|p|$ [GeV]", fontsize=12)
    ax.set_ylabel("Counts", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.text(0.97, 0.95,
            f"mean = {np.mean(radii):.1f} GeV\nstd = {np.std(radii):.1f} GeV",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    fig.tight_layout()
    _savefig(fig, save, "radius_distribution.png", output_dir)
    return fig

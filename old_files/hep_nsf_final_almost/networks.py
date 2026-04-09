"""
networks.py
===========
Flow model classes.

Naming convention
-----------------
s2  --  RecursiveSphereFlow
        Works in PHYSICAL (cos_theta, phi) coordinates.
        Base distribution: UNIFORM ON S2 (cos_theta ~ U[-1,1], phi ~ U[0,2pi]).
        cos_theta: free-parameter Cartesian RQS (no MLP).
        phi:       MLP-conditioned CIRCULAR RQS (d[0] == d[K]).
        No standardisation applied.

r2  --  AngularSphereFlow
        Works in STANDARDISED (cos_theta, phi) coordinates.
        Base distribution: STANDARD GAUSSIAN in R2.
        Both dimensions: MLP-conditioned standard RQS.

r3  --  CartesianNSF
        Works in STANDARDISED (px, py, pz) coordinates.
        Base distribution: STANDARD GAUSSIAN in R3.
        All dimensions: MLP-conditioned standard RQS.

All models support:
    forward(x, inverse=False)  ->  (z, log_det)  or  x_reconstructed
    log_prob(x)                ->  Tensor (B,)
    sample(n, device)          ->  Tensor (n, dim)

All models accept num_splines=N to stack N coupling blocks.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .splines import rqs, rqs_with_bounds, rqs_circular
from .mlps import build_conditioner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unpack(net_out, out_dims, num_bins):
    """Reshape flat conditioner output into (w, h, d) for rqs()."""
    p = net_out.view(-1, out_dims, 3 * num_bins + 1)
    return p[..., :num_bins], p[..., num_bins:2*num_bins], p[..., 2*num_bins:]


# ---------------------------------------------------------------------------
# S2  --  RecursiveSphereFlow  (physical coords, uniform S2 base)
# ---------------------------------------------------------------------------

class RecursiveSphereFlow(nn.Module):
    """Normalising flow on S2 with UNIFORM base distribution.

    Works in PHYSICAL coordinates:
        cos_theta in (-1, 1)
        phi       in (0, 2*pi)

    Base distribution:
        cos_theta ~ Uniform[-1, 1]
        phi       ~ Uniform[0, 2*pi]
        => log p(z) = -log(4*pi)  (constant, flat over the sphere)

    Architecture per block
    ----------------------
    cos_theta : FREE-PARAMETER Cartesian RQS. No MLP.
    phi       : MLP-conditioned CIRCULAR RQS. d[0] == d[K].

    Parameters
    ----------
    num_bins : int
    num_splines : int
    hidden_dim : int
    num_layers : int
    arch : str
    activation : str
    dropout : float
    """

    TWO_PI = 2.0 * np.pi
    LOG_BASE = -np.log(4.0 * np.pi)   # log(1 / 4pi) -- uniform on S2

    def __init__(self,
                 num_bins: int = 32,
                 num_splines: int = 1,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 arch: str = "mlp",
                 activation: str = "relu",
                 dropout: float = 0.0):
        super().__init__()
        self.num_bins    = num_bins
        self.num_splines = num_splines

        # Free parameters for cos_theta: (3K+1,) per block
        self.z_params = nn.ParameterList([
            nn.Parameter(torch.randn(3 * num_bins + 1))
            for _ in range(num_splines)
        ])

        # MLP conditioners for circular phi: output 3K (not 3K+1)
        self.phi_nets = nn.ModuleList([
            build_conditioner(arch, in_dim=1, out_dim=3 * num_bins,
                              hidden_dim=hidden_dim, num_layers=num_layers,
                              activation=activation, dropout=dropout)
            for _ in range(num_splines)
        ])

    def _z_whd(self, k, B):
        p = self.z_params[k]
        K = self.num_bins
        w = p[:K].unsqueeze(0).expand(B, -1)
        h = p[K:2*K].unsqueeze(0).expand(B, -1)
        d = p[2*K:].unsqueeze(0).expand(B, -1)
        return w, h, d

    def _phi_whd(self, k, z):
        out = self.phi_nets[k](z)
        K   = self.num_bins
        return out[:, :K], out[:, K:2*K], out[:, 2*K:]

    def forward(self, x, inverse=False):
        B = x.shape[0]
        log_det = torch.zeros(B, device=x.device, dtype=x.dtype)

        if not inverse:
            cos_theta = x[:, 0]
            phi       = x[:, 1]

            for k in range(self.num_splines):
                # Save DATA-SPACE cos_theta BEFORE transforming — phi_net
                # must always be conditioned on data-space cos_theta, matching
                # the original notebook's architecture.
                cos_theta_cond = cos_theta

                # Step A: transform cos_theta (data → base)
                w, h, d = self._z_whd(k, B)
                cos_theta, ldj = rqs_with_bounds(
                    cos_theta, w, h, d,
                    inverse=False, b_x=(-1, 1), b_y=(-1, 1))
                log_det += ldj

                # Step B: transform phi conditioned on PRE-TRANSFORM (data-space) cos_theta
                w, h, d = self._phi_whd(k, cos_theta_cond.unsqueeze(1))
                phi, ldj = rqs_circular(
                    phi, w, h, d,
                    inverse=False,
                    b_x=(0, self.TWO_PI), b_y=(0, self.TWO_PI))
                log_det += ldj

            return torch.stack([cos_theta, phi], dim=1), log_det

        else:
            cos_theta = x[:, 0]
            phi       = x[:, 1]

            for k in reversed(range(self.num_splines)):
                # Inverse of forward block k:
                # Forward was: cos_theta_cond = cos_theta_k (data-space)
                #              cos_theta_{k+1} = rqs_fwd(cos_theta_k)
                #              phi_{k+1}       = rqs_circ_fwd(phi_k | phi_net(cos_theta_k))
                #
                # Inverse: first recover data-space cos_theta_k via rqs_inv,
                #          then invert phi conditioned on that same cos_theta_k.
                w, h, d = self._z_whd(k, B)
                cos_theta_data, _ = rqs_with_bounds(
                    cos_theta, w, h, d,
                    inverse=True, b_x=(-1, 1), b_y=(-1, 1))

                # Condition phi on data-space cos_theta (consistent with forward)
                w, h, d = self._phi_whd(k, cos_theta_data.unsqueeze(1))
                phi, _ = rqs_circular(
                    phi, w, h, d,
                    inverse=True,
                    b_x=(0, self.TWO_PI), b_y=(0, self.TWO_PI))

                cos_theta = cos_theta_data

            return torch.stack([cos_theta, phi], dim=1)

    def log_prob(self, x):
        z, log_det = self.forward(x, inverse=False)
        # Base is uniform on S2: log p(z) = -log(4*pi) for all z
        log_pz = torch.full((x.shape[0],), self.LOG_BASE,
                            device=x.device, dtype=x.dtype)
        return log_pz + log_det

    @torch.no_grad()
    def sample(self, n, device=None):
        device = device or next(self.parameters()).device
        # Sample from uniform S2 base
        cos_theta = torch.rand(n, device=device) * 2.0 - 1.0
        phi       = torch.rand(n, device=device) * self.TWO_PI
        z = torch.stack([cos_theta, phi], dim=1)
        return self.forward(z, inverse=True)


# ---------------------------------------------------------------------------
# R2  --  AngularSphereFlow  (standardised coords, Gaussian R2 base)
# ---------------------------------------------------------------------------

class AngularSphereFlow(nn.Module):
    """Normalising flow in R2 for standardised (cos_theta, phi).

    Works in STANDARDISED angular space.
    Base distribution: standard Gaussian in R2.
    Both dimensions use MLP-conditioned standard RQS.

    Parameters
    ----------
    num_bins : int
    bound : float -- spline half-domain in standardised space.
    num_splines : int
    hidden_dim, num_layers, arch, activation, dropout : conditioner settings.
    """

    def __init__(self,
                 num_bins: int = 32,
                 bound: float = 5.0,
                 num_splines: int = 1,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 arch: str = "mlp",
                 activation: str = "relu",
                 dropout: float = 0.0):
        super().__init__()
        self.num_bins    = num_bins
        self.bound       = bound
        self.num_splines = num_splines
        out = 3 * num_bins + 1

        self.nets_phi = nn.ModuleList([
            build_conditioner(arch, 1, out, hidden_dim, num_layers,
                              activation, dropout)
            for _ in range(num_splines)])
        self.nets_costheta = nn.ModuleList([
            build_conditioner(arch, 1, out, hidden_dim, num_layers,
                              activation, dropout)
            for _ in range(num_splines)])

    def forward(self, x, inverse=False):
        B = x.shape[0]
        log_det = torch.zeros(B, device=x.device, dtype=x.dtype)

        if not inverse:
            ct, phi = x[:, 0:1], x[:, 1:2]
            for k in range(self.num_splines):
                w, h, d = _unpack(self.nets_phi[k](ct), 1, self.num_bins)
                phi, ldj = rqs(phi, w, h, d, inverse=False, bound=self.bound)
                log_det += ldj.squeeze(-1)
                w, h, d = _unpack(self.nets_costheta[k](phi), 1, self.num_bins)
                ct, ldj = rqs(ct, w, h, d, inverse=False, bound=self.bound)
                log_det += ldj.squeeze(-1)
            return torch.cat([ct, phi], dim=1), log_det
        else:
            ct, phi = x[:, 0:1], x[:, 1:2]
            for k in reversed(range(self.num_splines)):
                w, h, d = _unpack(self.nets_costheta[k](phi), 1, self.num_bins)
                ct, _ = rqs(ct, w, h, d, inverse=True, bound=self.bound)
                w, h, d = _unpack(self.nets_phi[k](ct), 1, self.num_bins)
                phi, _ = rqs(phi, w, h, d, inverse=True, bound=self.bound)
            return torch.cat([ct, phi], dim=1)

    def log_prob(self, x):
        z, log_det = self.forward(x, inverse=False)
        log_pz = -0.5 * (z ** 2 + np.log(2.0 * np.pi)).sum(dim=1)
        return log_pz + log_det

    @torch.no_grad()
    def sample(self, n, device=None):
        device = device or next(self.parameters()).device
        z = torch.randn(n, 2, device=device)
        return self.forward(z, inverse=True)


# ---------------------------------------------------------------------------
# R3  --  CartesianNSF  (standardised Cartesian, Gaussian R3 base)
# ---------------------------------------------------------------------------

class CartesianNSF(nn.Module):
    """Normalising flow in R3 for standardised (px, py, pz).

    Base distribution: standard Gaussian in R3.

    Parameters
    ----------
    num_bins : int
    bound : float
    num_splines : int
    hidden_dim, num_layers, arch, activation, dropout : conditioner settings.
    """

    def __init__(self,
                 num_bins: int = 32,
                 bound: float = 5.0,
                 num_splines: int = 1,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 arch: str = "mlp",
                 activation: str = "relu",
                 dropout: float = 0.0):
        super().__init__()
        self.num_bins    = num_bins
        self.bound       = bound
        self.num_splines = num_splines

        out2 = 2 * (3 * num_bins + 1)
        out1 = 1 * (3 * num_bins + 1)

        self.nets1 = nn.ModuleList([
            build_conditioner(arch, 1, out2, hidden_dim, num_layers,
                              activation, dropout)
            for _ in range(num_splines)])
        self.nets2 = nn.ModuleList([
            build_conditioner(arch, 2, out1, hidden_dim, num_layers,
                              activation, dropout)
            for _ in range(num_splines)])

    def forward(self, x, inverse=False):
        B = x.shape[0]
        log_det = torch.zeros(B, device=x.device, dtype=x.dtype)

        if not inverse:
            px, pyz = x[:, 0:1], x[:, 1:3]
            for k in range(self.num_splines):
                w, h, d = _unpack(self.nets1[k](px), 2, self.num_bins)
                pyz, ldj = rqs(pyz, w, h, d, inverse=False, bound=self.bound)
                log_det += ldj.sum(dim=-1)
                w, h, d = _unpack(self.nets2[k](pyz), 1, self.num_bins)
                px, ldj = rqs(px, w, h, d, inverse=False, bound=self.bound)
                log_det += ldj.sum(dim=-1)
            return torch.cat([px, pyz], dim=1), log_det
        else:
            px, pyz = x[:, 0:1], x[:, 1:3]
            for k in reversed(range(self.num_splines)):
                w, h, d = _unpack(self.nets2[k](pyz), 1, self.num_bins)
                px, _ = rqs(px, w, h, d, inverse=True, bound=self.bound)
                w, h, d = _unpack(self.nets1[k](px), 2, self.num_bins)
                pyz, _ = rqs(pyz, w, h, d, inverse=True, bound=self.bound)
            return torch.cat([px, pyz], dim=1)

    def log_prob(self, x):
        z, log_det = self.forward(x, inverse=False)
        log_pz = -0.5 * (z ** 2 + np.log(2.0 * np.pi)).sum(dim=1)
        return log_pz + log_det

    @torch.no_grad()
    def sample(self, n, device=None):
        device = device or next(self.parameters()).device
        z = torch.randn(n, 3, device=device)
        return self.forward(z, inverse=True)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "s2": RecursiveSphereFlow,   # physical (cos_theta, phi), uniform S2 base
    "r2": AngularSphereFlow,     # standardised (cos_theta, phi), Gaussian R2 base
    "r3": CartesianNSF,          # standardised (px, py, pz), Gaussian R3 base
}


def build_model(model_type: str, **kwargs) -> nn.Module:
    """Instantiate a flow model by registry key.

    Parameters
    ----------
    model_type : str  -- 's2', 'r2', or 'r3'.
    **kwargs          -- passed to the model constructor.

    Examples
    --------
    >>> model = build_model('s2', num_bins=32, num_splines=2)
    >>> model = build_model('r2', num_bins=32, num_splines=2, bound=5.0)
    >>> model = build_model('r3', num_bins=64, num_splines=3, hidden_dim=128)
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_type}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type](**kwargs)
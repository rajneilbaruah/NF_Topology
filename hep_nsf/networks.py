"""
networks.py
===========
Flow model classes.

Three models, each corresponding to one of your original notebooks:

RecursiveSphereFlow  --  registry key ``r2``
    The original Circular_Splines notebook model.
    Works directly in PHYSICAL coordinates -- no standardisation needed.
    - cos_theta in (-1, 1)  : free-parameter Cartesian RQS (no MLP)
    - phi in (0, 2*pi)      : MLP-conditioned CIRCULAR RQS (d[0]==d[K])
    This is the physically most correct angular model because phi is periodic.

AngularSphereFlow    --  registry key ``s2``
    The Cartesian_RQS_3MomS2 notebook model.
    Works in STANDARDISED (cos_theta, phi) space.
    Both dimensions use MLP-conditioned standard RQS.

CartesianNSF         --  registry key ``r3``
    The Cartesian_RQS_3MomR3 notebook model.
    Works in STANDARDISED (px, py, pz) space.
    All dimensions use MLP-conditioned standard RQS.

All models support:
    forward(x, inverse=False)  ->  (z, log_det)  or  x_reconstructed
    log_prob(x)                ->  Tensor (B,)
    sample(n, device)          ->  Tensor (n, dim)

All models accept num_splines=N to stack N coupling blocks.
num_splines=1 reproduces the original single-block notebook behaviour.
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


def _unpack_circular(net_out, out_dims, num_bins):
    """Reshape for rqs_circular(): d has K values not K+1."""
    # Total params per dim for circular: K + K + K = 3K  (no +1 on d)
    p = net_out.view(-1, out_dims, 3 * num_bins)
    return p[..., :num_bins], p[..., num_bins:2*num_bins], p[..., 2*num_bins:]


# ---------------------------------------------------------------------------
# R2  --  RecursiveSphereFlow  (circular phi + free-param cos_theta)
# ---------------------------------------------------------------------------

class RecursiveSphereFlow(nn.Module):
    """Normalising flow on S2 with a CIRCULAR spline for phi.

    Faithfully implements the RecursiveSphereFlow from Circular_Splines.ipynb.

    Architecture per block
    ----------------------
    cos_theta: transformed by a FREE-PARAMETER Cartesian RQS.
               No MLP -- the spline parameters are directly learned scalars.
               Domain: (-1, 1).

    phi:       transformed by an MLP-CONDITIONED CIRCULAR RQS.
               Conditioned on the (already transformed) cos_theta.
               Domain: (0, 2*pi).
               Periodicity enforced: d[0] == d[K].

    Note: this model works in PHYSICAL coordinates so no external
    normalisation / denormalisation is needed for this model alone.
    The base distribution is a 2D standard normal (after the spline maps
    to approximately Gaussian-distributed latents).

    Parameters
    ----------
    num_bins : int
        RQS bins (default 32).
    num_splines : int
        Number of stacked coupling blocks (default 1).
    hidden_dim : int
        Width of the phi conditioner MLP (default 64).
    num_layers : int
        Depth of the phi conditioner MLP (default 2).
    arch : str  -- 'mlp' or 'resnet'.
    activation : str
    dropout : float
    """

    TWO_PI = 2.0 * np.pi

    def __init__(self,
                 num_bins: int = 32,
                 num_splines: int = 1,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 arch: str = "mlp",
                 activation: str = "relu",
                 dropout: float = 0.0):
        super().__init__()
        self.num_bins    = num_bins
        self.num_splines = num_splines

        # Free parameters for cos_theta spline (one set per block)
        # Shape: (3*K+1,)  -- K widths, K heights, K+1 derivatives
        self.z_params = nn.ParameterList([
            nn.Parameter(torch.randn(3 * num_bins + 1))
            for _ in range(num_splines)
        ])

        # MLP conditioners for phi circular spline (one per block)
        # Output size: 3*K (NOT 3*K+1 because circular spline needs K derivs)
        self.phi_nets = nn.ModuleList([
            build_conditioner(arch, in_dim=1, out_dim=3 * num_bins,
                              hidden_dim=hidden_dim, num_layers=num_layers,
                              activation=activation, dropout=dropout)
            for _ in range(num_splines)
        ])

    def _z_whd(self, k, B):
        """Expand block-k free parameters to (B,) tensors for rqs_with_bounds."""
        p = self.z_params[k]                          # (3K+1,)
        K = self.num_bins
        w = p[:K].unsqueeze(0).expand(B, -1)          # (B, K)
        h = p[K:2*K].unsqueeze(0).expand(B, -1)       # (B, K)
        d = p[2*K:].unsqueeze(0).expand(B, -1)        # (B, K+1)
        return w, h, d

    def _phi_whd(self, k, z):
        """Get phi spline parameters from MLP conditioned on z."""
        out = self.phi_nets[k](z)                     # (B, 3K)
        K   = self.num_bins
        w   = out[:, :K]
        h   = out[:, K:2*K]
        d   = out[:, 2*K:]                            # (B, K) -- circular
        return w, h, d

    def forward(self, x, inverse=False):
        """Forward or inverse pass.

        forward  (inverse=False): physical (cos_theta, phi) -> latent (z1, z2)
        inverse  (inverse=True) : latent  (z1, z2) -> physical (cos_theta, phi)

        Returns
        -------
        forward : (output Tensor (B,2),  log_det Tensor (B,))
        inverse : output Tensor (B,2)
        """
        B = x.shape[0]
        log_det = torch.zeros(B, device=x.device, dtype=x.dtype)

        if not inverse:
            # ---- Forward: physical -> latent ------------------------------ #
            cos_theta = x[:, 0]                        # (B,)
            phi       = x[:, 1]                        # (B,)

            for k in range(self.num_splines):
                # Step A: transform cos_theta with free-param Cartesian RQS
                w, h, d = self._z_whd(k, B)
                cos_theta, ldj = rqs_with_bounds(
                    cos_theta, w, h, d, inverse=False, b_x=(-1, 1), b_y=(-1, 1))
                log_det += ldj

                # Step B: transform phi with MLP-conditioned circular RQS
                #         conditioned on the JUST-transformed cos_theta
                w, h, d = self._phi_whd(k, cos_theta.unsqueeze(1))
                phi, ldj = rqs_circular(
                    phi, w, h, d, inverse=False,
                    b_x=(0, self.TWO_PI), b_y=(0, self.TWO_PI))
                log_det += ldj

            return torch.stack([cos_theta, phi], dim=1), log_det

        else:
            # ---- Inverse: latent -> physical ------------------------------ #
            cos_theta = x[:, 0]
            phi       = x[:, 1]

            for k in reversed(range(self.num_splines)):
                # Undo Step B: invert circular phi spline
                # Need cos_theta BEFORE it was transformed in Step A of this block.
                # So we first invert cos_theta, then use it to condition phi.
                w, h, d = self._z_whd(k, B)
                cos_theta_prev, _ = rqs_with_bounds(
                    cos_theta, w, h, d, inverse=True, b_x=(-1, 1), b_y=(-1, 1))

                w, h, d = self._phi_whd(k, cos_theta_prev.unsqueeze(1))
                phi, _ = rqs_circular(
                    phi, w, h, d, inverse=True,
                    b_x=(0, self.TWO_PI), b_y=(0, self.TWO_PI))

                cos_theta = cos_theta_prev

            return torch.stack([cos_theta, phi], dim=1)

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
# S2  --  AngularSphereFlow  (standard RQS, standardised space)
# ---------------------------------------------------------------------------

class AngularSphereFlow(nn.Module):
    """Normalising flow on S2 using standard MLP-conditioned RQS.

    Works in STANDARDISED (cos_theta, phi) space. Both dimensions use the
    same standard symmetric RQS. No circular boundary enforcement.

    This is the Cartesian_RQS_3MomS2 notebook model.

    Parameters
    ----------
    num_bins : int
    bound : float -- spline half-domain in standardised space (default 5.0).
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

        self.nets_phi      = nn.ModuleList([
            build_conditioner(arch, 1, out, hidden_dim, num_layers, activation, dropout)
            for _ in range(num_splines)])
        self.nets_costheta = nn.ModuleList([
            build_conditioner(arch, 1, out, hidden_dim, num_layers, activation, dropout)
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
# R3  --  CartesianNSF  (standard RQS, standardised Cartesian space)
# ---------------------------------------------------------------------------

class CartesianNSF(nn.Module):
    """Normalising flow in R3 for (px, py, pz) using standard RQS.

    Works in STANDARDISED Cartesian space.
    This is the Cartesian_RQS_3MomR3 notebook model.

    Parameters
    ----------
    num_bins : int
    bound : float -- spline half-domain in standardised space (default 5.0).
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

        out2 = 2 * (3 * num_bins + 1)   # params for 2 dims (py, pz)
        out1 = 1 * (3 * num_bins + 1)   # params for 1 dim  (px)

        self.nets1 = nn.ModuleList([   # px -> params for (py, pz)
            build_conditioner(arch, 1, out2, hidden_dim, num_layers, activation, dropout)
            for _ in range(num_splines)])
        self.nets2 = nn.ModuleList([   # (py, pz) -> params for px
            build_conditioner(arch, 2, out1, hidden_dim, num_layers, activation, dropout)
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
    "r2": RecursiveSphereFlow,   # circular phi + free-param cos_theta
    "s2": AngularSphereFlow,     # standard RQS, normalised space
    "r3": CartesianNSF,          # standard RQS, Cartesian space
}


def build_model(model_type: str, **kwargs) -> nn.Module:
    """Instantiate a flow model by registry key.

    Parameters
    ----------
    model_type : str  -- 'r2', 's2', or 'r3'.
    **kwargs          -- passed directly to the model constructor.

    Examples
    --------
    >>> model = build_model('r2', num_bins=32, num_splines=2)
    >>> model = build_model('s2', num_bins=32, num_splines=2, bound=5.0)
    >>> model = build_model('r3', num_bins=64, num_splines=3, hidden_dim=128)
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_type}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type](**kwargs)

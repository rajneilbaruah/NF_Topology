"""
networks.py
===========
Normalising-flow model classes.

Three model variants are provided, all using the Rational-Quadratic
Spline (RQS) bijection from ``splines.py`` and conditioner networks from
``mlps.py``.

Models
------
AngularSphereFlow  (``s2``)
    2-D angular flow on S².  Input: ``(cos θ, φ)`` normalised to
    ``[-1, 1]``.  Designed for the muon directional dataset.

CartesianNSF  (``r3``)
    3-D Cartesian flow in R³.  Input: ``(px, py, pz)`` standardised to
    unit-variance.

RecursiveSphereFlow  (``s2_recursive``)
    Lightweight 2-D sphere flow that uses a single pair of unconditioned
    parameter vectors (no MLP) for maximum speed.

All models share the same ``forward`` signature:
    - ``forward(x, inverse=False)`` → ``(z, log_det)`` or ``x_sample``
    - ``log_prob(x)``               → scalar NLL per sample
    - ``sample(n)``                 → ``(n, dim)`` tensor of new samples

Stacking
--------
Pass ``num_splines=N`` to any model to stack N coupling-layer *blocks*.
Each block is a complete alternating-coupling pass over all dimensions.
``num_splines=1`` reproduces the original single-spline behaviour from
the notebooks.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .splines import rqs
from .mlps import build_conditioner


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unpack_params(net_out: torch.Tensor,
                   out_dims: int,
                   num_bins: int
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reshape flat conditioner output into ``(w, h, d)`` spline tensors.

    Parameters
    ----------
    net_out : Tensor, shape ``(B, out_dims * (3*K+1))``
    out_dims : int   Number of dimensions being transformed.
    num_bins : int   ``K`` — number of spline bins.

    Returns
    -------
    w : ``(B, out_dims, K)``
    h : ``(B, out_dims, K)``
    d : ``(B, out_dims, K+1)``
    """
    p = net_out.view(-1, out_dims, 3 * num_bins + 1)
    return p[..., :num_bins], p[..., num_bins: 2 * num_bins], p[..., 2 * num_bins:]


# ---------------------------------------------------------------------------
# S² angular flow  (2-D)
# ---------------------------------------------------------------------------

class AngularSphereFlow(nn.Module):
    """Normalising flow on the 2-sphere (cos θ, φ) using stacked RQS layers.

    The base distribution is a 2-D standard normal in the normalised
    coordinates ``z = (z_cosθ, z_φ)``.

    Each *spline block* consists of two alternating coupling layers:
      1. Transform φ conditioned on cos θ.
      2. Transform cos θ conditioned on (transformed) φ.

    Parameters
    ----------
    num_bins : int
        Number of RQS bins per coupling step (default 32).
    bound : float
        Spline domain ``[-bound, bound]`` in normalised space (default 5.0).
    num_splines : int
        Number of stacked spline blocks (default 1).
        More blocks increase expressiveness at the cost of compute.
    hidden_dim : int
        Width of each conditioner network (default 64).
    num_layers : int
        Depth of each conditioner network (default 2).
    arch : str
        Conditioner architecture: ``'mlp'`` or ``'resnet'`` (default ``'mlp'``).
    activation : str
        Nonlinearity used inside conditioner networks (default ``'relu'``).
    dropout : float
        Dropout probability in conditioners (default 0.0).
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
        self.num_bins = num_bins
        self.bound = bound
        self.num_splines = num_splines
        out_size = 3 * num_bins + 1

        # Build a ModuleList of (net_phi, net_costheta) pairs — one per block
        self.nets_phi      = nn.ModuleList()
        self.nets_costheta = nn.ModuleList()
        for _ in range(num_splines):
            self.nets_phi.append(build_conditioner(
                arch, in_dim=1, out_dim=out_size,
                hidden_dim=hidden_dim, num_layers=num_layers,
                activation=activation, dropout=dropout))
            self.nets_costheta.append(build_conditioner(
                arch, in_dim=1, out_dim=out_size,
                hidden_dim=hidden_dim, num_layers=num_layers,
                activation=activation, dropout=dropout))

    # ------------------------------------------------------------------ #
    def forward(self,
                x: torch.Tensor,
                inverse: bool = False
                ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward or inverse pass.

        Parameters
        ----------
        x : Tensor, shape ``(B, 2)``
            In forward mode: normalised ``(cos θ, φ)`` data.
            In inverse mode: samples from base distribution (std normal).
        inverse : bool
            ``False`` → data → base  (returns ``(z, log_det)``).
            ``True``  → base → data  (returns ``x_reconstructed``).

        Returns
        -------
        forward:  ``(z, log_det_total)``  — used during training.
        inverse:  ``x_reconstructed``     — used during sampling.
        """
        log_det_total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        if not inverse:
            # ---- Forward: data → base --------------------------------- #
            costheta, phi_scaled = x[:, 0:1], x[:, 1:2]

            for k in range(self.num_splines):
                # Step A: transform φ | cos θ
                w, h, d = _unpack_params(self.nets_phi[k](costheta), 1, self.num_bins)
                phi_scaled, ldj = rqs(phi_scaled, w, h, d,
                                      inverse=False, bound=self.bound)
                log_det_total += ldj.squeeze(-1)

                # Step B: transform cos θ | φ
                w, h, d = _unpack_params(self.nets_costheta[k](phi_scaled), 1, self.num_bins)
                costheta, ldj = rqs(costheta, w, h, d,
                                    inverse=False, bound=self.bound)
                log_det_total += ldj.squeeze(-1)

            z = torch.cat([costheta, phi_scaled], dim=1)
            return z, log_det_total

        else:
            # ---- Inverse: base → data --------------------------------- #
            costheta, phi_scaled = x[:, 0:1], x[:, 1:2]

            # Reverse through blocks in reverse order
            for k in reversed(range(self.num_splines)):
                # Undo Step B: cos θ | φ
                w, h, d = _unpack_params(self.nets_costheta[k](phi_scaled), 1, self.num_bins)
                costheta, _ = rqs(costheta, w, h, d,
                                  inverse=True, bound=self.bound)

                # Undo Step A: φ | cos θ
                w, h, d = _unpack_params(self.nets_phi[k](costheta), 1, self.num_bins)
                phi_scaled, _ = rqs(phi_scaled, w, h, d,
                                    inverse=True, bound=self.bound)

            return torch.cat([costheta, phi_scaled], dim=1)

    # ------------------------------------------------------------------ #
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log-probability of data ``x`` under the learned distribution.

        Parameters
        ----------
        x : Tensor, shape ``(B, 2)``  — normalised ``(cos θ, φ)``.

        Returns
        -------
        log_prob : Tensor, shape ``(B,)``
        """
        z, log_det = self.forward(x, inverse=False)
        log_pz = -0.5 * (z ** 2 + np.log(2.0 * np.pi)).sum(dim=1)
        return log_pz + log_det

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(self, n: int,
               device: torch.device | None = None) -> torch.Tensor:
        """Draw ``n`` samples from the learned distribution.

        Parameters
        ----------
        n : int
        device : torch.device, optional

        Returns
        -------
        Tensor, shape ``(n, 2)``  — normalised ``(cos θ, φ)``.
        """
        device = device or next(self.parameters()).device
        z = torch.randn(n, 2, device=device)
        return self.forward(z, inverse=True)


# ---------------------------------------------------------------------------
# R³ Cartesian flow  (3-D)
# ---------------------------------------------------------------------------

class CartesianNSF(nn.Module):
    """Normalising flow in R³ for 3-momentum ``(px, py, pz)`` using RQS.

    The base distribution is a 3-D standard normal in standardised space.

    Each *spline block* consists of two alternating coupling layers:
      1. Transform ``(py, pz)`` conditioned on ``px``.
      2. Transform ``px``       conditioned on ``(py, pz)``.

    Parameters
    ----------
    num_bins : int
        Number of RQS bins (default 32).
    bound : float
        Spline domain half-width in standardised coordinates (default 5.0).
    num_splines : int
        Number of stacked spline blocks (default 1).
    hidden_dim : int
        Conditioner hidden width (default 64).
    num_layers : int
        Conditioner depth (default 2).
    arch : str
        ``'mlp'`` or ``'resnet'``.
    activation : str
    dropout : float
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
        self.num_bins = num_bins
        self.bound = bound
        self.num_splines = num_splines
        out_size_2 = 2 * (3 * num_bins + 1)  # params for 2 dims
        out_size_1 = 1 * (3 * num_bins + 1)  # params for 1 dim

        # net1: px → params for (py, pz)
        # net2: (py, pz) → params for px
        self.nets1 = nn.ModuleList()
        self.nets2 = nn.ModuleList()
        for _ in range(num_splines):
            self.nets1.append(build_conditioner(
                arch, in_dim=1, out_dim=out_size_2,
                hidden_dim=hidden_dim, num_layers=num_layers,
                activation=activation, dropout=dropout))
            self.nets2.append(build_conditioner(
                arch, in_dim=2, out_dim=out_size_1,
                hidden_dim=hidden_dim, num_layers=num_layers,
                activation=activation, dropout=dropout))

    # ------------------------------------------------------------------ #
    def forward(self,
                x: torch.Tensor,
                inverse: bool = False
                ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward or inverse pass.

        Parameters
        ----------
        x : Tensor, shape ``(B, 3)``
        inverse : bool

        Returns
        -------
        forward:  ``(z, log_det_total)``
        inverse:  ``x_reconstructed``
        """
        log_det_total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        if not inverse:
            # ---- Forward: data → base --------------------------------- #
            px, pyz = x[:, 0:1], x[:, 1:3]

            for k in range(self.num_splines):
                # Step A: transform (py, pz) | px
                w, h, d = _unpack_params(self.nets1[k](px), 2, self.num_bins)
                pyz, ldj = rqs(pyz, w, h, d, inverse=False, bound=self.bound)
                log_det_total += ldj.sum(dim=-1)

                # Step B: transform px | (py, pz)
                w, h, d = _unpack_params(self.nets2[k](pyz), 1, self.num_bins)
                px, ldj = rqs(px, w, h, d, inverse=False, bound=self.bound)
                log_det_total += ldj.sum(dim=-1)

            z = torch.cat([px, pyz], dim=1)
            return z, log_det_total

        else:
            # ---- Inverse: base → data --------------------------------- #
            px, pyz = x[:, 0:1], x[:, 1:3]

            for k in reversed(range(self.num_splines)):
                # Undo Step B
                w, h, d = _unpack_params(self.nets2[k](pyz), 1, self.num_bins)
                px, _ = rqs(px, w, h, d, inverse=True, bound=self.bound)

                # Undo Step A
                w, h, d = _unpack_params(self.nets1[k](px), 2, self.num_bins)
                pyz, _ = rqs(pyz, w, h, d, inverse=True, bound=self.bound)

            return torch.cat([px, pyz], dim=1)

    # ------------------------------------------------------------------ #
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det = self.forward(x, inverse=False)
        log_pz = -0.5 * (z ** 2 + np.log(2.0 * np.pi)).sum(dim=1)
        return log_pz + log_det

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(self, n: int,
               device: torch.device | None = None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        z = torch.randn(n, 3, device=device)
        return self.forward(z, inverse=True)


# ---------------------------------------------------------------------------
# Recursive Sphere Flow  (parameter-only, no MLP)
# ---------------------------------------------------------------------------

class RecursiveSphereFlow(nn.Module):
    """Lightweight 2-D sphere flow with *free parameters* (no conditioner MLP).

    The spline parameters for both dimensions are learned as free
    ``nn.Parameter`` vectors rather than outputs of a neural network.
    This is the fastest possible single-pass flow and is useful as a
    baseline or when compute is very limited.

    Parameters
    ----------
    num_bins : int
        RQS bins (default 32).
    num_splines : int
        Number of stacked free-parameter spline layers (default 1).
    bound_costheta : tuple[float, float]
        Domain for cos θ component (default ``(-1, 1)``).
    bound_phi : tuple[float, float]
        Domain for φ component (default ``(0, 2π)``).
    """

    def __init__(self,
                 num_bins: int = 32,
                 num_splines: int = 1,
                 bound_costheta: tuple[float, float] = (-1.0, 1.0),
                 bound_phi: tuple[float, float] = (0.0, 2.0 * np.pi)):
        super().__init__()
        self.num_bins = num_bins
        self.num_splines = num_splines
        self.bound_costheta = bound_costheta
        self.bound_phi = bound_phi

        # Each layer has independent parameters for cos θ and φ
        self.z_params  = nn.ParameterList()
        self.phi_params = nn.ParameterList()
        for _ in range(num_splines):
            self.z_params.append(nn.Parameter(torch.randn(3 * num_bins + 1)))
            self.phi_params.append(nn.Parameter(torch.randn(3 * num_bins + 1)))

    def _expand(self, param: torch.Tensor,
                batch: int, out_dims: int = 1) -> tuple:
        """Expand a 1-D parameter vector to ``(B, out_dims, K)`` tensors."""
        p = param.unsqueeze(0).unsqueeze(0).expand(batch, out_dims, -1)
        K = self.num_bins
        return p[..., :K], p[..., K: 2 * K], p[..., 2 * K:]

    def forward(self,
                x: torch.Tensor,
                inverse: bool = False
                ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        B = x.shape[0]
        log_det = torch.zeros(B, device=x.device, dtype=x.dtype)
        costheta, phi = x[:, 0:1], x[:, 1:2]

        # Compute bound centres for the simple shift-to-[-1,1] mapping
        cz_half = (self.bound_costheta[1] - self.bound_costheta[0]) / 2.0
        phi_half = (self.bound_phi[1] - self.bound_phi[0]) / 2.0

        if not inverse:
            for k in range(self.num_splines):
                # cos θ layer
                w, h, d = self._expand(self.z_params[k], B)
                costheta, ldj = rqs(costheta, w, h, d,
                                    inverse=False, bound=cz_half)
                log_det += ldj.squeeze(-1)
                # φ layer
                w, h, d = self._expand(self.phi_params[k], B)
                phi, ldj = rqs(phi, w, h, d,
                               inverse=False, bound=phi_half)
                log_det += ldj.squeeze(-1)
            return torch.cat([costheta, phi], dim=1), log_det

        else:
            for k in reversed(range(self.num_splines)):
                w, h, d = self._expand(self.phi_params[k], B)
                phi, _ = rqs(phi, w, h, d, inverse=True, bound=phi_half)
                w, h, d = self._expand(self.z_params[k], B)
                costheta, _ = rqs(costheta, w, h, d, inverse=True, bound=cz_half)
            return torch.cat([costheta, phi], dim=1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det = self.forward(x, inverse=False)
        log_pz = -0.5 * (z ** 2 + np.log(2.0 * np.pi)).sum(dim=1)
        return log_pz + log_det

    @torch.no_grad()
    def sample(self, n: int,
               device: torch.device | None = None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        z = torch.randn(n, 2, device=device)
        return self.forward(z, inverse=True)


# ---------------------------------------------------------------------------
# Registry for easy instantiation by name
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type] = {
    "s2":           AngularSphereFlow,
    "r3":           CartesianNSF,
    "s2_recursive": RecursiveSphereFlow,
}


def build_model(model_type: str, **kwargs) -> nn.Module:
    """Instantiate a flow model by its registry key.

    Parameters
    ----------
    model_type : str
        One of ``'s2'``, ``'r3'``, ``'s2_recursive'``.
    **kwargs
        Keyword arguments forwarded to the model constructor.

    Returns
    -------
    nn.Module

    Examples
    --------
    >>> model = build_model('r3', num_bins=64, num_splines=3, hidden_dim=128)
    >>> model = build_model('s2', num_bins=32, num_splines=2, arch='resnet')
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_type}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type](**kwargs)

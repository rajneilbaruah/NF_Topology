"""
splines.py
==========
Core Rational-Quadratic Spline (RQS) implementation.

Both the Cartesian R³ flow and the angular S² flow share the same
monotone rational-quadratic bijection.  This module provides a single,
well-tested implementation that every network in this package relies on.

References
----------
Durkan et al., "Neural Spline Flows" (NeurIPS 2019)
https://arxiv.org/abs/1906.04032
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rqs_forward(inputs: torch.Tensor,
                w: torch.Tensor,
                h: torch.Tensor,
                d: torch.Tensor,
                bound: float = 1.0,
                eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass of the Rational-Quadratic Spline.

    Maps ``inputs`` from the *data* domain ``[-bound, bound]`` to the
    *base* domain ``[-bound, bound]`` and returns the log-absolute
    determinant of the Jacobian.

    Parameters
    ----------
    inputs : Tensor, shape ``(B, D)``
        Values to transform.  Each element must lie inside the interval
        ``(-bound, bound)``.  Out-of-range values are clamped with a small
        epsilon margin.
    w : Tensor, shape ``(B, D, K)``
        Unnormalised bin-width logits.  Passed through softmax internally.
    h : Tensor, shape ``(B, D, K)``
        Unnormalised bin-height logits.  Passed through softmax internally.
    d : Tensor, shape ``(B, D, K+1)``
        Unnormalised derivative logits at the ``K+1`` knot positions.
        Passed through softplus internally, with a floor of 1e-3.
    bound : float
        Half-width of the spline domain.  Input values are mapped from
        ``[-bound, bound]`` to ``[-bound, bound]``.
    eps : float
        Small numerical epsilon used when clamping inputs.

    Returns
    -------
    outputs : Tensor, shape ``(B, D)``
    logabsdet : Tensor, shape ``(B, D)``
        Log |dy/dx| per element.
    """
    left = right = -bound, bound  # unpack as left, right below
    left, right = -bound, bound
    bottom, top = -bound, bound
    num_bins = w.shape[-1]

    # Clamp inputs strictly inside the spline domain
    inputs = inputs.clamp(left + eps, right - eps)

    # Normalise spline parameters
    widths = F.softmax(w, dim=-1) * (right - left)          # (B, D, K)
    heights = F.softmax(h, dim=-1) * (top - bottom)         # (B, D, K)
    derivatives = F.softplus(d) + 1e-3                      # (B, D, K+1)

    # Cumulative knot positions
    cum_widths = F.pad(torch.cumsum(widths, dim=-1), (1, 0), value=0.0) + left   # (B, D, K+1)
    cum_heights = F.pad(torch.cumsum(heights, dim=-1), (1, 0), value=0.0) + bottom  # (B, D, K+1)

    # Locate each input in the bin grid
    bin_idx = (
        torch.searchsorted(cum_widths, inputs.unsqueeze(-1).contiguous(), right=True) - 1
    ).clamp(0, num_bins - 1)  # (B, D, 1)

    # Gather local parameters
    w_b   = torch.gather(widths,      -1, bin_idx)           # (B, D, 1)
    h_b   = torch.gather(heights,     -1, bin_idx)
    d_k   = torch.gather(derivatives, -1, bin_idx)
    d_kp1 = torch.gather(derivatives, -1, bin_idx + 1)
    x_k   = torch.gather(cum_widths,  -1, bin_idx)
    y_k   = torch.gather(cum_heights, -1, bin_idx)
    s_b   = h_b / w_b

    # Normalised position inside the bin
    xi = (inputs.unsqueeze(-1) - x_k) / w_b                 # (B, D, 1)

    # RQS transform
    den = s_b + (d_kp1 + d_k - 2.0 * s_b) * xi * (1.0 - xi)
    num_ = h_b * (s_b * xi ** 2 + d_k * xi * (1.0 - xi))
    outputs = (y_k + num_ / den).squeeze(-1)                 # (B, D)

    # Log |dy/dx|
    deriv_num = s_b ** 2 * (
        d_kp1 * xi ** 2
        + 2.0 * s_b * xi * (1.0 - xi)
        + d_k * (1.0 - xi) ** 2
    )
    logabsdet = (torch.log(deriv_num + 1e-9) - 2.0 * torch.log(den + 1e-9)).squeeze(-1)
    return outputs, logabsdet


def rqs_inverse(inputs: torch.Tensor,
                w: torch.Tensor,
                h: torch.Tensor,
                d: torch.Tensor,
                bound: float = 1.0,
                eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse pass of the Rational-Quadratic Spline.

    Maps ``inputs`` from the *base* domain ``[-bound, bound]`` back to
    the *data* domain ``[-bound, bound]``.  Uses the closed-form
    quadratic root to invert the bijection exactly.

    Parameters
    ----------
    inputs : Tensor, shape ``(B, D)``
        Latent values to invert.
    w, h, d : Tensors, shape ``(B, D, K)`` / ``(B, D, K+1)``
        Spline parameters (same convention as :func:`rqs_forward`).
    bound : float
    eps : float

    Returns
    -------
    outputs : Tensor, shape ``(B, D)``
    logabsdet : Tensor, shape ``(B, D)``
        Log |dx/dy| per element (negative of the forward LDJ).
    """
    left, right = -bound, bound
    bottom, top = -bound, bound
    num_bins = w.shape[-1]

    inputs = inputs.clamp(bottom + eps, top - eps)

    widths = F.softmax(w, dim=-1) * (right - left)
    heights = F.softmax(h, dim=-1) * (top - bottom)
    derivatives = F.softplus(d) + 1e-3

    cum_widths = F.pad(torch.cumsum(widths, dim=-1), (1, 0), value=0.0) + left
    cum_heights = F.pad(torch.cumsum(heights, dim=-1), (1, 0), value=0.0) + bottom

    # Search in height space for the inverse
    bin_idx = (
        torch.searchsorted(cum_heights, inputs.unsqueeze(-1).contiguous(), right=True) - 1
    ).clamp(0, num_bins - 1)

    w_b   = torch.gather(widths,      -1, bin_idx)
    h_b   = torch.gather(heights,     -1, bin_idx)
    d_k   = torch.gather(derivatives, -1, bin_idx)
    d_kp1 = torch.gather(derivatives, -1, bin_idx + 1)
    x_k   = torch.gather(cum_widths,  -1, bin_idx)
    y_k   = torch.gather(cum_heights, -1, bin_idx)
    s_b   = h_b / w_b

    # Solve the quadratic  a*xi^2 + b*xi + c = 0
    y_rel = inputs.unsqueeze(-1) - y_k
    a = h_b * (s_b - d_k) + y_rel * (d_kp1 + d_k - 2.0 * s_b)
    b = h_b * d_k - y_rel * (d_kp1 + d_k - 2.0 * s_b)
    c = -s_b * y_rel

    xi = 2.0 * c / (-b - torch.sqrt(b ** 2 - 4.0 * a * c + 1e-9))
    outputs = (xi * w_b + x_k).squeeze(-1)

    # LDJ for inverse = negative of forward LDJ
    den = s_b + (d_kp1 + d_k - 2.0 * s_b) * xi * (1.0 - xi)
    deriv_num = s_b ** 2 * (
        d_kp1 * xi ** 2
        + 2.0 * s_b * xi * (1.0 - xi)
        + d_k * (1.0 - xi) ** 2
    )
    logabsdet = -(torch.log(deriv_num + 1e-9) - 2.0 * torch.log(den + 1e-9)).squeeze(-1)
    return outputs, logabsdet


def rqs(inputs: torch.Tensor,
        w: torch.Tensor,
        h: torch.Tensor,
        d: torch.Tensor,
        inverse: bool = False,
        bound: float = 1.0,
        eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Unified entry point: calls :func:`rqs_forward` or :func:`rqs_inverse`.

    Parameters
    ----------
    inverse : bool
        If ``False`` (default) runs the forward pass (data → base).
        If ``True`` runs the inverse pass (base → data).
    """
    if inverse:
        return rqs_inverse(inputs, w, h, d, bound=bound, eps=eps)
    return rqs_forward(inputs, w, h, d, bound=bound, eps=eps)

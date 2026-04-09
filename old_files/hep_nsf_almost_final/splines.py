"""
splines.py
==========
Core Rational-Quadratic Spline (RQS) implementations.

Three public functions are provided:

rqs
    Standard symmetric spline: domain ``[-bound, bound]`` → ``[-bound, bound]``.
    Used by AngularSphereFlow (S2) and CartesianNSF (R3).

rqs_with_bounds
    General asymmetric spline: explicit ``(left, right)`` input domain and
    ``(bottom, top)`` output domain.  Used when the physical domain is not
    centred at zero, e.g. phi in (0, 2*pi).

rqs_circular
    Circular / periodic spline: same as rqs_with_bounds but enforces
    d[0] == d[K] (derivative at the left edge equals derivative at the
    right edge).  Used for the phi dimension in RecursiveSphereFlow (R2)
    because phi lives on a circle.

    The circular constraint is:
        dp = torch.cat([dp, dp[:, 0:1]], dim=-1)
    The MLP/parameter vector outputs only K derivatives (not K+1),
    and the K+1-th is automatically set equal to the first.

References
----------
Durkan et al., "Neural Spline Flows" (NeurIPS 2019)
https://arxiv.org/abs/1906.04032
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Internal core  (shared by all three public functions)
# ---------------------------------------------------------------------------

def _rqs_core_forward(inputs, widths, heights, derivatives,
                      cum_widths, cum_heights, num_bins):
    bin_idx = (
        torch.searchsorted(cum_widths, inputs.unsqueeze(-1).contiguous(), right=True) - 1
    ).clamp(0, num_bins - 1)

    w_b   = torch.gather(widths,      -1, bin_idx)
    h_b   = torch.gather(heights,     -1, bin_idx)
    d_k   = torch.gather(derivatives, -1, bin_idx)
    d_kp1 = torch.gather(derivatives, -1, bin_idx + 1)
    x_k   = torch.gather(cum_widths,  -1, bin_idx)
    y_k   = torch.gather(cum_heights, -1, bin_idx)
    s_b   = h_b / w_b

    xi  = (inputs.unsqueeze(-1) - x_k) / w_b
    den = s_b + (d_kp1 + d_k - 2.0 * s_b) * xi * (1.0 - xi)
    num_ = h_b * (s_b * xi ** 2 + d_k * xi * (1.0 - xi))
    outputs = (y_k + num_ / den).squeeze(-1)

    deriv_num = s_b ** 2 * (
        d_kp1 * xi ** 2 + 2.0 * s_b * xi * (1.0 - xi) + d_k * (1.0 - xi) ** 2
    )
    logabsdet = (torch.log(deriv_num + 1e-9) - 2.0 * torch.log(den + 1e-9)).squeeze(-1)
    return outputs, logabsdet


def _rqs_core_inverse(inputs, widths, heights, derivatives,
                      cum_widths, cum_heights, num_bins):
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

    y_rel = inputs.unsqueeze(-1) - y_k
    a = h_b * (s_b - d_k) + y_rel * (d_kp1 + d_k - 2.0 * s_b)
    b = h_b * d_k - y_rel * (d_kp1 + d_k - 2.0 * s_b)
    c = -s_b * y_rel

    xi      = 2.0 * c / (-b - torch.sqrt(b ** 2 - 4.0 * a * c + 1e-9))
    outputs = (xi * w_b + x_k).squeeze(-1)

    den = s_b + (d_kp1 + d_k - 2.0 * s_b) * xi * (1.0 - xi)
    deriv_num = s_b ** 2 * (
        d_kp1 * xi ** 2 + 2.0 * s_b * xi * (1.0 - xi) + d_k * (1.0 - xi) ** 2
    )
    logabsdet = -(torch.log(deriv_num + 1e-9) - 2.0 * torch.log(den + 1e-9)).squeeze(-1)
    return outputs, logabsdet


# ---------------------------------------------------------------------------
# 1.  Standard symmetric spline  (S2 and R3 models)
# ---------------------------------------------------------------------------

def rqs(inputs, w, h, d, inverse=False, bound=1.0, eps=1e-6):
    """Standard RQS on [-bound, bound].  Used by S2 and R3 models."""
    left, right = -bound, bound
    bottom, top = -bound, bound
    num_bins    = w.shape[-1]

    inputs  = inputs.clamp(left + eps, right - eps)
    widths  = F.softmax(w, dim=-1) * (right - left)
    heights = F.softmax(h, dim=-1) * (top - bottom)
    derivs  = F.softplus(d) + 1e-3
    cw = F.pad(torch.cumsum(widths,  dim=-1), (1, 0), value=0.0) + left
    ch = F.pad(torch.cumsum(heights, dim=-1), (1, 0), value=0.0) + bottom

    if inverse:
        return _rqs_core_inverse(inputs, widths, heights, derivs, cw, ch, num_bins)
    return _rqs_core_forward(inputs, widths, heights, derivs, cw, ch, num_bins)


# ---------------------------------------------------------------------------
# 2.  General asymmetric spline  (explicit b_x / b_y domain bounds)
# ---------------------------------------------------------------------------

def rqs_with_bounds(inputs, w, h, d, inverse=False,
                    b_x=(0.0, 1.0), b_y=(0.0, 1.0), eps=1e-6):
    """RQS with explicit asymmetric input/output domain.

    Parameters
    ----------
    inputs : Tensor (B,)  or (B, 1)
    w, h   : Tensor (B, K)
    d      : Tensor (B, K+1)
    b_x    : (left, right)  input domain
    b_y    : (bottom, top)  output domain

    Matches the rqs_logic(b_x=..., b_y=...) convention in the notebook.
    """
    left,   right  = b_x
    bottom, top    = b_y
    num_bins       = w.shape[-1]

    inputs  = inputs.clamp(left + eps, right - eps)
    widths  = F.softmax(w, dim=-1) * (right  - left)
    heights = F.softmax(h, dim=-1) * (top    - bottom)
    derivs  = F.softplus(d) + 1e-3
    cw = F.pad(torch.cumsum(widths,  dim=-1), (1, 0), value=0.0) + left
    ch = F.pad(torch.cumsum(heights, dim=-1), (1, 0), value=0.0) + bottom

    # Expand batch dim if using free (unbatched) parameters
    if cw.shape[0] == 1 and inputs.shape[0] > 1:
        B  = inputs.shape[0]
        cw      = cw.expand(B, -1)
        ch      = ch.expand(B, -1)
        widths  = widths.expand(B, -1)
        heights = heights.expand(B, -1)
        derivs  = derivs.expand(B, -1)

    # Core functions expect (B, D, K); here D=1 so we unsqueeze/squeeze
    def _u(t): return t.unsqueeze(1)
    if inverse:
        out, ldj = _rqs_core_inverse(
            _u(inputs), _u(widths), _u(heights), _u(derivs), _u(cw), _u(ch), num_bins)
    else:
        out, ldj = _rqs_core_forward(
            _u(inputs), _u(widths), _u(heights), _u(derivs), _u(cw), _u(ch), num_bins)

    return out.squeeze(1), ldj.squeeze(1)


# ---------------------------------------------------------------------------
# 3.  Circular / periodic spline  (phi dimension of R2 model)
# ---------------------------------------------------------------------------

def rqs_circular(inputs, w, h, d, inverse=False,
                 b_x=(0.0, 6.283185307179586),
                 b_y=(0.0, 6.283185307179586),
                 eps=1e-6):
    """Circular RQS with periodic derivative boundary condition.

    Enforces d[0] == d[K] by appending d[:, 0] as d[:, K] before calling
    rqs_with_bounds.  This makes the spline smooth at the wrap-around point
    (phi = 0 == phi = 2*pi).

    Parameters
    ----------
    inputs : Tensor (B,)
    w, h   : Tensor (B, K)
    d      : Tensor (B, K)   <-- only K values, NOT K+1
    b_x, b_y : domain bounds, default (0, 2*pi)

    The circular trick from the notebook:
        dp = torch.cat([dp, dp[:, 0:1]], dim=1)
    """
    # Enforce periodicity: append first derivative as last
    d_circular = torch.cat([d, d[:, 0:1]], dim=-1)   # (B, K) -> (B, K+1)
    return rqs_with_bounds(inputs, w, h, d_circular,
                           inverse=inverse, b_x=b_x, b_y=b_y, eps=eps)

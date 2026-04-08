"""
hep_nsf — HEP Neural Spline Flows
===================================
A clean, cluster-ready Python package for learning probability densities
of high-energy physics 3-momentum data using Rational-Quadratic Spline
normalising flows.

Quick start
-----------
>>> from hep_nsf.networks import build_model
>>> model = build_model('s2', num_bins=32, num_splines=2)
>>> from hep_nsf.networks import build_model
>>> model = build_model('r3', num_bins=64, num_splines=3, hidden_dim=128)

See Also
--------
main.py   — CLI entry point with full argument parsing.
README.md — Installation, usage, and cluster instructions.
"""

from .networks import (
    AngularSphereFlow,
    CartesianNSF,
    RecursiveSphereFlow,
    build_model,
    MODEL_REGISTRY,
)
from .splines import rqs, rqs_with_bounds, rqs_circular
from .mlps    import MLP, ResNet, build_conditioner
from .utils   import (
    set_seed,
    get_device,
    cartesian_to_spherical,
    spherical_to_cartesian,
    cartesian_to_physics,
    load_json_data,
    normalise,
    denormalise,
    make_dataloaders,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
)
from .train   import train_model
from .analysis import (
    kl_divergence_kde,
    effective_sample_size,
    wasserstein_1d,
    js_divergence_1d,
    consistency_report,
    evaluate,
    model_summary,
)

__version__ = "1.0.0"
__all__ = [
    # Networks
    "AngularSphereFlow", "CartesianNSF", "RecursiveSphereFlow",
    "build_model", "MODEL_REGISTRY",
    # Splines
    "rqs", "rqs_with_bounds", "rqs_circular",
    # MLPs
    "MLP", "ResNet", "build_conditioner",
    # Utils
    "set_seed", "get_device",
    "cartesian_to_spherical", "spherical_to_cartesian", "cartesian_to_physics",
    "load_json_data", "normalise", "denormalise",
    "make_dataloaders", "save_checkpoint", "load_checkpoint",
    "count_parameters",
    # Training
    "train_model",
    # Analysis
    "kl_divergence_kde", "effective_sample_size",
    "wasserstein_1d", "js_divergence_1d",
    "consistency_report", "evaluate", "model_summary",
]

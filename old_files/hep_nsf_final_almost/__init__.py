"""
hep_nsf — HEP Neural Spline Flows
===================================
Naming:
  s2 = RecursiveSphereFlow  — physical (cos_theta, phi), uniform S2 base
  r2 = AngularSphereFlow    — standardised (cos_theta, phi), Gaussian R2 base
  r3 = CartesianNSF         — standardised (px, py, pz), Gaussian R3 base
"""

from .networks import (
    RecursiveSphereFlow,
    AngularSphereFlow,
    CartesianNSF,
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
    save_losses,
    load_losses,
    log_std_correction,
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

__version__ = "2.0.0"
__all__ = [
    "RecursiveSphereFlow", "AngularSphereFlow", "CartesianNSF",
    "build_model", "MODEL_REGISTRY",
    "rqs", "rqs_with_bounds", "rqs_circular",
    "MLP", "ResNet", "build_conditioner",
    "set_seed", "get_device",
    "cartesian_to_spherical", "spherical_to_cartesian", "cartesian_to_physics",
    "load_json_data", "normalise", "denormalise",
    "make_dataloaders", "save_checkpoint", "load_checkpoint",
    "count_parameters", "save_losses", "load_losses", "log_std_correction",
    "train_model",
    "kl_divergence_kde", "effective_sample_size",
    "wasserstein_1d", "js_divergence_1d",
    "consistency_report", "evaluate", "model_summary",
]

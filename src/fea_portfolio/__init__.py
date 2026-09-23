"""Small, reproducible DOLFINx finite-element portfolio examples."""

from .flow import CylinderFlowSimulation
from .heat import CenterHeatSimulation
from .poisson import CircularCutoutPoissonSimulation

__all__ = [
    "CenterHeatSimulation",
    "CircularCutoutPoissonSimulation",
    "CylinderFlowSimulation",
]


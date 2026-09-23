"""Poisson solve with an immersed circular cutout constraint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from .base import Simulation
from .visualization import FieldVisualizer


@dataclass(frozen=True)
class PoissonConfig:
    cells: int = 72
    cutout_center: tuple[float, float] = (0.5, 0.5)
    cutout_radius: float = 0.16


class CircularCutoutPoissonSimulation(Simulation):
    """Uses a strongly constrained circular region as an embedded cutout boundary."""

    def __init__(self, project_root: Path, config: PoissonConfig = PoissonConfig()) -> None:
        super().__init__(project_root)
        self.config = config
        self.visualizer = FieldVisualizer()

    def run(self) -> None:
        config = self.config
        domain = mesh.create_unit_square(MPI.COMM_WORLD, config.cells, config.cells)
        space = fem.functionspace(domain, ("Lagrange", 1))
        source = fem.Function(space, name="poisson_source")
        source.interpolate(lambda x: 10.0 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        solution = fem.Function(space, name="potential")
        trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)

        exterior_facets = mesh.locate_entities_boundary(
            domain, domain.topology.dim - 1, lambda x: np.full(x.shape[1], True)
        )
        exterior_dofs = fem.locate_dofs_topological(space, domain.topology.dim - 1, exterior_facets)
        center_x, center_y = config.cutout_center
        cutout_dofs = fem.locate_dofs_geometrical(
            space,
            lambda x: np.hypot(x[0] - center_x, x[1] - center_y) <= config.cutout_radius,
        )
        boundary_dofs = np.unique(np.concatenate((exterior_dofs, cutout_dofs)))
        boundary = fem.dirichletbc(PETSc.ScalarType(0), boundary_dofs, space)
        problem = LinearProblem(
            ufl.dot(ufl.grad(trial), ufl.grad(test)) * ufl.dx,
            source * test * ufl.dx,
            bcs=[boundary],
            petsc_options_prefix="poisson_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        solution.x.array[:] = problem.solve().x.array
        solution.x.scatter_forward()
        if MPI.COMM_WORLD.rank == 0:
            self.visualizer.scalar_screenshot(solution, self.paths.images / "poisson_circular_cutout.png")

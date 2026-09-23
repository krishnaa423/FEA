"""Backward-Euler heat equation with a localized central source."""

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
class HeatConfig:
    cells: int = 48
    diffusivity: float = 0.08
    time_step: float = 0.025
    steps: int = 80
    source_strength: float = 40.0
    source_width: float = 0.08


class CenterHeatSimulation(Simulation):
    """Solves a 2D heat problem until the source-driven field nearly settles."""

    def __init__(self, project_root: Path, config: HeatConfig = HeatConfig()) -> None:
        super().__init__(project_root)
        self.config = config
        self.visualizer = FieldVisualizer()

    def run(self) -> None:
        config = self.config
        domain = mesh.create_unit_square(MPI.COMM_WORLD, config.cells, config.cells)
        space = fem.functionspace(domain, ("Lagrange", 1))
        temperature = fem.Function(space, name="temperature")
        previous = fem.Function(space, name="previous_temperature")
        source = fem.Function(space, name="central_heat_source")
        source.interpolate(
            lambda x: config.source_strength
            * np.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / config.source_width**2)
        )

        facets = mesh.locate_entities_boundary(
            domain, domain.topology.dim - 1, lambda x: np.full(x.shape[1], True)
        )
        dofs = fem.locate_dofs_topological(space, domain.topology.dim - 1, facets)
        boundary = fem.dirichletbc(PETSc.ScalarType(0), dofs, space)
        trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
        a = (trial * test + config.time_step * config.diffusivity * ufl.dot(ufl.grad(trial), ufl.grad(test))) * ufl.dx
        rhs = (previous * test + config.time_step * source * test) * ufl.dx
        problem = LinearProblem(
            a,
            rhs,
            bcs=[boundary],
            petsc_options_prefix="heat_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )

        frames: list[np.ndarray] = []
        times: list[float] = []
        for step in range(config.steps):
            temperature.x.array[:] = problem.solve().x.array
            temperature.x.scatter_forward()
            previous.x.array[:] = temperature.x.array
            if step % 2 == 0:
                frames.append(temperature.x.array.real.copy())
                times.append((step + 1) * config.time_step)

        if MPI.COMM_WORLD.rank == 0:
            self.visualizer.scalar_screenshot(temperature, self.paths.images / "heat_steady_state.png", "inferno")
            cells = domain.topology.connectivity(domain.topology.dim, 0).array.reshape((-1, 3))
            self.visualizer.scalar_gif(domain.geometry.x, cells, frames, times, self.paths.images / "heat_evolution.gif", "2D heat equation")

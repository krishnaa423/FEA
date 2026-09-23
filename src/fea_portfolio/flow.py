"""A projection-method Navier-Stokes example for flow past an immersed cylinder."""

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
class FlowConfig:
    length: float = 2.2
    height: float = 0.6
    cells_x: int = 72
    cells_y: int = 24
    viscosity: float = 0.01
    density: float = 1.0
    time_step: float = 0.01
    steps: int = 70
    cylinder_center: tuple[float, float] = (0.75, 0.3)
    cylinder_radius: float = 0.10
    solid_penalty: float = 1500.0


class CylinderFlowSimulation(Simulation):
    """Runs a small 2D incompressible projection solve around an immersed ball."""

    def __init__(self, project_root: Path, config: FlowConfig = FlowConfig()) -> None:
        super().__init__(project_root)
        self.config = config
        self.visualizer = FieldVisualizer()

    def run(self) -> None:
        config = self.config
        domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [np.array([0.0, 0.0]), np.array([config.length, config.height])],
            [config.cells_x, config.cells_y],
            cell_type=mesh.CellType.triangle,
        )
        velocity_space = fem.functionspace(domain, ("Lagrange", 1, (2,)))
        pressure_space = fem.functionspace(domain, ("Lagrange", 1))
        velocity = fem.Function(velocity_space, name="velocity")
        velocity_old = fem.Function(velocity_space, name="previous_velocity")
        tentative = fem.Function(velocity_space, name="tentative_velocity")
        pressure = fem.Function(pressure_space, name="pressure")
        penalty = fem.Function(pressure_space, name="solid_penalty")
        cx, cy = config.cylinder_center
        penalty.interpolate(
            lambda x: config.solid_penalty
            * (np.hypot(x[0] - cx, x[1] - cy) <= config.cylinder_radius)
        )

        fdim = domain.topology.dim - 1
        inlet_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
        wall_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], config.height))
        inlet_dofs = fem.locate_dofs_topological(velocity_space, fdim, inlet_facets)
        wall_dofs = fem.locate_dofs_topological(velocity_space, fdim, wall_facets)
        inlet = fem.Function(velocity_space)
        inlet.interpolate(
            lambda x: np.vstack(
                (1.2 * 4.0 * x[1] * (config.height - x[1]) / config.height**2, np.zeros(x.shape[1]))
            )
        )
        velocity_bcs = [
            fem.dirichletbc(inlet, inlet_dofs),
            fem.dirichletbc(np.array((0.0, 0.0), dtype=PETSc.ScalarType), wall_dofs, velocity_space),
        ]
        outlet_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], config.length))
        outlet_dofs = fem.locate_dofs_topological(pressure_space, fdim, outlet_facets)
        pressure_bc = fem.dirichletbc(PETSc.ScalarType(0), outlet_dofs, pressure_space)

        u, v = ufl.TrialFunction(velocity_space), ufl.TestFunction(velocity_space)
        q, p = ufl.TrialFunction(pressure_space), ufl.TestFunction(pressure_space)
        convection = ufl.dot(velocity_old, ufl.nabla_grad(velocity_old))
        a_velocity = (
            config.density / config.time_step * ufl.inner(u, v)
            + config.viscosity * ufl.inner(ufl.grad(u), ufl.grad(v))
            + penalty * ufl.inner(u, v)
        ) * ufl.dx
        l_velocity = (
            config.density / config.time_step * ufl.inner(velocity_old, v)
            - config.density * ufl.inner(convection, v)
            + pressure * ufl.div(v)
        ) * ufl.dx
        tentative_problem = LinearProblem(
            a_velocity,
            l_velocity,
            bcs=velocity_bcs,
            petsc_options_prefix="tentative_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        a_pressure = ufl.dot(ufl.grad(q), ufl.grad(p)) * ufl.dx
        l_pressure = -config.density / config.time_step * ufl.div(tentative) * p * ufl.dx
        pressure_problem = LinearProblem(
            a_pressure,
            l_pressure,
            bcs=[pressure_bc],
            petsc_options_prefix="pressure_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        a_correction = ufl.inner(u, v) * ufl.dx
        l_correction = (
            ufl.inner(tentative, v)
            - config.time_step / config.density * ufl.inner(ufl.grad(pressure), v)
        ) * ufl.dx
        correction_problem = LinearProblem(
            a_correction,
            l_correction,
            bcs=velocity_bcs,
            petsc_options_prefix="correction_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )

        frames: list[np.ndarray] = []
        times: list[float] = []
        for step in range(config.steps):
            tentative.x.array[:] = tentative_problem.solve().x.array
            tentative.x.scatter_forward()
            pressure.x.array[:] = pressure_problem.solve().x.array
            pressure.x.scatter_forward()
            velocity.x.array[:] = correction_problem.solve().x.array
            velocity.x.scatter_forward()
            velocity_old.x.array[:] = velocity.x.array
            velocity_old.x.scatter_forward()
            if step % 2 == 0:
                components = velocity_space.dofmap.index_map_bs
                frames.append(np.linalg.norm(velocity.x.array.real.reshape((-1, components)), axis=1))
                times.append((step + 1) * config.time_step)

        if MPI.COMM_WORLD.rank == 0:
            self.visualizer.vector_screenshot(velocity, self.paths.images / "cylinder_flow.png", "2D flow past an immersed cylinder")
            cells = domain.topology.connectivity(domain.topology.dim, 0).array.reshape((-1, 3))
            self.visualizer.scalar_gif(domain.geometry.x, cells, frames, times, self.paths.images / "cylinder_flow.gif", "Speed around an immersed cylinder", "viridis")

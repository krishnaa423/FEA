"""PyVista screenshots and Matplotlib GIF helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pyvista
from dolfinx import fem
from dolfinx.plot import vtk_mesh


class FieldVisualizer:
    """Writes portable raster artifacts without requiring an interactive display."""

    def __init__(self) -> None:
        pyvista.OFF_SCREEN = True

    @staticmethod
    def _scalar_grid(field: fem.Function) -> pyvista.UnstructuredGrid:
        topology, cell_types, geometry = vtk_mesh(field.function_space)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        grid.point_data[field.name] = field.x.array.real
        grid.set_active_scalars(field.name)
        return grid

    def scalar_screenshot(
        self, field: fem.Function, output: Path, title: str, cmap: str = "viridis"
    ) -> None:
        grid = self._scalar_grid(field)
        plotter = pyvista.Plotter(off_screen=True, window_size=(1200, 720))
        plotter.add_mesh(grid, scalars=field.name, cmap=cmap, show_edges=False)
        plotter.add_text(title, font_size=14)
        plotter.view_xy()
        plotter.enable_parallel_projection()
        plotter.screenshot(str(output))
        plotter.close()

    def vector_screenshot(
        self, field: fem.Function, output: Path, title: str
    ) -> None:
        topology, cell_types, geometry = vtk_mesh(field.function_space)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        components = field.function_space.dofmap.index_map_bs
        values = field.x.array.real.reshape((-1, components))
        vectors = np.zeros((values.shape[0], 3))
        vectors[:, : min(components, 3)] = values[:, : min(components, 3)]
        grid.point_data["velocity"] = vectors
        speed = np.linalg.norm(vectors[:, :2], axis=1)
        grid.point_data["speed"] = speed
        arrows = grid.glyph(orient="velocity", scale=False, factor=0.16)
        plotter = pyvista.Plotter(off_screen=True, window_size=(1200, 520))
        plotter.add_mesh(grid, scalars="speed", cmap="plasma", opacity=0.75)
        plotter.add_mesh(arrows, scalars="speed", cmap="plasma")
        plotter.add_text(title, font_size=14)
        plotter.view_xy()
        plotter.enable_parallel_projection()
        plotter.screenshot(str(output))
        plotter.close()

    @staticmethod
    def scalar_gif(
        coordinates: np.ndarray,
        cells: np.ndarray,
        frames: Sequence[np.ndarray],
        times: Sequence[float],
        output: Path,
        title: str,
        cmap: str = "inferno",
    ) -> None:
        matplotlib.use("Agg", force=True)
        triangulation = mtri.Triangulation(coordinates[:, 0], coordinates[:, 1], cells)
        vmin = min(float(np.min(frame)) for frame in frames)
        vmax = max(float(np.max(frame)) for frame in frames)
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        artist = ax.tripcolor(triangulation, frames[0], shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(artist, ax=ax, label="field value")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        def update(index: int):
            artist.set_array(frames[index])
            ax.set_title(f"{title}, t = {times[index]:.3f}")
            return (artist,)

        movie = animation.FuncAnimation(fig, update, frames=len(frames), interval=100, blit=False)
        movie.save(output, writer=animation.PillowWriter(fps=10))
        plt.close(fig)


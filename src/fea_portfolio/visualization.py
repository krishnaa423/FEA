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
from PIL import Image
from dolfinx import fem
from dolfinx.plot import vtk_mesh


class FieldVisualizer:
    """Writes portable raster artifacts without requiring an interactive display."""

    def __init__(self) -> None:
        pyvista.OFF_SCREEN = True

    @staticmethod
    def _window_size(grid: pyvista.UnstructuredGrid) -> tuple[int, int]:
        """Keep the rendered domain close to the image aspect ratio."""
        width = grid.bounds[1] - grid.bounds[0]
        height = grid.bounds[3] - grid.bounds[2]
        aspect = min(max(width / height, 0.8), 3.4)
        image_height = 720
        return (round(image_height * aspect), image_height)

    @staticmethod
    def _frame_xy(plotter: pyvista.Plotter, grid: pyvista.UnstructuredGrid) -> None:
        """Use a deterministic orthographic frame instead of PyVista's loose fit."""
        width = grid.bounds[1] - grid.bounds[0]
        height = grid.bounds[3] - grid.bounds[2]
        viewport_aspect = plotter.window_size[0] / plotter.window_size[1]
        plotter.view_xy()
        plotter.enable_parallel_projection()
        plotter.camera.focal_point = (
            (grid.bounds[0] + grid.bounds[1]) / 2,
            (grid.bounds[2] + grid.bounds[3]) / 2,
            0,
        )
        plotter.camera.parallel_scale = max(height, width / viewport_aspect) * 1.12
        plotter.render()

    @staticmethod
    def _trim_white_margin(output: Path) -> None:
        """Remove PyVista's unused white viewport border from a screenshot."""
        image = Image.open(output).convert("RGB")
        pixels = np.asarray(image)
        occupied = np.any(pixels < 248, axis=2)
        rows, columns = np.where(occupied)
        padding = 16
        left = max(int(columns.min()) - padding, 0)
        right = min(int(columns.max()) + padding + 1, image.width)
        top = max(int(rows.min()) - padding, 0)
        bottom = min(int(rows.max()) + padding + 1, image.height)
        image.crop((left, top, right, bottom)).save(output)

    @staticmethod
    def _scalar_grid(field: fem.Function) -> pyvista.UnstructuredGrid:
        topology, cell_types, geometry = vtk_mesh(field.function_space)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        grid.point_data[field.name] = field.x.array.real
        grid.set_active_scalars(field.name)
        return grid

    def scalar_screenshot(
        self, field: fem.Function, output: Path, cmap: str = "viridis"
    ) -> None:
        grid = self._scalar_grid(field)
        plotter = pyvista.Plotter(off_screen=True, window_size=self._window_size(grid))
        plotter.add_mesh(
            grid,
            scalars=field.name,
            cmap=cmap,
            show_edges=False,
            scalar_bar_args={"title": "", "vertical": False, "position_x": 0.3, "position_y": 0.03},
        )
        self._frame_xy(plotter, grid)
        plotter.screenshot(str(output))
        plotter.close()
        self._trim_white_margin(output)

    def vector_screenshot(
        self, field: fem.Function, output: Path
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
        plotter = pyvista.Plotter(off_screen=True, window_size=self._window_size(grid))
        plotter.add_mesh(
            grid,
            scalars="speed",
            cmap="plasma",
            opacity=0.75,
            scalar_bar_args={"title": "", "vertical": False, "position_x": 0.3, "position_y": 0.03},
        )
        plotter.add_mesh(arrows, scalars="speed", cmap="plasma", show_scalar_bar=False)
        self._frame_xy(plotter, grid)
        plotter.screenshot(str(output))
        plotter.close()
        self._trim_white_margin(output)

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
        triangulation = mtri.Triangulation(coordinates[:, 0], coordinates[:, 1], cells)
        vmin = min(float(np.min(frame)) for frame in frames)
        vmax = max(float(np.max(frame)) for frame in frames)
        fig, ax = plt.subplots(constrained_layout=True)
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

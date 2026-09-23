"""Shared lifecycle and filesystem helpers for the simulation examples."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from mpi4py import MPI


@dataclass(frozen=True)
class OutputPaths:
    """Repository-local artifact paths, created once by each simulation."""

    root: Path

    @property
    def images(self) -> Path:
        return self.root / "images"

    @property
    def results(self) -> Path:
        return self.root / "results"

    def ensure(self) -> None:
        if MPI.COMM_WORLD.rank == 0:
            self.images.mkdir(parents=True, exist_ok=True)
            self.results.mkdir(parents=True, exist_ok=True)
        MPI.COMM_WORLD.barrier()


class Simulation(ABC):
    """A small interface that keeps setup, solve, and artifact creation explicit."""

    def __init__(self, project_root: Path) -> None:
        self.paths = OutputPaths(project_root)
        self.paths.ensure()

    @abstractmethod
    def run(self) -> None:
        """Solve the problem and write its documented artifacts."""


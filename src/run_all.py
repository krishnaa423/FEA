from pathlib import Path

from fea_portfolio import CenterHeatSimulation, CircularCutoutPoissonSimulation, CylinderFlowSimulation


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for simulation in (CircularCutoutPoissonSimulation(root), CenterHeatSimulation(root), CylinderFlowSimulation(root)):
        simulation.run()


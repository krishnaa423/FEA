from pathlib import Path

from fea_portfolio import CylinderFlowSimulation


if __name__ == "__main__":
    CylinderFlowSimulation(Path(__file__).resolve().parents[1]).run()


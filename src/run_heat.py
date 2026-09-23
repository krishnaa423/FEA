from pathlib import Path

from fea_portfolio import CenterHeatSimulation


if __name__ == "__main__":
    CenterHeatSimulation(Path(__file__).resolve().parents[1]).run()


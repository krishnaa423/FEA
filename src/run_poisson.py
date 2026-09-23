from pathlib import Path

from fea_portfolio import CircularCutoutPoissonSimulation


if __name__ == "__main__":
    CircularCutoutPoissonSimulation(Path(__file__).resolve().parents[1]).run()


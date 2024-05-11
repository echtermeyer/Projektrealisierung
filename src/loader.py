import pandas as pd

from pathlib import Path


class TripLoader:
    def __init__(self, dataset: str = "trips") -> None:
        if dataset not in ["trips"]:
            raise ValueError("Invalid dataset. Please choose from 'trips'.")

        self.ROOT = Path(__file__).parent.parent
        self.data = self.load(dataset)

    def load(self, dataset: str) -> pd.DataFrame:
        if dataset == "trips":
            return pd.read_csv(self.ROOT / f"src/data/tripfile.csv", delimiter=";")

    @property
    def trips(self):
        return self.data

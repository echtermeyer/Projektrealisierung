import pandas as pd

from pathlib import Path


class TripLoader:
    def __init__(self, dataset: str = "trips") -> None:
        valid_datasets = ["trips", "trips_ABCD", "trips_MNOP", "trips_ZYXW"]

        if dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset. Please choose from {valid_datasets}.")

        self.ROOT = Path(__file__).parent.parent
        self.data = self.load(dataset)

    def load(self, dataset: str) -> pd.DataFrame:
        file_paths = {
            "trips": "src/data/tripfile.csv",
            "trips_ABCD": "src/data/ABCD_tripfiles.csv",
            "trips_MNOP": "src/data/MNOP_tripfiles.csv",
            "trips_ZYXW": "src/data/ZYXW_tripfiles.csv",
        }
        return pd.read_csv(self.ROOT / file_paths[dataset], delimiter=";")

    @property
    def trips(self) -> pd.DataFrame:
        if hasattr(self, "_trips_data"):
            return self._trips_data
        self._trips_data = self.load("trips")
        return self._trips_data

    @property
    def trips_ABCD(self) -> pd.DataFrame:
        if hasattr(self, "_trips_ABCD_data"):
            return self._trips_ABCD_data
        self._trips_ABCD_data = self.load("trips_ABCD")
        return self._trips_ABCD_data

    @property
    def trips_MNOP(self) -> pd.DataFrame:
        if hasattr(self, "_trips_MNOP_data"):
            return self._trips_MNOP_data
        self._trips_MNOP_data = self.load("trips_MNOP")
        return self._trips_MNOP_data

    @property
    def trips_ZYXW(self) -> pd.DataFrame:
        if hasattr(self, "_trips_ZYXW_data"):
            return self._trips_ZYXW_data
        self._trips_ZYXW_data = self.load("trips_ZYXW")
        return self._trips_ZYXW_data

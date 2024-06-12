import csv
import pandas as pd

from pathlib import Path

from src.preprocessing import Preprocessor, CSV_Cleaner


class TripLoader:
    def __init__(self) -> None:
        self.ROOT = Path(__file__).parent.parent
        self.file_paths = {
            "trips": "src/data/tripfile.csv",
            "trips_ABCD": "src/data/ABCD_tripfiles.csv",
            "trips_MNOP": "src/data/MNOP_tripfiles.csv",
            "trips_ZYXW": "src/data/ZYXW_tripfiles.csv",
        }

        self.csv_cleaner = CSV_Cleaner()
        self.preprocessor = Preprocessor(self.ROOT)

    def __load(self, dataset: str, delimiter: str = ",") -> pd.DataFrame:
        unprocessed_file = self.ROOT / self.file_paths[dataset]

        processed_file = self.ROOT / self.file_paths[dataset]
        processed_file = processed_file.with_name(
            processed_file.stem + "_preprocessed" + processed_file.suffix
        )
        processed_file = self.ROOT / processed_file

        if processed_file.exists():
            return pd.read_csv(processed_file)

        print("#" * 15)
        print("Cleaning CSV file (this might take a while)...")
        self.csv_cleaner.fix_csv(unprocessed_file)

        unprocessed_df = pd.read_csv(
            self.ROOT / self.file_paths[dataset], delimiter=delimiter
        )
        processed_df = self.preprocessor.preprocess(unprocessed_df, dataset)
        processed_df.to_csv(processed_file, index=False)

        return processed_df

    @property
    def trips(self) -> pd.DataFrame:
        if hasattr(self, "_trips_data"):
            return self._trips_data
        self._trips_data = self.__load("trips", delimiter=";")
        return self._trips_data

    @property
    def trips_ABCD(self) -> pd.DataFrame:
        if hasattr(self, "_trips_ABCD_data"):
            return self._trips_ABCD_data

        self._trips_ABCD_data = self.__load("trips_ABCD")
        return self._trips_ABCD_data

    @property
    def trips_MNOP(self) -> pd.DataFrame:
        if hasattr(self, "_trips_MNOP_data"):
            return self._trips_MNOP_data

        self._trips_MNOP_data = self.__load("trips_MNOP")
        return self._trips_MNOP_data

    @property
    def trips_ZYXW(self) -> pd.DataFrame:
        if hasattr(self, "_trips_ZYXW_data"):
            return self._trips_ZYXW_data

        self._trips_ZYXW_data = self.__load("trips_ZYXW")
        return self._trips_ZYXW_data
